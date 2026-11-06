"""Adaptive racing and shared-world NEAT training. See README.md."""
import argparse
import configparser
import json
import math
from pathlib import Path
import pickle
import random
import tempfile

import neat
import numpy as np
from racing import Car, curvature, generate_urban_gp, prepare_segments
from train_neat import build_observation

INPUTS = 36  # existing 26, two opponents x 4, target speed, learning phase
DT = 1 / 30


class TrackMemory:
    """Per-car, per-race speed limits. Only visited sectors are refined."""
    def __init__(self, track):
        self.center = np.asarray(track['center'])
        self.widths = np.asarray(track['widths'])
        self.steps = np.linalg.norm(np.roll(self.center, -1, axis=0) - self.center, axis=1)
        self.length = float(self.steps.sum())
        self.limits = np.clip(np.sqrt(5.0 / np.maximum(curvature(self.center), .001)), 4, 45)
        self.visits = np.zeros(len(self.center), dtype=int)
        self.laps = 0
        self.distance = 0.0
        self.last_index = None
        self.lap_times = []
        self.last_lap_time = 0.0

    def observe(self, index, speed, off, contact, elapsed):
        if self.last_index is not None:
            delta = (index - self.last_index + len(self.center)//2) % len(self.center) - len(self.center)//2
            # Signed progress: reversing or oscillating cannot manufacture laps.
            if abs(delta) < len(self.center) / 4:
                indices = (self.last_index + np.arange(abs(delta))) % len(self.center) if delta >= 0 else (index + np.arange(-delta)) % len(self.center)
                self.distance += float(self.steps[indices].sum()) * (1 if delta >= 0 else -1)
        self.last_index = index
        while self.distance >= (self.laps + 1) * self.length:
            self.laps += 1
            self.lap_times.append(elapsed - self.last_lap_time)
            self.last_lap_time = elapsed
        # Refine only at sector entry, rather than once per frame.
        if self.visits[index] == 0 or off > 0 or contact:
            if off > 0 or contact:
                self.limits[index] = max(3., min(self.limits[index] * .85, max(3., speed * .9)))
        self.visits[index] += 1

    def target(self, index):
        # Conservative survey, refinement, then learned race profile.
        factor = (.65, .82, 1.)[min(self.laps, 2)]
        best, distance = 45., 0.
        for k in range(min(40, len(self.center))):
            j = (index + k) % len(self.center)
            best = min(best, math.sqrt((self.limits[j] * factor)**2 + 2 * 7 * distance))
            distance += self.steps[j]
        return best


def opponent_observation(car, others, sensing_range=80.):
    """Nearest two cars: local x/y, relative forward speed, presence flag."""
    forward = np.array([math.cos(car.heading), math.sin(car.heading)])
    side = np.array([-forward[1], forward[0]])
    nearby = sorted(((math.hypot(o.x-car.x, o.y-car.y), o) for o in others if o is not car), key=lambda p: p[0])
    out = []
    for distance, other in nearby:
        if distance > sensing_range:
            continue
        delta = np.array([other.x-car.x, other.y-car.y])
        velocity = other.v * np.array([math.cos(other.heading), math.sin(other.heading)]) - car.v * forward
        out.extend([float(delta @ forward / sensing_range), float(delta @ side / sensing_range), float(np.clip(velocity @ forward / 45, -1, 1)), 1.])
        if len(out) == 8:
            break
    return np.array(out + [0.] * (8-len(out)))


def resolve_collisions(cars, radius=1.3):
    """Equal-mass inelastic disc contacts; returns per-car contact flags.

    Simulation substeps prevent cars crossing one another between contact checks.
    This intentionally simple model is not rigid-body vehicle physics.
    """
    contacts = [False] * len(cars)
    for i, a in enumerate(cars):
        for j in range(i+1, len(cars)):
            b = cars[j]
            delta = np.array([b.x-a.x, b.y-a.y])
            distance = float(np.linalg.norm(delta))
            if distance >= 2*radius:
                continue
            normal = delta / distance if distance > 1e-9 else np.array([1., 0.])
            correction = normal * (2*radius-distance + 1e-6)/2
            a.x -= correction[0]; a.y -= correction[1]
            b.x += correction[0]; b.y += correction[1]
            fa = np.array([math.cos(a.heading), math.sin(a.heading)])
            fb = np.array([math.cos(b.heading), math.sin(b.heading)])
            va, vb = fa*a.v, fb*b.v
            closing = float((vb-va) @ normal)
            if closing < 0:
                va += closing*.5*normal
                vb -= closing*.5*normal
                a.v = max(0., float(va @ fa))
                b.v = max(0., float(vb @ fb))
            contacts[i] = contacts[j] = True
    return contacts


class Race:
    def __init__(self, track, networks, collisions=False):
        self.track = track
        self.center = np.asarray(track['center'])
        self.segments = prepare_segments(track['left'], track['right'])
        self.networks = networks
        self.collisions = collisions
        self.cars, self.memories = [], []
        self.elapsed = 0.
        self.contacts = np.zeros(len(networks), dtype=int)
        self.off_time = np.zeros(len(networks))
        self.previous_contacts = np.zeros(len(networks), dtype=bool)
        for i in range(len(networks)):
            # Stagger by arc distance so initial grid does not overlap.
            distance, index = 0., 0
            steps = np.linalg.norm(np.roll(self.center,-1,axis=0)-self.center,axis=1)
            while distance < i*7:
                index = (index-1) % len(self.center)
                distance += steps[index]
            p = self.center[index]
            tangent = self.center[(index+1)%len(self.center)] - p
            self.cars.append(Car(p, math.atan2(tangent[1], tangent[0])))
            memory = TrackMemory(track)
            memory.last_index = index
            self.memories.append(memory)

    def step(self, dt=DT):
        controls = []
        for i, (car, memory, net) in enumerate(zip(self.cars, self.memories, self.networks)):
            index = int(np.argmin(np.sum((self.center-[car.x,car.y])**2, axis=1)))
            off = max(0., float(np.linalg.norm(self.center[index]-[car.x,car.y])) - self.track['widths'][index]/2)
            memory.observe(index, car.v, off, self.previous_contacts[i], self.elapsed)
            self.off_time[i] += dt * (off > 0)
            target = memory.target(index)
            look_distance = max(6., car.v*.65)
            look, traveled = index, 0.
            while traveled < look_distance:
                traveled += memory.steps[look]
                look = (look+1) % len(self.center)
            delta = self.center[look] - [car.x,car.y]
            angle = math.atan2(delta[1],delta[0]) - car.heading
            wheel = math.atan2(2*2.6*math.sin(angle), max(float(np.linalg.norm(delta)), 1.))
            steering_scale = car.max_steer * (.55+.45*max(0.,1-car.v/90))
            steer = wheel / max(steering_scale, .01)
            opponents = opponent_observation(car, self.cars)
            residual = np.zeros(3)
            if net is not None:
                obs = np.concatenate([build_observation(car,self.center,self.track['widths'],self.track['left'],self.track['right'],self.segments), opponents, [target/45, min(memory.laps,2)/2]])
                residual = np.clip(net.activate(obs), -1, 1)
            # Learned residuals adjust line and speed, while survey stays conservative.
            steer += .25*residual[0]
            target = max(2., target*(1+.2*residual[1]))
            # Basic emergency following; network still sees opponents and can steer.
            for k in (0,4):
                x,y,_,present = opponents[k:k+4]
                if present and 0 < x*80 < 6+car.v*.7 and abs(y*80) < 2.8:
                    target = min(target, max(0., (x*80-3)*.6))
            error = target-car.v
            gas = float(np.clip(error*.5,0,1))
            brake = float(np.clip(-error*.3 + max(0.,residual[2])*.2,0,1))
            car.drag = .10*(1+off*.8)
            controls.append((gas,brake,float(np.clip(steer,-1,1))))
        # Fixed high-frequency contact checks, also bounded by movement distance.
        substeps = max(4, math.ceil(max((c.v for c in self.cars),default=0)*dt/.5))
        touched = np.zeros(len(self.cars), dtype=bool)
        for _ in range(substeps):
            for car, (gas,brake,steer) in zip(self.cars,controls):
                car.update(dt/substeps,gas,brake,steer)
            if self.collisions:
                touched |= resolve_collisions(self.cars)
        self.contacts += touched & ~self.previous_contacts
        self.previous_contacts = touched
        self.elapsed += dt

    def results(self):
        return [dict(distance=m.distance,laps=m.laps,lap_times=m.lap_times,contacts=int(self.contacts[i]),off_time=float(self.off_time[i]),fitness=m.distance - self.off_time[i]*20-self.contacts[i]*25) for i,m in enumerate(self.memories)]


def load_config(path, population=None):
    parser = configparser.ConfigParser()
    parser.read(path)
    parser['DefaultGenome']['num_inputs'] = str(INPUTS)
    if population is not None:
        parser['NEAT']['pop_size'] = str(population)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.ini') as f:
        parser.write(f); f.flush()
        return neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,f.name)


def run_race(race, seconds, render=False):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    if render:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((1100,800))
        clock = pygame.time.Clock()
        font = pygame.font.Font(None,24)
        lo,hi = race.center.min(axis=0),race.center.max(axis=0)
        scale = min(1000/max(hi[0]-lo[0],1),680/max(hi[1]-lo[1],1))
        def point(p):
            return tuple(((np.asarray(p)-lo)*scale+[40,70]).astype(int))
    try:
        for _ in range(math.ceil(seconds/DT)):
            if render and any(e.type == pygame.QUIT for e in pygame.event.get()):
                break
            race.step()
            if render:
                screen.fill((15,18,24))
                for edge in ('left','right'):
                    pygame.draw.lines(screen,(60,180,200),True,[point(p) for p in race.track[edge]],2)
                for i,c in enumerate(race.cars):
                    pygame.draw.circle(screen,((80+i*67)%255,180,(240-i*37)%255),point([c.x,c.y]),5)
                info=' | '.join(f'{i}: lap {m.laps} contacts {race.contacts[i]}' for i,m in enumerate(race.memories))
                screen.blit(font.render(info,True,(255,255,255)),(12,12))
                pygame.display.flip(); clock.tick(30)
    finally:
        if render:
            pygame.quit()
    return race.results()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['train','race'])
    ap.add_argument('--config',default=str(Path(__file__).with_name('neat_config.ini')))
    ap.add_argument('--seed',type=int,default=424242)
    ap.add_argument('--track',help='JSON containing center, left, right, widths (otherwise generated)')
    ap.add_argument('--seconds',type=float,default=180)
    ap.add_argument('--collisions',action='store_true')
    ap.add_argument('--cars',type=int,default=4)
    ap.add_argument('--pop',type=int,default=24)
    ap.add_argument('--gens',type=int,default=20)
    ap.add_argument('--tracks-per-gen',type=int,default=3)
    ap.add_argument('--genome',help='Trusted adaptive_race model pickle only')
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--outdir',default='runs/adaptive')
    args=ap.parse_args()
    if args.cars < 1 or args.pop < 2 or args.gens < 1 or args.tracks_per_gen < 1 or args.seconds <= 0 or not math.isfinite(args.seconds):
        ap.error('cars/tracks/gens/seconds must be positive, pop at least 2')
    random.seed(args.seed); np.random.seed(args.seed)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    def track(seed):
        if args.track:
            data=json.loads(Path(args.track).read_text())
            data={k:np.asarray(data[k],dtype=float) for k in ('center','left','right','widths')}
            n=len(data['center'])
            if n < 4 or any(data[k].shape != (n,2) for k in ('center','left','right')) or data['widths'].shape != (n,) or any(not np.isfinite(v).all() for v in data.values()) or np.any(data['widths']<=0) or np.any(np.linalg.norm(np.roll(data['center'],-1,axis=0)-data['center'],axis=1)<1e-6):
                raise ValueError('Invalid track geometry')
            return data
        return generate_urban_gp(seed=seed)
    if args.mode == 'race':
        nets=[None]*args.cars
        if args.genome:
            with open(args.genome,'rb') as f:
                payload=pickle.load(f)
            cfg=payload['config']
            if cfg.genome_config.num_inputs != INPUTS:
                ap.error('Model requires adaptive_race 36-input format; legacy models are incompatible')
            nets=[neat.nn.FeedForwardNetwork.create(payload['genome'],cfg) for _ in nets]
        results=run_race(Race(track(args.seed),nets,args.collisions),args.seconds,args.render)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))
    else:
        cfg=load_config(args.config,args.pop)
        pop=neat.Population(cfg)
        pop.add_reporter(neat.StdOutReporter(False))
        generation=0
        def evaluate(genomes,config):
            nonlocal generation
            scores={gid:[] for gid,_ in genomes}
            # Same tracks per generation for fair comparison, new tracks next generation.
            for t in range(args.tracks_per_gen):
                circuit=track(args.seed+generation*1009+t)
                order=list(genomes)
                random.shuffle(order)
                for start in range(0,len(order),args.cars):
                    batch=order[start:start+args.cars]
                    nets=[neat.nn.FeedForwardNetwork.create(g,config) for _,g in batch]
                    results=run_race(Race(circuit,nets,args.collisions),args.seconds)
                    for (gid,_),result in zip(batch,results):
                        scores[gid].append(result['fitness'])
            for gid,g in genomes:
                g.fitness=float(np.mean(scores[gid]))
            generation+=1
        winner=pop.run(evaluate,args.gens)
        with (out/'adaptive_winner.pkl').open('wb') as f:
            pickle.dump(dict(genome=winner,config=cfg,format='adaptive-race-v1'),f)
        print('Saved',out/'adaptive_winner.pkl')


if __name__ == '__main__':
    main()
