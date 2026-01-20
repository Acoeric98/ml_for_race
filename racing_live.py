# racing_live.py
# Visual NEAT training with GPU-accelerated batched raycasting (CuPy fallback to CPU).
# HUD: speedometer, current agent ID/score; right-side leaderboard + minimap
# Start-screen: choose New population OR Load best genome (.pkl) with file dialog; inject N clones in gen 0.
#
# Run:
#   python racing_live.py --config neat_config.ini --gens 20 --pop 120 --seed 424242 --fps 60
# Keys in run:
#   + / -  zoom
#   F      toggle follow Leader vs manual focus
#   TAB    next alive (manual focus mode)
#   ESC    quit

import math, argparse, os, copy, numpy as np
import neat
import pygame
import pickle

# Environment module (does NOT init pygame on import)
from racing import Car, generate_urban_gp, nearest_point_on_polyline, prepare_segments, raycast_batch

# ------------ optional native file dialog (tkinter) ------------
def pick_pkl_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select best genome .pkl",
            filetypes=[("Pickle files","*.pkl"), ("All files","*.*")]
        )
        root.destroy()
        return path or None
    except Exception:
        return None

# ---------------- seed helpers ----------------
def maybe_load_best(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # accept either {"genome": g, "config": cfg} or {"gen":..,"fitness":..,"genome":g,"config":cfg}
        return payload.get("genome", payload)
    except Exception as e:
        print("[SEED] Failed to load best:", e)
        return None

def inject_best_clones(pop, best_g, n_clones):
    if (best_g is None) or (n_clones <= 0):
        return
    keys = list(pop.population.keys())
    n = min(n_clones, len(keys))
    for k in keys[:n]:
        gnew = copy.deepcopy(best_g)
        gnew.key = k
        gnew.fitness = None
        pop.population[k] = gnew

# ---------------- GUI: start screen ----------------
def start_screen(screen, font, W, H, default_clones):
    """
    Returns (mode, best_path, clones)
    mode: 'new' or 'load'
    best_path: str or None
    clones: int (used only when mode == 'load')
    """
    clock = pygame.time.Clock()
    small = pygame.font.SysFont("consolas", 18)
    big   = pygame.font.SysFont("consolas", 36)

    mode = None
    best_path = None
    clones = max(1, int(default_clones))

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit
                if mode is None:
                    if e.key == pygame.K_n:
                        return ('new', None, 0)
                    if e.key == pygame.K_l:
                        # choose file
                        best_path = pick_pkl_file()
                        if best_path:
                            mode = 'load'
                else:
                    # adjust clones and confirm
                    if e.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                        clones = min(999, clones + 1)
                    elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        clones = max(1, clones - 1)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return ('load', best_path, clones)

        screen.fill((15,18,24))
        title = big.render("NEAT Urban GP — Start", True, (230,230,230))
        screen.blit(title, (W//2 - title.get_width()//2, 80))

        if mode is None:
            lines = [
                "Press  N  →  New population",
                "Press  L  →  Load best genome (.pkl)…",
                "",
                "ESC to quit"
            ]
            y = 160
            for s in lines:
                r = small.render(s, True, (200,200,200))
                screen.blit(r, (W//2 - r.get_width()//2, y)); y += 30
        else:
            # load mode: show file and let user set clones
            ptxt = small.render(f"Selected: {best_path}", True, (200,200,200))
            screen.blit(ptxt, (W//2 - min(ptxt.get_width(), W-80)//2, 160))
            r1 = small.render("Number of clones to inject in Generation 0:", True, (200,200,200))
            screen.blit(r1, (W//2 - r1.get_width()//2, 210))
            r2 = big.render(f"{clones}", True, (240,140,60))
            screen.blit(r2, (W//2 - r2.get_width()//2, 250))
            hint = small.render("[+/-] adjust   Enter: Start   ESC: Quit", True, (160,160,160))
            screen.blit(hint, (W//2 - hint.get_width()//2, 310))

        pygame.display.flip()
        clock.tick(60)

# ---------------- Observation ----------------
def build_observation(car, center, widths, left, right, rays_deg, seg_pack, max_range=150.0):
    pos = np.array([car.x, car.y], dtype=np.float32)
    p_closest, i, t, d2 = nearest_point_on_polyline(center, pos)
    tangent = center[(i+1)%len(center)] - center[(i-1)%len(center)]
    tangent = tangent / (np.linalg.norm(tangent)+1e-9)
    forward = np.array([math.cos(car.heading), math.sin(car.heading)], dtype=np.float32)
    heading_err = np.arctan2(forward[0]*tangent[1]-forward[1]*tangent[0],
                             forward[0]*tangent[0]+forward[1]*tangent[1]) / math.pi
    lateral = (np.cross(tangent, pos - p_closest)) / (0.5*widths[i] + 1e-6)
    speed = np.clip(car.v/90.0, 0.0, 1.0)

    angs = np.asarray(rays_deg, dtype=np.float32) * (np.pi/180.0)
    dirs = np.stack([np.cos(car.heading + angs),
                     np.sin(car.heading + angs)], axis=1).astype(np.float32)
    pts = np.repeat(pos[None,:], len(rays_deg), axis=0)

    ax_np, ay_np, sx_np, sy_np = seg_pack
    ray_vals = raycast_batch(pts, dirs, ax_np, ay_np, sx_np, sy_np, max_range=max_range)
    return np.array(list(ray_vals) + [speed, heading_err, float(np.clip(lateral, -1.0, 1.0))], dtype=float)

# ---------------- Physics helper ----------------
def step_car(car, center, widths, left, right, steer, gas, brake, dt):
    pos = np.array([car.x, car.y])
    p_closest, ii, tt, d2 = nearest_point_on_polyline(center, pos)
    halfw = 0.5*widths[ii]
    dist = math.sqrt(d2)
    off = max(0.0, dist - halfw)
    car.grip = 1.0 / (1.0 + off*0.6)
    car.max_steer = math.radians(28) * car.grip
    car.drag = 0.10 * (1.0 + off*0.8)
    car.update(dt, gas, brake, steer)
    tangent = center[(ii+1)%len(center)] - center[(ii-1)%len(center)]
    tangent = tangent / (np.linalg.norm(tangent)+1e-9)
    vel = np.array([math.cos(car.heading), math.sin(car.heading)]) * car.v
    forward_dot = float(np.dot(vel, tangent))
    return max(0.0, forward_dot), off

# ---------------- Color helper ----------------
def lerp(a,b,t): return a + (b-a)*t
def fitness_color(f, fmin, fmax):
    t = 0.0 if fmax<=fmin else np.clip((f - fmin)/(fmax - fmin), 0.0, 1.0)
    if t < 0.5:
        u = t/0.5
        r = int(lerp(70, 60, u)); g = int(lerp(120, 220, u)); b = int(lerp(255, 120, u))
    else:
        u = (t-0.5)/0.5
        r = int(lerp(60, 250, u)); g = int(lerp(220, 230, u)); b = int(lerp(120, 80, u))
    return (r,g,b)

def draw_speedometer(screen, font, x, y, w, h, v, v_max=90.0):
    pygame.draw.rect(screen, (28,32,40), (x,y,w,h), border_radius=6)
    frac = max(0.0, min(1.0, v / v_max))
    pygame.draw.rect(screen, (240,140,60), (x+2, y+2, int((w-4)*frac), h-4), border_radius=6)
    txt = font.render(f"Speed {v:5.1f} m/s", True, (230,230,230))
    screen.blit(txt, (x+8, y+h//2 - txt.get_height()//2))

def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neat_config.ini")
    ap.add_argument("--gens", type=int, default=800)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--sim-time", type=float, default=20.0, help="seconds per generation (visual)")
    ap.add_argument("--fps", type=int, default=120)
    args = ap.parse_args()

    np.random.seed(args.seed)

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      args.config)
    cfg.pop_size = args.pop

    # Window & fonts (for start screen)
    W, H = 1400, 860
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("NEAT Urban GP — Live Training Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    font_small = pygame.font.SysFont("consolas", 16)

    # Start screen: choose mode
    mode, best_path, clones = start_screen(screen, font, W, H, default_clones=max(1, args.pop//10))
    print(f"[START] mode={mode}  best={best_path}  clones={clones}")

    # NEAT pop + reporters
    pop = neat.Population(cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Fixed track for the live session (randomly generated)
    T = generate_urban_gp(seed=args.seed)
    center = T["center"]; left = T["left"]; right = T["right"]; widths = T["widths"]
    seg_pack = prepare_segments(left, right)

    # World fit (leader viewport ~70% width)
    all_pts = np.vstack([left, right])
    mn = np.min(all_pts, axis=0); mx = np.max(all_pts, axis=0)
    world_c = (mn+mx)/2
    world_size = max(mx[0]-mn[0], mx[1]-mn[1])
    base_scale = 0.85 * min(W*0.7, H) / world_size
    leader_zoom = 1.15  # closer default

    # Sidebar layout
    sidebar = pygame.Rect(int(W*0.7)+12, 12, W - int(W*0.7) - 24, H - 24)
    board_rect = pygame.Rect(sidebar.x, sidebar.y, sidebar.w, int(sidebar.h*0.55))
    mm_rect    = pygame.Rect(sidebar.x, board_rect.bottom+8, sidebar.w, sidebar.bottom - (board_rect.bottom+8))
    def to_mm(p):
        sx = (p[0]-world_c[0]) * (mm_rect.w/world_size) + (mm_rect.x + mm_rect.w/2)
        sy = (p[1]-world_c[1]) * (mm_rect.h/world_size) + (mm_rect.y + mm_rect.h/2)
        return int(sx), int(sy)
    track_mm = [to_mm(p) for p in left], [to_mm(p) for p in right], [to_mm(p) for p in center]

    # ----- scoring -----
    V_REF  = 28.0
    K_FAST = 0.10
    K_SLOW = 0.10
    K_OFF  = 0.80
    K_FWD  = 0.095
    K_BRAKE= 0.50

    # ----- performance knobs -----
    VISIBLE_DOT_SKIP = 2
    DRAW_EVERY_N = 2
    TRACK_STRIDE = 3
    RAYS_DEG = [-85, -60, -35, -15, 0, 15, 35, 60, 85]  # 9 rays + 3 extras -> 12 inputs
    RAY_CACHE_HOLD = 3
    BATCH_SIZE = 40
    RAY_MAX_RANGE = 150.0

    # Early KO
    EARLY_KO_SPEED = 3.5
    EARLY_KO_TIME  = 3.8
    OFF_KO         = 8.0

    # View / follow
    auto_follow_leader = True
    view_idx = 0

    # >>> One-time injection at Generation 0 if user picked 'load'
    best_seed = maybe_load_best(best_path) if mode == 'load' else None
    if best_seed is not None and clones > 0:
        inject_best_clones(pop, best_seed, clones)
        print(f"[SEED] Injected {clones} clones from {best_path}")

    for gen in range(args.gens):
        genomes = list(pop.population.items())
        nets, cars = [], []
        scores = np.full(len(genomes), 10.0, dtype=float)
        laps = np.zeros(len(genomes), dtype=int)
        started = np.zeros(len(genomes), dtype=bool)
        active = np.ones(len(genomes), dtype=bool)
        slow_timer = np.zeros(len(genomes), dtype=float)
        ray_cache = [None] * len(genomes)


        for idx, (gid, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            nets.append(net)
            p0, p1 = center[0], center[1]
            heading0 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
            cars.append(Car((p0[0], p0[1]), heading0))

        dt = 1.0 / args.fps
        frames = int(args.sim_time * args.fps)

        for frame in range(frames):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); return
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_k, pygame.K_KP_PLUS, pygame.K_EQUALS):
                        leader_zoom = min(18.0, leader_zoom * 1.07)
                    elif e.key in (pygame.K_l, pygame.K_KP_MINUS):
                        leader_zoom = max(0.6, leader_zoom / 1.07)
                    elif e.key == pygame.K_f:
                        auto_follow_leader = not auto_follow_leader
                    elif e.key == pygame.K_TAB:
                        auto_follow_leader = False
                        if len(genomes) > 0:
                            start = (view_idx + 1) % len(genomes)
                            for k in range(len(genomes)):
                                i = (start + k) % len(genomes)
                                if active[i]:
                                    view_idx = i; break

            N = len(genomes)
            start_idx = (frame * BATCH_SIZE) % N
            count = min(BATCH_SIZE, N)
            step_indices = [(start_idx + k) % N for k in range(count)]

            for i in step_indices:
                if not active[i]: continue
                car = cars[i]

                use_cache = (frame % RAY_CACHE_HOLD) != 0 and (ray_cache[i] is not None)
                if use_cache:
                    obs = ray_cache[i]
                else:
                    obs = build_observation(car, center, widths, left, right, RAYS_DEG, seg_pack, max_range=RAY_MAX_RANGE)
                    ray_cache[i] = obs

                # ------- INPUT SIZE COMPAT FIX --------
                # A NEAT háló elvárt inputszámát használjuk (régi genome-oknál eltérhet).
                expected_inputs = len(nets[i].input_nodes)
                if len(obs) > expected_inputs:
                    obs = obs[:expected_inputs]
                elif len(obs) < expected_inputs:
                    # nullával kipótlás
                    pad = expected_inputs - len(obs)
                    obs = np.pad(obs, (0, pad), mode='constant', constant_values=0.0)
                # --------------------------------------

                steer, gas, brake = nets[i].activate(obs)
                steer = float(np.clip(steer, -1.0, 1.0))
                gas   = float(np.clip((gas+1)/2, 0.0, 1.0))
                brake = float(np.clip((brake+1)/2, 0.0, 1.0))

                v_pre = car.v
                gas = min(1.0, gas * (1.20 if v_pre < V_REF else 1.05))

                fwd, off = step_car(car, center, widths, left, right, steer, gas, brake, dt)

                v = car.v
                if v >= V_REF: scores[i] += (v - V_REF) * K_FAST * dt
                else:          scores[i] -= (V_REF - v) * K_SLOW * dt
                scores[i] += fwd * K_FWD * dt
                if off > 0.0:  scores[i] -= off * K_OFF * dt
                scores[i] -= brake * K_BRAKE * dt

                # laps
                pos = np.array([car.x, car.y])
                a, b = center[0], center[1]
                def side(p,a,b): return np.cross(b-a, p-a)
                if side(pos, a, b) < 0 and side(np.array([car.x,car.y]), a, b) >= 0:
                    if started[i]: laps[i] += 1
                    started[i] = True

                # early KO
                if v < EARLY_KO_SPEED: slow_timer[i] += dt
                else:                  slow_timer[i] = max(0.0, slow_timer[i] - dt*0.5)
                if slow_timer[i] >= EARLY_KO_TIME or off >= OFF_KO:
                    active[i] = False
                    continue

            if int(np.sum(active)) == 0: break

            fmin, fmax = float(np.min(scores)), float(np.max(scores))
            leader_idx = int(np.argmax(scores))
            if auto_follow_leader: view_idx = leader_idx
            else:
                if not active[view_idx]:
                    for k in range(len(genomes)):
                        j = (view_idx + 1 + k) % len(genomes)
                        if active[j]: view_idx = j; break

            pygame.event.pump()
            if frame % DRAW_EVERY_N != 0:
                clock.tick(args.fps); continue

            # ------- draw -------
            screen.fill((15,18,24))
            focus = cars[view_idx]
            lookahead = min(80.0, 20.0 + 0.6*focus.v)
            cam = (focus.x + math.cos(focus.heading)*lookahead,
                   focus.y + math.sin(focus.heading)*lookahead)
            scale = base_scale * leader_zoom

            def w2s(p):
                return int((p[0]-cam[0])*scale + W*0.35), int((p[1]-cam[1])*scale + H*0.5)

            pygame.draw.aalines(screen, (50,180,220), True, [w2s(p) for p in left[::TRACK_STRIDE]], True)
            pygame.draw.aalines(screen, (50,180,220), True, [w2s(p) for p in right[::TRACK_STRIDE]], True)
            pygame.draw.aalines(screen, (90,90,120), True, [w2s(p) for p in center[::TRACK_STRIDE]], True)
            pygame.draw.line(screen, (255,255,255), w2s(center[0]), w2s(center[1]), 2)

            # focused car (poly car, nincs szükség sprite-ra)
            Lpix = 4.2*scale*1.1; Wpix = 1.9*scale*1.1
            cx, cy = w2s((focus.x, focus.y))
            ang = focus.heading; ca, sa = math.cos(ang), math.sin(ang)
            corners = []
            for dx, dy in [( Lpix/2,  Wpix/2), ( Lpix/2, -Wpix/2),
                           (-Lpix/2, -Wpix/2), (-Lpix/2,  Wpix/2)]:
                x = cx + dx*ca - dy*sa
                y = cy + dx*sa + dy*ca
                corners.append((x,y))
            pygame.draw.polygon(screen, (240,140,60), corners)

            # leaderboard
            pygame.draw.rect(screen, (25,28,36), board_rect, border_radius=8)
            title = font.render("Leaderboard (score)", True, (230,230,230))
            screen.blit(title, (board_rect.x+10, board_rect.y+8))
            order = list(np.argsort(-scores))
            rows = min(len(order), 14)
            y0 = board_rect.y + 36
            for r in range(rows):
                i = order[r]; gid = genomes[i][0]; sc = scores[i]
                txt = f"{r+1:2d}.  ID {gid:<6}  {sc:7.2f}  {'ALIVE' if active[i] else 'KO'}"
                col = fitness_color(sc, fmin, fmax)
                if i == view_idx:
                    pygame.draw.rect(screen, (45,52,64), (board_rect.x+6, y0-2, board_rect.w-12, 22), border_radius=4)
                screen.blit(font_small.render(txt, True, col), (board_rect.x+12, y0))
                y0 += 22

            # minimap
            pygame.draw.rect(screen, (25,28,36), mm_rect, border_radius=8)
            pygame.draw.aalines(screen, (60,200,240), True, track_mm[0], True)
            pygame.draw.aalines(screen, (60,200,240), True, track_mm[1], True)
            pygame.draw.aalines(screen, (110,110,140), True, track_mm[2], True)
            for i, car in enumerate(cars):
                if not active[i]: continue
                if i % VISIBLE_DOT_SKIP != 0: continue
                col = fitness_color(scores[i], fmin, fmax)
                x, y = to_mm((car.x, car.y))
                pygame.draw.circle(screen, col, (x,y), 3)
            fx, fy = to_mm((focus.x, focus.y))
            pygame.draw.circle(screen, (255,255,255), (fx,fy), 5, 2)

            # HUD
            alive = int(np.sum(active))
            hud1 = f"Gen {gen+1}/{args.gens}   Pop {args.pop} (alive {alive})   Leader score {fmax:6.2f}   Laps {int(np.max(laps))}"
            pygame.display.set_caption(hud1)
            screen.blit(font.render(hud1, True, (230,230,230)), (12,12))
            gid_focus = genomes[view_idx][0]
            hud2 = f"Zoom {leader_zoom:.2f}   View: {'LEADER' if auto_follow_leader else f'ID {gid_focus}'}   Controls: + / - zoom, F follow, TAB next, ESC quit"
            screen.blit(font.render(hud2, True, (160,160,160)), (12,34))
            draw_speedometer(screen, font, 14, H-74, 280, 32, focus.v, v_max=90.0)
            box = pygame.Rect(310, H-74, 220, 32)
            pygame.draw.rect(screen, (28,32,40), box, border_radius=6)
            sc_now = scores[view_idx]
            screen.blit(font.render(f"ID {gid_focus}   Score {sc_now:6.2f}", True, (230,230,230)), (box.x+10, box.y+6))

            pygame.display.flip()
            clock.tick(args.fps)

        # assign fitness & save best
        for i, (gid, genome) in enumerate(genomes):
            genome.fitness = float(max(0.0, scores[i]))

        best_idx = int(np.argmax(scores))
        best_genome = genomes[best_idx][1]

        os.makedirs('runs/live_best', exist_ok=True)
        if not hasattr(pop, '_best_live_fitness'):
            pop._best_live_fitness = -1e18
        fit = float(best_genome.fitness or -1e18)
        if fit >= pop._best_live_fitness:
            pop._best_live_fitness = fit
            payload = {"gen": pop.generation, "fitness": fit, "genome": best_genome, "config": cfg}
            fname = os.path.join('runs/live_best', f"best_gen_{pop.generation:04d}_{fit:.2f}.pkl")
            with open(fname, 'wb') as f: pickle.dump(payload, f)
            with open(os.path.join('runs/live_best', 'best_so_far.pkl'), 'wb') as f: pickle.dump(payload, f)
            print(f"[LIVE] Saved best-so-far @ gen {pop.generation} fitness={fit:.2f} -> {fname}")


        for gid, genome in pop.population.items():
            if genome.fitness is None:
                genome.fitness = 0.0

        # Ugyanez minden faj tagjára
        for sid, species in pop.species.species.items():
            for gid, genome in species.members.items():
                if genome.fitness is None:
                    genome.fitness = 0.0
        # evolve
        pop.reporters.post_evaluate(cfg, pop.population, pop.species, best_genome)
        pop.population = pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
        pop.species.speciate(cfg, pop.population, pop.generation)
        pop.generation += 1

    pygame.quit()

if __name__ == "__main__":
    main()
