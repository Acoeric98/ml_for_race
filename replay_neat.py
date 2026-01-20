
# replay_neat.py
# Visualize a trained NEAT agent with GPU batched raycasting.
# Run:
#   python replay_neat.py --genome runs/urban_gp/neat_winner.pkl --track runs/urban_gp/replay_track.json

import math, json, pickle, argparse
import numpy as np
import neat
import pygame

from racing import Car, nearest_point_on_polyline, prepare_segments, raycast_batch

def world_bbox(points_list):
    P = np.vstack(points_list)
    mn = np.min(P, axis=0); mx = np.max(P, axis=0)
    return mn, mx

def observe(car, center, widths, left, right, seg_pack):
    pos = np.array([car.x, car.y], dtype=np.float32)
    p_closest, i, t, d2 = nearest_point_on_polyline(center, pos)
    tangent = center[(i+1)%len(center)] - center[(i-1)%len(center)]
    tangent = tangent / (np.linalg.norm(tangent)+1e-9)
    forward = np.array([math.cos(car.heading), math.sin(car.heading)], dtype=np.float32)
    heading_err = np.arctan2(forward[0]*tangent[1]-forward[1]*tangent[0],
                             forward[0]*tangent[0]+forward[1]*tangent[1]) / math.pi
    lateral = (np.cross(tangent, pos - p_closest)) / (0.5*widths[i] + 1e-6)
    speed = np.clip(car.v/90.0, 0.0, 1.0)

    rays_deg = [-60, -35, -15, 0, 15, 35, 60]
    angs = np.asarray(rays_deg, dtype=np.float32) * (np.pi/180.0)
    dirs = np.stack([np.cos(car.heading + angs),
                     np.sin(car.heading + angs)], axis=1).astype(np.float32)
    pts = np.repeat(pos[None,:], len(rays_deg), axis=0)

    ax_np, ay_np, sx_np, sy_np = seg_pack
    ray_vals = raycast_batch(pts, dirs, ax_np, ay_np, sx_np, sy_np, max_range=150.0)

    # lines for debug viz (denormalize by 150m range)
    ray_lines = []
    for dval, ang in zip(ray_vals, angs):
        dlen = float(dval) * 150.0
        dvec = np.array([math.cos(car.heading + ang), math.sin(car.heading + ang)], dtype=np.float32)
        ray_lines.append((pos, pos + dvec*dlen))

    obs = np.array(list(ray_vals) + [speed, heading_err, float(np.clip(lateral, -1.0, 1.0))], dtype=float)
    return obs, i, ray_lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neat_config.ini")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--track", required=True)
    args = ap.parse_args()

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      args.config)
    with open(args.genome, "rb") as f:
        winner_payload = pickle.load(f)
    winner = winner_payload["genome"] if isinstance(winner_payload, dict) and "genome" in winner_payload else winner_payload
    net = neat.nn.FeedForwardNetwork.create(winner, cfg)

    with open(args.track, "r", encoding="utf-8") as f:
        T = json.load(f)
    center = np.array(T["center"], dtype=float)
    left   = np.array(T["left"], dtype=float)
    right  = np.array(T["right"], dtype=float)
    widths = np.array(T["widths"], dtype=float)
    seg_pack = prepare_segments(left, right)

    W, H = 1280, 800
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    mn, mx = world_bbox([left, right])
    world_c = (mn+mx)/2
    world_size = max(mx[0]-mn[0], mx[1]-mn[1])
    base_scale = 0.85 * min(W, H) / world_size

    # minimap
    mm_size = 220
    mm = pygame.Surface((mm_size, mm_size), pygame.SRCALPHA)
    def to_mm(p):
        sx = (p[0]-world_c[0]) * (mm_size/world_size) + mm_size/2
        sy = (p[1]-world_c[1]) * (mm_size/world_size) + mm_size/2
        return int(sx), int(sy)
    pygame.draw.aalines(mm, (60,200,240), True, [to_mm(p) for p in left], True)
    pygame.draw.aalines(mm, (60,200,240), True, [to_mm(p) for p in right], True)
    pygame.draw.aalines(mm, (110,110,140), True, [to_mm(p) for p in center], True)

    # spawn car
    p0, p1 = center[0], center[1]
    heading0 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    car = Car((p0[0], p0[1]), heading0)

    lap=0; started=False; progress=0.0
    zoom = 1.9; auto_zoom = True

    running=True
    while running:
        dt = clock.tick(60)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running=False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS): zoom *= 1.1
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS): zoom /= 1.1
                elif e.key == pygame.K_z: auto_zoom = not auto_zoom
                elif e.key == pygame.K_ESCAPE: running=False

        obs, idx, ray_lines = observe(car, center, widths, left, right, seg_pack)
        steer, gas, brake = net.activate(obs)
        steer = float(np.clip(steer, -1.0, 1.0))
        gas   = float(np.clip((gas+1)/2, 0.0, 1.0))
        brake = float(np.clip((brake+1)/2, 0.0, 1.0))

        pos = np.array([car.x, car.y])
        p_closest, ii, tt, d2 = nearest_point_on_polyline(center, pos)
        halfw = 0.5*widths[ii]
        dist = math.sqrt(d2)
        off = max(0.0, dist - halfw)
        car.grip = 1.0 / (1.0 + off*0.6)
        car.max_steer = math.radians(28) * car.grip
        car.drag = 0.18 * (1.0 + off*0.8)

        car.update(dt, gas, brake, steer)

        tangent = center[(ii+1)%len(center)] - center[(ii-1)%len(center)]
        tangent = tangent / (np.linalg.norm(tangent)+1e-9)
        vel = np.array([math.cos(car.heading), math.sin(car.heading)]) * car.v
        forward_dot = float(np.dot(vel, tangent))
        progress += max(0.0, forward_dot) * dt

        a, b = center[0], center[1]
        def side(p,a,b): return np.cross(b-a, p-a)
        if side(pos, a, b) < 0 and side(np.array([car.x,car.y]), a, b) >= 0:
            if started: lap += 1
            started = True

        # camera & zoom
        lookahead = min(80.0, 20.0 + 0.6*car.v)
        cam = (car.x + math.cos(car.heading)*lookahead,
               car.y + math.sin(car.heading)*lookahead)
        scale = base_scale
        if auto_zoom:
            target_zoom = 2.2 - (min(90.0, car.v)/90.0) * (2.2 - 0.9)
            zoom = zoom*0.90 + target_zoom*0.10
        zoom = max(0.6, min(3.5, zoom))
        scale *= zoom

        def w2s(p): return int((p[0]-cam[0])*scale + W/2), int((p[1]-cam[1])*scale + H/2)

        # draw
        screen.fill((15,18,24))
        pygame.draw.aalines(screen, (50,180,220), True, [w2s(p) for p in left], True)
        pygame.draw.aalines(screen, (50,180,220), True, [w2s(p) for p in right], True)
        pygame.draw.aalines(screen, (90,90,120), True, [w2s(p) for p in center], True)
        pygame.draw.line(screen, (255,255,255), w2s(center[0]), w2s(center[1]), 2)

        for a,bp in ray_lines:
            pygame.draw.line(screen, (255,220,120), w2s(a), w2s(bp), 1)

        car_len, car_w = 4.2, 1.9
        Lpix = car_len*scale*1.1; Wpix = car_w*scale*1.1
        cx, cy = w2s((car.x, car.y))
        ang = car.heading
        cosA, sinA = math.cos(ang), math.sin(ang)
        corners=[]
        for dx, dy in [( Lpix/2,  Wpix/2), ( Lpix/2, -Wpix/2),
                       (-Lpix/2, -Wpix/2), (-Lpix/2,  Wpix/2)]:
            x = cx + dx*cosA - dy*sinA
            y = cy + dx*sinA + dy*cosA
            corners.append((x,y))
        pygame.draw.polygon(screen, (240,140,60), corners)

        # minimap
        screen.blit(mm, (W - mm_size - 12, 12))
        car_mm = (int((car.x-world_c[0])*(mm_size/world_size) + (W-mm_size-12) + mm_size/2),
                  int((car.y-world_c[1])*(mm_size/world_size) + 12 + mm_size/2))
        pygame.draw.circle(screen, (255,80,80), car_mm, 3)

        # HUD
        screen.blit(pygame.font.SysFont('consolas',18).render(
            f"Speed: {car.v:5.1f} m/s   Lap: {lap}   Progress: {progress:7.1f}", True, (230,230,230)), (12,12))
        screen.blit(pygame.font.SysFont('consolas',18).render(
            f"Zoom: {zoom:.2f}  AutoZoom: {'ON' if auto_zoom else 'OFF'}   [+/-] Zoom, Z Toggle, ESC Quit", True, (160,160,160)), (12,34))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
