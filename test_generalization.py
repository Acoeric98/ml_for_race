# test_generalization.py
# Evaluate a trained NEAT agent on unseen random tracks.
# Example:
#   python test_generalization.py --genome runs/urban_gp/neat_winner.pkl --tests 10 --seed-offset 1000000

import argparse
import math
import pickle
import numpy as np
import neat

from racing import (
    Car,
    generate_urban_gp,
    nearest_point_on_polyline,
    prepare_segments,
    raycast_batch,
    lookahead_centerline_points,
)

RAYS_DEG = [-60, -35, -15, 0, 15, 35, 60]
N_LOOKAHEAD = 8
LOOKAHEAD_STRIDE = 5
LOOKAHEAD_SCALE = 50.0
OFF_KO = 8.0


def build_observation(car, center, widths, left, right, seg_pack):
    pos = np.array([car.x, car.y], dtype=np.float32)
    p_closest, i, t, d2 = nearest_point_on_polyline(center, pos)
    tangent = center[(i + 1) % len(center)] - center[(i - 1) % len(center)]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    forward = np.array([math.cos(car.heading), math.sin(car.heading)], dtype=np.float32)
    heading_err = np.arctan2(
        forward[0] * tangent[1] - forward[1] * tangent[0],
        forward[0] * tangent[0] + forward[1] * tangent[1],
    ) / math.pi
    lateral = (np.cross(tangent, pos - p_closest)) / (0.5 * widths[i] + 1e-6)
    speed = np.clip(car.v / 90.0, 0.0, 1.0)

    angs = np.asarray(RAYS_DEG, dtype=np.float32) * (np.pi / 180.0)
    dirs = np.stack(
        [np.cos(car.heading + angs), np.sin(car.heading + angs)], axis=1
    ).astype(np.float32)
    pts = np.repeat(pos[None, :], len(RAYS_DEG), axis=0)

    ax_np, ay_np, sx_np, sy_np = seg_pack
    ray_vals = raycast_batch(pts, dirs, ax_np, ay_np, sx_np, sy_np, max_range=150.0)
    lookahead = lookahead_centerline_points(
        center,
        i,
        pos,
        car.heading,
        n_lookahead=N_LOOKAHEAD,
        stride=LOOKAHEAD_STRIDE,
        scale=LOOKAHEAD_SCALE,
    )
    obs = list(ray_vals) + list(lookahead) + [speed, heading_err, float(np.clip(lateral, -1.0, 1.0))]
    return np.array(obs, dtype=float), i


def simulate_track(net, track, max_time=20.0, dt=1 / 60.0):
    center = track["center"]
    left = track["left"]
    right = track["right"]
    widths = track["widths"]
    seg_pack = prepare_segments(left, right)

    p0, p1 = center[0], center[1]
    heading0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    car = Car((p0[0], p0[1]), heading0)

    progress = 0.0
    max_off = 0.0
    laps = 0
    started = False

    for _ in range(int(max_time / dt)):
        obs, idx = build_observation(car, center, widths, left, right, seg_pack)
        expected_inputs = len(net.input_nodes)
        if len(obs) > expected_inputs:
            obs = obs[:expected_inputs]
        elif len(obs) < expected_inputs:
            obs = np.pad(obs, (0, expected_inputs - len(obs)), mode="constant", constant_values=0.0)
        steer, gas, brake = net.activate(obs)
        steer = float(np.clip(steer, -1.0, 1.0))
        gas = float(np.clip((gas + 1) / 2, 0.0, 1.0))
        brake = float(np.clip((brake + 1) / 2, 0.0, 1.0))

        pos = np.array([car.x, car.y])
        p_closest, ii, tt, d2 = nearest_point_on_polyline(center, pos)
        halfw = 0.5 * widths[ii]
        dist = math.sqrt(d2)
        off = max(0.0, dist - halfw)
        max_off = max(max_off, off)

        car.grip = 1.0 / (1.0 + off * 0.6)
        car.max_steer = math.radians(28) * car.grip
        car.drag = 0.10 * (1.0 + off * 0.8)
        car.update(dt, gas, brake, steer)

        tangent = center[(ii + 1) % len(center)] - center[(ii - 1) % len(center)]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        vel = np.array([math.cos(car.heading), math.sin(car.heading)]) * car.v
        forward_dot = float(np.dot(vel, tangent))
        if forward_dot > 0.0:
            progress += forward_dot * dt

        a, b = center[0], center[1]

        def side(p, a, b):
            return np.cross(b - a, p - a)

        if side(pos, a, b) < 0 and side(np.array([car.x, car.y]), a, b) >= 0:
            if started:
                laps += 1
            started = True

        if off >= OFF_KO:
            break

    success = (laps >= 1) and (max_off < OFF_KO)
    return progress, max_off, laps, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neat_config.ini")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--tests", type=int, default=10)
    ap.add_argument("--seed-range", type=int, default=100000)
    ap.add_argument("--seed-offset", type=int, default=1000000)
    ap.add_argument("--max-time", type=float, default=20.0)
    args = ap.parse_args()

    cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    with open(args.genome, "rb") as f:
        payload = pickle.load(f)
    genome = payload.get("genome", payload) if isinstance(payload, dict) else payload
    net = neat.nn.FeedForwardNetwork.create(genome, cfg)

    rng = np.random.default_rng(args.seed_offset)
    seeds = rng.integers(args.seed_offset, args.seed_offset + args.seed_range, size=args.tests)

    results = []
    for seed in seeds:
        track = generate_urban_gp(seed=int(seed))
        metrics = simulate_track(net, track, max_time=args.max_time)
        results.append((int(seed),) + metrics)

    header = f"{'seed':>9} | {'progress':>10} | {'max_off':>7} | {'laps':>4} | {'success':>7}"
    print(header)
    print("-" * len(header))
    successes = 0
    for seed, progress, max_off, laps, success in results:
        successes += int(success)
        print(
            f"{seed:9d} | {progress:10.1f} | {max_off:7.2f} | {laps:4d} | {str(success):>7}"
        )

    rate = successes / max(1, len(results))
    print(f"\nSuccess rate: {successes}/{len(results)} = {rate:.1%}")


if __name__ == "__main__":
    main()
