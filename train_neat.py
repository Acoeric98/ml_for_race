
# train_neat.py
# Headless NEAT training with aggressive scoring and GPU batched raycasting (CuPy fallback to CPU).
# Saves: neat_winner.pkl, replay_track.json, stats.csv
# Run examples:
#   python train_neat.py --gens 40 --pop 120 --seed 424242 --fixed-track --outdir runs/urban_gp
#   python train_neat.py --gens 60 --pop 160 --parallel --outdir runs/urban_gp_fast

import os, time, math, json, csv, argparse, pickle
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

# ---- Scoring constants (aggressive) ----
V_REF  = 28.0   # ~100+ km/h target
K_FAST = 0.10   # reward above V_REF
K_SLOW = 0.22   # penalty below V_REF
K_OFF  = 0.50   # off-track penalty
K_FWD  = 0.025  # forward progress reward
K_BRAKE= 0.50   # braking penalty

# ---- Early KO thresholds ----
EARLY_KO_SPEED = 2.5   # m/s
EARLY_KO_TIME  = 1.5   # s
OFF_KO         = 8.0   # m

# ---- Sensor set ----
RAYS_DEG = [-60, -35, -15, 0, 15, 35, 60]  # 7 rays + lookahead + speed/heading/lateral
N_LOOKAHEAD = 8
LOOKAHEAD_STRIDE = 5
LOOKAHEAD_SCALE = 50.0

def build_observation(car, center, widths, left, right, seg_pack):
    pos = np.array([car.x, car.y], dtype=np.float32)
    p_closest, i, t, d2 = nearest_point_on_polyline(center, pos)
    tangent = center[(i+1)%len(center)] - center[(i-1)%len(center)]
    tangent = tangent / (np.linalg.norm(tangent)+1e-9)
    forward = np.array([math.cos(car.heading), math.sin(car.heading)], dtype=np.float32)
    heading_err = np.arctan2(forward[0]*tangent[1]-forward[1]*tangent[0],
                             forward[0]*tangent[0]+forward[1]*tangent[1]) / math.pi
    lateral = (np.cross(tangent, pos - p_closest)) / (0.5*widths[i] + 1e-6)
    speed = np.clip(car.v/90.0, 0.0, 1.0)

    angs = np.asarray(RAYS_DEG, dtype=np.float32) * (np.pi/180.0)
    dirs = np.stack([np.cos(car.heading + angs),
                     np.sin(car.heading + angs)], axis=1).astype(np.float32)
    pts = np.repeat(pos[None,:], len(RAYS_DEG), axis=0)

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
    return np.array(obs, dtype=float)

def eval_one_genome(genome, config, track, max_time=12.0):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    center = track["center"]; left = track["left"]; right = track["right"]; widths = track["widths"]
    seg_pack = prepare_segments(left, right)

    p0, p1 = center[0], center[1]
    heading0 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    car = Car((p0[0], p0[1]), heading0)

    score = 10.0  # start at 10
    dt = 1/60.0
    slow_timer = 0.0

    for _ in range(int(max_time / dt)):
        obs = build_observation(car, center, widths, left, right, seg_pack)
        expected_inputs = len(net.input_nodes)
        if len(obs) > expected_inputs:
            obs = obs[:expected_inputs]
        elif len(obs) < expected_inputs:
            obs = np.pad(obs, (0, expected_inputs - len(obs)), mode="constant", constant_values=0.0)
        steer, gas, brake = net.activate(obs)
        steer = float(np.clip(steer, -1.0, 1.0))
        gas   = float(np.clip((gas+1)/2, 0.0, 1.0))
        brake = float(np.clip((brake+1)/2, 0.0, 1.0))

        # throttle bias (more if under target)
        v_pre = car.v
        gas = min(1.0, gas * (1.20 if v_pre < V_REF else 1.05))

        # compute off before update for penalty & KO
        pos = np.array([car.x, car.y])
        p_closest, ii, tt, d2 = nearest_point_on_polyline(center, pos)
        halfw = 0.5*widths[ii]
        dist = math.sqrt(d2)
        off = max(0.0, dist - halfw)

        # --- Scoring ---
        if v_pre >= V_REF:
            score += (v_pre - V_REF) * K_FAST * dt
        else:
            score -= (V_REF - v_pre) * K_SLOW * dt
        if off > 0.0:
            score -= off * K_OFF * dt
        score -= brake * K_BRAKE * dt

        # early KO
        if v_pre < EARLY_KO_SPEED:
            slow_timer += dt
        else:
            slow_timer = max(0.0, slow_timer - dt*0.5)
        if slow_timer >= EARLY_KO_TIME or off >= OFF_KO:
            break

        # physics step (lower base drag when off)
        car.grip = 1.0 / (1.0 + off*0.6)
        car.max_steer = math.radians(28) * car.grip
        car.drag = 0.10 * (1.0 + off*0.8)  # was 0.18
        car.update(dt, gas, brake, steer)

        # forward progress reward after update
        tangent = center[(ii+1)%len(center)] - center[(ii-1)%len(center)]
        tangent = tangent / (np.linalg.norm(tangent)+1e-9)
        vel = np.array([math.cos(car.heading), math.sin(car.heading)]) * car.v
        fwd = float(np.dot(vel, tangent))
        if fwd > 0.0:
            score += fwd * K_FWD * dt

    return float(max(0.0, score))

def make_track(seed):
    T = generate_urban_gp(seed=seed)
    return {"seed": seed, "center": T["center"], "left": T["left"], "right": T["right"], "widths": T["widths"]}

def _sample_track_seeds(genome_key, base_seed, gen_idx, num_tracks, seed_range):
    seed_val = (base_seed + gen_idx * 1000003 + int(genome_key)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_val)
    return rng.integers(0, seed_range, size=num_tracks, endpoint=False).tolist()

EVAL_CONTEXT = {}

def eval_genome_multi(genome, config):
    ctx = EVAL_CONTEXT
    if ctx.get("fixed_track"):
        tracks = [ctx["fixed_track"]]
    else:
        seeds = _sample_track_seeds(
            genome.key,
            ctx["base_seed"],
            ctx["gen_idx"],
            ctx["train_seeds_per_gen"],
            ctx["seed_range"],
        )
        tracks = [make_track(seed=s) for s in seeds]
    scores = []
    for track in tracks:
        scores.append(eval_one_genome(genome, config, track, max_time=ctx["max_time"]))
    return float(np.mean(scores)) if scores else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="neat_config.ini")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--fixed-track", action="store_true")
    ap.add_argument("--outdir", default="runs/urban_gp")
    ap.add_argument("--parallel", action="store_true", help="use neat.ParallelEvaluator")
    ap.add_argument("--save-dir", type=str, default=None, help="where to save best genomes (default=outdir)")
    ap.add_argument("--save-best-every", type=int, default=1, help="save best genome every N generations (0=off)")
    ap.add_argument("--checkpointer", type=int, default=0, help="save full NEAT state every N generations (0=off)")
    ap.add_argument("--train-seeds-per-gen", type=int, default=3,
                    help="number of random tracks per genome evaluation")
    ap.add_argument("--seed-range", type=int, default=100000,
                    help="track seeds sampled uniformly from [0, seed-range)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      args.config)
    if args.pop is not None:
        cfg.pop_size = int(args.pop)

    os.makedirs(args.outdir, exist_ok=True)
    stats_csv = os.path.join(args.outdir, "stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["generation","best_score","mean_score","stdev_score","elapsed_sec"])

    # Fixed replay track so the viewer knows what to load
    replay_track = make_track(seed=args.seed)
    with open(os.path.join(args.outdir, "replay_track.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": replay_track["seed"],
            "center": replay_track["center"].tolist(),
            "left":   replay_track["left"].tolist(),
            "right":  replay_track["right"].tolist(),
            "widths": replay_track["widths"].tolist()
        }, f)

    pop = neat.Population(cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()

    save_dir = args.save_dir or args.outdir
    os.makedirs(save_dir, exist_ok=True)
    if args.checkpointer and args.checkpointer > 0:
        pop.add_reporter(neat.Checkpointer(generation_interval=args.checkpointer,
                                           filename_prefix=os.path.join(save_dir, "chkpt-")))
    best_so_far = -1e18
    pop.add_reporter(stats)

    t0 = time.time()
    for gen in range(args.gens):
        fixed_track = replay_track if args.fixed_track else None
        max_time = 10.0 if (args.train_seeds_per_gen > 1 and not args.fixed_track) else 12.0
        EVAL_CONTEXT.update({
            "fixed_track": fixed_track,
            "train_seeds_per_gen": max(1, int(args.train_seeds_per_gen)),
            "seed_range": int(args.seed_range),
            "base_seed": int(args.seed),
            "gen_idx": int(gen),
            "max_time": float(max_time),
        })

        # Evaluate
        if args.parallel:
            from neat.parallel import ParallelEvaluator
            import os as _os
            pe = ParallelEvaluator(_os.cpu_count() or 4, eval_genome_multi)
            pe.evaluate(list(pop.population.items()))  # fills genome.fitness
        else:
            genomes = list(pop.population.items())
            for gid, genome in genomes:
                genome.fitness = eval_genome_multi(genome, cfg)

        # Log
        scores = [g.fitness for g in pop.population.values()]
        best, mean, stdev = float(np.max(scores)), float(np.mean(scores)), float(np.std(scores))
        elapsed = time.time() - t0
        with open(stats_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([gen, f"{best:.4f}", f"{mean:.4f}", f"{stdev:.4f}", f"{elapsed:.2f}"])
        print(f"[Gen {gen+1}/{args.gens}] best={best:.2f} mean={mean:.2f} stdev={stdev:.2f}", flush=True)

        # Save best genome (per schedule) and update best_so_far
        if args.save_best_every and ((gen + 1) % args.save_best_every == 0):
            best_gid, best_genome = max(pop.population.items(), key=lambda kv: (kv[1].fitness or -1e18))
            current_best = float(best_genome.fitness or -1e18)
            if current_best >= best_so_far:
                best_so_far = current_best
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"best_gen_{gen+1:04d}_{current_best:.2f}.pkl")
                with open(fname, "wb") as f:
                    pickle.dump({"gen": gen+1, "fitness": current_best, "genome": best_genome, "config": cfg}, f)
                with open(os.path.join(save_dir, "best_so_far.pkl"), "wb") as f:
                    pickle.dump({"gen": gen+1, "fitness": current_best, "genome": best_genome, "config": cfg}, f)
                print(f"[SAVE] Best-so-far improved: gen={gen+1} fitness={current_best:.2f} -> {fname}", flush=True)

        # ---- Evolve (always) ----
        pop.reporters.post_evaluate(cfg, pop.population, pop.species, None)
        pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
        pop.species.speciate(cfg, pop.population, pop.generation)

    winner = stats.best_genome()
    with open(os.path.join(args.outdir, "neat_winner.pkl"), "wb") as f:
        pickle.dump({"genome": winner, "config": cfg}, f)
    print("Saved to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
