
# racing.py
# Core track + car + geometry utilities, now with optional CUDA/CuPy-accelerated batched raycasting.
# Dependencies: numpy (optional: cupy for GPU), pygame (only for the standalone demo main()).
# NOTE: Importing this module does NOT initialize pygame.

import math, random, sys
import numpy as _np

# ---- Optional CUDA via CuPy ----
try:
    import cupy as _cp
    xp = _cp
    GPU_AVAILABLE = True
except Exception:
    xp = _np
    GPU_AVAILABLE = False

# ---------------- Geometry helpers ----------------

def lookahead_centerline_points(center, nearest_idx, car_pos, car_heading,
                                n_lookahead=8, stride=5, scale=50.0):
    """
    Compute lookahead centerline points in the car's local frame.

    Returns flat array: [x1, y1, x2, y2, ...] normalized by `scale`.
    """
    center = _np.asarray(center, dtype=float)
    pos = _np.asarray(car_pos, dtype=float)
    n = len(center)
    out = _np.zeros((n_lookahead, 2), dtype=_np.float32)
    cos_h = math.cos(car_heading)
    sin_h = math.sin(car_heading)
    for k in range(n_lookahead):
        idx = (nearest_idx + (k + 1) * stride) % n
        dx, dy = center[idx] - pos
        # rotate into car frame (heading forward = +x)
        x_local = cos_h * dx + sin_h * dy
        y_local = -sin_h * dx + cos_h * dy
        out[k, 0] = x_local / scale
        out[k, 1] = y_local / scale
    return out.reshape(-1)

def catmull_rom_spline(P, samples_per_seg=20, alpha=0.5):
    """Closed Catmull–Rom (centripetal) through P (N x 2)."""
    P = _np.asarray(P, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        p0 = P[(i-1)%n]; p1 = P[i]; p2 = P[(i+1)%n]; p3 = P[(i+2)%n]
        def tj(ti, pa, pb):
            return ti + (_np.linalg.norm(pb-pa) ** alpha)
        t0=0.0; t1=tj(t0,p0,p1); t2=tj(t1,p1,p2); t3=tj(t2,p2,p3)
        ts = _np.linspace(t1, t2, samples_per_seg, endpoint=False)
        for t in ts:
            A1 = (t1 - t)/(t1 - t0)*p0 + (t - t0)/(t1 - t0)*p1
            A2 = (t2 - t)/(t2 - t1)*p1 + (t - t1)/(t2 - t1)*p2
            A3 = (t3 - t)/(t3 - t2)*p2 + (t - t2)/(t3 - t2)*p3
            B1 = (t2 - t)/(t2 - t0)*A1 + (t - t0)/(t2 - t0)*A2
            B2 = (t3 - t)/(t3 - t1)*A2 + (t - t1)/(t3 - t1)*A3
            C  = (t2 - t)/(t2 - t1)*B1 + (t - t1)/(t2 - t1)*B2
            out.append(C)
    return _np.asarray(out)

def polyline_length(P):
    P = _np.asarray(P)
    return float(_np.sum(_np.linalg.norm(P[(_np.arange(len(P))+1)%len(P)] - P, axis=1)))

def resample_equal_arclength(P, step):
    """Closed polyline resample by equal arc length."""
    P = _np.asarray(P)
    diffs = P[(_np.arange(len(P))+1)%len(P)] - P
    segs = _np.linalg.norm(diffs, axis=1)
    cum = _np.concatenate([[0.0], _np.cumsum(segs)])
    total = cum[-1]
    n = max(32, int(round(total/step)))
    targets = _np.linspace(0.0, total, n, endpoint=False)
    out = []
    k = 0
    for t in targets:
        while cum[k+1] < t:
            k += 1
        s = (t - cum[k]) / (segs[k] if segs[k] > 1e-9 else 1.0)
        out.append(P[k] + s*diffs[k])
    return _np.asarray(out)

def curvature(P):
    """Approx curvature kappa at each point (closed polyline)."""
    P = _np.asarray(P)
    n = len(P)
    K = _np.zeros(n)
    for i in range(n):
        a = P[(i-1)%n]; b = P[i]; c = P[(i+1)%n]
        ab = _np.linalg.norm(b-a); bc = _np.linalg.norm(c-b); ac = _np.linalg.norm(c-a)
        s = 0.5*(ab+bc+ac)
        area2 = max(s*(s-ab)*(s-bc)*(s-ac), 0.0)
        if area2 <= 1e-16 or ab < 1e-9 or bc < 1e-9:
            K[i] = 0.0
        else:
            R = (ab*bc*ac)/(4.0*math.sqrt(area2) + 1e-12)
            K[i] = 0.0 if R == 0 else 1.0/R
    return K

def left_right_edges(center, widths):
    """Compute left/right edges from centerline and per-point width."""
    n = len(center)
    L = _np.zeros_like(center); R = _np.zeros_like(center)
    for i in range(n):
        prev = center[(i-1)%n]; nxt = center[(i+1)%n]
        t = nxt - prev
        tl = _np.linalg.norm(t)
        if tl < 1e-9:
            nrm = _np.array([0.0,1.0])
        else:
            t = t/tl
            nrm = _np.array([-t[1], t[0]])
        half = 0.5*widths[i]
        L[i] = center[i] + nrm*half
        R[i] = center[i] - nrm*half
    return L, R

# --------------- Track generator (fast, automatic) ---------------

def generate_urban_gp(seed=1234,
                      bbox=(1800,1800),
                      n_ctrl=12,
                      resample_step=10.0,
                      min_radius_fast=55.0,
                      width_range=(10.0,16.0)):
    rng = random.Random(seed)
    W,H = bbox
    # roughly circular control points with jitter
    angles = sorted([rng.uniform(0, 2*math.pi) for _ in range(n_ctrl)])
    rad = min(W,H)*0.35
    jit = min(W,H)*0.14
    cx,cy = W/2, H/2
    ctrl = []
    for a in angles:
        r = rad + rng.uniform(-jit, jit)
        x = cx + r*math.cos(a) + rng.uniform(-40,40)
        y = cy + r*math.sin(a) + rng.uniform(-40,40)
        ctrl.append((x,y))
    spline = catmull_rom_spline(_np.array(ctrl), samples_per_seg=14, alpha=0.5)
    center = resample_equal_arclength(spline, resample_step)

    # Gentle "Urban GP" shaping: stretch two longest straight-ish runs
    K = curvature(center)
    straight = (K < 0.002)  # R > ~500m
    best_run = 0; br_len = 0; run=0
    for i in range(len(straight)*2):
        if straight[i%len(straight)]:
            run += 1
            if run > br_len:
                br_len = run; best_run = (i-run+1)%len(straight)
        else:
            run=0
    if br_len > 0:
        for j in range(br_len):
            idx = (best_run + j) % len(center)
            prev = center[(idx-1)%len(center)]; nxt = center[(idx+1)%len(center)]
            t = nxt - prev; tl = _np.linalg.norm(t) or 1.0
            t = t/tl
            center[idx] += t*2.0  # tiny nudge

    # Width from curvature (tighter -> wider)
    K = curvature(center)
    k_norm = _np.clip(K / (1.0/min_radius_fast), 0.0, 1.0)
    wmin,wmax = width_range
    widths = wmin + (1.0 - k_norm) * (wmax - wmin)
    for _ in range(2):
        widths = 0.25*_np.roll(widths,-1) + 0.5*widths + 0.25*_np.roll(widths,1)

    left,right = left_right_edges(center, widths)
    return {
        "center": center,
        "left": left,
        "right": right,
        "widths": widths,
        "bbox": (W,H)
    }

# --------------- Simple car model ----------------

class Car:
    def __init__(self, pos, heading=0.0):
        self.x, self.y = pos
        self.heading = heading  # radians
        self.v = 0.0
        self.max_steer = math.radians(28)
        self.accel = 11.0       # m/s^2
        self.brake = 18.0
        self.drag = 0.18
        self.grip = 1.0         # scaled if off-track

    def update(self, dt, throttle, brake, steer_input):
        # speed-dependent steering (less at high speed)
        steer = self.max_steer * steer_input * (0.55 + 0.45*max(0.0, 1.0 - self.v/90.0))
        # longitudinal
        a = throttle*self.accel - brake*self.brake - self.drag*self.v
        self.v = max(0.0, self.v + a*dt)
        # integrate simple bicycle kinematics
        self.heading += self.v * math.tan(steer) / 2.6 * dt  # ~2.6m wheelbase
        self.x += math.cos(self.heading) * self.v * dt
        self.y += math.sin(self.heading) * self.v * dt

# nearest point utilities
def nearest_point_on_polyline(P, q):
    # brute force segment projection (fine for demo)
    Pn = len(P)
    best_d2 = 1e18; best_s = 0; best_t = 0.0
    for i in range(Pn):
        a = P[i]; b = P[(i+1)%Pn]
        ab = b - a
        t = float(_np.dot(q - a, ab) / (_np.dot(ab, ab) + 1e-12))
        t = max(0.0, min(1.0, t))
        p = a + t*ab
        d2 = float(_np.dot(q - p, q - p))
        if d2 < best_d2:
            best_d2 = d2; best_s = i; best_t = t
    p = P[best_s] + best_t*(P[(best_s+1)%len(P)] - P[best_s])
    return p, best_s, best_t, best_d2

# --------- Batched raycasting (GPU if available) ----------

def prepare_segments(left, right):
    """Flatten track edges into segment endpoint arrays (NumPy CPU)."""
    segs = []
    for E in (left, right):
        for i in range(len(E)):
            a = E[i]; b = E[(i+1) % len(E)]
            segs.append((a[0], a[1], b[0], b[1]))
    segs = _np.asarray(segs, dtype=_np.float32)
    ax, ay, bx, by = [segs[:,k] for k in range(4)]
    sx = bx - ax; sy = by - ay
    return ax, ay, sx, sy  # numpy float32

def _raycast_batch_gpu(points_xy, dirs_xy, seg_ax, seg_ay, seg_sx, seg_sy, max_range=150.0):
    # normalize directions
    d = dirs_xy / (xp.linalg.norm(dirs_xy, axis=1, keepdims=True) + 1e-9)
    px = points_xy[:,0][:,None]; py = points_xy[:,1][:,None]
    rx = d[:,0][:,None];        ry = d[:,1][:,None]

    ax = seg_ax[None,:]; ay = seg_ay[None,:]
    sx = seg_sx[None,:]; sy = seg_sy[None,:]

    denom = rx * (-sy) + ry * (sx)
    valid = xp.abs(denom) > 1e-9

    t = ((ax - px) * (-sy) + (ay - py) * (sx)) / (denom + (~valid)*1e9)
    u = ((ax - px) * ry - (ay - py) * rx) / (denom + (~valid)*1e9)

    hit = valid & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t_masked = xp.where(hit, t, xp.full_like(t, 1e9))

    best_t = xp.min(t_masked, axis=1)
    best_t = xp.clip(best_t, 0.0, max_range)
    return (best_t / max_range).astype(xp.float32)

def raycast_batch(points_xy_np, dirs_xy_np, seg_ax_np, seg_ay_np, seg_sx_np, seg_sy_np, max_range=150.0):
    """
    Accepts & returns numpy arrays. Uses GPU (CuPy) if available, else NumPy.
    points_xy_np: (R,2) float32
    dirs_xy_np:   (R,2) float32
    Returns: (R,) float32 in [0,1]
    """
    if GPU_AVAILABLE:
        p = _cp.asarray(points_xy_np, dtype=_cp.float32)
        d = _cp.asarray(dirs_xy_np,   dtype=_cp.float32)
        ax = _cp.asarray(seg_ax_np); ay = _cp.asarray(seg_ay_np)
        sx = _cp.asarray(seg_sx_np); sy = _cp.asarray(seg_sy_np)
        out = _raycast_batch_gpu(p, d, ax, ay, sx, sy, max_range=max_range)
        return _cp.asnumpy(out)
    else:
        # pure NumPy
        p = points_xy_np.astype(_np.float32, copy=False)
        d = dirs_xy_np.astype(_np.float32, copy=False)
        d = d / (_np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        px = p[:,0][:,None]; py = p[:,1][:,None]
        rx = d[:,0][:,None]; ry = d[:,1][:,None]
        ax = seg_ax_np[None,:]; ay = seg_ay_np[None,:]
        sx = seg_sx_np[None,:]; sy = seg_sy_np[None,:]
        denom = rx * (-sy) + ry * (sx)
        valid = _np.abs(denom) > 1e-9
        t = ((ax - px) * (-sy) + (ay - py) * (sx)) / (denom + (~valid)*1e9)
        u = ((ax - px) * ry - (ay - py) * rx) / (denom + (~valid)*1e9)
        hit = valid & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t_masked = _np.where(hit, t, _np.full_like(t, 1e9))
        best_t = _np.min(t_masked, axis=1)
        best_t = _np.clip(best_t, 0.0, max_range)
        return (best_t / max_range).astype(_np.float32)

# --------------- Optional standalone demo ---------------
# Only runs if invoked directly. Avoid import-time pygame init.

def main():
    import pygame
    random.seed(20250809)
    _np.random.seed(0)

    track = generate_urban_gp(seed=20250809)
    center = track["center"]; left = track["left"]; right = track["right"]; widths = track["widths"]

    # screen
    W, H = 1280, 800
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # fit track in view
    all_pts = _np.vstack([left, right])
    minx, miny = _np.min(all_pts, axis=0); maxx, maxy = _np.max(all_pts, axis=0)
    world_c = ((minx+maxx)/2, (miny+maxy)/2)
    world_size = max(maxx-minx, maxy-miny)
    base_scale = 0.85 * min(W, H) / world_size

    # Camera & zoom
    zoom = 1.8
    auto_zoom = True
    def world_to_screen(p, cam, scale):
        px = (p[0]-cam[0])*scale + W/2
        py = (p[1]-cam[1])*scale + H/2
        return int(px), int(py)

    # init car
    p0 = center[0]; p1 = center[1]
    heading0 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    car = Car((p0[0], p0[1]), heading0)

    lap = 0; started = False
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    zoom *= 1.10
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom /= 1.10
                elif e.key == pygame.K_z:
                    auto_zoom = not auto_zoom

        keys = pygame.key.get_pressed()
        throttle = 1.0 if keys[pygame.K_UP] else 0.0
        brake    = 1.0 if keys[pygame.K_DOWN] else 0.0
        steer_in = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0)

        # off-track grip/drag penalty
        q = _np.array([car.x, car.y])
        px, idx, t, d2 = nearest_point_on_polyline(center, q)
        halfw = 0.5*widths[idx]
        dist = math.sqrt(d2)
        off = max(0.0, dist - halfw)
        car.grip = 1.0 / (1.0 + off*0.6)
        car.max_steer = math.radians(28) * car.grip
        car.drag = 0.18 * (1.0 + off*0.8)

        car.update(dt, throttle, brake, steer_in)

        # lap detection
        a = center[0]; b = center[1]
        def side(p, a, b): return _np.cross(b-a, p-a)
        if side(q, a, b) < 0 and side(_np.array([car.x,car.y]), a, b) >= 0:
            if started: lap += 1
            started = True

        # Camera follow
        lookahead_m = min(80.0, 20.0 + 0.6*car.v)
        cam = (car.x + math.cos(car.heading)*lookahead_m,
               car.y + math.sin(car.heading)*lookahead_m)

        # Auto zoom
        scale = base_scale
        if auto_zoom:
            target_zoom = 2.2 - (min(90.0, car.v)/90.0) * (2.2 - 0.9)
            zoom = zoom*0.90 + target_zoom*0.10
        zoom = max(0.6, min(3.5, zoom))
        scale *= zoom

        # Draw
        screen.fill((15, 18, 24))
        ptsL = [world_to_screen(p, cam, scale) for p in left]
        ptsR = [world_to_screen(p, cam, scale) for p in right]
        pygame.draw.aalines(screen, (50,180,220), True, ptsL, True)
        pygame.draw.aalines(screen, (50,180,220), True, ptsR, True)
        ptsC = [world_to_screen(p, cam, scale) for p in center]
        pygame.draw.aalines(screen, (90,90,120), True, ptsC, True)
        s0 = world_to_screen(center[0], cam, scale)
        s1 = world_to_screen(center[1], cam, scale)
        pygame.draw.line(screen, (255,255,255), s0, s1, 2)

        # car
        car_len, car_w = 4.2, 1.9
        Lpix = car_len*scale * 1.1
        Wpix = car_w*scale * 1.1
        cx, cy = world_to_screen((car.x, car.y), cam, scale)
        ang = car.heading
        cosA, sinA = math.cos(ang), math.sin(ang)
        corners = []
        for dx, dy in [( Lpix/2,  Wpix/2), ( Lpix/2, -Wpix/2),
                       (-Lpix/2, -Wpix/2), (-Lpix/2,  Wpix/2)]:
            x = cx + dx*cosA - dy*sinA
            y = cy + dx*sinA + dy*cosA
            corners.append((x,y))
        pygame.draw.polygon(screen, (240,140,60), corners)
        pygame.draw.circle(screen, (255,255,255), (cx,cy), max(2, int(2*zoom)))

        # HUD
        info1 = f"Speed: {car.v:5.1f} m/s   Lap: {lap}"
        info2 = f"Zoom: {zoom:.2f}  AutoZoom: {'ON' if auto_zoom else 'OFF'}   [+/-] zoom, Z toggle, ESC quit"
        pygame.font.init()
        screen.blit(pygame.font.SysFont("consolas", 18).render(info1, True, (230,230,230)), (12,12))
        screen.blit(pygame.font.SysFont("consolas", 18).render(info2, True, (160,160,160)), (12,34))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import pygame
        pygame.quit(); sys.exit(0)
