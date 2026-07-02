#!/usr/bin/env python3
"""Render the "falling pixels" magnetic-pendulum video.

Particles seed a circle circumscribing the view (no rectangular border
artifacts), colored pure R/G/B by capturing magnet, and render as additive
light. The video opens on the three settled points, plays the collapse in
reverse (unfurling into the full fractal), holds, then collapses forward —
first and last frames match, so it loops.

Physics, palette, and view match pendulum/index.html. Simulation runs on a
supersampled particle grid (torch, MPS if available); frames are splatted
bilinearly and piped raw into ffmpeg.

Usage:
  python3 render_fall.py out.mp4          # full render: 1080x1920, 60 fps, 20 s
  python3 render_fall.py out.mp4 --test   # quick 270x480, 30 fps, 3 s sanity run
"""
import math
import subprocess
import sys
import time

import numpy as np
import torch

TEST = "--test" in sys.argv
OUT = next((a for a in sys.argv[1:] if not a.startswith("-")), "pendulum_fall.mp4")

W_OUT, H_OUT = (270, 480) if TEST else (1080, 1920)
FPS = 30 if TEST else 60
FWD_S = 2.5 if TEST else 12.5
HOLD_END_S = 0.3 if TEST else 0.5
HOLD_FRACTAL_S = 0.4 if TEST else 1.25
REV_STRIDE = 2
SS = 2.0
STEPS_PER_FRAME = 12 if TEST else 3

HWX = 4.8
HWY = HWX * H_OUT / W_OUT
HW_MID, HW_END = 2.0, 1.4
ZOOM_SPLIT = 0.6
DT, SPRING, HH, FRIC = 0.02, 0.5, 0.1225, 0.15
MAGS = [(0.0, 1.0), (-0.866, -0.5), (0.866, -0.5)]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
GRAY, BG = (40, 40, 44), (16, 16, 20)
BASIN_MAX_STEPS = 4500

PW, PH = int(W_OUT * SS), int(H_OUT * SS)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device={DEV} out={W_OUT}x{H_OUT}@{FPS} forward={FWD_S}s "
      f"(dots {HOLD_END_S}s / unfurl / fractal {HOLD_FRACTAL_S}s / collapse)", flush=True)


R_SEED = math.hypot(HWX, HWY)


def grid():
    dx = 2 * HWX / PW
    n = int(round(2 * R_SEED / dx))
    c = ((torch.arange(n, dtype=torch.float32, device=DEV) + 0.5) / n - 0.5) * 2 * R_SEED
    wy, wx = torch.meshgrid(-c, c, indexing="ij")
    wx, wy = wx.flatten(), wy.flatten()
    keep = wx * wx + wy * wy <= R_SEED * R_SEED
    return wx[keep].clone(), wy[keep].clone()


def step(x, y, vx, vy):
    ax = -SPRING * x - FRIC * vx
    ay = -SPRING * y - FRIC * vy
    for mx, my in MAGS:
        dx, dy = mx - x, my - y
        d2 = dx * dx + dy * dy + HH
        w = 1.0 / (d2 * torch.sqrt(d2))
        ax = ax + dx * w
        ay = ay + dy * w
    vx = vx + ax * DT
    vy = vy + ay * DT
    return x + vx * DT, y + vy * DT, vx, vy


def basin_colors():
    x, y = grid()
    vx = torch.zeros_like(x)
    vy = torch.zeros_like(y)
    near = torch.full_like(x, -1, dtype=torch.int32)
    cnt = torch.zeros_like(near)
    result = torch.full_like(near, -1)
    t0 = time.time()
    for s in range(BASIN_MAX_STEPS):
        x, y, vx, vy = step(x, y, vx, vy)
        slow = (vx * vx + vy * vy) < 0.05
        hit = torch.full_like(near, -1)
        for m, (mx, my) in enumerate(MAGS):
            dx, dy = mx - x, my - y
            hit = torch.where(slow & ((dx * dx + dy * dy) < 0.1) & (hit < 0),
                              torch.full_like(near, m), hit)
        cnt = torch.where((hit >= 0) & (hit == near), cnt + 1, torch.zeros_like(cnt))
        near = hit
        result = torch.where((cnt > 25) & (result < 0), hit, result)
        if s % 250 == 249:
            done = (result >= 0).float().mean().item()
            print(f"  basin pass: step {s+1}/{BASIN_MAX_STEPS}, settled {done*100:.1f}%, "
                  f"{time.time()-t0:.0f}s", flush=True)
            if done > 0.9995:
                break
    res = result.cpu().numpy()
    col = np.empty((res.size, 3), dtype=np.float32)
    col[:] = GRAY
    for m, c in enumerate(COLORS):
        col[res == m] = c
    print(f"basin pass done in {time.time()-t0:.0f}s; gray {np.mean(res < 0)*100:.2f}%",
          flush=True)
    return col


def splat(xn, yn, col, hw):
    hh = hw * H_OUT / W_OUT
    sx = (xn / (2 * hw) + 0.5) * W_OUT - 0.5
    sy = (0.5 - yn / (2 * hh)) * H_OUT - 0.5
    ix0 = np.floor(sx).astype(np.int64)
    iy0 = np.floor(sy).astype(np.int64)
    fx = (sx - ix0).astype(np.float32)
    fy = (sy - iy0).astype(np.float32)
    accc = [np.zeros(H_OUT * W_OUT, dtype=np.float64) for _ in range(3)]
    for ox, oy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        gx, gy = ix0 + ox, iy0 + oy
        ok = (gx >= 0) & (gx < W_OUT) & (gy >= 0) & (gy < H_OUT)
        idx = (gy[ok] * W_OUT + gx[ok])
        wo = w[ok].astype(np.float64)
        for c in range(3):
            accc[c] += np.bincount(idx, weights=wo * col[ok, c], minlength=H_OUT * W_OUT)
    energy = 1.0 / (SS * SS)
    frame = np.empty((H_OUT * W_OUT, 3), dtype=np.float32)
    for c in range(3):
        frame[:, c] = BG[c] + accc[c] * energy
    return np.clip(frame, 0, 255).astype(np.uint8).reshape(H_OUT, W_OUT, 3)


def main():
    col = basin_colors()
    x, y = grid()
    print(f"particles in circle: {x.numel()/1e6:.1f}M (seed radius {R_SEED:.2f})", flush=True)
    vx = torch.zeros_like(x)
    vy = torch.zeros_like(y)
    fwd_frames = round(FWD_S * FPS)
    frames = [splat(x.cpu().numpy(), y.cpu().numpy(), col, HWX)]
    t0 = time.time()
    for f in range(fwd_frames):
        for _ in range(STEPS_PER_FRAME):
            x, y, vx, vy = step(x, y, vx, vy)
        u = (f + 1) / fwd_frames
        if u < ZOOM_SPLIT:
            s = u / ZOOM_SPLIT
            s = s * s * (3 - 2 * s)
            hw = HWX * (HW_MID / HWX) ** s
        else:
            s = (u - ZOOM_SPLIT) / (1 - ZOOM_SPLIT)
            s = s * s * (3 - 2 * s)
            hw = HW_MID * (HW_END / HW_MID) ** s
        frames.append(splat(x.cpu().numpy(), y.cpu().numpy(), col, hw))
        if f % 60 == 59:
            el = time.time() - t0
            print(f"  frame {f+1}/{fwd_frames}  {el:.0f}s elapsed, "
                  f"ETA {el/(f+1)*(fwd_frames-f-1):.0f}s", flush=True)
    last = len(frames) - 1
    rev = list(range(last, -1, -REV_STRIDE))
    if rev[-1] != 0:
        rev.append(0)
    seq = ([last] * round(HOLD_END_S * FPS) + rev
           + [0] * round(HOLD_FRACTAL_S * FPS) + list(range(len(frames))))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W_OUT}x{H_OUT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in seq:
        ff.stdin.write(frames[i].tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {OUT} ({len(seq)} frames, {len(seq)/FPS:.1f}s: "
          f"dots {HOLD_END_S}s -> unfurl {len(rev)/FPS:.1f}s -> "
          f"fractal {HOLD_FRACTAL_S}s -> collapse {len(frames)/FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
