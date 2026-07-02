#!/usr/bin/env python3
"""Grade and cut an HDR forward master into an mp4 — no re-simulation.

The HDR master (written by render_fall.py) is raw float16 linear light,
frame 0 = full fractal, last frame = settled points, with a .json sidecar.
Grading (exposure/crosstalk/bloom) and cut assembly happen here, so look
tweaks take minutes.

Usage:
  grade_cut.py MASTER.raw OUT.mp4 SPEC... [--exposure X] [--gamma G]
               [--crosstalk C] [--bloom-thresh T] [--trim-end SECONDS]

SPEC tokens (playback order): fwd | rev | hold0:SECONDS | holdN:SECONDS
--trim-end drops the last S seconds of the forward master (the near-converged
tail), shortening fwd, rev, and the frame holdN refers to.
--auto enables auto-exposure: per frame, base k is set so the 99th percentile
of lit-pixel brightness sits just below white (top ~1% of the light may clip),
clamped at --exposure (default 1.9, the wide-open aperture), and smoothed over
~0.75 s in log space. Exposure is a pure function of frame content, so the
loop starts and ends at the same k automatically.

Example (fractal-first cut, softer highlights):
  grade_cut.py m_hdr_master.raw out.mp4 hold0:1 fwd holdN:0.5 rev hold0:2 --exposure 1.6
"""
import json
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from grading import grade

args = sys.argv[1:]
opts = {}
trim_end = 0.0
auto = "--auto" in args
if auto:
    args.remove("--auto")
if "--trim-end" in args:
    i = args.index("--trim-end")
    trim_end = float(args[i + 1])
    del args[i:i + 2]
for flag, key in (("--exposure", "exposure"), ("--gamma", "gamma"),
                  ("--crosstalk", "crosstalk"), ("--bloom-thresh", "bloom_thresh")):
    if flag in args:
        i = args.index(flag)
        opts[key] = float(args[i + 1])
        del args[i:i + 2]
MASTER, OUT = args[0], args[1]
SPEC = args[2:]

with open(MASTER.rsplit(".", 1)[0] + ".json") as jf:
    meta = json.load(jf)
W, H, FPS, N_ALL = meta["width"], meta["height"], meta["fps"], meta["frames"]
hdr = np.memmap(MASTER, dtype=np.float16, mode="r", shape=(N_ALL, H, W, 3))
N = max(2, N_ALL - round(trim_end * FPS))
print(f"master: {N_ALL} frames {W}x{H}@{FPS}, using first {N} "
      f"(trim-end {trim_end}s); grade opts: {opts or 'defaults'}", flush=True)

expo = None
if auto:
    k_max = opts.pop("exposure", 1.9)
    g = opts.get("gamma", 0.65)
    ct = opts.get("crosstalk", 0.05)
    bt = opts.get("bloom_thresh", 2.5)
    t0 = time.time()
    p99 = np.ones(N)
    for i in range(N):
        a = np.asarray(hdr[i][::4, ::4], dtype=np.float32)
        tot = a.sum(axis=-1, keepdims=True)
        L = (1 - ct) * a + (ct / 2) * (tot - a)
        ex = np.maximum(L - bt, 0.0)
        L = L + 0.4 * gaussian_filter(ex, (1.5, 1.5, 0)) + 0.18 * gaussian_filter(ex, (5, 5, 0))
        mx = L.max(axis=-1)
        lit = mx[mx > 0.02]
        if lit.size:
            p99[i] = max(np.percentile(lit, 99), 1e-3)
    k_raw = np.clip(4.2 / np.power(p99, g), 0.15, k_max)
    expo = np.exp(gaussian_filter1d(np.log(k_raw), FPS * 0.75, mode="nearest"))
    print(f"auto-exposure: metered {N} frames in {time.time()-t0:.0f}s; "
          f"k range {expo.min():.2f}..{expo.max():.2f} "
          f"(raw {k_raw.min():.2f}..{k_raw.max():.2f})", flush=True)

t0 = time.time()
graded = []
for i in range(N):
    o = dict(opts, exposure=float(expo[i])) if auto else opts
    graded.append(grade(np.asarray(hdr[i], dtype=np.float32), **o))
    if i % 200 == 199:
        el = time.time() - t0
        print(f"  graded {i+1}/{N}  {el:.0f}s, ETA {el/(i+1)*(N-i-1):.0f}s", flush=True)

seq = []
for tok in SPEC:
    if tok == "fwd":
        seq.extend(range(N))
    elif tok == "rev":
        seq.extend(range(N - 1, -1, -1))
    elif tok.startswith("hold0:"):
        seq.extend([0] * round(float(tok.split(":")[1]) * FPS))
    elif tok.startswith("holdN:"):
        seq.extend([N - 1] * round(float(tok.split(":")[1]) * FPS))
    else:
        sys.exit(f"unknown spec token: {tok}")

enc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
     "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE)
for i in seq:
    enc.stdin.write(graded[i].tobytes())
enc.stdin.close()
enc.wait()
print(f"wrote {OUT} ({len(seq)} frames, {len(seq)/FPS:.1f}s)", flush=True)
