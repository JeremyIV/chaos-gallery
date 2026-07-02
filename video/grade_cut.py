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

Example (fractal-first cut, softer highlights):
  grade_cut.py m_hdr_master.raw out.mp4 hold0:1 fwd holdN:0.5 rev hold0:2 --exposure 1.6
"""
import json
import subprocess
import sys
import time

import numpy as np

from grading import grade

args = sys.argv[1:]
opts = {}
trim_end = 0.0
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

t0 = time.time()
graded = []
for i in range(N):
    graded.append(grade(np.asarray(hdr[i], dtype=np.float32), **opts))
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
