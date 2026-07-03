#!/usr/bin/env python3
"""Variable-speed retime of a finished video, with smooth speed ramps.

speed(t) is 1 outside the affected window, `flat` inside the inner region,
with smoothstep ramps of `ramp` seconds on each side — symmetric about
`center`. Output stays at the source fps; faster sections drop source
frames (monotonic, at most one use per source frame for speeds >= 1).

Usage:
  retime.py IN.mp4 OUT.mp4 --center 43 --flat 2.0 --inner 7.5 --ramp 5
  (affected window = center +- (inner + ramp) seconds)
"""
import subprocess
import sys

import numpy as np

args = sys.argv[1:]
opt = {"center": 43.0, "flat": 2.0, "inner": 7.5, "ramp": 5.0}
for k in list(opt):
    f = "--" + k
    if f in args:
        i = args.index(f)
        opt[k] = float(args[i + 1])
        del args[i:i + 2]
SRC, OUT = args[0], args[1]

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
     "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
     "-of", "csv=p=0", SRC], capture_output=True, text=True, check=True
).stdout.strip().split(",")
W, H = int(probe[0]), int(probe[1])
num, den = probe[2].split("/")
FPS = int(round(int(num) / int(den)))
N = int(probe[3])
FB = W * H * 3


def speed(t):
    d = abs(t - opt["center"])
    if d <= opt["inner"]:
        w = 1.0
    elif d >= opt["inner"] + opt["ramp"]:
        w = 0.0
    else:
        u = 1.0 - (d - opt["inner"]) / opt["ramp"]
        w = u * u * (3 - 2 * u)
    return 1.0 + (opt["flat"] - 1.0) * w


src_t = (np.arange(N) + 0.5) / FPS
out_dt = (1.0 / FPS) / np.array([speed(t) for t in src_t])
out_time = np.concatenate([[0.0], np.cumsum(out_dt)])
n_out = int(np.floor(out_time[-1] * FPS))
need = np.searchsorted(out_time, (np.arange(n_out) + 0.5) / FPS, side="right") - 1
need = np.clip(need, 0, N - 1)
print(f"{N} src frames ({N/FPS:.1f}s) -> {n_out} out frames ({n_out/FPS:.1f}s), "
      f"window {opt['center']-opt['inner']-opt['ramp']:.1f}..{opt['center']+opt['inner']+opt['ramp']:.1f}s "
      f"at up to {opt['flat']:.1f}x", flush=True)

dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", SRC, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-preset", "medium", "-movflags", "+faststart", OUT],
                       stdin=subprocess.PIPE)
j = 0
for i in range(N):
    buf = dec.stdout.read(FB)
    if len(buf) < FB:
        break
    while j < n_out and need[j] == i:
        enc.stdin.write(buf)
        j += 1
dec.wait()
enc.stdin.close()
enc.wait()
print(f"wrote {OUT} ({j} frames, {j/FPS:.1f}s)", flush=True)
