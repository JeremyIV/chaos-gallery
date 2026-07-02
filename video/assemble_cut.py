#!/usr/bin/env python3
"""Assemble a cut of the falling-pixels video from the lossless forward master.

The master (written by render_fall.py, or extracted from a finished mp4) holds
the forward collapse only: frame 0 = full fractal, last frame = settled points.
Any cut is an ordering over those frames — no re-simulation needed.

Usage:
  assemble_cut.py MASTER OUT.mp4 SPEC...

SPEC tokens, in playback order:
  fwd         all master frames, first to last (fractal -> points)
  rev         all master frames, last to first (points -> fractal)
  hold0:S     hold the first master frame (fractal) for S seconds
  holdN:S     hold the last master frame (points) for S seconds

Examples:
  assemble_cut.py master.mkv out.mp4 holdN:0.5 rev hold0:1.5 fwd
  assemble_cut.py master.mkv out.mp4 hold0:1.5 fwd holdN:0.5 rev
"""
import subprocess
import sys

MASTER, OUT = sys.argv[1], sys.argv[2]
SPEC = sys.argv[3:]

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height,r_frame_rate",
     "-of", "csv=p=0", MASTER],
    capture_output=True, text=True, check=True).stdout.strip().split(",")
W, H = int(probe[0]), int(probe[1])
num, den = probe[2].split("/")
FPS = int(round(int(num) / int(den)))
FRAME_BYTES = W * H * 3
print(f"master: {W}x{H}@{FPS}", flush=True)

dec = subprocess.Popen(
    ["ffmpeg", "-loglevel", "error", "-i", MASTER,
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    stdout=subprocess.PIPE)
frames = []
while True:
    buf = dec.stdout.read(FRAME_BYTES)
    if len(buf) < FRAME_BYTES:
        break
    frames.append(buf)
dec.wait()
print(f"loaded {len(frames)} master frames", flush=True)

seq = []
for tok in SPEC:
    if tok == "fwd":
        seq.extend(range(len(frames)))
    elif tok == "rev":
        seq.extend(range(len(frames) - 1, -1, -1))
    elif tok.startswith("hold0:"):
        seq.extend([0] * round(float(tok.split(":")[1]) * FPS))
    elif tok.startswith("holdN:"):
        seq.extend([len(frames) - 1] * round(float(tok.split(":")[1]) * FPS))
    else:
        sys.exit(f"unknown spec token: {tok}")

enc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
     "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE)
for i in seq:
    enc.stdin.write(frames[i])
enc.stdin.close()
enc.wait()
print(f"wrote {OUT} ({len(seq)} frames, {len(seq)/FPS:.1f}s)", flush=True)
