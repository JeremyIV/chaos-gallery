"""Shared HDR-to-display grading for the falling-pixels videos.

`acc` is linear HDR light per pixel (H, W, 3), in coverage units: 1.0 means one
full particle's worth of that channel. The grade simulates a camera: channel
crosstalk (sensor filter tails), two-scale bloom of the over-threshold excess
(lens glare), then a soft exposure curve t = 1 - exp(-exposure * L^gamma) that
saturates each channel smoothly. Lower `exposure` / higher `bloom_thresh`
preserve more highlight structure; `gamma` < 1 widens the shoulder.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def grade(acc, exposure=0.9, gamma=0.65, knee=None, crosstalk=0.05,
          bloom_thresh=2.5, bloom_near=0.4, bloom_wide=0.18, bg=(16, 16, 20)):
    """knee=M switches the curve to t = 1 - exp(-k * f(L)/f(1)), f(L) = L/(1+L/M):
    linear below single coverage (dim detail and the base fractal untouched, so
    t(1) = 1-exp(-exposure) exactly), rational compression above. Smaller M =
    stronger highlight squeeze. When knee is None the L^gamma curve applies."""
    acc = acc.astype(np.float32, copy=False)
    tot = acc.sum(axis=-1, keepdims=True)
    L = (1.0 - crosstalk) * acc + (crosstalk / 2.0) * (tot - acc)
    excess = np.maximum(L - bloom_thresh, 0.0)
    L = L + bloom_near * gaussian_filter(excess, sigma=(6, 6, 0))
    wide = gaussian_filter(np.ascontiguousarray(excess[::4, ::4]), sigma=(5, 5, 0))
    H, W = L.shape[:2]
    L = L + bloom_wide * zoom(wide, (H / wide.shape[0], W / wide.shape[1], 1), order=1)
    L = np.maximum(L, 0.0)
    if knee is not None:
        f = L / (1.0 + L / knee)
        t = 1.0 - np.exp(-exposure * f * (1.0 + 1.0 / knee))
    else:
        t = 1.0 - np.exp(-exposure * np.power(L, gamma))
    bgv = np.asarray(bg, dtype=np.float32)
    return np.clip(bgv + (255.0 - bgv) * t, 0, 255).astype(np.uint8)
