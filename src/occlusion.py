"""
Synthetic joint occlusion — the controlled test bed for this research.

Since real occlusion events are rare and hard to label, we simulate
missing joints by randomly zeroing out joint coordinates in otherwise-
complete skeleton clips. This lets us measure, under a known ground
truth, how much accuracy a recognition model loses when joints go
missing, and how much a recovery method wins back.

Two masking modes, matching the two realistic occlusion patterns:
  - "random":    each (frame, joint) independently occluded (jitter-like)
  - "bursty":     a joint goes missing for a contiguous run of frames
                  (the more realistic case — a hand leaving frame for half
                  a second, not flickering in and out every frame)
"""

import numpy as np
import torch


def occlude_random(clip, p_missing=0.15, rng=None):
    """
    clip: (C, T, V) tensor/array
    Returns (occluded_clip, mask) where mask is (T, V) bool, True = missing.
    """
    rng = rng or np.random.default_rng()
    C, T, V = clip.shape
    mask = rng.random((T, V)) < p_missing
    occluded = clip.clone() if torch.is_tensor(clip) else clip.copy()
    for t in range(T):
        for v in range(V):
            if mask[t, v]:
                occluded[:, t, v] = 0.0
    return occluded, mask


def occlude_bursty(clip, p_joint_affected=0.3, burst_len_range=(3, 10), rng=None):
    """
    Picks a subset of joints and, for each, occludes one contiguous
    run of frames (simulating e.g. a hand leaving the camera's view).
    """
    rng = rng or np.random.default_rng()
    C, T, V = clip.shape
    mask = np.zeros((T, V), dtype=bool)
    occluded = clip.clone() if torch.is_tensor(clip) else clip.copy()

    for v in range(V):
        if rng.random() >= p_joint_affected:
            continue
        burst_len = min(T, rng.integers(burst_len_range[0], burst_len_range[1] + 1))
        start = rng.integers(0, max(1, T - burst_len + 1))
        mask[start:start + burst_len, v] = True
        occluded[:, start:start + burst_len, v] = 0.0

    return occluded, mask


def missing_rate(mask):
    """Fraction of (frame, joint) entries that are missing — useful for logging compute savings later."""
    return float(mask.mean())
