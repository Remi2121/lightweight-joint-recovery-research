"""
Baseline joint recovery: plain temporal interpolation.

This is deliberately the *simplest possible* recovery method — it exists
so there is a working, measurable baseline before building the actual
research contribution (compute-selective recovery using temporal +
skeletal-graph context). It also serves as the "dense" comparison point:
this version checks and fills every joint uniformly, with no selectivity
at all, which is exactly the inefficiency this research targets.

For each joint whose coordinates are entirely zero at frame t (per the
`mask` produced by src/occlusion.py), fill it by linearly interpolating
between the nearest preceding and following frames where that joint was
present. Falls back to nearest-neighbour fill at sequence boundaries.
"""

import numpy as np
import torch


def recover_temporal_interpolation(clip, mask):
    """
    clip: (C, T, V) tensor/array, zeroed at missing entries
    mask: (T, V) bool, True = missing
    Returns a new (C, T, V) array with missing joints filled in.
    """
    is_tensor = torch.is_tensor(clip)
    arr = clip.detach().cpu().numpy().copy() if is_tensor else clip.copy()
    C, T, V = arr.shape

    for v in range(V):
        present_frames = np.where(~mask[:, v])[0]
        if present_frames.size == 0:
            continue  # joint missing in every frame of this clip -- nothing to interpolate from
        if present_frames.size == T:
            continue  # never missing

        for c in range(C):
            series = arr[c, :, v]
            arr[c, :, v] = np.interp(
                np.arange(T), present_frames, series[present_frames]
            )

    return torch.from_numpy(arr).float() if is_tensor else arr
