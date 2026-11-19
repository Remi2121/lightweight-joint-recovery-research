"""
PKU-MMD skeleton dataset loader.

Expected raw layout (place files here, see data/pku_mmd/raw/README.md):
    data/pku_mmd/raw/Skeleton/XXXX-L.skeleton   (or -M / -R camera view)
    data/pku_mmd/raw/Label/XXXX-L.txt

Skeleton file format: one line per frame, 3 x 25 x 2 floats
    (25 Kinect v2 joints, 3D xyz, up to 2 tracked subjects per frame).
Label file format: one row per action instance
    label_id, start_frame, end_frame, confidence

This loader cuts each continuous, untrimmed recording into per-action
clips using the label file's start/end frames, producing fixed-length
(single-person) training samples for action recognition.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_JOINTS = 25
NUM_CHANNELS = 3
MAX_BODIES_RAW = 2


def _parse_skeleton_file(path):
    """Returns array of shape (num_frames, MAX_BODIES_RAW, NUM_JOINTS, NUM_CHANNELS)."""
    frames = []
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            vals = np.array(vals, dtype=np.float32)
            expected = MAX_BODIES_RAW * NUM_JOINTS * NUM_CHANNELS
            if vals.size != expected:
                # tolerate truncated/padded lines by clipping or zero-padding
                fixed = np.zeros(expected, dtype=np.float32)
                fixed[: min(vals.size, expected)] = vals[: expected]
                vals = fixed
            frames.append(vals.reshape(MAX_BODIES_RAW, NUM_JOINTS, NUM_CHANNELS))
    return np.stack(frames, axis=0) if frames else np.zeros((0, MAX_BODIES_RAW, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)


def _parse_label_file(path):
    """Returns list of (label_id, start_frame, end_frame, confidence)."""
    instances = []
    with open(path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) < 3:
                continue
            label_id, start, end = int(parts[0]), int(parts[1]), int(parts[2])
            conf = float(parts[3]) if len(parts) > 3 else 1.0
            instances.append((label_id, start, end, conf))
    return instances


class PKUMMDDataset(Dataset):
    """
    Single-person, fixed-length clips cut from PKU-MMD continuous recordings.

    Each sample: (skeleton, label)
        skeleton: FloatTensor (NUM_CHANNELS, clip_len, NUM_JOINTS)  -- ST-GCN input layout
        label:    int class id
    """

    def __init__(self, raw_dir, clip_len=30, min_confidence=0.5, body_idx=0):
        self.raw_dir = raw_dir
        self.clip_len = clip_len
        self.body_idx = body_idx
        self.samples = []  # list of (skeleton_file, start, end, label_id)

        skeleton_dir = os.path.join(raw_dir, "Skeleton")
        label_dir = os.path.join(raw_dir, "Label")
        if not os.path.isdir(skeleton_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"Expected 'Skeleton/' and 'Label/' under {raw_dir}. "
                "See data/pku_mmd/raw/README.md for the expected layout."
            )

        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".txt"):
                continue
            seq_name = fname[:-4]
            skel_path = os.path.join(skeleton_dir, seq_name + ".skeleton")
            if not os.path.exists(skel_path):
                continue
            for label_id, start, end, conf in _parse_label_file(os.path.join(label_dir, fname)):
                if conf < min_confidence or end <= start:
                    continue
                self.samples.append((skel_path, start, end, label_id))

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, skel_path, start, end):
        frames = _parse_skeleton_file(skel_path)  # (T_total, bodies, joints, 3)
        segment = frames[start:end, self.body_idx, :, :]  # (T_seg, joints, 3)
        T_seg = segment.shape[0]

        if T_seg == 0:
            segment = np.zeros((self.clip_len, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
        elif T_seg < self.clip_len:
            pad = np.tile(segment[-1:], (self.clip_len - T_seg, 1, 1))
            segment = np.concatenate([segment, pad], axis=0)
        else:
            # uniform sample clip_len frames across the segment
            idx = np.linspace(0, T_seg - 1, self.clip_len).astype(int)
            segment = segment[idx]

        # (T, J, C) -> (C, T, J)  to match ST-GCN's expected input layout
        return segment.transpose(2, 0, 1)

    def __getitem__(self, i):
        skel_path, start, end, label_id = self.samples[i]
        clip = self._load_clip(skel_path, start, end)
        return torch.from_numpy(clip).float(), label_id
