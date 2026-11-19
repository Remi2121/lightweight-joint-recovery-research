"""
Step 2 of the research plan: measure how much accuracy is lost when
joints go missing, and how much a (currently naive) recovery method
wins back. This produces the first real evidence for the research
question -- before building the compute-selective version.

Compares three conditions on the same held-out clips:
    1. clean            -- no occlusion (upper bound)
    2. occluded          -- synthetic occlusion, no recovery (lower bound)
    3. occluded+recovered -- synthetic occlusion, then temporal-interpolation recovery

Usage:
    python -m src.evaluate_occlusion --raw-dir data/pku_mmd/raw \
        --checkpoint checkpoints/st_gcn_baseline.pt
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.pku_mmd import PKUMMDDataset
from src.models.st_gcn import STGCN
from src.occlusion import occlude_bursty, missing_rate
from src.recovery_baseline import recover_temporal_interpolation


def run_condition(model, loader, device, mode, rng):
    correct, seen = 0, 0
    rates = []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)

            if mode != "clean":
                occluded_batch = []
                for i in range(clips.size(0)):
                    occluded, mask = occlude_bursty(clips[i].cpu(), rng=rng)
                    rates.append(missing_rate(mask))
                    if mode == "occluded+recovered":
                        occluded = recover_temporal_interpolation(occluded, mask)
                    occluded_batch.append(occluded)
                clips = torch.stack(occluded_batch).to(device)

            logits = model(clips)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += clips.size(0)

    acc = correct / max(seen, 1)
    avg_missing = float(np.mean(rates)) if rates else 0.0
    return acc, avg_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/pku_mmd/raw")
    parser.add_argument("--clip-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default="checkpoints/st_gcn_baseline.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    dataset = PKUMMDDataset(args.raw_dir, clip_len=args.clip_len)
    num_classes = max(label for _, _, _, label in dataset.samples) + 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = STGCN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"{'Condition':<22} {'Accuracy':>10} {'Avg. missing rate':>20}")
    for mode in ["clean", "occluded", "occluded+recovered"]:
        acc, missing = run_condition(model, loader, device, mode, rng)
        print(f"{mode:<22} {acc:>10.3f} {missing:>20.1%}")


if __name__ == "__main__":
    main()
