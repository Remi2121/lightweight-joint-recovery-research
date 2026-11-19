"""
Step 1 of the research plan: get a working ST-GCN baseline trained on
PKU-MMD. Nothing about joint recovery happens here yet -- this just
proves the data -> model -> prediction pipeline works end to end.

Usage:
    python -m src.train --raw-dir data/pku_mmd/raw --epochs 30
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.pku_mmd import PKUMMDDataset
from src.models.st_gcn import STGCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/pku_mmd/raw")
    parser.add_argument("--clip-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--checkpoint", default="checkpoints/st_gcn_baseline.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = PKUMMDDataset(args.raw_dir, clip_len=args.clip_len)
    num_classes = max(label for _, _, _, label in full_dataset.samples) + 1
    print(f"Loaded {len(full_dataset)} clips, {num_classes} action classes")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = STGCN(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * clips.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += clips.size(0)

        scheduler.step()
        train_acc = total_correct / max(total_seen, 1)

        model.eval()
        val_correct, val_seen = 0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_seen += clips.size(0)
        val_acc = val_correct / max(val_seen, 1)

        print(f"Epoch {epoch+1}: train_loss={total_loss/max(total_seen,1):.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  -> saved new best checkpoint ({val_acc:.3f}) to {args.checkpoint}")

    print(f"Done. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
