"""Train the U-Net subhalo detector on a Tier-0 population.

    source ~/myenv/bin/activate
    python scripts/train_detector.py --train data/train_fixed60 --out checkpoints/unet_v0 --epochs 25

RQ6 reproducibility note: Tsang+2024 trained on 5x10^5 images; we train on
8,000 (a compute/time budget decision, not an attempt to match their scale).
Results should be read as "does a small reimplementation show the same
*qualitative* failure modes", not as a number directly comparable to theirs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector.dataset import LensPatchDataset
from detector.unet import UNet


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--pos-weight", type=float, default=150.0, help="BCE positive-class weight, offsetting the ~1000:1 pixel imbalance (a ~4-5px-radius disk in a 64x64 frame)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = device()
    print(f"device: {dev}")

    ds = LensPatchDataset(args.train)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_ch=1, base=16).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=dev))

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for img, mask, _, _ in train_dl:
            img, mask = img.to(dev), mask.to(dev)
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, mask, _, _ in val_dl:
                img, mask = img.to(dev), mask.to(dev)
                val_loss += loss_fn(model(img), mask).item() * img.size(0)
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": time.time() - t0})
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({time.time()-t0:.0f}s)")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    torch.save({"model_state": model.state_dict(), "scale": ds.scale, "mask_radius_px": 2.0, "base": 16, "epoch": args.epochs}, args.out / "model_last.pt")
    torch.save({"model_state": best_state, "scale": ds.scale, "mask_radius_px": 2.0, "base": 16, "epoch": best_epoch, "val_loss": best_val}, args.out / "model_best.pt")
    print(f"best val_loss={best_val:.4f} at epoch {best_epoch}")
    with open(args.out / "training_log.json", "w") as f:
        json.dump({"args": vars(args) | {"train": str(args.train), "out": str(args.out)}, "history": history, "n_train": n_train, "n_val": n_val, "scale": ds.scale, "best_epoch": best_epoch, "best_val_loss": best_val}, f, indent=2, default=str)
    print(f"\nsaved model + training log to {args.out}/")


if __name__ == "__main__":
    main()
