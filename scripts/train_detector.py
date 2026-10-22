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
from detector.unet import UNet, build_model


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, nargs="+", required=True, help="one or more training populations (concatenated; the first sets the normalization scale)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--pos-weight", type=float, default=150.0, help="BCE positive-class weight, offsetting the ~1000:1 pixel imbalance (a ~4-5px-radius disk in a 64x64 frame)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", choices=["seg", "image", "image_lse"], default="seg",
                   help="seg: per-pixel BCE on the disc mask (the Tsang+2024 recipe); image: BCE on the image-level label applied to the "
                        "frame's maximum logit, the same score the evaluation thresholds (a multiple-instance objective, no position label); "
                        "image_lse: the same with a log-sum-exp (smooth maximum, temperature 1) pooling, so every pixel receives gradient")
    p.add_argument("--mask-radius", type=float, default=2.0, help="disc-mask radius in pixels for --objective seg")
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint's weights (fine-tuning, e.g. c=60 -> c=15)")
    p.add_argument("--arch", choices=["unet", "convnext_unet"], default="unet", help="convnext_unet: the U-Net decoder on a ConvNeXt encoder")
    p.add_argument("--augment", action="store_true", help="random dihedral transform (flips and 90-degree rotations about the lens centre) applied to "
                   "each training batch, image and target mask together")
    p.add_argument("--cosine", action="store_true", help="cosine learning-rate decay from --lr to 1%% of it over the run")
    args = p.parse_args()

    dev = device()
    print(f"device: {dev}")

    parts = [LensPatchDataset(args.train[0], mask_radius_px=args.mask_radius)]
    parts += [LensPatchDataset(t, scale=parts[0].scale, mask_radius_px=args.mask_radius) for t in args.train[1:]]
    if len(parts) == 1:
        ds = parts[0]
    else:
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset(parts); ds.scale = parts[0].scale; ds.truths = [t for q in parts for t in q.truths]
    print(f"training images: {len(ds)}")
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(arch=args.arch).to(dev)
    if args.init is not None:
        ck = torch.load(args.init, map_location=dev); model.load_state_dict(ck["model_state"])
        print(f"initialized from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.01 * args.lr) if args.cosine else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=dev))
    if args.objective in ("image", "image_lse"):
        frac = float(np.mean([t.get("subhalo") is not None for t in ds.truths]))
        img_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - frac) / frac, device=dev))   # balances the 9:1 subhalo fraction
        print(f"image-level objective, subhalo fraction {frac:.3f}, pos_weight {(1 - frac) / frac:.3f}")
        if args.objective == "image":
            loss_fn = lambda logits, mask, has_sub: img_loss(logits.amax(dim=(1, 2, 3)), has_sub)  # noqa: E731
        else:   # log-sum-exp minus log(N): equals the max for one dominant pixel, the mean-logit + const for a flat map
            npx = None
            def loss_fn(logits, mask, has_sub):
                flat = logits.flatten(1)
                return img_loss(torch.logsumexp(flat, dim=1) - np.log(flat.shape[1]), has_sub)
    else:
        _seg = loss_fn
        loss_fn = lambda logits, mask, has_sub: _seg(logits, mask)  # noqa: E731

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for img, mask, has_sub, _ in train_dl:
            img, mask, has_sub = img.to(dev), mask.to(dev), has_sub.float().to(dev)
            if args.augment:
                k = int(torch.randint(0, 4, (1,))); flip = bool(torch.randint(0, 2, (1,)))
                img, mask = torch.rot90(img, k, dims=(2, 3)), torch.rot90(mask, k, dims=(2, 3))
                if flip:
                    img, mask = torch.flip(img, dims=(3,)), torch.flip(mask, dims=(3,))
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask, has_sub)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= n_train
        if sched is not None:
            sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, mask, has_sub, _ in val_dl:
                img, mask, has_sub = img.to(dev), mask.to(dev), has_sub.float().to(dev)
                val_loss += loss_fn(model(img), mask, has_sub).item() * img.size(0)
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": time.time() - t0})
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({time.time()-t0:.0f}s)")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    torch.save({"model_state": model.state_dict(), "scale": ds.scale, "mask_radius_px": args.mask_radius, "objective": args.objective, "base": 16, "arch": args.arch, "epoch": args.epochs}, args.out / "model_last.pt")
    torch.save({"model_state": best_state, "scale": ds.scale, "mask_radius_px": args.mask_radius, "objective": args.objective, "base": 16, "arch": args.arch, "epoch": best_epoch, "val_loss": best_val}, args.out / "model_best.pt")
    print(f"best val_loss={best_val:.4f} at epoch {best_epoch}")
    with open(args.out / "training_log.json", "w") as f:
        json.dump({"args": vars(args) | {"train": [str(t) for t in args.train], "out": str(args.out)}, "history": history, "n_train": n_train, "n_val": n_val, "scale": ds.scale, "best_epoch": best_epoch, "best_val_loss": best_val}, f, indent=2, default=str)
    print(f"\nsaved model + training log to {args.out}/")


if __name__ == "__main__":
    main()
