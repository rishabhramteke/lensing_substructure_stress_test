"""Two learned detectors beside the U-Net, trained on the same c=60 images and scored on the same test populations.

  nre             a per-lens neural ratio estimator (NRE-A, Hermans et al. 2020; the amortized-inference route of
                  Brehmer et al. 2019 and Anau Montel et al. 2023): a network f(x, theta) trained to separate joint
                  (image, parameter) pairs from shuffled ones, so exp f ~ p(theta|x)/p(theta), with theta = (subhalo
                  present, log10 M200, x, y) and a single "absent" value for subhalo-free images. The detection
                  statistic is the Bayes factor
                      ln B = ln E_{theta ~ p(theta|H1)} exp f(x, theta) - f(x, theta_H0),
                  estimated with 2048 prior draws, and the position is that of the highest-ratio draw.
  convnext_polar  a ConvNeXt classifier (Liu et al. 2022, timm `convnext_atto`) on the image resampled to polar
                  coordinates about the lens centre, so the arc lies along one axis and rotations are shifts (after
                  Campbell et al. 2026), trained with random rotations on the image-level label.

    python scripts/train_image_detector.py --arch nre --train data/train_fixed60_100k --out checkpoints/nre_100k_c60_seed0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import load_population, normalize  # noqa: E402

MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
EVAL_POPS = ["no_subhalo", "test_fixed60", "test_fixed15", "test_fixed30", "multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3",
             "ellmultipole_m4_a3", "decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"]
LOGM_MU, LOGM_SD, POS_SD = 9.5, 0.9, 0.8


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def theta_of(truths):
    th = np.zeros((len(truths), 4), np.float32)
    for i, t in enumerate(truths):
        s = t.get("subhalo")
        if s:
            th[i] = (1.0, (s["log10_M200"] - LOGM_MU) / LOGM_SD, s["x"] / POS_SD, s["y"] / POS_SD)
    return th


# --------------------------------------------------------------------------------------------- networks
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c_out)
        self.sk = nn.Sequential() if stride == 1 and c_in == c_out else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sk(x))


class NRE(nn.Module):
    """Spatially resolved ratio estimator: a U-Net-style encoder-decoder gives a feature map, the head reads it at the
    queried subhalo position (bilinear) and combines it with pooled global features and the queried mass; the
    subhalo-free hypothesis sees the global features only."""
    def __init__(self, width=32, half_extent=2.52):
        super().__init__()
        w = width; self.half = half_extent
        self.e1 = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1), nn.BatchNorm2d(w), nn.ReLU(), ResBlock(w, w, 1))           # 64
        self.e2 = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w, 1))                                     # 32
        self.e3 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w, 1))                                 # 16
        self.e4 = nn.Sequential(ResBlock(4 * w, 4 * w, 2), ResBlock(4 * w, 4 * w, 1))                                 # 8
        self.u3 = nn.Sequential(ResBlock(8 * w, 4 * w, 1)); self.u2 = nn.Sequential(ResBlock(6 * w, 2 * w, 1)); self.u1 = ResBlock(3 * w, w, 1)
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(w + 64 + 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def embed(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        f1 = self.e1(x); f2 = self.e2(f1); f3 = self.e3(f2); f4 = self.e4(f3)
        d3 = self.u3(torch.cat([up(f4), f3], 1)); d2 = self.u2(torch.cat([up(d3), f2], 1)); d1 = self.u1(torch.cat([up(d2), f1], 1))
        return d1, self.glob(f4)

    def ratio(self, e, th, rep=1):
        fmap, g = e
        if rep > 1:
            fmap = fmap.repeat_interleave(rep, 0); g = g.repeat_interleave(rep, 0)
        grid = (th[:, 2:4] * POS_SD / self.half).view(-1, 1, 1, 2)   # (x, y) in arcsec -> [-1, 1]; columns ~ x, rows ~ y
        loc = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True).view(len(th), -1) * th[:, :1]
        return self.head(torch.cat([loc, g, th[:, :2]], 1))[:, 0]


def polar_grid(n=64, n_r=64, n_phi=128, r_max_pix=31.5):
    """grid_sample grid mapping a (n x n) image to (n_r x n_phi) polar samples about the frame centre."""
    r = torch.linspace(0.5, r_max_pix, n_r)
    phi = torch.linspace(0, 2 * math.pi, n_phi + 1)[:-1]
    R, P = torch.meshgrid(r, phi, indexing="ij")
    c = (n - 1) / 2
    gx = (c + R * torch.cos(P)) / (n - 1) * 2 - 1
    gy = (c + R * torch.sin(P)) / (n - 1) * 2 - 1
    return torch.stack([gx, gy], -1)[None]


class ConvNeXtPolar(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.register_buffer("grid", polar_grid())
        self.net = timm.create_model("convnext_atto", pretrained=False, in_chans=1, num_classes=1)

    def forward(self, x, roll=None):
        p = F.grid_sample(x, self.grid.expand(len(x), -1, -1, -1), align_corners=True)
        if roll is not None:
            p = torch.stack([torch.roll(pi, int(k), dims=-1) for pi, k in zip(p, roll)])
        return self.net(p)[:, 0]


# --------------------------------------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, arch, images, prior_theta, dev, bs=16):
    model.eval()
    out, pos = np.zeros(len(images), np.float32), np.zeros((len(images), 2), np.float32)
    th1 = torch.from_numpy(prior_theta).to(dev); K = len(th1)
    th0 = torch.zeros(1, 4, device=dev)
    for a in range(0, len(images), bs):
        x = torch.from_numpy(images[a:a + bs]).unsqueeze(1).to(dev)
        if arch == "convnext_polar":
            out[a:a + len(x)] = model(x).cpu().numpy(); continue
        e = model.embed(x); nb = len(x)
        f0 = model.ratio(e, th0.expand(nb, -1))
        f1 = model.ratio(e, th1.repeat(nb, 1), rep=K).view(nb, K)
        out[a:a + len(x)] = (torch.logsumexp(f1, 1) - math.log(K) - f0).cpu().numpy()
        best = th1[f1.argmax(1)]
        pos[a:a + len(x)] = (best[:, 2:4] * POS_SD).cpu().numpy()
    return out, pos


def evaluate(model, arch, scale, prior_theta, dev, out_dir, data_root, pop_map=None):
    """pop_map: {standard name: population on disk}, to score a network on a re-rendered arm under the standard names."""
    S, P, T = {}, {}, {}
    pop_map = pop_map or {}
    for std in (list(pop_map) if pop_map else EVAL_POPS):
        pop = std
        disk = pop_map.get(std, std)
        if not (data_root / disk / "manifest.json").exists():
            continue
        imgs, truths, *_ = load_population(data_root / disk)
        S[pop], P[pop] = score(model, arch, normalize(imgs, scale), prior_theta, dev)
        T[pop] = truths
        np.save(out_dir / f"scores_{pop}.npy", S[pop])
        if arch == "nre":
            np.save(out_dir / f"positions_{pop}.npy", P[pop])
    thr = float(np.quantile(S["no_subhalo"], 0.90))
    res = {"arch": arch, "threshold_at_10pct_fpr": thr}

    def comp(pop):
        o = {}
        for lo, hi in MASS_BINS:
            sel = [i for i, t in enumerate(T[pop]) if t.get("subhalo") and lo <= t["subhalo"]["log10_M200"] < hi]
            o[f"{lo}-{hi}"] = {"n": len(sel), "completeness": float(np.mean(S[pop][sel] >= thr)) if sel else None}
        return o
    for pop in ("test_fixed60", "test_fixed15", "test_fixed30"):
        if pop in S:
            res[f"completeness_{pop}"] = comp(pop)
            pos = S[pop][[i for i, t in enumerate(T[pop]) if t.get("subhalo")]]
            neg = S["no_subhalo"]
            res[f"auc_{pop}"] = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    if "test_fixed15" in S:
        idx = [i for i, t in enumerate(T["test_fixed60"]) if t.get("subhalo")]
        d60, d15 = S["test_fixed60"][idx] >= thr, S["test_fixed15"][idx] >= thr
        res["paired_flip"] = {"c60_not_c15": int((d60 & ~d15).sum()), "c15_not_c60": int((d15 & ~d60).sum())}
    res["fpr"] = {pop: float(np.mean(S[pop] >= thr)) for pop in S if not pop.startswith("test_")}
    if arch == "nre":
        loc = {}
        for pop in ("test_fixed60", "test_fixed15"):
            ids = [i for i, t in enumerate(T[pop]) if t.get("subhalo") and S[pop][i] >= thr]
            d = [np.hypot(P[pop][i, 0] - T[pop][i]["subhalo"]["x"], P[pop][i, 1] - T[pop][i]["subhalo"]["y"]) for i in ids]
            loc[pop] = {"n_det": len(ids), "frac_within_0p16": float(np.mean(np.array(d) < 0.16)) if d else None,
                        "frac_within_0p3": float(np.mean(np.array(d) < 0.3)) if d else None}
        res["localization"] = loc
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------------------------- training
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["nre", "convnext_polar"], required=True)
    p.add_argument("--train", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--results", type=Path, default=None, help="default results/<out name>")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--eval-pops", nargs=4, metavar=("CLEAN", "C60", "C15", "MP"), help="score on these populations (a re-rendered arm) instead of Tier 0")
    args = p.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = device()
    args.out.mkdir(parents=True, exist_ok=True)
    res_dir = args.results or ROOT / "results" / args.out.name.replace("checkpoints/", "")
    res_dir.mkdir(parents=True, exist_ok=True)

    imgs, truths = [], []
    for t in args.train:
        a, tr, *_ = load_population(t); imgs.append(a); truths += tr
    raw = np.concatenate(imgs); del imgs
    scale = float(np.percentile(raw[:20000][raw[:20000] > 0], 90))
    X = normalize(raw, scale); del raw
    TH = theta_of(truths); Y = TH[:, 0].copy()
    perm = rng.permutation(len(X)); n_val = len(X) // 20
    va, tr_idx = perm[:n_val], perm[n_val:]
    prior_theta = TH[TH[:, 0] == 1][rng.choice(int((TH[:, 0] == 1).sum()), 2048, replace=False)]
    np.save(args.out / "prior_theta.npy", prior_theta)

    model = (NRE() if args.arch == "nre" else ConvNeXtPolar()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4 if args.arch == "nre" else 5e-2)
    steps = args.epochs * math.ceil(len(tr_idx) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    w_pos = float((Y[tr_idx] == 0).mean() / max((Y[tr_idx] == 1).mean(), 1e-9))   # balances the 9:1 classes

    def batch_loss(ix, train):
        x = torch.from_numpy(X[ix]).unsqueeze(1).to(dev)
        if train:   # the 4 dihedral flips about the lens centre; theta positions flipped with the image
            fx, fy = rng.random() < 0.5, rng.random() < 0.5
        else:
            fx = fy = False
        th = TH[ix].copy()
        if fx:
            x = torch.flip(x, [3]); th[:, 2] *= -1
        if fy:
            x = torch.flip(x, [2]); th[:, 3] *= -1
        if args.arch == "nre":
            th = torch.from_numpy(th).to(dev)
            e = model.embed(x)
            fj = model.ratio(e, th); fm = model.ratio(e, th[torch.randperm(len(th), device=dev)])
            return F.softplus(-fj).mean() + F.softplus(fm).mean()
        y = torch.from_numpy(Y[ix]).to(dev)
        roll = rng.integers(0, 128, len(ix)) if train else None
        logit = model(x, roll)
        wt = torch.where(y > 0, torch.full_like(y, w_pos), torch.ones_like(y))
        return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * wt).mean()

    log, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        model.train(); rng.shuffle(tr_idx); tl = []
        for a in range(0, len(tr_idx), args.batch_size):
            loss = batch_loss(np.sort(tr_idx[a:a + args.batch_size]), True)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            torch.manual_seed(1234)
            vl = float(np.mean([batch_loss(np.sort(va[a:a + 512]), False).item() for a in range(0, len(va), 512)]))
        log.append({"epoch": ep, "train": float(np.mean(tl)), "val": vl, "t": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if vl < best:
            best = vl
            torch.save({"model_state": model.state_dict(), "scale": scale, "arch": args.arch}, args.out / "model_best.pt")
    (args.out / "training_log.json").write_text(json.dumps({"n_train": len(tr_idx), "n_val": n_val, "scale": scale, "log": log}, indent=2))

    model.load_state_dict(torch.load(args.out / "model_best.pt", map_location=dev)["model_state"])
    pm = dict(zip(("no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a3"), args.eval_pops)) if args.eval_pops else None
    res = evaluate(model, args.arch, scale, prior_theta, dev, res_dir, args.data_root, pm)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
