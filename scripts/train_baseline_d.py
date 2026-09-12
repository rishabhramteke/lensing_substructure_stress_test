"""Train the Family-D NPE posterior estimator on a Tier-0 population.

    source ~/myenv/bin/activate
    python scripts/train_baseline_d.py --train data/train_fixed60_big --out checkpoints/npe_v1

Target theta = log10(subhalo mass), floor=6.0 (+jitter) for no-subhalo images
(see src/baseline_d/__init__.py for why this, not a hierarchical population
fit).

v1 (2026-09-11), after v0's posterior collapsed near the training-set
marginal (first_results.md's "Family D" section): three changes bundled --
(1) the floor target is no longer an *exact* repeated constant (753 identical
copies of 6.0 in v0's 8,000 images); a continuous normalizing flow is the
wrong tool for a literal point mass mixed into a continuum, so a small jitter
(FLOOR +/- FLOOR_JITTER) turns it into a narrow-but-finite bump the flow can
actually represent. (2) `ImageEmbedding` deepened (see that module's
docstring). (3) more flow capacity (hidden_features, num_transforms) and,
by default, a larger training set via --train pointing at a bigger population.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_d.embedding import ImageEmbedding
from detector.dataset import load_population, normalize
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

FLOOR = 6.0
FLOOR_JITTER = 0.3  # v1 fix: avoid an exact repeated point-mass target (see module docstring)
PRIOR_LOW, PRIOR_HIGH = 6.0, 11.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    images, truths, pixel_scale, num_pix, manifest = load_population(args.train)
    scale = float(np.percentile(images[images > 0], 90))
    x = normalize(images, scale).reshape(len(truths), -1)
    rng = np.random.default_rng(args.seed)
    theta = np.array([[(t["subhalo"]["log10_M200"] if t.get("subhalo")
                         else FLOOR + rng.uniform(-FLOOR_JITTER, FLOOR_JITTER))] for t in truths], dtype=np.float32)

    x_t = torch.from_numpy(x).float()
    theta_t = torch.from_numpy(theta).float()
    n_val = int(len(theta_t) * args.val_frac)
    print(f"n={len(theta_t)}  n_subhalo={(theta[:,0] > FLOOR + FLOOR_JITTER).sum()}  "
          f"n_floor={(theta[:,0] <= FLOOR + FLOOR_JITTER).sum()}  scale={scale:.4f}")

    prior = BoxUniform(low=torch.tensor([PRIOR_LOW]), high=torch.tensor([PRIOR_HIGH]))
    embed = ImageEmbedding(num_pix=num_pix, out_dim=64)
    density_estimator = posterior_nn("maf", embedding_net=embed, hidden_features=64, num_transforms=8, z_score_x="none")

    inference = NPE(prior=prior, density_estimator=density_estimator, device="cpu", show_progress_bars=True)
    inference.append_simulations(theta_t, x_t)

    t0 = time.time()
    de = inference.train(training_batch_size=64, max_num_epochs=args.max_epochs, validation_fraction=args.val_frac, stop_after_epochs=20)
    elapsed = time.time() - t0
    posterior = inference.build_posterior(de)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    log = {"n_train": len(theta_t) - n_val, "n_val": n_val, "scale": scale, "elapsed_s": elapsed,
           "summary": {k: v for k, v in inference.summary.items()} if hasattr(inference, "summary") else {}}
    (args.out / "training_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"\ntrained in {elapsed:.0f}s, saved posterior to {args.out}/posterior.pkl")
    print("summary:", log["summary"])


if __name__ == "__main__":
    main()
