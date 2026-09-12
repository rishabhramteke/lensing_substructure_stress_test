"""Visual sanity check for the Tier-1 COSMOS source upgrade.

    source ~/myenv/bin/activate
    python scripts/sanity_render_cosmos.py

Renders the same lens system with a Sersic source vs a real COSMOS postage
stamp (same macro lens, same subhalo, same noise draw -- only the source
changes), plus a grid of raw COSMOS stamps on their own so the deconvolution/
regularizing-PSF step (`../src/lensing/cosmos_source.py`) can be checked by
eye before it's trusted in a dataset. Writes to data/sanity/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lensing.config import tier0_tsang, tier1_cosmos
from lensing.simulate import LensRenderer, sample_truth
from lensing.cosmos_source import draw_cosmos_stamp

OUT = Path(__file__).resolve().parents[1] / "data" / "sanity"
OUT.mkdir(parents=True, exist_ok=True)


def show(ax, img, title, vmax=None):
    ax.imshow(np.arcsinh(img / (vmax or (img.max() or 1)) * 10), cmap="inferno", origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    rng = np.random.default_rng(11)

    # --- panel A: raw COSMOS stamps, to check the deconvolution step ---
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for i, ax in enumerate(axes.flat):
        stamp = draw_cosmos_stamp(rng, cosmos_index=int(rng.integers(0, 50000)))
        show(ax, stamp.image, f"idx={stamp.cosmos_index} rot={stamp.rotation_deg:.0f} deg")
    fig.suptitle("Raw COSMOS stamps (deconvolved + regularized), before lensing")
    fig.tight_layout()
    fig.savefig(OUT / "cosmos_raw_stamps.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'cosmos_raw_stamps.png'}")

    # --- panel B: identical lens system, Sersic vs COSMOS source ---
    cfg_s = tier0_tsang("fixed60")
    cfg_s.subhalo.presence_prob = 1.0
    cfg_c = tier1_cosmos("fixed60")
    cfg_c.subhalo.presence_prob = 1.0

    truth_s = sample_truth(cfg_s, np.random.default_rng(3))
    truth_s["lens_macro"].update({"q": 0.75, "e1": 0.10, "e2": 0.05, "phi_deg": 25.0})
    truth_s["source"].update({"x": 0.28, "y": 0.16})
    truth_s["subhalo"]["x"], truth_s["subhalo"]["y"] = 0.55, 0.75

    truth_c = sample_truth(cfg_c, np.random.default_rng(3))
    truth_c["lens_macro"].update({"q": 0.75, "e1": 0.10, "e2": 0.05, "phi_deg": 25.0})
    truth_c["source"].update({**truth_c["source"], "x": 0.28, "y": 0.16, "amp_scale": 1.5})
    truth_c["subhalo"]["x"], truth_c["subhalo"]["y"] = 0.55, 0.75

    r_s = LensRenderer(cfg_s).render(truth_s, seed=1)
    r_c = LensRenderer(cfg_c).render(truth_c, seed=1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    show(axes[0, 0], r_s["noisy"]["full"], "Sersic source: full (noisy)")
    show(axes[0, 1], r_s["noiseless"]["full"], "Sersic: noiseless")
    show(axes[0, 2], r_s["noiseless"]["full"] - r_s["noiseless"]["no_subhalo"], "Sersic: subhalo residual")
    show(axes[1, 0], r_c["noisy"]["full"], "COSMOS source: full (noisy)")
    show(axes[1, 1], r_c["noiseless"]["full"], "COSMOS: noiseless")
    show(axes[1, 2], r_c["noiseless"]["full"] - r_c["noiseless"]["no_subhalo"], "COSMOS: subhalo residual")
    fig.suptitle("Same macro lens + subhalo, Sersic vs real COSMOS source")
    fig.tight_layout()
    fig.savefig(OUT / "cosmos_vs_sersic.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'cosmos_vs_sersic.png'}")


if __name__ == "__main__":
    main()
