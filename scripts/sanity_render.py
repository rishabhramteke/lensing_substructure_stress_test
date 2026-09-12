"""Render illustrative Tier-0 examples and save PNGs for a visual sanity check.

    source ~/myenv/bin/activate
    python scripts/sanity_render.py

Writes to data/sanity/. Not part of the dataset generation pipeline.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lensing.config import tier0_tsang, tier2_multipole_confounder
from lensing.simulate import LensRenderer, sample_truth

OUT = Path(__file__).resolve().parents[1] / "data" / "sanity"
OUT.mkdir(parents=True, exist_ok=True)


def stats(diff, ring):
    """Robust detectability numbers: raw max-pixel is dominated by single
    fold pixels near an Einstein ring and is a poor summary on its own."""
    l1 = float(np.abs(diff).sum())
    l2 = float(np.sqrt((diff ** 2).sum()))
    frac_of_ring_flux = l1 / max(float(ring.sum()), 1e-12)
    return {"max": float(np.abs(diff).max()), "L1": l1, "L2": l2, "frac_of_ring_flux": frac_of_ring_flux}


def show(ax, img, title, vmax=None):
    ax.imshow(np.arcsinh(img / (vmax or img.max() or 1) * 10), cmap="inferno", origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    rng = np.random.default_rng(7)

    # --- panel A: c=60 subhalo case, full vs no-subhalo vs residual ---
    # A deliberately off-axis, elliptical example rather than a random draw:
    # a near-perfect Einstein ring makes max-pixel residuals meaningless
    # (dominated by fold pixels), and arcs are the more illustrative,
    # more common case anyway.
    cfg = tier0_tsang("fixed60")
    cfg.subhalo.presence_prob = 1.0  # force a subhalo for this illustration
    renderer = LensRenderer(cfg)
    truth = sample_truth(cfg, rng)
    truth["lens_macro"].update({"q": 0.75, "e1": 0.10, "e2": 0.05, "phi_deg": 25.0})
    truth["source"].update({"x": 0.28, "y": 0.16})
    truth["subhalo"]["x"], truth["subhalo"]["y"] = 0.55, 0.75
    truth["subhalo"]["r_from_center"] = float(np.hypot(0.55, 0.75))
    r = renderer.render(truth, seed=1)
    print("=== subhalo truth ===")
    print({k: truth[k] for k in ("lens_macro", "subhalo")})

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    show(axes[0], r["noisy"]["full"], "noisy image (with subhalo)")
    show(axes[1], r["noiseless"]["full"], "noiseless (with subhalo)")
    show(axes[2], r["noiseless"]["no_subhalo"], "noiseless (no subhalo)")
    diff = r["noiseless"]["full"] - r["noiseless"]["no_subhalo"]
    st = stats(diff, r["noiseless"]["full"])
    axes[3].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[3].set_title(f"residual (subhalo signature)\nL1/ring flux={st['frac_of_ring_flux']*100:.2f}%, max={st['max']:.3f}", fontsize=8)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.suptitle("Tier 0 (Tsang+2024 operating point, c=60) — subhalo signature")
    fig.tight_layout()
    fig.savefig(OUT / "panel_A_subhalo.png", dpi=140)
    plt.close(fig)

    # --- panel B: low concentration (c=15) — same subhalo mass, weaker signature ---
    cfg15 = tier0_tsang("fixed15")
    cfg15.subhalo.presence_prob = 1.0
    r15 = LensRenderer(cfg15)
    truth15 = dict(truth)  # same macro+source+position, only concentration changes
    truth15["subhalo"] = dict(truth["subhalo"])
    from lensing.concentration import sample_concentration
    truth15["subhalo"]["concentration"] = float(sample_concentration(np.array(truth["subhalo"]["log10_M200"]), "fixed15", rng))
    truth15["subhalo"]["concentration_mode"] = "fixed15"
    r_lo = r15.render(truth15, seed=1)
    diff_lo = r_lo["noiseless"]["full"] - r_lo["noiseless"]["no_subhalo"]

    st60, st15 = stats(diff, r["noiseless"]["full"]), stats(diff_lo, r_lo["noiseless"]["full"])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    axes[0].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[0].set_title(f"c=60, same mass\nL1/ring flux={st60['frac_of_ring_flux']*100:.2f}%", fontsize=9)
    axes[1].imshow(diff_lo, cmap="RdBu_r", origin="lower", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[1].set_title(f"c=15, same mass\nL1/ring flux={st15['frac_of_ring_flux']*100:.2f}%", fontsize=9)
    for a in axes: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Concentration realism (RQ4): same M={10**truth['subhalo']['log10_M200']:.1e} Msun, same position")
    fig.tight_layout()
    fig.savefig(OUT / "panel_B_concentration.png", dpi=140)
    plt.close(fig)
    print(f"integrated signature ratio c60/c15 = {st60['L1']/max(st15['L1'],1e-12):.2f}x  "
          f"(max-pixel ratio {st60['max']/max(st15['max'],1e-12):.2f}x is misleading near a fold)")

    # --- panel C: multipole confounder, no subhalo present ---
    cfg_mp = tier2_multipole_confounder(am_over_thetaE=0.03, m=4)
    r_mp_render = LensRenderer(cfg_mp)
    truth_mp = sample_truth(cfg_mp, rng)
    r_mp = r_mp_render.render(truth_mp, seed=2)
    diff_mp = r_mp["noiseless"]["full"] - r_mp["noiseless"]["no_multipole"]
    st_mp = stats(diff_mp, r_mp["noiseless"]["full"])
    print("=== multipole truth (no subhalo present) ===")
    print({k: truth_mp[k] for k in ("lens_macro", "subhalo", "multipole")})

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    show(axes[0], r_mp["noiseless"]["full"], "noiseless (m=4 multipole, no subhalo)")
    show(axes[1], r_mp["noiseless"]["no_multipole"], "noiseless (smooth ellipse)")
    axes[2].imshow(diff_mp, cmap="RdBu_r", origin="lower", vmin=-np.abs(diff_mp).max(), vmax=np.abs(diff_mp).max())
    axes[2].set_title(f"residual (multipole signature)\nL1/ring flux={st_mp['frac_of_ring_flux']*100:.2f}%, max={st_mp['max']:.3f}", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.suptitle("Tier 2 confounder: am/thetaE=0.03, m=4 — the multipole 'impostor' (RQ2)")
    fig.tight_layout()
    fig.savefig(OUT / "panel_C_multipole.png", dpi=140)
    plt.close(fig)

    print(f"\nsubhalo   (c=60): L1/ring flux = {st['frac_of_ring_flux']*100:.2f}%   max={st['max']:.4f}")
    print(f"multipole (3%):   L1/ring flux = {st_mp['frac_of_ring_flux']*100:.2f}%   max={st_mp['max']:.4f}")
    print(f"per-pixel noise std: {r['noise_std']:.4f}")
    print(f"\nwrote 3 PNGs to {OUT}/")


if __name__ == "__main__":
    main()
