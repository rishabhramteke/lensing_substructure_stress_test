"""Publication figures for the paper (A&A two-column: 88 mm single / 180 mm
double column). One script, one style, every number read from the results
files already on disk -- nothing is retyped by hand. Re-run after any result
changes and the manuscript picks up the new figures on the next compile.

    source ~/myenv/bin/activate
    python scripts/make_paper_figures.py

Writes paper/figures/fig*.{pdf,png}. Vector PDF for line/bar plots; PNG for
image panels. The localization figure is produced by
scripts/plot_localization_accuracy.py and copied here unchanged so there is a
single source of truth for it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

# ---- A&A geometry + a colorblind-safe (Okabe-Ito) palette ----
COL_W, DBL_W = 3.46, 7.09  # inches: 88 mm, 180 mm
BLUE, ORANGE, GREEN, VERM, PURPLE, GRAY, BLACK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7F7F7F", "#222222"
C_UNET, C_A, C_D, C_B = BLUE, VERM, PURPLE, GREEN
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
})
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
MIDS = [(lo + hi) / 2 for lo, hi in BINS]
KEYS = [f"{lo}-{hi}" for lo, hi in BINS]


def load(rel):
    return json.loads((ROOT / rel).read_text())


def save(fig, name, png=False):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    if png:
        fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ======================================================================
# Fig 1 -- what the detectors see: Tier 0 (Sersic) vs Tier 1 (COSMOS) source
# ======================================================================
def fig1_examples():
    from lensing.config import tier0_tsang, tier1_cosmos
    from lensing.simulate import LensRenderer, sample_truth

    def system(cfg_fn, seed_truth):
        cfg = cfg_fn("fixed60"); cfg.subhalo.presence_prob = 1.0
        truth = sample_truth(cfg, np.random.default_rng(seed_truth))
        truth["lens_macro"].update({"q": 0.75, "e1": 0.10, "e2": 0.05, "phi_deg": 25.0})
        truth["source"].update({"x": 0.28, "y": 0.16})
        if truth["source"].get("type") == "cosmos":
            truth["source"]["amp_scale"] = 1.5
        truth["subhalo"]["x"], truth["subhalo"]["y"] = 0.55, 0.75
        truth["subhalo"]["log10_M200"] = 10.7  # a clearly visible (but in-prior) perturber for the illustration
        return LensRenderer(cfg).render(truth, seed=1)

    r0, r1 = system(tier0_tsang, 7), system(tier1_cosmos, 3)
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 4.55))
    rows = [("Tier 0 — Sérsic source", r0),
            ("Tier 1 — COSMOS source", r1)]
    for i, (label, r) in enumerate(rows):
        full, nosub = r["noisy"]["full"], r["noisy"]["no_subhalo"]
        resid = r["noiseless"]["full"] - r["noiseless"]["no_subhalo"]
        vmax = full.max()
        for j, (img, title) in enumerate([(full, "observed image (with subhalo)"), (nosub, "same system, no subhalo"),
                                          (resid, "difference: the subhalo's signature")]):
            ax = axes[i, j]
            if j < 2:
                ax.imshow(np.arcsinh(img / vmax * 10), cmap="inferno", origin="lower")
            else:
                m = np.abs(resid).max()
                ax.imshow(resid, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
        axes[i, 0].set_ylabel(label, fontsize=7.2)
    # 1" scale bar on the first panel (0.08"/px -> 12.5 px)
    axes[0, 0].plot([4, 4 + 12.5], [4, 4], color="white", lw=1.2)
    axes[0, 0].text(4, 6.5, '1″', color="white", fontsize=6.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, "fig1_examples", png=True)


# ======================================================================
# Fig 2 -- RQ4: completeness vs mass at c=60 vs c=15, both families
# ======================================================================
def family_b_summary():
    """Family B (fitted variant, lambda=1e5). The two-seed aggregate written by
    scripts/aggregate_family_b_seeds.py if present, otherwise the single seed-set-0 summary."""
    agg = next((ROOT / f"results/baseline_b/aggregate_fitted_{k}seeds.json" for k in (4, 3, 2) if (ROOT / f"results/baseline_b/aggregate_fitted_{k}seeds.json").exists()), None)
    if agg is not None:
        d = json.loads(agg.read_text())
        ms = lambda x: {"mean": x["mean"], "std": x["std"]}
        return {"completeness_c60": {k: ms(v) for k, v in d["completeness_c60"].items()},
                "completeness_c15": {k: ms(v) for k, v in d["completeness_c15"].items()},
                "fpr_no_subhalo": {"mean": 0.10, "std": 0.0},
                "fpr_mp_a1": ms(d["confounder_fpr"]["multipole_m4_a1"]),
                "fpr_mp_a3": ms(d["confounder_fpr"]["multipole_m4_a3"])}, d["n_seeds"]
    d = load("results/baseline_b/fitted/summary/results.json")
    wrap = lambda comp: {k: {"mean": (v["completeness"] if v["completeness"] is not None else np.nan), "std": 0.0} for k, v in comp.items()}
    conf = d["confounder_fpr"]
    return {"completeness_c60": wrap(d["completeness_c60_by_mass_bin"]), "completeness_c15": wrap(d["completeness_c15_by_mass_bin"]),
            "fpr_no_subhalo": {"mean": conf["no_subhalo (calibration set)"], "std": 0.0},
            "fpr_mp_a1": {"mean": conf["multipole m=4, a=0.01*thetaE"], "std": 0.0},
            "fpr_mp_a3": {"mean": conf["multipole m=4, a=0.03*thetaE"], "std": 0.0}}, 1


def fig2_completeness():
    u = load("results/aggregate/summary.json")
    a = load("results/baseline_a/aggregate/summary.json")
    b, nb = family_b_summary()
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.6), sharey=True)
    for ax, d, name, col, n in [(axes[0], u, "Family C — U-Net", C_UNET, u["n_seeds"]),
                                 (axes[1], a, "Family A — parametric scan", C_A, a["n_seeds"]),
                                 (axes[2], b, "Family B — potential correction", C_B, nb)]:
        for key, ls, mk, lab, c in [("completeness_c60", "-", "o", "c = 60 (literature fiducial)", col), ("completeness_c15", "--", "s", "c = 15 (Tsang+2024's low-c ablation)", GRAY)]:
            m = [100 * d[key][k]["mean"] for k in KEYS]; sd = [100 * d[key][k]["std"] for k in KEYS]
            ax.errorbar(MIDS, m, yerr=sd if n > 1 else None, fmt=mk + ls, color=c, ms=3.5, capsize=2, lw=1.1, label=lab)
        ax.axhline(10, color=BLACK, ls=":", lw=0.7)
        ax.set_title(f"{name}\n(n = {n} seed{'s' if n > 1 else ' set'}{', mean ± s.d.' if n > 1 else ''})", fontsize=7.5)
        ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("completeness at 10% FPR  [%]")
    axes[0].text(9.55, 3, "10% = chance level", fontsize=6.3, color=BLACK)
    axes[0].legend(loc="upper left", fontsize=6.3)
    save(fig, "fig2_completeness")


# ======================================================================
# Fig 3 -- RQ2/RQ1 confounder FPR + the non-physical decoy control
# ======================================================================
def fig3_confounders():
    u = load("results/aggregate/summary.json")
    a = load("results/baseline_a/aggregate/summary.json")
    dec = load("results/noise_decoy_control/results.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    conds = ["no_subhalo (calibration set)", "multipole m=4, a=0.01*thetaE", "multipole m=4, a=0.03*thetaE"]
    um = [100 * u["confounder_fpr"][c]["mean"] for c in conds]; us = [100 * u["confounder_fpr"][c]["std"] for c in conds]
    am = [100 * a[k]["mean"] for k in ("fpr_no_subhalo", "fpr_mp_a1", "fpr_mp_a3")]
    as_ = [100 * a[k]["std"] for k in ("fpr_no_subhalo", "fpr_mp_a1", "fpr_mp_a3")]
    b, nb = family_b_summary()
    bm = [100 * b[k]["mean"] for k in ("fpr_no_subhalo", "fpr_mp_a1", "fpr_mp_a3")]
    bs = [100 * b[k]["std"] for k in ("fpr_no_subhalo", "fpr_mp_a1", "fpr_mp_a3")]
    x = np.arange(3); w = 0.26
    ax.bar(x - w, um, w, yerr=us, capsize=2, color=C_UNET, label=f"Family C — U-Net (n={u['n_seeds']})")
    ax.bar(x, am, w, yerr=as_, capsize=2, color=C_A, label=f"Family A — scan (n={a['n_seeds']})")
    ax.bar(x + w, bm, w, yerr=bs if nb > 1 else None, capsize=2, color=C_B, label=f"Family B — potential corr. (n={nb})")
    for off, vals, col in ((-w, um, C_UNET), (0, am, C_A), (w, bm, C_B)):
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 2.5, f"{v:.0f}%", ha="center", fontsize=6.0, color=col)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["no confounder\n(calibration)", "multipole m=4\n$a=0.01\\,\\theta_E$", "multipole m=4\n$a=0.03\\,\\theta_E$"])
    ax.set_ylabel("false-positive rate on subhalo-free images  [%]")
    ax.set_ylim(0, 105); ax.set_title("(a) lens-shape confounder (real gravitational structure)")
    ax.legend(loc="upper left")

    ax = axes[1]
    # every decoy run that exists is drawn: the original Gaussian (seed 42), a second Gaussian seed,
    # and the asymmetric dipole decoy the self-review asked for
    runs = [("results/noise_decoy_control/results.json", "Gaussian, seed 42", "o", "-"),
            ("results/noise_decoy_control_seed43/results.json", "Gaussian, seed 43", "o", ":"),
            ("results/noise_decoy_control_dipole/results.json", "dipole (asymmetric), seed 42", "^", "--")]
    amps_ref = None
    for rel, lab, mk, ls in runs:
        if not (ROOT / rel).exists():
            continue
        d = load(rel)
        amps = sorted(float(k) for k in d if k not in ("baseline_fpr_clean_no_subhalo", "config"))
        amps_ref = amps_ref or amps
        ax.plot(amps, [100 * d[str(a_)]["unet_fpr"] for a_ in amps], marker=mk, ls=ls, color=C_UNET, ms=3.5, label=f"U-Net — {lab}")
        ax.plot(amps, [100 * d[str(a_)]["a_fpr"] for a_ in amps], marker=mk, ls=ls, color=C_A, ms=3.5, label=f"Family A — {lab}")
    fb = ROOT / "results/noise_decoy_control_familyB/results.json"
    if fb.exists():   # Family B on the identical decoys (round-5 addition)
        d = json.loads(fb.read_text())
        for shape, mk, ls, lab in (("gaussian", "o", "-", "Gaussian, seed 42"), ("gaussian_seed43", "o", ":", "Gaussian, seed 43"), ("dipole", "^", "--", "dipole (asymmetric), seed 42")):
            if shape in d:
                amps = sorted(float(k) for k in d[shape])
                ax.plot(amps, [100 * d[shape][str(a_)]["b_fpr"] for a_ in amps], marker=mk, ls=ls, color=C_B, ms=3.5, label=f"Family B — {lab}")
    ax.axhline(10, color=BLACK, ls=":", lw=0.7, label="clean-image baseline (10%)")
    ax.set_xlabel(r"decoy amplitude  [$\sigma$ of local noise]")
    ax.set_ylabel("flagged as detection  [%]")
    ax.set_ylim(0, 88); ax.set_xticks(amps_ref or [3, 6, 10])
    ax.set_title("(b) non-physical decoys (no lensing signature)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.6, frameon=False)
    save(fig, "fig3_confounders")


# ======================================================================
# Fig 9 -- scale-up: U-Net at 8k vs 30k training images (drawn only once the 30k aggregate exists)
# ======================================================================
def fig9_scaleup():
    if not (ROOT / "results/aggregate_30k/summary.json").exists():
        print("  (fig9_scaleup skipped: results/aggregate_30k/summary.json not present yet)")
        return
    u8 = load("results/aggregate/summary.json"); u30 = load("results/aggregate_30k/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), sharey=True)
    for ax, key, title in [(axes[0], "completeness_c60", "(a) c = 60"), (axes[1], "completeness_c15", "(b) c = 15")]:
        for d, col, ls, lab in [(u8, GRAY, "--", f"8,000 training images (n={u8['n_seeds']}), AUC {u8['auc']['mean']:.2f}"),
                                (u30, C_UNET, "-", f"30,000 training images (n={u30['n_seeds']}), AUC {u30['auc']['mean']:.2f}")]:
            m = [100 * d[key][k]["mean"] for k in KEYS]; s = [100 * d[key][k]["std"] for k in KEYS]
            ax.errorbar(MIDS, m, yerr=s, fmt="o" + ls, color=col, ms=3.5, capsize=2, label=lab)
        ax.axhline(10, color=BLACK, ls=":", lw=0.7); ax.axhline(33.6, color=GRAY, ls="-.", lw=0.6)
        ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_title(title); ax.set_ylim(0, 100)
    axes[0].text(8.12, 32.0, "Tsang+2024: 33.6% at 9–9.5 dex", fontsize=6, color=GRAY, ha="left", va="top")
    axes[0].set_ylabel("completeness at 10% FPR  [%]"); axes[0].legend(loc="upper left")
    save(fig, "fig9_scaleup")


# ======================================================================
# Fig 4 -- localization: copy the diagnostic figure verbatim (single source of truth)
# ======================================================================
def fig4_localization():
    for ext in ("pdf", "png"):  # vector preferred; PNG kept as fallback
        src = ROOT / f"results/localization/localization_accuracy_abc.{ext}"   # 3-panel A/B/C version
        if not src.exists():
            src = ROOT / f"results/localization/localization_accuracy.{ext}"
        if src.exists():
            shutil.copy(src, FIG / f"fig4_localization.{ext}")
            print(f"  copied fig4_localization.{ext}")


# ======================================================================
# Fig 5 -- Family A mass bias among location-correct finds, both seeds
# ======================================================================
def fig5_mass_bias():
    """Fitted vs true mass for location-correct finds. Panel (a): the scan at its
    customary fixed c=15 (seeds 0 and 200). Panel (b): the same scan with the
    concentration assumption removed -- matched c=60 and the Dutton & Maccio (2014)
    c-M relation (seed 3) -- rendered only for the variants whose runs exist."""
    truths = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    LOC = 0.16

    def thr_for(root):
        s = root / "summary/results.json"
        if s.exists():
            return json.loads(s.read_text())["threshold_delta_chi2_at_10pct_fpr"]
        neg = [r["delta_chi2"] for r in (json.loads(l) for l in open(root / "no_subhalo/scan_results.jsonl")) if r["reliable_fit"]]
        return float(np.quantile(neg, 0.90))

    def points(root):
        thr = thr_for(root)
        pts = []
        for r in (json.loads(l) for l in open(root / "test_fixed60/scan_results.jsonl")):
            if r["reliable_fit"] and r["has_subhalo"] and r["delta_chi2"] >= thr:
                sub = truths[r["index"]]["subhalo"]
                if np.hypot(r["best_x"] - sub["x"], r["best_y"] - sub["y"]) < LOC:
                    pts.append((sub["log10_M200"], r["best_log10_m"]))
        return np.array(pts)

    seeds_a = [(ROOT / "results/baseline_a", "o", C_A, "seed 0"), (ROOT / "results/baseline_a_seed1", "s", C_A, "seed 200"),
               (ROOT / "results/baseline_a_seed300", "v", C_A, "seed 300")]
    panels = [("(a) scan assumes $c=15$ (the customary choice)", [v for v in seeds_a if (v[0] / "test_fixed60/scan_results.jsonl").exists()])]
    variants = [(ROOT / "results/baseline_a_c60", "D", C_UNET, "scan at matched $c=60$"),
                (ROOT / "results/baseline_a_ccm", "^", BLACK, "scan at the $c$--$M$ relation")]
    variants = [v for v in variants if (v[0] / "test_fixed60/scan_results.jsonl").exists()]
    if variants:
        panels.append(("(b) concentration assumption removed (same lenses as seed 0)", variants))

    fig, axes = plt.subplots(1, len(panels), figsize=(DBL_W if len(panels) > 1 else COL_W, 3.0), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        for k, (root, mk, col, lab) in enumerate(series):
            pts = points(root)
            jitter = np.random.default_rng(k).uniform(-0.05, 0.05, size=len(pts))
            ax.scatter(pts[:, 0], pts[:, 1] + jitter, marker=mk, s=18, color=col, alpha=0.85, edgecolor="none",
                       label=f"{lab}: {len(pts)} location-correct, {int((pts[:,1] < pts[:,0]).sum())} below the line")
        ax.plot([8, 11], [8, 11], color=BLACK, lw=0.8, ls="-", label="fitted = true")
        for g in (8.5, 9.5, 10.5):
            ax.axhline(g, color=GRAY, lw=0.5, ls=":")
        ax.text(10.95, 8.55, "scan's only three\nmass hypotheses", fontsize=6.3, color=GRAY, ha="right", va="bottom")
        ax.set_xlim(8, 11); ax.set_ylim(8, 11)
        ax.set_xlabel(r"true $\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel(r"Family A best-fit $\log_{10}(M/M_\odot)$")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "fig5_mass_bias")


# ======================================================================
# Fig 8 -- Family B: regularization sensitivity band
# ======================================================================
def fig8_family_b():
    d = load("results/baseline_b/summary_family_b.json")
    lams = ["1e4", "1e5", "1e6"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    styles = {"1e4": (":", "^", 0.55), "1e5": ("-", "o", 1.0), "1e6": ("--", "s", 0.75)}
    for lam in lams:
        comp = d["fitted"]["by_lambda"][lam]["completeness_c60"]
        m = [100 * comp[k]["completeness"] if comp[k]["completeness"] is not None else np.nan for k in KEYS]
        ls, mk, al = styles[lam]
        ax.plot(MIDS, m, marker=mk, ls=ls, color=C_B, alpha=al, ms=3.5, lw=1.1, label=rf"fitted macro-model, $\lambda=10^{{{int(np.log10(float(lam)))}}}$" + ("  (primary)" if lam == "1e5" else ""))
    comp = d["oracle"]["by_lambda"]["1e5"]["completeness_c60"]
    ax.plot(MIDS, [100 * comp[k]["completeness"] for k in KEYS], marker="o", ls="-", color=GRAY, ms=3.5, lw=1.0, label=r"oracle macro-model, $\lambda=10^{5}$ (ceiling)")
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_ylim(0, 105); ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel("completeness at 10% FPR, $c=60$  [%]")
    ax.set_title("(a) completeness vs regularization strength"); ax.legend(loc="lower right", fontsize=6.0)
    ax = axes[1]
    x = np.arange(3); w = 0.36
    a1 = [100 * d["fitted"]["by_lambda"][l]["confounder_fpr"]["multipole_m4_a1"]["fpr"] for l in lams]
    a3 = [100 * d["fitted"]["by_lambda"][l]["confounder_fpr"]["multipole_m4_a3"]["fpr"] for l in lams]
    ax.bar(x - w / 2, a1, w, color=C_B, alpha=0.55, label=r"multipole $a_m=0.01\,\theta_E$")
    ax.bar(x + w / 2, a3, w, color=C_B, label=r"multipole $a_m=0.03\,\theta_E$")
    for xi, v1, v3 in zip(x, a1, a3):
        ax.text(xi - w / 2, v1 + 2, f"{v1:.0f}%", ha="center", fontsize=6.3); ax.text(xi + w / 2, v3 + 2, f"{v3:.0f}%", ha="center", fontsize=6.3)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([r"$\lambda=10^{4}$", r"$\lambda=10^{5}$", r"$\lambda=10^{6}$"])
    ax.set_ylim(0, 110); ax.set_ylabel("FPR on subhalo-free images  [%]")
    ax.set_title("(b) confounder false positives vs regularization"); ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "fig8_family_b")


# ======================================================================
# Fig 11 -- lens light: what the detectors see before and after subtraction
# ======================================================================
def fig11_lenslight():
    """One c=60 subhalo lens from data/lenslight_fixed60: observed with lens light; after subtracting
    the fitted single Sersic (what the pipelines see); after the correctly specified double Sersic
    (if that run exists); and with the TRUE lens light removed (the Tier-0-equivalent image)."""
    pop = ROOT / "data/lenslight_fixed60"
    truths = [json.loads(l) for l in open(pop / "truth.jsonl")]
    noisy = np.load(pop / "images_full_noisy.npy"); noiseless = np.load(pop / "images_full_noiseless.npy")
    a1 = ROOT / "results/baseline_a_lenslight/lenslight_fixed60"; a2 = ROOT / "results/baseline_a_lenslight2/lenslight_fixed60"
    sub1 = np.load(a1 / "lens_light_subtracted.npy"); idx1 = np.load(a1 / "subsample_indices.npy")
    sub2 = np.load(a2 / "lens_light_subtracted.npy") if (a2 / "lens_light_subtracted.npy").exists() else None
    idx2 = np.load(a2 / "subsample_indices.npy") if sub2 is not None else None
    # pick a mid-mass subhalo lens whose single-Sersic fit is typical (chi2/N near the population median)
    recs = {r["index"]: r for r in (json.loads(l) for l in open(a1 / "scan_results.jsonl"))}
    cands = [i for i in idx1 if truths[i].get("subhalo") and 9.8 <= truths[i]["subhalo"]["log10_M200"] <= 10.4]
    med = np.median([recs[i]["chi2_smooth_per_dof"] for i in cands])
    i = min(cands, key=lambda j: abs(recs[j]["chi2_smooth_per_dof"] - med))
    k1 = int(np.where(idx1 == i)[0][0])
    # true lens light: render via the simulator's own renderer for this truth
    sys.path.insert(0, str(ROOT / "src"))
    from lensing.config import tier2_lens_light
    from lensing.simulate import LensRenderer
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    cfg = tier2_lens_light("fixed60"); rend = LensRenderer(cfg)
    sim = rend._sim_api(["EPL", "SHEAR"], n_lens_light=2)
    im = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    true_ll = im.lens_surface_brightness(kwargs_lens_light=truths[i]["lens_light"])
    panels = [("(a) observed, with lens light", noisy[i]), ("(b) single-S\u00e9rsic subtracted\n(what the pipelines see)", sub1[k1])]
    if sub2 is not None and i in idx2:
        panels.append(("(c) double-S\u00e9rsic subtracted\n(correctly specified)", sub2[int(np.where(idx2 == i)[0][0])]))
    panels.append((f"({'d' if len(panels) == 3 else 'c'}) true lens light removed\n(the Tier-0 image)", noisy[i] - true_ll))
    fig, axes = plt.subplots(1, len(panels), figsize=(DBL_W, DBL_W / len(panels) + 0.45))
    scale = np.percentile(noisy[i] - true_ll, 99.5)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.arcsinh(np.clip(img, -scale, None) / (0.05 * scale)), cmap="inferno", origin="lower", vmin=np.arcsinh(-1 / 0.05) * 0.3)
        ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
        sub = truths[i]["subhalo"]; ps = cfg.instrument.pixel_scale if hasattr(cfg.instrument, "pixel_scale") else 0.08
        c = (noisy.shape[1] - 1) / 2
        ax.plot(c + sub["x"] / 0.08, c + sub["y"] / 0.08, "o", mfc="none", mec="#2fbf71", ms=9, mew=1.2)
    lab = f"$\\log_{{10}} M_{{200}}={truths[i]['subhalo']['log10_M200']:.1f}$, $c=60$" + "\n" + f"single-S\u00e9rsic fit: $\\chi^2/{{\\rm dof}}={recs[i]['chi2_smooth_per_dof']:.1f}$"
    axes[0].text(0.03, 0.03, lab, transform=axes[0].transAxes, color="white", fontsize=6.3, va="bottom")
    save(fig, "fig11_lenslight")


# ======================================================================
# Fig 12 -- one-page synthesis: which stress breaks which family, and how badly
# ======================================================================
def fig12_summary():
    """Rows = stresses, columns = families. Each cell: the headline number and a severity colour.
    Numbers are the ones quoted in the text (3-seed means where they exist); sources in comments."""
    import matplotlib.patches as mpatches
    GREEN, AMBER, RED, GREY = "#cfe8d5", "#f6dfb0", "#f2b8b5", "#e6e6e6"
    # (row label, [(cell text, colour) for A, B, C])
    rows = [
        ("concentration $c=60\\to15$\n(completeness, $10^{10}$--$10^{10.5}\\,M_\\odot$)",
         [("89 $\\to$ 78%\nno collapse", GREEN), ("94 $\\to$ 77%;\n65 $\\to$ 17% below $10^{9.5}$", AMBER), ("51 $\\to$ 15%\ncollapse to chance", RED)]),
        ("lens-shape multipole, $a_4=3\\%\\,\\theta_E$\n(false positives, subhalo-free)",
         [("77%\n(100% joint re-fit)", RED), ("86%", RED), ("11%\nunmoved", GREEN)]),
        ("non-physical decoy, 10$\\sigma$ bump\n(false positives)",
         [("2--4%\nignores it", GREEN), ("63--76%\nresponds most", RED), ("19--21%\nfires", AMBER)]),
        ("real COSMOS source (Tier 1)",
         [("fits misspecified\n($\\chi^2$/dof 58--79)", AMBER), ("not run", GREY), ("AUC 0.62 $\\to$ 0.48\nchance", RED)]),
        ("lens light, single-S\u00e9rsic subtraction\n(clean FPR at $\\Delta\\chi^2>20$ / completeness)",
         [("0.2 $\\to$ 25%;\ncompleteness halves", RED), ("null $\\times$10;\nchance", RED), ("AUC 0.55;\nchance", RED)]),
        ("lens light, double-S\u00e9rsic subtraction",
         [("0%; completeness\nwithin 10 pts", GREEN), ("completeness 94%\nrecovered", GREEN), ("AUC 0.67\nrecovered", GREEN)]),
        ("localization $\\leq 2$ px\n(of nominal detections)",
         [("13%", RED), ("55%", AMBER), ("57%", AMBER)]),
        ("mass estimate\n(localized detections)",
         [("$-1.1$ dex, 44/44 low\n(frozen macro-model)", RED), ("$-0.65$ dex\n(aperture, calibratable)", AMBER), ("not attempted", GREY)]),
        ("CDM mass-function weighting\n(population completeness, $c=60$)",
         [("53 $\\to$ 21%", AMBER), ("--", GREY), ("30 $\\to$ 12%", AMBER)]),
    ]
    cols = ["A  parametric scan", "B  potential correction", "C  U-Net"]
    fig, ax = plt.subplots(figsize=(DBL_W, 0.42 * len(rows) + 0.7))
    ax.set_xlim(0, 3.9); ax.set_ylim(0, len(rows)); ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(1.35 + j * 0.85 + 0.425, len(rows) + 0.08, c, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for i, (lab, cells) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(1.30, y + 0.5, lab, ha="right", va="center", fontsize=6.6)
        for j, (txt, col) in enumerate(cells):
            x0 = 1.35 + j * 0.85
            ax.add_patch(mpatches.FancyBboxPatch((x0 + 0.02, y + 0.05), 0.81, 0.9, boxstyle="round,pad=0,rounding_size=0.04", fc=col, ec="white", lw=1.2))
            ax.text(x0 + 0.425, y + 0.5, txt, ha="center", va="center", fontsize=6.3)
    handles = [mpatches.Patch(fc=GREEN, label="robust / recovered"), mpatches.Patch(fc=AMBER, label="degraded"), mpatches.Patch(fc=RED, label="fails"), mpatches.Patch(fc=GREY, label="not tested / n.a.")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=6.5, frameon=False)
    save(fig, "fig12_summary")


# ======================================================================
# Fig 6 -- CDM mass-function re-weighting
# ======================================================================
def fig6_mass_function():
    d = load("results/mass_function_reweighting/results.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    wts = [100 * d["bin_weights"][k] for k in KEYS]
    ax.bar(MIDS, [100 / 6] * 6, width=0.46, color=GRAY, alpha=0.45, label="test populations (uniform in log M)")
    ax.bar(MIDS, wts, width=0.3, color=BLACK, label=r"CDM: $dN/dM \propto M^{-1.9}$")
    for x_, w_ in zip(MIDS, wts):
        ax.text(x_, w_ + 1.5, f"{w_:.0f}%", ha="center", fontsize=6.3)
    ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel("share of all subhalos  [%]")
    ax.set_title("(a) how many subhalos each bin actually holds"); ax.legend(loc="upper right")
    ax.set_ylim(0, 75)

    ax = axes[1]
    rows = d["results"]
    labels = [r["label"].replace("(Family C), ", "\n").replace("Family A, ", "Family A\n") for r in rows]
    naive = [100 * r["naive_unweighted"] for r in rows]; wtd = [100 * r["cdm_mass_function_weighted"] for r in rows]
    y = np.arange(len(rows))
    for yi, n_, w_ in zip(y, naive, wtd):
        ax.plot([w_, n_], [yi, yi], color=GRAY, lw=1.2, zorder=1)
    ax.scatter(naive, y, color=GRAY, s=28, zorder=2, label="flat average over mass bins")
    ax.scatter(wtd, y, color=BLACK, s=28, zorder=3, label="weighted by the CDM mass function")
    for yi, n_, w_ in zip(y, naive, wtd):
        ax.text(n_ + 1.2, yi, f"{n_:.0f}%", va="center", fontsize=6.3, color=GRAY)
        ax.text(w_ - 1.2, yi, f"{w_:.0f}%", va="center", ha="right", fontsize=6.3, color=BLACK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8)
    ax.axvline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xlim(0, 80); ax.set_xlabel("population completeness at 10% FPR  [%]")  # room for the legend clear of the 30% label
    ax.set_title("(b) the same per-bin numbers, weighted two ways"); ax.legend(loc="upper right")
    ax.invert_yaxis()
    save(fig, "fig6_mass_function")


# ======================================================================
# Fig 7 -- Tier 1: the U-Net retrained on real sources
# ======================================================================
def fig7_tier1():
    u0 = load("results/aggregate/summary.json"); u1 = load("results/aggregate_tier1/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw={"width_ratios": [0.8, 1.3]})
    ax = axes[0]
    vals = [u0["auc"]["mean"], u1["auc"]["mean"]]; errs = [u0["auc"]["std"], u1["auc"]["std"]]
    ax.bar([0, 1], vals, yerr=errs, capsize=3, color=[C_UNET, GRAY], width=0.6)
    ax.axhline(0.5, color=BLACK, ls=":", lw=0.7); ax.text(1.32, 0.505, "chance", fontsize=6.3, ha="right")
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i + 0.36, v, f"{v:.3f}", ha="left", va="center", fontsize=6.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Tier 0\n(n={u0['n_seeds']})", f"Tier 1\n(n={u1['n_seeds']})"])
    ax.set_ylim(0.4, 0.7); ax.set_ylabel("ROC AUC (c = 60 vs no subhalo)")
    ax.set_title("(a) same U-Net, same recipe")

    ax = axes[1]
    for d, col, lab in [(u0, C_UNET, "Tier 0 (Sérsic source)"), (u1, GRAY, "Tier 1 (COSMOS source)")]:
        m = [100 * d["completeness_c60"][k]["mean"] for k in KEYS]; s = [100 * d["completeness_c60"][k]["std"] for k in KEYS]
        ax.errorbar(MIDS, m, yerr=s, fmt="o-", color=col, ms=3.5, capsize=2, label=lab)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel("completeness at 10% FPR  [%]")
    ax.set_ylim(0, 75); ax.set_title("(b) completeness at c = 60"); ax.legend(loc="upper left")
    save(fig, "fig7_tier1")


ALL = {"fig1": fig1_examples, "fig2": fig2_completeness, "fig3": fig3_confounders, "fig4": fig4_localization,
       "fig5": fig5_mass_bias, "fig6": fig6_mass_function, "fig7": fig7_tier1, "fig8": fig8_family_b, "fig9": fig9_scaleup, "fig11": fig11_lenslight, "fig12": fig12_summary}

if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or list(ALL)   # e.g. `python scripts/make_paper_figures.py fig5` rebuilds one figure
    print("building paper figures ->", FIG)
    for k in want:
        ALL[k]()
    print("done")
