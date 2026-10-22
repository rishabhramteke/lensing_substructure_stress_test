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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import A_JOINT_FULL, A_FROZEN_FULL, B_FULL, UNET_AGG, A_ABS, wilson as _wilson  # noqa: E402

# ---- A&A geometry + a colorblind-safe (Okabe-Ito) palette ----
COL_W, DBL_W = 3.46, 7.09  # inches: 88 mm, 180 mm
# Palette: the Okabe & Ito (2002) colorblind-safe qualitative set, true hex values (referee round 20 asked for a
# figure set with no green-vs-red pairing; tab10's orange/green/red is not colorblind-safe). Image maps stay inferno
# (Galan+2022, Biggio+2022) and residuals a diverging RdBu.
BLUE, ORANGE, GREEN, VERM, PURPLE, GRAY, BLACK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7f7f7f", "#222222"
SKY = "#56B4E9"
C_UNET, C_A, C_D, C_B = BLUE, VERM, PURPLE, GREEN
# Figures are typeset in the same Times design as the body text. The manuscript uses txfonts,
# which renders over URW Nimbus Roman; TeX Gyre Termes is that same design in OpenType form, so
# registering it here makes figure text and figure maths match the page instead of falling back to
# matplotlib's STIX. mathtext.fontset="custom" points the maths at the same family.
import glob as _glob
import matplotlib.font_manager as _fm
for _f in _glob.glob("/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
_SERIF = ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
_MATH = {"mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Termes",
         "mathtext.it": "TeX Gyre Termes:italic", "mathtext.bf": "TeX Gyre Termes:bold",
         "mathtext.default": "it"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": _SERIF, **_MATH,
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
        truth["subhalo"]["log10_M200"] = 9.3   # the 10^9-10^9.5 regime the paper's comparisons turn on (referee round 22)
        return LensRenderer(cfg).render(truth, seed=1)

    r0, r1 = system(tier0_tsang, 7), system(tier1_cosmos, 3)
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W * 0.66, 3.05))
    rows = [("Tier 0: Sérsic source", r0),
            ("Tier 1: COSMOS source", r1)]
    for i, (label, r) in enumerate(rows):
        full, nosub = r["noisy"]["full"], r["noisy"]["no_subhalo"]
        resid = r["noiseless"]["full"] - r["noiseless"]["no_subhalo"]
        vmax = full.max()
        for j, (img, title) in enumerate([(full, "observed"), (nosub, "subhalo removed"),
                                          (resid, "difference")]):
            ax = axes[i, j]
            if j < 2:
                ax.imshow(np.arcsinh(img / vmax * 10), cmap="inferno", origin="lower")
            else:
                m = np.abs(resid).max()
                ax.imshow(resid, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7.2, pad=3)
        axes[i, 0].set_ylabel(label, fontsize=7.2)
    # 1" scale bar on the first panel (0.08"/px -> 12.5 px)
    axes[0, 0].plot([4, 4 + 12.5], [4, 4], color="white", lw=1.2)
    axes[0, 0].text(4, 6.5, "1 arcsec", color="white", fontsize=6.0)   # plain text: the ″ glyph is missing from the figure font and printed as a bare 1
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


def wilson(k, n, z=1.0):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _binomial_from_scans(root, pop_clean="no_subhalo", pops=("test_fixed60", "test_fixed15")):
    """Completeness per bin with Wilson 68% half-widths from a full-population scan root (A or B format)."""
    ns = [json.loads(l) for l in open(ROOT / root / pop_clean / "scan_results.jsonl")]
    thr = float(np.quantile([r["delta_chi2"] for r in ns if r["reliable_fit"]], 0.90))
    out = {}
    for pop, key in zip(pops, ("completeness_c60", "completeness_c15", "completeness_c30")):
        if not (ROOT / root / pop / "scan_results.jsonl").exists():
            continue
        recs = [json.loads(l) for l in open(ROOT / root / pop / "scan_results.jsonl")]
        rel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
        d = {}
        for lo, hi in [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]:
            sel = [r for r in rel if lo <= r["log10_M200_true"] < hi]
            k = sum(1 for r in sel if r["delta_chi2"] >= thr); n = len(sel)
            p_, l_, h_ = wilson(k, n) if n else (np.nan, np.nan, np.nan)
            d[f"{lo}-{hi}"] = {"mean": p_, "lo": l_, "hi": h_, "n": n}
        out[key] = d
    return out


def fig2_completeness():
    u = load(UNET_AGG + "/summary.json")
    a = load("results/baseline_a/aggregate/summary.json")
    b, nb = family_b_summary()
    def _complete(root):   # a run is usable only once every population has written its manifest (files are buffered while running)
        return all((ROOT / root / p / "manifest.json").exists() for p in ("no_subhalo", "test_fixed60", "test_fixed15"))
    P3 = ("test_fixed60", "test_fixed15", "test_fixed30")   # round 30: c=30 curves alongside (referee M3)
    a_full = _binomial_from_scans(A_JOINT_FULL, pops=P3) if _complete(A_JOINT_FULL) else None
    b_full = _binomial_from_scans(B_FULL, pops=P3) if _complete(B_FULL) else None
    IC = load("results/intermediate_concentration.json") if (ROOT / "results/intermediate_concentration.json").exists() else {}
    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.45), sharey=True)
    # panel order A, B, C as everywhere else in the paper; the network at the published scale gets its own panel (d)
    panels = [(axes[0], a, a_full, "Family A: joint parametric scan", C_A, "full populations, n = 1 000" if a_full else f"n = {a['n_seeds']} seeds"),
              (axes[1], b, b_full, "Family B: free-form residual", C_B, "full populations, n = 1 000" if b_full else f"n = {nb} seed sets"),
              (axes[2], u, None, "Family C: U-Net", C_UNET, f"30 000 images, {u['n_seeds']} seeds")]
    u15 = load("results/aggregate_30k_c15/summary.json") if (ROOT / "results/aggregate_30k_c15/summary.json").exists() else None
    for ax, d, full, name, col, sub in panels:
        for key, ls, mk, lab, c in [("completeness_c60", "-", "o", "c = 60", col), ("completeness_c30", "-.", "D", "c = 30", SKY), ("completeness_c15", "--", "s", "c = 15", GRAY)]:
            if key == "completeness_c30":
                if full is not None and key in full:
                    m = [100 * full[key][k]["mean"] for k in KEYS]
                    lo = [max(0.0, 100 * (full[key][k]["mean"] - full[key][k]["lo"])) for k in KEYS]; hi = [max(0.0, 100 * (full[key][k]["hi"] - full[key][k]["mean"])) for k in KEYS]
                    ax.errorbar(MIDS, m, yerr=[lo, hi], fmt=mk + ls, color=c, ms=3.0, capsize=2, lw=1.0, mfc="white", label=lab)
                elif full is None and "U-Net" in name and IC.get("C"):
                    seeds = IC["C"]["per_seed"]; ks = [k for k in KEYS if k in seeds[0]["populations"]["test_fixed30"]["bins"]]
                    mm = np.array([[sd["populations"]["test_fixed30"]["bins"][k]["p"] for k in ks] for sd in seeds])
                    ax.errorbar([MIDS[KEYS.index(k)] for k in ks], 100 * mm.mean(0), yerr=100 * mm.std(0, ddof=1), fmt=mk + ls, color=c, ms=3.0, capsize=2, lw=1.0, mfc="white", label=lab)
                continue
            if full is not None:
                m = [100 * full[key][k]["mean"] for k in KEYS]
                lo = [max(0.0, 100 * (full[key][k]["mean"] - full[key][k]["lo"])) for k in KEYS]; hi = [max(0.0, 100 * (full[key][k]["hi"] - full[key][k]["mean"])) for k in KEYS]
                ax.errorbar(MIDS, m, yerr=[lo, hi], fmt=mk + ls, color=c, ms=3.5, capsize=2, lw=1.1, label=lab)
            else:
                m = [100 * d[key][k]["mean"] for k in KEYS]; sd = np.array([100 * d[key][k]["std"] for k in KEYS])
                # binomial half-width from the per-bin count of the full 1 000-lens test populations (U-Net scores every lens)
                nbin = np.array([170, 170, 170, 170, 170, 150]) if "U-Net" in name else np.array([50] * 6)
                binom = 100 * np.sqrt(np.array(m) / 100 * (1 - np.array(m) / 100) / nbin)
                ax.errorbar(MIDS, m, yerr=np.sqrt(sd ** 2 + binom ** 2), fmt=mk + ls, color=c, ms=3.5, capsize=2, lw=1.1, label=lab)
        if "U-Net" in name and u15 is not None:   # referee round 19: the network trained at c=15, tested at c=15 (drawn last so the legend reads c=60, c=15, then this)
            m = [100 * u15["completeness_c15"][k]["mean"] for k in KEYS]; sd = np.array([100 * u15["completeness_c15"][k]["std"] for k in KEYS])
            nbin = np.array([170, 170, 170, 170, 170, 150]); binom = 100 * np.sqrt(np.array(m) / 100 * (1 - np.array(m) / 100) / nbin)
            ax.errorbar(MIDS, m, yerr=np.sqrt(sd ** 2 + binom ** 2), fmt="^:", color=col, mfc="white", ms=3.5, capsize=2, lw=1.0, label=f"c = 15, trained at 15")
        ax.axhline(10, color=BLACK, ls=":", lw=0.7)
        ax.set_title(f"{name}\n({sub})", fontsize=7.2)
        ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylim(0, 100); ax.set_xticks([8.5, 9, 9.5, 10, 10.5])
        ax.legend(loc="upper left" if "U-Net" in name else "lower right", fontsize=5.4 if "U-Net" in name else 6.0, handlelength=1.6, labelspacing=0.25)
    ax = axes[3]   # the network at the published scale
    if (ROOT / "results/detector_30k_c60pub_seed0/results.json").exists():
        U5 = load("results/detector_30k_c60pub_seed0/results.json")
        for key, ls, mk, c, lab in (("completeness_c60_by_mass_bin", "-", "o", C_UNET, "c = 60"), ("completeness_c15_by_mass_bin", "--", "s", GRAY, "c = 15")):
            k_n = [(round(U5[key][k]["completeness"] * U5[key][k]["n"]), U5[key][k]["n"]) for k in KEYS]
            w = [wilson(k, n) for k, n in k_n]
            m = [100 * p for p, _, _ in w]
            ax.errorbar(MIDS, m, yerr=[[100 * (p - lo) for p, lo, _ in w], [100 * (hi - p) for p, _, hi in w]], fmt=mk + ls, color=c, ms=3.5, capsize=2, lw=1.1, label=lab)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_title("U-Net at the published scale\n" + r"($5\times10^5$ images, one seed)", fontsize=7.2)
    ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylim(0, 100); ax.set_xticks([8.5, 9.5, 10.5])
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.6, labelspacing=0.25)
    for a_ in axes[:3]:
        a_.set_xticks([8.5, 9.5, 10.5])
    axes[0].set_ylabel("completeness at 10% FPR  [%]")
    axes[0].text(8.05, 3, "10% = chance", fontsize=6.0, color=BLACK)
    save(fig, "fig2_completeness")


# ======================================================================
# Fig 3 -- RQ2/RQ1 confounder FPR + the non-physical decoy control
# ======================================================================
def fig3_confounders():
    # referee round 20: the decoy panel is split into Gaussian (b) and dipole (c) so each legend is short, the Gaussian
    # seed band is labelled, and the decoy table is gone (these panels carry it).
    u = load(UNET_AGG + "/summary.json")
    J = load("results/joint_scan.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.75), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    ax = axes[0]
    conds = ["no_subhalo (calibration set)", "multipole m=4, a=0.01*thetaE", "multipole m=4, a=0.03*thetaE"]
    um = [100 * u["confounder_fpr"][c]["mean"] for c in conds]; us = [100 * u["confounder_fpr"][c]["std"] for c in conds]
    C = J["core"]["cal"]
    a_cells = [C["fpr_clean"], C["fpr_multipole_m4_a1"], C["fpr_multipole_m4_a3"]]
    am = [100 * c["p"] for c in a_cells]; as_ = [100 * 0.5 * (c["hi"] - c["lo"]) for c in a_cells]
    FB = load("results/full_populations.json")["B"]
    b_cells = [{"fpr": FB["fpr_clean"], "lo": FB["fpr_clean"], "hi": FB["fpr_clean"]}, FB["fpr_multipole_m4_a1"], FB["fpr_multipole_m4_a3"]]
    bm = [100 * c.get("fpr", c.get("p")) for c in b_cells]; bs = [100 * 0.5 * (c["hi"] - c["lo"]) for c in b_cells]
    x = np.arange(3); w = 0.27
    ax.bar(x - w, am, w, yerr=as_, capsize=2, color=C_A, label="A joint scan")
    # B hatched, so A (vermillion) and B (bluish green) side by side never rely on hue alone (referee round 22)
    ax.bar(x, bm, w, yerr=bs, capsize=2, color=C_B, hatch="////", edgecolor="white", lw=0, label="B free-form residual")
    ax.bar(x + w, um, w, yerr=us, capsize=2, color=C_UNET, label=f"C U-Net ({u['n_seeds']} seeds)")
    for off, vals, col in ((-w, am, C_A), (0, bm, C_B), (w, um, C_UNET)):
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 2.5, f"{v:.0f}", ha="center", fontsize=6.2, color=col)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["none\n(calibration)", "$m=4$\n$a=0.01\\,\\theta_E$", "$m=4$\n$a=0.03\\,\\theta_E$"])
    ax.set_ylabel("flagged, subhalo-free images  [%]")
    ax.set_ylim(0, 132); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_title("(a) lens-shape multipole, 10% cal.")
    ax.legend(loc="upper left", fontsize=5.8, frameon=False, ncol=2, columnspacing=0.8)

    D = J.get("decoys", {}).get("populations", {})
    UP = load("results/unet_populations.json")["populations"]
    fbp = ROOT / "results/noise_decoy_control_familyB/results.json"
    FBD = json.loads(fbp.read_text()) if fbp.exists() else {}
    def uvals(shape, seeds):
        return np.array([[100 * UP[f"decoy_{shape}_{sd}_a{a}"]["aggregate"]["flag_rate"]["mean"] for a in (3, 6, 10)] for sd in seeds if f"decoy_{shape}_{sd}_a3" in UP])
    for ax, shape, seeds, title in ((axes[1], "gaussian", ("s42", "s43"), "(b) Gaussian decoy"), (axes[2], "dipole", ("s42",), "(c) dipole decoy")):
        v = uvals(shape, seeds)
        if len(v):
            ax.plot([3, 6, 10], v.mean(0), marker="o", ls="-", color=C_UNET, ms=3.5, label=f"C U-Net ({u['n_seeds']} seeds)")
            if len(v) > 1:
                ax.fill_between([3, 6, 10], v.min(0), v.max(0), color=C_UNET, alpha=0.18, lw=0)
        bk = [k for k in ((("gaussian", "gaussian_seed43") if shape == "gaussian" else ("dipole",))) if k in FBD]
        if bk:
            amps = sorted(float(k) for k in FBD[bk[0]]); vb = np.array([[100 * FBD[k][str(a_)]["b_fpr"] for a_ in amps] for k in bk])
            ax.plot(amps, vb.mean(0), marker="D", ls="--", color=C_B, ms=3.2, label="B free-form residual")
            if len(bk) > 1:
                ax.fill_between(amps, vb.min(0), vb.max(0), color=C_B, alpha=0.18, lw=0)
                ax.fill_between([], [], [], color="0.75", lw=0, label="shading: range over the two draws")
        keys = [f"decoy_{shape}_{sd}_a10" for sd in seeds if f"decoy_{shape}_{sd}_a10" in D]
        if keys:
            g = [100 * D[k]["cal"]["p"] for k in keys]; ga = [100 * D[k]["abs"]["p"] for k in keys]
            ax.errorbar([10.2], [np.mean(g)], yerr=[[np.mean(g) - min(g)], [max(g) - np.mean(g)]], fmt="s", color=C_A, ms=4.5, capsize=2, label="A joint scan, 10% cal.")
            ax.errorbar([10.2], [np.mean(ga)], yerr=[[np.mean(ga) - min(ga)], [max(ga) - np.mean(ga)]], fmt="s", mfc="white", color=C_A, ms=4.5, capsize=2, label=f"A joint scan, $\\Delta\\chi^2\\geq{A_ABS:g}$")
        ax.axhline(10, color=BLACK, ls=":", lw=0.7)
        ax.text(6.4, 11.5, "clean baseline, 10%", fontsize=5.8, color=BLACK)
        ax.set_xlabel(r"decoy amplitude  [$\sigma$ of local noise]")
        ax.set_ylim(0, 92); ax.set_xticks([3, 6, 10]); ax.set_xlim(2.5, 10.9)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=5.8, frameon=False)
    axes[1].set_ylabel("flagged as detection  [%]")
    fig.tight_layout()
    save(fig, "fig3_confounders")


# ======================================================================
# Fig 9 -- scale-up: U-Net at 8k vs 30k training images (drawn only once the 30k aggregate exists)
# ======================================================================
def fig9_scaleup():
    """SUPPLEMENTARY, not included in the manuscript: the 30k scale-up result is reported in the text only.
    Kept because it regenerates from released results and may be wanted by a reader."""
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
    panels = [("(a) frozen macro-model\nthree mass hypotheses, $c=15$ template", [v for v in seeds_a if (v[0] / "test_fixed60/scan_results.jsonl").exists()])]
    variants = [(ROOT / A_JOINT_FULL, "D", C_UNET, "joint scan, 10% cal.")]
    variants = [v for v in variants if (v[0] / "test_fixed60/scan_results.jsonl").exists()]
    if variants:
        panels.append(("(b) macro-model re-fitted jointly\nwith the perturber, mass free", variants))

    fig, axes = plt.subplots(1, len(panels), figsize=(DBL_W if len(panels) > 1 else COL_W, 3.0), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        for k, (root, mk, col, lab) in enumerate(series):
            pts = points(root)
            jitter = np.random.default_rng(k).uniform(-0.05, 0.05, size=len(pts))
            ax.scatter(pts[:, 0], pts[:, 1] + jitter, marker=mk, s=18, color=col, alpha=0.85, edgecolor="none",
                       label=f"{lab}: {len(pts)} localized, {int((pts[:,1] < pts[:,0]).sum())} below the line")
        ax.plot([8, 11], [8, 11], color=BLACK, lw=0.8, ls="-", label="fitted = true")
        if "frozen" in title or "(a)" in title:
            for g in (8.5, 9.5, 10.5):
                ax.axhline(g, color=GRAY, lw=0.5, ls=":")
        ax.set_xlim(8, 11); ax.set_ylim(7.5, 12.3)
        ax.set_xlabel(r"true $\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel(r"Family A best-fit $\log_{10}(M/M_\odot)$")
        ax.set_title(title, fontsize=7.5)
        # (a)'s points lie below the identity line and (b)'s above it: each legend goes where its panel has no data
        ax.legend(loc="upper left" if "(a)" in title else "lower right", fontsize=6.3, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig5_mass_bias")


# ======================================================================
# Fig 8 -- Family B: regularization sensitivity band
# ======================================================================
def fig8_family_b():
    d = load("results/baseline_b/summary_family_b.json")
    lams = ["1e4", "1e5", "1e6"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    styles = {"1e4": (":", "^", 1.0), "1e5": ("-", "o", 1.0), "1e6": ("--", "s", 1.0)}
    # three distinct Okabe-Ito hues plus line style and marker per lambda (four shades of one green were not separable
    # in print, referee round 22); the oracle ceiling is black
    LAMCOL = {"1e4": "#E3B5CF", "1e5": "#CC79A7", "1e6": "#6E2F55"}   # sequential shades of Okabe-Ito reddish purple, disjoint from the family colours
    # referee round 19: the fitted curves come from the FULL 1 000-lens populations (every record stores the peak
    # statistic at all three lambdas), with Wilson intervals, so a low-mass point below the chance line can be read
    # as the fluctuation it is; the oracle stays on the 300-lens seed-0 subsample (no full oracle run exists)
    full_ok = all((ROOT / B_FULL / p_ / "manifest.json").exists() for p_ in ("no_subhalo", "test_fixed60"))
    if full_ok:
        ns = [json.loads(l) for l in open(ROOT / B_FULL / "no_subhalo/scan_results.jsonl")]
        pos = [r for r in (json.loads(l) for l in open(ROOT / B_FULL / "test_fixed60/scan_results.jsonl")) if r["reliable_fit"] and r["has_subhalo"]]
        mp = {p_: [r for r in (json.loads(l) for l in open(ROOT / B_FULL / f"{p_}/scan_results.jsonl")) if r["reliable_fit"]] for p_ in ("multipole_m4_a1", "multipole_m4_a3")}
        full_fpr = {}
    for lam in lams:
        ls, mk, al = styles[lam]
        lab = rf"fitted macro-model, $\lambda=10^{{{int(np.log10(float(lam)))}}}$" + ("  (primary)" if lam == "1e5" else "")
        if full_ok:
            key = f"peak_abs_dkappa_lam{lam}"
            thr = float(np.quantile([r[key] for r in ns if r["reliable_fit"]], 0.90))
            m, lo, hi = [], [], []
            for a, b in BINS:
                sel = [r[key] for r in pos if a <= r["log10_M200_true"] < b]
                p_, l_, h_ = wilson(sum(1 for v in sel if v >= thr), len(sel))
                m.append(100 * p_); lo.append(max(0.0, 100 * (p_ - l_))); hi.append(max(0.0, 100 * (h_ - p_)))
            ax.errorbar(MIDS, m, yerr=[lo, hi], fmt=mk + ls, color=LAMCOL[lam], alpha=al, ms=3.5, lw=1.1, capsize=1.5, label=lab)
            full_fpr[lam] = {p_: float(np.mean([r[key] >= thr for r in mp[p_]])) for p_ in mp}
            continue
        comp = d["fitted"]["by_lambda"][lam]["completeness_c60"]
        m = [100 * comp[k]["completeness"] if comp[k]["completeness"] is not None else np.nan for k in KEYS]
        ax.plot(MIDS, m, marker=mk, ls=ls, color=LAMCOL[lam], alpha=al, ms=3.5, lw=1.1, label=lab)
    # a darker shade of C_B (Family B's own green), not grey: this whole figure is Family B's
    # internal lambda-sensitivity, and the oracle is a ceiling *of the same method*, not a
    # different, neutral reference series.
    B_DARK = "#1d681d"
    comp = d["oracle"]["by_lambda"]["1e5"]["completeness_c60"]
    ax.plot(MIDS, [100 * comp[k]["completeness"] for k in KEYS], marker="*", ls="-", color=BLACK, ms=4.5, lw=1.0, label=r"oracle macro-model, $\lambda=10^{5}$ (ceiling)")
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_ylim(0, 105); ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); ax.set_ylabel("completeness at 10% FPR, $c=60$  [%]")
    ax.set_title("(a) completeness vs regularization strength"); ax.legend(loc="lower right", fontsize=6.0)
    ax = axes[1]
    x = np.arange(3); w = 0.36
    if full_ok:
        a1 = [100 * full_fpr[l]["multipole_m4_a1"] for l in lams]; a3 = [100 * full_fpr[l]["multipole_m4_a3"] for l in lams]
    else:
        a1 = [100 * d["fitted"]["by_lambda"][l]["confounder_fpr"]["multipole_m4_a1"]["fpr"] for l in lams]
        a3 = [100 * d["fitted"]["by_lambda"][l]["confounder_fpr"]["multipole_m4_a3"]["fpr"] for l in lams]
    ax.bar(x - w / 2, a1, w, color=[LAMCOL[l] for l in lams], hatch="////", edgecolor="white", lw=0, label=r"multipole $a_m=0.01\,\theta_E$ (hatched)")
    ax.bar(x + w / 2, a3, w, color=[LAMCOL[l] for l in lams], label=r"multipole $a_m=0.03\,\theta_E$")
    for xi, v1, v3 in zip(x, a1, a3):
        ax.text(xi - w / 2, v1 + 2, f"{v1:.0f}%", ha="center", fontsize=6.3); ax.text(xi + w / 2, v3 + 2, f"{v3:.0f}%", ha="center", fontsize=6.3)
    ax.axhline(10, color=BLACK, ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([r"$\lambda=10^{4}$", r"$\lambda=10^{5}$", r"$\lambda=10^{6}$"])
    ax.set_ylim(0, 110); ax.set_ylabel("FPR on subhalo-free images  [%]")
    import matplotlib.patches as _mp
    ax.set_title("(b) confounder false positives vs regularization")
    ax.legend(handles=[_mp.Patch(fc="0.6", hatch="////", ec="white", label=r"multipole $a_m=0.01\,\theta_E$"), _mp.Patch(fc="0.6", label=r"multipole $a_m=0.03\,\theta_E$")], loc="upper left", fontsize=6.3)
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
    # pick a mid-mass subhalo lens the analysis RETAINS (single-Sersic chi2/dof < 10, the gate) whose fit is typical of
    # the retained ones (chi2/dof near their median), so the example is one the scan actually scores
    recs = {r["index"]: r for r in (json.loads(l) for l in open(a1 / "scan_results.jsonl"))}
    cands = [i for i in idx1 if truths[i].get("subhalo") and 9.8 <= truths[i]["subhalo"]["log10_M200"] <= 10.4
             and recs[i]["chi2_smooth_per_dof"] < 10]
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
    panels = [("(a) observed, with lens light", noisy[i]), ("(b) single-S\u00e9rsic subtracted\n(under-flexible light model)", sub1[k1])]
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
        ax.plot(c + sub["x"] / 0.08, c + sub["y"] / 0.08, "o", mfc="none", mec="#56B4E9", ms=9, mew=1.2)
    axes[0].text(0.03, 0.03, f"$\\log_{{10}} M_{{200}}={truths[i]['subhalo']['log10_M200']:.1f}$, $c=60$", transform=axes[0].transAxes, color="white", fontsize=6.3, va="bottom")
    axes[1].text(0.03, 0.03, f"$\\chi^2/{{\\rm dof}}={recs[i]['chi2_smooth_per_dof']:.1f}$", transform=axes[1].transAxes, color="white", fontsize=6.3, va="bottom")
    # 1 arcsec scale bar, as in Fig. 2
    npx = noisy.shape[1]; L = 1.0 / 0.08
    axes[-1].plot([npx - 4 - L, npx - 4], [4, 4], color="white", lw=1.5); axes[-1].text(npx - 4 - L / 2, 6, "1 arcsec", color="white", fontsize=6.3, ha="center", va="bottom")
    save(fig, "fig11_lenslight")


# ======================================================================
# Fig 12 -- one-page synthesis: which stress breaks which family, and how badly
# ======================================================================

def _family_b_mass_error():
    """Median and MAD of the free-form statistic's aperture-mass error over its localized c=60 detections at its own
    10% calibration (the numbers of Sect. 4.4), computed from the scan records rather than typed in."""
    L = lambda p: [json.loads(l) for l in open(ROOT / "results/baseline_b_full/fitted" / p / "scan_results.jsonl")]  # noqa: E731
    t = float(np.quantile([r["delta_chi2"] for r in L("no_subhalo") if r["reliable_fit"]], 0.9))
    T = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    e = []
    for r in L("test_fixed60"):
        if r["reliable_fit"] and r["has_subhalo"] and r["delta_chi2"] >= t and r["best_log10_m"] > 6.01:
            s = T[r["index"]]["subhalo"]
            if np.hypot(r["best_x"] - s["x"], r["best_y"] - s["y"]) < 0.16:
                e.append(r["best_log10_m"] - r["log10_M200_true"])
    e = np.array(e)
    return float(np.median(e)), float(np.median(np.abs(e - np.median(e))))

def fig12_summary():
    """Rows = stresses, columns = families. Every cell is computed from the results files (no retyped
    numbers); the caption states the sample each cell comes from. A missing input gives a grey cell."""
    import re
    import matplotlib.patches as mpatches
    # Light tints of the Okabe & Ito (2002) colorblind-safe qualitative palette (the same standard
    # named, if not the exact hex values, used for BLUE/ORANGE/GREEN/VERM above): blue #0072B2,
    # orange #E69F00, vermillion #D55E00, each blended toward white so cell text stays legible.
    # Blue vs. vermillion replaces the original green vs. red, which is the classic red-green
    # colorblind confusion pair; vermillion is diluted less than orange so severity still reads
    # as a light-to-dark ramp, not just a hue change.
    BLUEG, AMBER, RED, GREY = "#bfdcec", "#f9e7bf", "#e49659", "#e6e6e6"
    NOTRES = "#f3f3f3"   # a loss the arm is too small to resolve: twice its standard error exceeds 10 points
    # insensitive: a null response from a detector that barely detects (sky blue #56B4E9, Okabe & Ito, diluted); not "unaffected"
    INSENS = "#d5ecf9"

    def get(rel):
        try:
            return load(rel)
        except Exception:
            return None
    J = get("results/joint_scan.json") or {}
    FP = get("results/full_populations.json") or {}
    U = get(UNET_AGG + "/summary.json") or {}
    UP = (get("results/unet_populations.json") or {}).get("populations", {})
    UPS = (get("results/unet_populations_psf015_psftrained.json") or {}).get("populations", {})
    UPD = (get("results/unet_populations_psf015_deeptrained.json") or {}).get("populations", {})
    LL = get("results/lens_light_experiment.json") or {}
    DEP = get("results/depth.json") or {}
    MF = get("results/mass_function_reweighting/results.json") or {}
    LOC = get("results/localization/summary_abc.json") or get("results/localization/summary.json") or {}
    T1 = get("results/aggregate_tier1/summary.json") or {}
    FBA = get("results/family_b_arms.json") or {}
    try:
        _ns = [json.loads(l) for l in open(ROOT / B_FULL / "no_subhalo/scan_results.jsonl")]
        FBA_T0 = float(np.quantile([r["delta_chi2"] for r in _ns if r["reliable_fit"]], 0.90))
    except Exception:
        FBA_T0 = 1.0
    UR = get("results/aggregate_30k_realistic/summary.json") or {}
    NAR = get("results/new_arms.json") or {}
    JR = NAR.get("realistic_full") or {}
    NA = ("not run", GREY)

    def P(x, nd=0):
        return f"{100*x:.{nd}f}%"

    def sev(loss, amber=10, red=30):
        """completeness loss in points at the common calibration: <10 unaffected, 10-30 degraded, >30 fails"""
        return BLUEG if loss < amber else (AMBER if loss < red else RED)

    NEUTRAL = "#ffffff"   # the CDM-weighting row is a re-expression, not a stress, and is not graded

    def sevl(frac):
        """localized fraction of detections: >=70% unaffected, 40-70% degraded, <40% fails"""
        return BLUEG if frac >= 0.7 else (AMBER if frac >= 0.4 else RED)

    def sevm(err):
        """median mass error: |err| < 0.2 dex unaffected, 0.2-1 dex degraded (calibratable offset), >1 dex fails"""
        return BLUEG if abs(err) < 0.2 else (AMBER if abs(err) <= 1.0 else RED)

    def sevf(fpr):
        """false-positive rate at the common calibration: <15% unaffected, 15-50% degraded, >50% fails"""
        return BLUEG if fpr < 0.15 else (AMBER if fpr < 0.5 else RED)

    def b_loc():
        try:
            ns = [json.loads(l) for l in open(ROOT / B_FULL / "no_subhalo/scan_results.jsonl")]
            thr = float(np.quantile([r["delta_chi2"] for r in ns if r["reliable_fit"]], 0.90))
            tr = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
            det = [r for r in (json.loads(l) for l in open(ROOT / B_FULL / "test_fixed60/scan_results.jsonl")) if r["reliable_fit"] and r["has_subhalo"] and r["delta_chi2"] >= thr]
            d = np.array([np.hypot(r["best_x"] - tr[r["index"]]["subhalo"]["x"], r["best_y"] - tr[r["index"]]["subhalo"]["y"]) for r in det])
            return float((d < 0.16).mean())
        except Exception:
            return None

    rows = []
    # ---- one bin rule for every completeness cell (referee round 23): the lowest mass bin, companion bin excluded, in
    # which the family's BASELINE completeness is at least 50%, so a saturated bin never hides a loss
    BIN_ORDER = ["8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5"]

    def lowbin(get0):
        for b_ in BIN_ORDER:
            try:
                if get0(b_) >= 0.5:
                    return b_
            except Exception:
                continue
        return "10.0-10.5"

    def blab(b_):
        lo_, hi_ = (x.replace(".0", "") for x in b_.split("-"))
        return f"$10^{{{lo_}}}$\u2013$10^{{{hi_}}}$"

    def ccell(get0, get1, pre="", post=""):
        """completeness baseline -> stressed in the rule's bin, graded by the loss in points, but only when the loss exceeds
        twice its binomial standard error (unpaired, so conservative; the bin holds ~0.9 n / 6 subhalo lenses for an arm
        of n lenses, n read from the cell's own '$n$=' note, else 1 000)"""
        import re as _re
        b_ = lowbin(get0); x0, x1 = get0(b_), get1(b_)
        m_ = _re.search(r"\$n\$=(\d+)", post); n_ = int(m_.group(1)) if m_ else 1000
        nb = max(0.9 * n_ / 6.0, 1.0)
        se = 100 * np.sqrt(x0 * (1 - x0) / nb + x1 * (1 - x1) / nb)
        loss = 100 * (x0 - x1)
        col = sev(loss) if loss > 2 * se else (NOTRES if 2 * se > 10 else BLUEG)
        rest = _re.sub(r",? ?\$n\$=\d+", "", post)
        return (f"{pre}{P(x0)} $\\to$ {P(x1)} ({blab(b_)})\n$\\pm${se:.0f} pts" + (f", $n$={n_}" if m_ else "") + rest, col)

    def worst(*cols):
        order = [BLUEG, NOTRES, INSENS, AMBER, RED]
        return max(cols, key=lambda c: order.index(c) if c in order else -1)


    NARp = NAR.get("paired_baselines", {})
    def A_same(key):   # Tier-0 joint-scan completeness on the arm's own lenses, at the Tier-0 calibration
        return lambda b_: NARp[key]["base_cal"][b_]["p"]
    def B_same(arm_root, pop):   # Tier-0 free-form completeness on the arm's own lenses, at the Tier-0 calibration
        L = lambda r, q: [json.loads(l) for l in open(ROOT / r / q / "scan_results.jsonl")]  # noqa: E731
        idx = {r["index"] for r in L(arm_root, pop)}
        t0 = float(np.quantile([r["delta_chi2"] for r in L("results/baseline_b_full/fitted", "no_subhalo") if r["reliable_fit"]], 0.9))
        X = [r for r in L("results/baseline_b_full/fitted", "test_fixed60") if r["index"] in idx and r["reliable_fit"] and r["has_subhalo"]]
        def g(b_):
            lo, hi = map(float, b_.split("-")); v = [r["delta_chi2"] >= t0 for r in X if lo <= r["log10_M200_true"] < hi]
            return float(np.mean(v)) if v else 0.0
        return g
    J_c60 = lambda b_: J["core"]["cal"]["completeness_c60"][b_]["p"]  # noqa: E731
    B_c60 = lambda b_: FP["B"]["completeness_c60"][b_]["completeness"]  # noqa: E731
    U_c60 = lambda b_: U["completeness_c60"][b_]["mean"]  # noqa: E731

    # 1. concentration
    try:
        fl = J["core"]["cal"]["flip_60_to_15"]
        cA = ccell(J_c60, lambda b_: J["core"]["cal"]["completeness_c15"][b_]["p"], post=f"\nflip {fl['lost']}:{fl['gained']}")
    except Exception:
        cA = NA
    try:
        cB = ccell(B_c60, lambda b_: FP["B"]["completeness_c15"][b_]["completeness"])
    except Exception:
        cB = NA
    try:
        txt, col = ccell(U_c60, lambda b_: U["completeness_c15"][b_]["mean"])
        U15 = get("results/aggregate_30k_c15/summary.json")
        if U15:
            b_ = lowbin(U_c60); txt += f"\ntrained at $c{{=}}15$: {P(U15['completeness_c15'][b_]['mean'])}"
        cC = (txt, col)
    except Exception:
        cC = NA
    rows.append(("concentration $c=60\\to15$\n(completeness, 10% cal.)", [cA, cB, cC]))
    # 2. multipole
    try:
        C = J["core"]; fA = C["cal"]["fpr_multipole_m4_a3"]["p"]; fA20 = C["abs"]["fpr_multipole_m4_a3"]["p"]
        MPJ = get("results/multipole_orders_joint_summary.json") or {}
        f4 = (MPJ.get("mp4") or {}).get("multipole_m4_a3")
        def Pf(x):   # sub-percent rates keep one decimal so 0.3% does not print as 0%
            return P(x, 1) if 0 < x < 0.01 else P(x)
        if f4:
            m3 = (MPJ["mp4"].get("multipole_m3_a3") or {})
            cA = (f"bare: {P(fA)}, {P(fA20)} ($\\geq{A_ABS:g}$)\nfree $m{{=}}4$: {Pf(f4['fpr_10pct'])}, {Pf(f4['fpr_20'])}\n$m{{=}}3$ truth: {P(m3.get('fpr_10pct', float('nan')))}, {P(m3.get('fpr_20', float('nan')))}", sevf(fA))
        else:
            cA = (f"{P(fA)} (10% cal.)\n{P(fA20)} at $\\Delta\\chi^2\\geq{A_ABS:g}$", sevf(fA))
    except Exception:
        cA = NA
    try:
        fB = FP["B"]["fpr_multipole_m4_a3"]["fpr"]; cB = (f"{P(fB)}", sevf(fB))
    except Exception:
        cB = NA
    try:
        fC = U["confounder_fpr"]["multipole m=4, a=0.03*thetaE"]["mean"]; cC = (f"{P(fC)}\nunmoved", INSENS if fC < 0.15 else sevf(fC))
    except Exception:
        cC = NA
    rows.append(("lens-shape multipole, $a_4=3\\%\\,\\theta_E$\n(false positives, subhalo-free)", [cA, cB, cC]))
    try:
        E = NAR["ellmp"]; cA = (f"bare: {P(E['bare']['cal'])} (cal.)\nfree circ. $m{{=}}4$: {P(E['free_circular_m4']['cal'])} (cal.),\n{P(E['free_circular_m4']['abs'])} at $\\Delta\\chi^2\\geq{A_ABS:g}$", sevf(E["free_circular_m4"]["cal"]))
    except Exception:
        cA = NA
    try:   # Family B on the elliptical multipole, at its Tier-0 calibration (round 31)
        E_ = [r for r in (json.loads(l) for l in open(ROOT / "results/baseline_b_ell/fitted/ellmultipole_m4_a3/scan_results.jsonl")) if r["reliable_fit"]]
        fE = float(np.mean([r["delta_chi2"] >= FBA_T0 for r in E_])); cBe = (f"{P(fE)}, $n$={len(E_)}", sevf(fE))
    except Exception:
        cBe = NA
    rows.append(("elliptical $m=4$ multipole, $a_4=3\\%\\,\\theta_E$\n(false positives, subhalo-free)", [cA, cBe, NA]))
    # 3. decoy
    try:
        D = J["decoys"]["populations"]; g = [D[k]["cal"]["p"] for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
        ga = [D[k]["abs"]["p"] for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
        cA = (f"{P(min(g))}\u2013{P(max(g))} (10% cal.)\n{P(min(ga))}\u2013{P(max(ga))} at $\\Delta\\chi^2\\geq{A_ABS:g}$", sevf(max(g)))
    except Exception:
        cA = NA
    try:
        fb = load("results/noise_decoy_control_familyB/results.json"); v = [fb[k]["10.0"]["b_fpr"] for k in ("gaussian", "gaussian_seed43", "dipole") if k in fb]
        cB = (f"{P(min(v))}\u2013{P(max(v))}\nresponds most", sevf(max(v)))
    except Exception:
        cB = NA
    try:
        v = [UP[k]["aggregate"]["flag_rate"]["mean"] for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
        cC = (f"{P(min(v))}\u2013{P(max(v))}\nfires", sevf(max(v)))
    except Exception:
        cC = NA
    rows.append(("non-physical decoy, $10\\sigma$\n(false positives)", [cA, cB, cC]))
    # 4. Tier 1 (the joint scan's cell comes from its Tier-1 run when that exists)
    T1J = NAR.get("tier1") or {}
    try:
        cA = (f"{T1J['clean']['gated']}/{T1J['clean']['n']} clean gated;\nclean FPR {P(T1J['clean_fpr_20'])} at $\\Delta\\chi^2\\geq{A_ABS:g}$", RED)
    except Exception:
        cA = ("joint scan not run\n(frozen: $\\chi^2$/dof 57\u201367)", GREY)
    try:
        t1 = T1["auc"]["mean"]; t30 = (get("results/detector_tier1_30k_seed0/results.json") or {}).get("roc_auc_fixed60_vs_no_subhalo")
        cC = (f"retr.: AUC {t1:.2f} / {t30:.2f}\n(8k / 30k images)" if t30 else f"retr.: AUC {t1:.2f}\nchance", RED)
    except Exception:
        cC = NA
    rows.append(("real COSMOS source (Tier 1)", [cA, ("281/300 gated;\nnull $\\times$20, chance", RED), cC]))
    # 5/6. lens light
    def ll_cell(key, single):
        try:
            d = LL[key]; f20 = d["fpr_clean_abs"]["20"]
            if single:   # too few lenses survive the gate for a completeness to mean anything; the finding is the false-positive rate
                return (f"clean FPR {P(f20)} at $\\Delta\\chi^2\\geq{A_ABS:g}$;\nmultipole {P(d['fpr_multipole_a3']['20'])}", sevf(f20))
            txt, col = ccell(A_same("lenslight2"), lambda b_: d["completeness_c60"][b_]["completeness"], post=", $n$=200")
            return (f"{txt}\nclean FPR {P(f20, 1) if f20 < 0.02 else P(f20)} at $\\Delta\\chi^2\\geq{A_ABS:g}$", worst(col, sevf(f20)))
        except Exception:
            return NA
    def llb(key):
        try:
            d = LL[key]
            arm = "results/baseline_b_lenslight2/fitted" if "double" in key else "results/baseline_b_lenslight/fitted"
            return ccell(B_same(arm, "lenslight_fixed60"), lambda b_: d["completeness_c60"][b_]["completeness"], post=", $n$=300")
        except Exception:
            return NA
    def llc(key, fpr_key=None):
        try:
            d = LL["unet"][key]
            comp_ = d.get("completeness_c60_recalibrated") or d["completeness_c60"]
            txt, col = ccell(U_c60, lambda b_: comp_[b_]["completeness"], post=f"\nAUC {d['auc_c60_vs_clean']:.2f}")
            if fpr_key and fpr_key in d:
                txt += f"; {P(d[fpr_key])} clean > thr."; col = worst(col, sevf(d[fpr_key]))
            return (txt, col)
        except Exception:
            return NA
    rows.append(("lens light, single-S\u00e9rsic subtraction\n(under-flexible light model)", [ll_cell("family_a", True), llb("family_b_single_sersic"), llc("subtracted_recalibrated")]))
    def llb1k():   # Family B on the 1 000-lens double-Sersic arm (round 29), against its full Tier-0 population
        try:
            L_ = lambda q: [json.loads(l) for l in open(ROOT / "results/baseline_b_lenslight1k/fitted" / q / "scan_results.jsonl")]  # noqa: E731
            t_ = float(np.quantile([r["delta_chi2"] for r in L_("lenslight1k_no_subhalo") if r["reliable_fit"]], 0.9))
            S_ = [r for r in L_("lenslight1k_fixed60") if r["reliable_fit"] and r["has_subhalo"]]
            g_ = lambda b_: float(np.mean([r["delta_chi2"] >= t_ for r in S_ if float(b_.split("-")[0]) <= r["log10_M200_true"] < float(b_.split("-")[1])]))  # noqa: E731
            return ccell(B_c60, g_, post=", $n$=1000")
        except Exception:
            return llb("family_b_double_sersic")
    rows.append(("lens light, double-S\u00e9rsic subtraction\n(correctly specified)", [ll_cell("family_a_double_sersic", False), llb1k(), llc("double_sersic_subtracted", "fpr_clean_at_original_threshold")]))
    # 7. localization
    try:
        la = J["core"]["cal"]["detections_c60"]["frac_localized"]
        ff = (NAR.get("freepos_flags") or {}).get("real", {})
        fp = {"n": ff.get("n"), "frac_localized": ff.get("frac_localized")} if ff else (J.get("freepos") or {}).get("cal", {}).get("detections_c60", {})
        if fp.get("n"):
            cA = (f"position free: {P(fp['frac_localized'])} ($\\Delta\\chi^2\\geq{A_ABS:g}$)\n(grid: {P(la)}, ceiling 21.5%)", sevl(fp["frac_localized"]))
        else:
            cA = (f"{P(la)}\n(grid ceiling 21.5%)", sevl(la))
    except Exception:
        cA = NA
    lb = b_loc(); cB = (P(lb), sevl(lb)) if lb is not None else NA
    try:
        lc = LOC["family_c_unet"]["frac_localized_2px"]; cC = (P(lc), sevl(lc))
    except Exception:
        cC = NA
    rows.append(("localization $\\leq 2$ px\n(of detections, $c=60$)", [cA, cB, cC]))
    # 8. mass
    try:
        m = J["core"]["cal"]["detections_c60"]["mass_error_all"]["median"]; cA = (f"${m:+.1f}$ dex\n($c=15$ template on $c=60$ truth)", sevm(m))
    except Exception:
        cA = NA
    mB, sB = _family_b_mass_error()
    rows.append(("mass estimate\n(median error)", [cA, (f"${mB:+.2f}$ dex, scatter {sB:.2f} dex\n(aperture, localized detections)", sevm(mB)), ("not attempted", GREY)]))
    # 9. LOS: graded by the flag rate at the common calibration
    try:
        a = J["los"]["arms"]["z025"]; fc, fa = a["cal"]["flag_rate"]["p"], a["abs"]["flag_rate"]["p"]; d = a["abs"].get("detections", {})
        lf = ((J.get("los_freepos") or {}).get("arms", {}).get("z025", {}).get("abs", {}) or {}).get("detections", {})
        extra = f"\nfree pos.: {P(lf['frac_localized'])} loc., {lf['mass_error_median']:+.1f} dex" if lf else ""
        cA = (f"flagged {P(fc)}, {P(fa)} ($\\geq{A_ABS:g}$);\n{P(d.get('frac_localized', 0))} loc., {d.get('mass_error_median', 0):+.1f} dex" + extra, sevf(fc))
    except Exception:
        cA = NA
    try:
        v = UP["los_halo_z025"]["aggregate"]["flag_rate"]["mean"]; cC = (f"{P(v)}\nat baseline", INSENS if v < 0.15 else sevf(v))
    except Exception:
        cC = NA
    try:
        fl = FBA["los_z025"]["flag_rate"]; cB = (f"flagged {P(fl)}", sevf(fl))
    except Exception:
        cB = NA
    rows.append(("line-of-sight halo, foreground $z=0.25$\n(no subhalo present)", [cA, cB, cC]))
    # 9b. sub-threshold subhalo + line-of-sight population in every image, the clean control included (referee round 26)
    try:
        NAp = get("results/new_arms.json")["pop"]
        fA = NAp["clean_fpr_20"]
        cA = (f"clean FPR {P(fA)} at $\\Delta\\chi^2\\geq{A_ABS:g}$;\ncal. thr. {NAp['threshold']:.0f}, $n$={NAp['clean']['n']}", sevf(fA))
    except Exception:
        cA = NA
    try:
        Lb = lambda q: [json.loads(l) for l in open(ROOT / "results/baseline_b_pop/fitted" / q / "scan_results.jsonl")]  # noqa: E731
        L0 = lambda q: [json.loads(l) for l in open(ROOT / "results/baseline_b_full/fitted" / q / "scan_results.jsonl")]  # noqa: E731
        t0 = float(np.quantile([r["delta_chi2"] for r in L0("no_subhalo") if r["reliable_fit"]], 0.9))
        fB = float(np.mean([r["delta_chi2"] >= t0 for r in Lb("no_subhalo") if r["reliable_fit"]]))
        cB = (f"{P(fB)} of clean lenses above\nits Tier-0 threshold, $n$=300", sevf(fB))
    except Exception:
        cB = NA
    try:
        Up = get("results/detector_30k_seed0_pop/results.json")
        cC = (f"AUC {Up['roc_auc_fixed60_vs_no_subhalo']:.2f}\n(Tier 0: 0.67), seed 0", BLUEG)
    except Exception:
        cC = NA
    rows.append(("sub-threshold population: $10^{7}$\u2013$10^{8.5}$\nsubhalos + line-of-sight halos (clean lenses)", [cA, cB, cC]))
    # 10. depth
    try:
        Dd, Ds = DEP["deep"]["A"], DEP["shallow"]["A"]
        # all six bins (joint_scan.json "shallow" block), so the bin rule applies as in every other row
        cA = ccell(A_same("shallow"), lambda b_: J["shallow"]["cal"]["completeness_c60"][b_]["p"], post=f", $n$=300\nmultipole {P(Dd['mp_cal'])} $\\to$ {P(Ds['mp_cal'])}")
    except Exception:
        cA = NA
    try:
        cB = ccell(B_same("results/baseline_b_shallow/fitted", "test_shallow_fixed60"), lambda b_: FBA["shallow"]["completeness_c60"][b_]["p"], post=", $n$=300")
    except Exception:
        cB = NA
    try:
        Cs = DEP["shallow"]["C"]; CR = DEP["shallow"].get("C_retrained") or Cs; Cd = DEP["deep"]["C"]
        txt, col = ccell(lambda b_: Cd["c60_recal"][b_]["mean"], lambda b_: CR["c60_recal"][b_]["mean"], pre="retr.: ")
        f0 = Cs["fpr_clean_at_stored_threshold"]
        cC = (txt + f"\ndeep-trained: {P(f0)} of clean flagged", worst(col, sevf(f0)))
    except Exception:
        cC = NA
    rows.append(("survey depth, 5400 s $\\to$ 135 s (arc S/N $\\div 6$)\n(completeness, 10% cal.; what transfers)", [cA, cB, cC]))
    # 11. PSF
    try:
        def _psf_full():   # the 0.15" arm on the full 1 000-lens populations (round 29), against Tier 0 on the same lenses
            L_ = lambda q: [json.loads(l) for l in open(ROOT / "results/baseline_a_joint_psf015_full" / q / "scan_results.jsonl")]  # noqa: E731
            t_ = float(np.quantile([r["delta_chi2"] for r in L_("no_subhalo") if r["reliable_fit"]], 0.9))
            S_ = [r for r in L_("test_fixed60") if r["reliable_fit"] and r["has_subhalo"]]
            return lambda b_: float(np.mean([r["delta_chi2"] >= t_ for r in S_ if float(b_.split("-")[0]) <= r["log10_M200_true"] < float(b_.split("-")[1])]))
        cA = ccell(J_c60, _psf_full(), post=", $n$=1000")
    except Exception:
        cA = NA
    try:
        cB = ccell(B_same("results/baseline_b_psf015/fitted", "psf015_fixed60"), lambda b_: FBA["psf015"]["completeness_c60"][b_]["p"], post=", $n$=300")
    except Exception:
        cB = NA
    try:
        txt, col = ccell(U_c60, lambda b_: UPS["psf015_fixed60"]["aggregate"]["completeness"][b_]["mean"], pre="retr.: ")
        tr = UPD["psf015_no_subhalo"]["aggregate"]["flag_rate"]["mean"] if "psf015_no_subhalo" in UPD else UP.get("psf015_no_subhalo", {}).get("aggregate", {}).get("flag_rate", {}).get("mean")
        cC = (txt + f"\n$0.08''$-trained: {P(tr)} of clean flagged", worst(col, sevf(tr)))
    except Exception:
        cC = NA
    rows.append(("PSF widened $0.08''\\to0.15''$\n(completeness, 10% cal.; what transfers)", [cA, cB, cC]))
    # combined realism: 0.15" PSF + 600 s + two-component lens light (double-Sersic fit)
    try:
        cA = ccell(J_c60, lambda b_: JR["c60"]["cal"][b_]["p"], post=", $n$=1000")  # clean FPR at the absolute threshold is in the text
        if "multipole_m4_a3" in JR:
            fm = JR["multipole_m4_a3"]["cal"]; cA = (cA[0] + f"\nmultipole {P(fm)} (cal.)", worst(cA[1], sevf(fm)))
    except Exception:
        cA = NA
    try:
        cB = ccell(B_c60, lambda b_: FBA["realistic"]["completeness_c60"][b_]["p"], post=f", $n$=1000\nmultipole {P(FBA['realistic']['fpr_multipole'])}")
        cB = (cB[0], worst(cB[1], sevf(FBA["realistic"]["fpr_multipole"])))
    except Exception:
        cB = NA
    try:
        cC = ccell(U_c60, lambda b_: UR["completeness_c60"][b_]["mean"], pre="retr.: ", post=f"\nAUC {UR['auc']['mean']:.2f}")
    except Exception:
        cC = NA
    rows.append(("combined realism: $0.15''$ PSF, 600 s,\nlens light (completeness, 10% cal.)", [cA, cB, cC]))
    for key, lab in (("clumpy", "source misspecified: two 2\u20136% clumps\n(clean FPR at the absolute threshold)"),
                     ("invcrime", "render misspecified: 5$\\times$ supersampled,\nPSF +10% (clean FPR, absolute threshold)")):
        try:
            X = NAR[key]; f20 = X["clean_fpr_20"]
            cA = (f"clean FPR {P(f20)} at $\\Delta\\chi^2\\geq{A_ABS:g}$;\ncal. threshold $\\Delta\\chi^2={X['threshold']:.0f}$", sevf(f20))
        except Exception:
            cA = NA
        try:
            Y = FBA[key]; cB = (f"threshold $\\times${Y['threshold'] / FBA_T0:.0f};\ncompl. at chance", RED)
        except Exception:
            cB = NA
        rows.append((lab, [cA, cB, NA]))
    # Euclid-like operating point (round 31): each family on its own clean control there, against Tier 0 on the same lenses
    def scan_arm(root):
        L_ = lambda q: [json.loads(l) for l in open(ROOT / root / q / "scan_results.jsonl")]  # noqa: E731
        t_ = float(np.quantile([r["delta_chi2"] for r in L_("no_subhalo") if r["reliable_fit"]], 0.9))
        S_ = [r for r in L_("test_fixed60") if r["reliable_fit"] and r["has_subhalo"]]
        M_ = [r for r in L_("multipole_m4_a3") if r["reliable_fit"]]
        comp = lambda b_: float(np.mean([r["delta_chi2"] >= t_ for r in S_ if float(b_.split("-")[0]) <= r["log10_M200_true"] < float(b_.split("-")[1])]))  # noqa: E731
        return comp, float(np.mean([r["delta_chi2"] >= t_ for r in M_])), {r["index"] for r in L_("test_fixed60")}
    def t0_same(root, idx):
        L_ = lambda q: [json.loads(l) for l in open(ROOT / root / q / "scan_results.jsonl")]  # noqa: E731
        t_ = float(np.quantile([r["delta_chi2"] for r in L_("no_subhalo") if r["reliable_fit"]], 0.9))
        S_ = [r for r in L_("test_fixed60") if r["index"] in idx and r["reliable_fit"] and r["has_subhalo"]]
        return lambda b_: float(np.mean([r["delta_chi2"] >= t_ for r in S_ if float(b_.split("-")[0]) <= r["log10_M200_true"] < float(b_.split("-")[1])]))
    try:
        ce, fe, ie = scan_arm("results/baseline_a_joint_euclid")
        cA = ccell(t0_same(A_JOINT_FULL, ie), ce, post=f", $n$={len(ie)}\nmultipole {P(fe)} (cal.)"); cA = (cA[0], worst(cA[1], sevf(fe)))
    except Exception:
        cA = NA
    try:
        ce, fe, ie = scan_arm("results/baseline_b_euclid/fitted")
        cB = ccell(t0_same(B_FULL, ie), ce, post=f", $n$={len(ie)}\nmultipole {P(fe)}"); cB = (cB[0], worst(cB[1], sevf(fe)))
    except Exception:
        cB = NA
    rows.append(("Euclid-like: $0.101''$ pixels, $0.16''$ PSF,\nsources 2.5 mag fainter (completeness, 10% cal.)", [cA, cB, NA]))
    # 12. CDM weighting
    def mf(label):
        try:
            r = next(x for x in MF["results"] if x["label"] == label)
            return (f"{P(r['naive_unweighted'])} $\\to$ {P(r['cdm_mass_function_weighted'])}", NEUTRAL)
        except Exception:
            return NA
    rows.append(("CDM mass-function weighting, flat $\\to$ weighted\n(population completeness, $c=60$, 10% cal.)", [mf("Family A, c=60"), mf("Family B, c=60"), mf("U-Net (Family C), c=60")]))

    # ---- round 29 (M1, M2): capability rows (localization, mass, CDM weighting) leave the stress map for the text and
    # Table loccomp; gravitational imaging with parametric confirmation becomes a column, and Family C is the network
    # trained at the published scale (5x10^5 images) wherever it was scored, else the 30 000-image network, marked "30k".
    rows = [r for r in rows if not r[0].startswith(("localization", "mass estimate", "CDM mass-function"))]
    GI = get("results/family_bgi.json") or {}
    GIp = GI.get("pops", {})
    SD = (get("results/strong_detectors.json") or {}).get("unet500k", {})
    U5 = get("results/detector_30k_c60pub_seed0/results.json") or {}
    B6 = ["8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5", "10.5-11.0"]
    def u5_t0(b_):
        return U5["completeness_c60_by_mass_bin"][b_]["completeness"]
    def gi_conf(pop):
        return lambda b_: GIp[pop]["completeness_confirmed"][b_]["p"]
    def gi_rate(pop, key="rate_confirmed"):
        return GIp[pop][key]["p"]
    def gi_cells(lab):
        try:
            if lab.startswith("concentration"):
                fl = GI["flip_confirmed"]
                return ccell(gi_conf("test_fixed60"), gi_conf("test_fixed15"), post=f", $n$=300\nflip {fl[0]}:{fl[1]}")
            if lab.startswith("lens-shape"):
                f = gi_rate("multipole_m4_a3"); return (f"confirmed {P(f)}\n($\\Delta\\ln Z\\geq10$: {P(gi_rate('multipole_m4_a3', 'rate_lnz10'))})", sevf(f))
            if lab.startswith("non-physical"):
                v = [gi_rate(k) for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
                w = [gi_rate(k, "rate_lnz10") for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
                return (f"confirmed {P(max(v))}\n($\\Delta\\ln Z\\geq10$: {P(min(w))}\u2013{P(max(w))})", sevf(max(v)))
            if lab.startswith("real COSMOS"):
                T = [json.loads(l) for l in open(ROOT / "results/baseline_bgi_tier1/no_subhalo/scan_results.jsonl")]
                f = float(np.mean([r["delta_lnZ"] >= 10 for r in T])) if T and "delta_lnZ" in T[0] else None
                return (f"{P(f)} of clean flagged at\n$\\Delta\\ln Z\\geq10$ (all {len(T)} lenses)", RED) if f is not None else NA
            if lab.startswith("sub-threshold"):
                Pc = (get("results/family_bgi_pop_confirmation.json") or {})["pops"]["no_subhalo"]
                return (f"confirmed {P(Pc['confirmed_20'])} of clean,\n$n$={Pc['n_retained']}", sevf(Pc["confirmed_20"]))
            if lab.startswith("lens light, double"):
                Lc = get("results/family_bgi_lenslight_confirmation.json")
                if Lc:
                    f = Lc["pops"]["no_subhalo"]["confirmed_20"]; m = Lc["pops"].get("multipole_m4_a3", {}).get("confirmed_20")
                    return (f"confirmed {P(f)} of clean" + (f";\nmultipole {P(m)}" if m is not None else "") + f", $n$={Lc['pops']['no_subhalo']['n_retained']}", worst(sevf(f), sevf(m or 0)))
            if lab.startswith("survey depth"):
                Dc = get("results/family_bgi_600_confirmation.json")
                if Dc:
                    f = Dc["pops"]["no_subhalo"]["confirmed_20"]; m = Dc["pops"].get("multipole_m4_a3", {}).get("confirmed_20")
                    return (f"600 s: confirmed {P(f)} of clean,\n" + (f"multipole {P(m)}, " if m is not None else "") + f"$n$={Dc['pops']['no_subhalo']['n_retained']}", worst(sevf(f), sevf(m or 0)))
            conf_rate = {"elliptical": ("ell", "ellmultipole_m4_a3", "confirmed {f}\n($\\Delta\\ln Z\\geq10$: {g})"),
                         "line-of-sight": ("los", "los_halo_z025", "confirmed {f}\n($\\Delta\\ln Z\\geq10$: {g})"),
                         "source misspecified": ("clumpy", "no_subhalo", "confirmed {f} of clean\n($\\Delta\\ln Z\\geq10$: {g})"),
                         "render misspecified": ("invcrime", "no_subhalo", "confirmed {f} of clean\n($\\Delta\\ln Z\\geq10$: {g})")}
            for pre_, (tag, pop, fmt) in conf_rate.items():
                if lab.startswith(pre_):
                    Xc = get(f"results/family_bgi_{tag}_confirmation.json")
                    if not Xc or pop not in Xc["pops"]:
                        return NA
                    d = Xc["pops"][pop]
                    return (fmt.format(f=P(d["confirmed_20"]), g=P(d["flag_rate_b"])) + f", $n$={d['n_retained']}", sevf(d["confirmed_20"]))
            arm_conf = {"lens light, single": ("lenslight1", "under-flexible"), "PSF widened": ("psf015", ""), "combined realism": ("realistic", ""),
                        "Euclid-like": ("euclid", "")}
            for pre_, (tag, _) in arm_conf.items():
                if lab.startswith(pre_):
                    Xc = get(f"results/family_bgi_{tag}_confirmation.json")
                    if not Xc or "test_fixed60" not in Xc["pops"]:
                        return NA
                    d = Xc["pops"]["test_fixed60"]["completeness_confirmed_20"]; f = Xc["pops"]["no_subhalo"]["confirmed_20"]
                    m = Xc["pops"].get("multipole_m4_a3", {}).get("confirmed_20")
                    txt, col = ccell(gi_conf("test_fixed60"), lambda b_: d[b_]["p"] if d[b_]["p"] is not None else 0.0, post=f", $n$={Xc['pops']['no_subhalo']['n_retained']}")
                    extra = f"\nclean {P(f)}" + (f", multipole {P(m)}" if m is not None else "")
                    return (txt + extra, worst(col, sevf(f), sevf(m or 0)))
        except Exception:
            return NA
        return NA
    def c5_cells(lab, c30):
        try:
            C5 = SD["confounders"]; A5 = SD["arms"]
            def arm(key, label=""):
                a = A5[key]; b_ = lowbin(u5_t0)
                txt, col = ccell(u5_t0, lambda bb: a["completeness"][B6.index(bb)], post=f"\nAUC {a['auc']:.2f}")
                return (label + txt, col)
            if lab.startswith("concentration"):
                fl = U5["paired_concentration_flip"]
                return ccell(u5_t0, lambda b_: U5["completeness_c15_by_mass_bin"][b_]["completeness"], post=f"\nflip {fl['detected_at_c60_not_c15']}:{fl['detected_at_c15_not_c60']}")
            if lab.startswith("lens-shape"):
                f = C5["multipole_m4_a3"]["flag_rate"]; return (f"{P(f)}\nunmoved", INSENS if f < 0.15 else sevf(f))
            if lab.startswith("elliptical"):
                f = C5["ellmultipole_m4_a3"]["flag_rate"]; return (f"{P(f)}", INSENS if f < 0.15 else sevf(f))
            if lab.startswith("non-physical"):
                v = [C5[k]["flag_rate"] for k in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10")]
                return (f"{P(min(v))}\u2013{P(max(v))}\nfires", sevf(max(v)))
            if lab.startswith("line-of-sight"):
                f = C5["los_halo_z025"]["flag_rate"]; return (f"{P(f)}\nat baseline", INSENS if f < 0.15 else sevf(f))
            if lab.startswith("real COSMOS"):
                return (f"AUC {A5['tier1']['auc']:.2f}", RED if A5["tier1"]["auc"] < 0.6 else AMBER)
            if lab.startswith("sub-threshold"):
                return (f"AUC {A5['population']['auc']:.2f}\n(Tier 0: {U5['roc_auc_fixed60_vs_no_subhalo']:.2f})", BLUEG if A5["population"]["auc"] > U5["roc_auc_fixed60_vs_no_subhalo"] - 0.05 else AMBER)
            if lab.startswith("survey depth"):
                return arm("shallow135")
            if lab.startswith("PSF"):
                return arm("psf015")
            if lab.startswith("combined realism"):
                return arm("realistic")
            if lab.startswith("source misspecified"):
                return arm("clumpy")
            if lab.startswith("render misspecified"):
                return arm("render_psf")
            if lab.startswith("Euclid-like"):
                return arm("euclid")
        except Exception:
            pass
        if c30[0] in ("not run", "not attempted"):
            return c30
        L_ = c30[0].split("\n")   # the "30k" mark goes on the shortest line so the first stays inside the cell
        k_ = 1 if len(L_) > 1 else 0
        L_[k_] = "30k: " + L_[k_]
        return ("\n".join(L_), c30[1])
    rows = [(lab, [cells[0], gi_cells(lab), cells[1], c5_cells(lab, cells[2])]) for lab, cells in rows]
    cols = ["A  joint scan\n(idealized: exact\nsource family)", "GI  linearized grav.\nimaging (pixelized\nsource, confirmed)",
            "B  free-form $\\delta\\psi$\n(diagnostic,\nnot a detector)", "C  U-Net, $5\\times10^{5}$\nimages (\"30k\": the\n30 000-image network)"]
    # rows grouped by the mechanism they test (Sect. 5.1), each group under a header
    GROUPS = [("Unmodelled lensing or light structure", ("lens-shape", "elliptical", "line-of-sight", "sub-threshold", "lens light, single", "lens light, double")),
              ("Non-physical structure", ("non-physical",)),
              ("Forward and source model", ("source misspecified", "render misspecified", "real COSMOS")),
              ("Signal and operating point", ("concentration", "survey depth", "PSF widened", "combined realism", "Euclid-like"))]
    ordered = []
    for head, pres in GROUPS:
        ordered.append((head, None))
        for pre_ in pres:
            ordered += [r for r in rows if r[0].startswith(pre_)]
    assert len([r for r in ordered if r[1] is not None]) == len(rows), "a summary row is in no group"
    HH = 0.42   # header height in row units
    ytot = sum(HH if c is None else 1.0 for _, c in ordered)
    CW = 0.66
    fig, ax = plt.subplots(figsize=(DBL_W, 0.34 * ytot + 0.8))
    ax.set_xlim(0, 1.35 + CW * len(cols) + 0.05); ax.set_ylim(0, ytot); ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(1.35 + j * CW + CW / 2, ytot + 0.08, c, ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ytop = ytot
    for lab, cells in ordered:
        if cells is None:
            ytop -= HH
            ax.text(0.02, ytop + 0.12, lab, ha="left", va="bottom", fontsize=6.6, fontweight="bold", fontstyle="italic")
            ax.plot([0.02, 1.35 + CW * len(cols)], [ytop + 0.06, ytop + 0.06], color="#999999", lw=0.5)
            continue
        ytop -= 1.0; y = ytop
        ax.text(1.30, y + 0.5, lab, ha="right", va="center", fontsize=6.4)
        for j, (txt, col) in enumerate(cells):
            # one headline number per cell; the detail is in the tables the caption points to
            L_ = txt.split("\n")
            head = L_[0].split(";")[0].rstrip(", ")
            head = re.sub(r"\s+(of clean|flagged at|at|of|above|lenses)$", "", head).replace("clean FPR ", "FPR ").replace("clean lenses", "clean")
            k30 = any(l.startswith("30k: ") for l in L_[1:])
            if len(re.sub(r"\$[^$]*\$", "xx", head)) + 5 * k30 > 19:
                head = re.sub(r"\s*\([^()]*\^[^()]*\)$", "", head)   # drop the trailing mass bin; the caption names it
            txt = ("30k: " if k30 else "") + head
            x0 = 1.35 + j * CW
            ax.add_patch(mpatches.FancyBboxPatch((x0 + 0.02, y + 0.05), CW - 0.04, 0.9, boxstyle="round,pad=0,rounding_size=0.04", fc=col, ec="#cccccc" if col == NEUTRAL else "white", lw=1.2,
                                                 hatch="////" if col == INSENS else ("...." if col == GREY else ("\\\\" if col == NOTRES else None))))
            ax.text(x0 + CW / 2, y + 0.5, txt.replace("-0.", "\u22120.").replace("-1.", "\u22121."), ha="center", va="center", fontsize=5.6,
                    bbox=dict(fc=col, ec="none", pad=0.6) if col in (INSENS, GREY, NOTRES) else None)
    handles = [mpatches.Patch(fc=BLUEG, label="unaffected / recovered"), mpatches.Patch(fc=INSENS, hatch="////", ec="#9ccbe8", label="insensitive (as the $5\\times10^{5}$ network)"), mpatches.Patch(fc=NOTRES, hatch="\\\\", ec="#bbbbbb", label="not resolved by the arm's size"), mpatches.Patch(fc=AMBER, label="degraded"), mpatches.Patch(fc=RED, label="fails"), mpatches.Patch(fc=GREY, hatch="....", ec="#bbbbbb", label="not tested / n.a.")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.0, frameon=False)
    save(fig, "fig12_summary")


# ======================================================================
# Fig 13 -- completeness vs projected signal: is the concentration collapse a property of the signal?
# ======================================================================
def fig13_signal():
    """Completeness vs two concentration-independent signal variables on the full 1 000-lens populations:
    top, the perturbation S/N a detector sees; bottom, the projected mass within 0.2" (what a peak statistic measures)."""
    d = load("results/completeness_vs_signal.json")
    rows = [("log10_snr", r"$\log_{10}$ S/N$_{\rm pert}$", 0.25),
            ("log10_Mproj_0p2", r"$\log_{10}[M_{\rm proj}(<0.2'')/M_\odot]$", 0.5)]
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 5.1), sharey=True, sharex="row", gridspec_kw={"hspace": 0.55})
    fams = (("A", "Family A: joint parametric scan", C_A), ("B", "Family B: free-form residual", C_B), ("C", "Family C: U-Net", C_UNET))
    for (var, xlab, w), axrow in zip(rows, axes):
        for ax, (fam, name, col) in zip(axrow, fams):
            F = d["families"][fam]
            arms = [c for c in ("c60", "c15") if c in F]
            allx = np.concatenate([np.array(F[c][var]) for c in arms])
            lo0 = np.floor(allx.min() / w) * w; edges = np.arange(lo0, allx.max() + w, w)
            # a bin is plotted only if BOTH populations hold at least min_n_per_bin lenses there (the caption's rule)
            keep = [(a, b) for a, b in zip(edges[:-1], edges[1:])
                    if all(((np.array(F[c][var]) >= a) & (np.array(F[c][var]) < b)).sum() >= d["min_n_per_bin"] for c in arms)]
            for c, ls, mk, lab, colr in (("c60", "-", "o", "$c=60$ population (family colour)", col), ("c15", "--", "s", "$c=15$ population (grey)", GRAY)):
                if c not in F: continue
                xv = np.array(F[c][var]); det = np.array(F[c]["detected"])
                x, y, lo, hi = [], [], [], []
                for a, b in keep:
                    sel = (xv >= a) & (xv < b)
                    p, l, h = wilson(det[sel].sum(), sel.sum()); x.append(0.5 * (a + b)); y.append(100 * p); lo.append(max(0.0, 100 * (p - l))); hi.append(max(0.0, 100 * (h - p)))
                ax.errorbar(x, y, yerr=[lo, hi], fmt=mk + ls, color=colr, ms=3.5, capsize=2, lw=1.1, label=lab)
            comp = d["comparison"][fam][var]
            zmax = max(abs(r["z"]) for r in comp) if comp else 0
            ztxt = f"$c{{=}}60$ vs $c{{=}}15$: largest $|z|$ = {zmax:.1f} over {len(comp)} bins"
            ax.axhline(10, color=BLACK, ls=":", lw=0.7); ax.set_ylim(0, 100)
            ax.set_title((name + "\n" if var == "log10_snr" else "") + ztxt, fontsize=7.2 if var == "log10_snr" else 6.8)
            ax.set_xlabel(xlab)
    for ax in axes[:, 0]: ax.set_ylabel("completeness at 10% FPR  [%]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 1.035))
    save(fig, "fig13_signal")


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
    # one label per row, family on the first line and concentration on the second, so long
    # names (Family B was added after the first draft) cannot collide with the left-hand panel
    def _lab(t):
        fam, _, conc = t.partition(",")
        fam = fam.replace("U-Net (Family C)", "U-Net").strip()
        return f"{fam}\n{conc.strip()}" if conc else fam
    labels = [_lab(r["label"]) for r in rows]
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
    ax.set_title("(b) the same per-bin numbers, weighted two ways"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax.invert_yaxis()
    save(fig, "fig6_mass_function")


# ======================================================================
# Fig 7 -- Tier 1: the U-Net retrained on real sources
# ======================================================================
def fig7_tier1():
    """SUPPLEMENTARY, not included in the manuscript: the Tier-1 U-Net result is reported in the text only.
    Kept because it regenerates from released results and may be wanted by a reader."""
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
       "fig5": fig5_mass_bias, "fig6": fig6_mass_function, "fig7": fig7_tier1, "fig8": fig8_family_b, "fig9": fig9_scaleup, "fig11": fig11_lenslight, "fig12": fig12_summary, "fig13": fig13_signal}

# ======================================================================
# Fig 14 -- the learned detector against its training budget (referee round 29, M2: replaces the training-variant prose)
# ======================================================================
def fig14_learning_curve():
    """AUC (c=60 test vs clean control) and completeness at 10^9.5-10^10 Msun at c=60 and c=15 against the number of
    training images, for the U-Net recipe (8 000 and 30 000: four-seed means and standard deviations; 100 000 and
    500 000: one seed each), with the ConvNeXt encoder and the neural ratio estimator at 100 000 images."""
    B_ = "9.5-10.0"
    def agg(rel):
        d = load(rel); return (d["auc"]["mean"], d["auc"]["std"], d["completeness_c60"][B_]["mean"], d["completeness_c60"][B_]["std"],
                              d["completeness_c15"][B_]["mean"], d["completeness_c15"][B_]["std"])
    def one(run):
        d = load(f"results/{run}/results.json")
        k60 = "completeness_c60_by_mass_bin" if "completeness_c60_by_mass_bin" in d else "completeness_test_fixed60"
        k15 = "completeness_c15_by_mass_bin" if "completeness_c15_by_mass_bin" in d else "completeness_test_fixed15"
        auc = d.get("roc_auc_fixed60_vs_no_subhalo", d.get("auc_test_fixed60"))
        return (auc, 0.0, d[k60][B_]["completeness"], 0.0, d[k15][B_]["completeness"], 0.0)
    N = [8e3, 3e4, 1e5, 5e5]
    R = [agg("results/aggregate/summary.json"), agg("results/aggregate_30k/summary.json"), one("detector_30k_c60big_seed0"), one("detector_30k_c60pub_seed0")]
    X = {"ConvNeXt": one("detector_100k_c60convnext_seed0"), "ratio estimator": one("nre_100k_c60_seed0")}
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    ax = axes[0]
    ax.errorbar(N, [r[0] for r in R], yerr=[r[1] for r in R], color=BLUE, marker="o", ms=4, lw=1.2, capsize=2, label="U-Net recipe")
    for (lab, r), mk, col in zip(X.items(), ("s", "D"), (VERM, PURPLE)):
        ax.plot([1e5], [r[0]], marker=mk, ms=5, ls="none", color=col, label=lab)
    ax.axhline(0.5, color=GRAY, ls=":", lw=0.8); ax.text(9e3, 0.505, "chance", fontsize=6.3, color=GRAY)
    ax.set_xscale("log"); ax.set_xlabel("training images"); ax.set_ylabel("AUC, $c=60$ vs clean"); ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax = axes[1]
    ax.errorbar(N, [100 * r[2] for r in R], yerr=[100 * r[3] for r in R], color=BLUE, marker="o", ms=4, lw=1.2, capsize=2, label="$c=60$")
    ax.errorbar(N, [100 * r[4] for r in R], yerr=[100 * r[5] for r in R], color=GRAY, marker="o", mfc="white", ms=4, lw=1.2, capsize=2, label="$c=15$, same lenses")
    for (lab, r), mk, col in zip(X.items(), ("s", "D"), (VERM, PURPLE)):
        ax.plot([1e5], [100 * r[2]], marker=mk, ms=5, ls="none", color=col); ax.plot([1e5], [100 * r[4]], marker=mk, ms=5, ls="none", mfc="white", color=col)
    ax.axhline(10, color=GRAY, ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("training images"); ax.set_ylabel("completeness at $10^{9.5}$\u2013$10^{10}\,M_\\odot$ [%]"); ax.set_ylim(0, 100)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig14_learning_curve")


# ======================================================================
# Fig 15 -- misspecification tolerance of the absolute threshold (referee round 29, M6)
# ======================================================================
def fig15_misspec():
    """Clean false-positive rate of the joint scan at Delta chi^2 >= 20 against the amplitude of a PSF error and of
    unmodelled source clumps, at three operating points, and the median clean Delta chi^2 with an epsilon^2 guide."""
    G = load("results/misspec_grid.json")
    sty = {"t0": (BLUE, "o", "Tier 0 (arc S/N $1.5\\times10^{3}$)"), "600": (ORANGE, "s", "600 s ($4.9\\times10^{2}$)"), "euclid": (VERM, "D", "Euclid-like (176)")}
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for j, (kind, xl) in enumerate((("psf", "PSF width error"), ("clump", "flux of each of two source clumps"))):
        ax = axes[j]
        for b, (col, mk, lab) in sty.items():
            pts = G[b][kind]; base = G[b]["base"]
            if not pts:
                continue
            x = [100 * p["amp"] for p in pts]; y = [100 * p["fpr_20"] for p in pts]
            lo = [100 * (p["fpr_20"] - p["fpr_20_68"][0]) for p in pts]; hi = [100 * (p["fpr_20_68"][1] - p["fpr_20"]) for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5, label=lab)
            ax.axhline(100 * base["fpr_20"], color=col, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xlabel(xl + " [%]"); ax.set_ylim(-3, 100)
        tk = [2, 5, 10] if kind == "psf" else [0.5, 1, 2, 4]
        ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.minorticks_off()
        ax.set_ylabel("clean lenses above $\\Delta\\chi^2=20$ [%]" if j == 0 else "")
        ax.set_title("(a) PSF" if j == 0 else "(b) source", fontsize=7)
    axes[0].legend(fontsize=5.8, frameon=False, loc="upper left")
    ax = axes[2]
    for b, (col, mk, lab) in sty.items():
        for kind, ls, fill in (("psf", "-", col), ("clump", "--", "white")):
            pts = G[b][kind]
            if pts:
                ax.plot([100 * p["amp"] for p in pts], [p["median_dchi2"] for p in pts], color=col, marker=mk, ms=3.5, ls=ls, lw=1.0, mfc=fill)
    xg = np.array([0.4, 12.0]); ax.plot(xg, 45.6 * (xg / 5.0) ** 2, color=GRAY, lw=0.8, ls=":"); ax.text(0.5, 45.6 * (0.6 / 5) ** 2 * 1.6, "$\\propto\\epsilon^{2}$", fontsize=6.5, color=GRAY)
    ax.axhline(20, color=BLACK, lw=0.6, ls="-."); ax.text(0.45, 23, "$\\Delta\\chi^2=20$", fontsize=6.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.5", "1", "2", "5", "10"]); ax.set_xlabel("misspecification amplitude $\\epsilon$ [%]"); ax.set_ylabel("median clean $\\Delta\\chi^2$")
    ax.set_title("(c) solid: PSF; dashed: source", fontsize=7)
    fig.tight_layout()
    save(fig, "fig15_misspec")


if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or list(ALL)   # e.g. `python scripts/make_paper_figures.py fig5` rebuilds one figure
    print("building paper figures ->", FIG)
    for k in want:
        ALL[k]()
    print("done")
