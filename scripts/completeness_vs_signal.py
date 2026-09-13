"""Is the concentration collapse a property of the SIGNAL? (referee rounds 3 and 4)

For every subhalo in the matched c=60 / c=15 test populations compute three concentration-independent
signal variables and record whether each family detected it at the common 10%-FPR threshold:

  * log10 M_proj(<0.1")  -- TNFW projected mass inside 0.1" of the subhalo (round-3 variable)
  * log10 M_proj(<0.2")  -- the same inside 0.2"
  * log10 S/N_pert       -- the perturbation signal-to-noise a detector actually sees:
                            sqrt( sum_pixels [(I_full - I_no_subhalo) / sigma]^2 ) from the population's own
                            noiseless twin images and lenstronomy's per-pixel noise model.

If c=60 and c=15 fall on one curve against a variable, completeness is a function of that signal and the
concentration comparison is a statement about how much signal a diffuse perturber delivers.

    python scripts/completeness_vs_signal.py            # full 1 000-lens populations for A and B; U-Net scores every lens
-> results/completeness_vs_signal.json, paper/tables/signal_variables.tex
   (the paper figure is drawn by make_paper_figures.py::fig13_signal)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.LensModel.Profiles.tnfw import TNFW  # noqa: E402
from lenstronomy.SimulationAPI.sim_api import SimAPI  # noqa: E402
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

LC = LensCosmo(z_lens=0.5, z_source=1.0)
PROF = TNFW()
VARIABLES = {"log10_Mproj_0p1": r"$\log_{10} M_{\rm proj}(<0.1'')$", "log10_Mproj_0p2": r"$\log_{10} M_{\rm proj}(<0.2'')$", "log10_snr": r"$\log_{10}$ S/N$_{\rm pert}$"}
BIN_WIDTH = {"log10_Mproj_0p1": 0.5, "log10_Mproj_0p2": 0.5, "log10_snr": 0.25}
MIN_N = 20


def projected_mass_msun(sub, R_arcsec):
    Rs, alpha_Rs = LC.nfw_physical2angle(M=10 ** sub["log10_M200"], c=sub["concentration"])
    rho0 = PROF.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
    return float(PROF.mass_2d(R_arcsec, Rs, rho0, sub["tau"] * Rs) * LC.sigma_crit_angle)


def perturbation_snr(pop_dir):
    """Per-lens perturbation S/N from the noiseless full and no-subhalo twins (NaN for lenses without a subhalo)."""
    man = json.loads((pop_dir / "manifest.json").read_text())
    full = np.load(pop_dir / "images_full_noiseless.npy", mmap_mode="r"); ctrl = np.load(pop_dir / "images_control_noiseless.npy", mmap_mode="r")
    sim = SimAPI(num_pix=man["image_shape"][0], kwargs_single_band=man["kwargs_band"], kwargs_model={"lens_model_list": ["EPL"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    out = np.full(len(full), np.nan)
    for i in range(len(full)):
        f = np.asarray(full[i]); sigma = sim.estimate_noise(f)
        out[i] = np.sqrt(np.sum(((f - np.asarray(ctrl[i])) / sigma) ** 2))
    return out


def family_a_like(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    if not p.exists():
        return None
    ns = [json.loads(l) for l in open(ROOT / root / "no_subhalo" / "scan_results.jsonl")]
    thr = float(np.quantile([r["delta_chi2"] for r in ns if r["reliable_fit"]], 0.90))
    return {r["index"]: (r["delta_chi2"] >= thr) for r in (json.loads(l) for l in open(p)) if r["reliable_fit"] and r["has_subhalo"]}


def unet_detections(pop, ckpt=ROOT / "checkpoints/unet_v0/model_best.pt", results=ROOT / "results/detector_v0_best/results.json"):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ck = torch.load(ckpt, map_location=dev); model = UNet(in_ch=1, base=ck.get("base", 16)).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
    thr = json.loads(results.read_text())["threshold_at_10pct_fpr"]
    ds = LensPatchDataset(ROOT / "data" / pop, scale=ck["scale"]); out = {}
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, has_sub, _ = ds[i]
            if not ds.truths[i].get("subhalo"):
                continue
            out[i] = float(torch.sigmoid(model(img.unsqueeze(0).to(dev))).max()) >= thr
    return out


def wilson(k, n, z=1.0):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, c - h, c + h


def binned(x, det, width):
    lo0 = np.floor(np.nanmin(x) / width) * width; hi0 = np.ceil(np.nanmax(x) / width) * width
    edges = np.arange(lo0, hi0 + width / 2, width)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        rows.append({"lo": float(a), "hi": float(b), "n": int(sel.sum()), "k": int(det[sel].sum())})
    return rows


def compare(d, var):
    """Per-bin c15 - c60 completeness at equal signal, with a two-proportion z, over bins holding >= MIN_N lenses in both."""
    x60, k60 = np.array(d["c60"][var]), np.array(d["c60"]["detected"]); x15, k15 = np.array(d["c15"][var]), np.array(d["c15"]["detected"])
    w = BIN_WIDTH[var]
    lo0 = np.floor(min(np.nanmin(x60), np.nanmin(x15)) / w) * w; hi0 = np.ceil(max(np.nanmax(x60), np.nanmax(x15)) / w) * w
    rows = []
    for a in np.arange(lo0, hi0, w):
        s60 = (x60 >= a) & (x60 < a + w); s15 = (x15 >= a) & (x15 < a + w)
        if s60.sum() < MIN_N or s15.sum() < MIN_N:
            continue
        p60, p15 = k60[s60].mean(), k15[s15].mean()
        se = np.sqrt(p60 * (1 - p60) / s60.sum() + p15 * (1 - p15) / s15.sum())
        rows.append({"lo": float(a), "hi": float(a + w), "n60": int(s60.sum()), "n15": int(s15.sum()), "p60": float(p60), "p15": float(p15), "diff": float(p15 - p60), "z": float((p15 - p60) / se) if se > 0 else 0.0})
    return rows


def joint_chi2(rows):
    """A pooled test over bins, so that variables binned differently can be compared.

    The per-bin two-proportion z are independent (disjoint lenses), so sum z^2 is chi^2 with
    one degree of freedom per bin under the null that the two concentrations have the same
    completeness at equal signal. This replaces the max|z| of the first version, whose null
    expectation grows with the number of bins (referee round 7, technical point 1)."""
    if not rows:
        return None
    z = np.array([r["z"] for r in rows]); chi2 = float((z ** 2).sum()); k = len(rows)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, k))
    except Exception:
        p = None
    return {"chi2": chi2, "dof": k, "chi2_per_dof": chi2 / k, "p_value": p,
            "max_abs_z": float(np.abs(z).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-root", default="results/baseline_a_full"); ap.add_argument("--b-root", default="results/baseline_b_full/fitted")
    ap.add_argument("--out", default="results/completeness_vs_signal.json"); ap.add_argument("--table", default="paper/tables/signal_variables.tex")
    args = ap.parse_args()
    truths = {c: [json.loads(l) for l in open(ROOT / f"data/test_fixed{c}/truth.jsonl")] for c in (60, 15)}
    snr = {c: perturbation_snr(ROOT / f"data/test_fixed{c}") for c in (60, 15)}
    res = {"variables": VARIABLES, "bin_width": BIN_WIDTH, "min_n_per_bin": MIN_N, "roots": {"A": args.a_root, "B": args.b_root, "C": "results/detector_v0_best (single seed, every lens)"}, "families": {}}
    for fam, getter in (("A", lambda pop: family_a_like(args.a_root, pop)), ("B", lambda pop: family_a_like(args.b_root, pop)), ("C", unet_detections)):
        res["families"][fam] = {}
        for c in (60, 15):
            det = getter(f"test_fixed{c}")
            if det is None:
                continue
            idx = sorted(det)
            subs = [truths[c][i]["subhalo"] for i in idx]
            res["families"][fam][f"c{c}"] = {
                "index": idx, "log10_M200": [s["log10_M200"] for s in subs],
                "log10_Mproj_0p1": [float(np.log10(projected_mass_msun(s, 0.1))) for s in subs],
                "log10_Mproj_0p2": [float(np.log10(projected_mass_msun(s, 0.2))) for s in subs],
                "log10_snr": [float(np.log10(snr[c][i])) for i in idx],
                "detected": [bool(det[i]) for i in idx]}
    res["comparison"] = {fam: {var: compare(d, var) for var in VARIABLES} for fam, d in res["families"].items() if "c60" in d and "c15" in d}
    res["joint"] = {fam: {var: joint_chi2(rows) for var, rows in byvar.items()} for fam, byvar in res["comparison"].items()}
    (ROOT / args.out).write_text(json.dumps(res))

    # ---- text summary + LaTeX table: c15 - c60 at equal signal, per family and variable
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"family & signal variable & bins & mean $\Delta$ & $\chi^2$/dof ($p$) \\", r"\midrule"]
    names = {"A": "A scan", "B": "B pot.\\ corr.", "C": "C U-Net"}
    for fam, byvar in res["comparison"].items():
        print(f"== Family {fam}")
        for k, (var, rows) in enumerate(byvar.items()):
            if not rows:
                continue
            diffs = np.array([r["diff"] for r in rows]); zs = np.array([r["z"] for r in rows]); j = int(np.argmax(np.abs(zs)))
            print(f"  {var:16s} " + " | ".join(f"{r['lo']:.2f}-{r['hi']:.2f}: {100*r['p60']:.0f} vs {100*r['p15']:.0f}% (n {r['n60']}/{r['n15']}, z {r['z']:+.1f})" for r in rows))
            sign = "+" if diffs.mean() >= 0 else "$-$"
            J = res["joint"][fam][var]
            pv = "--" if J["p_value"] is None else (f"{J['p_value']:.3f}" if J["p_value"] >= 0.001 else "$<0.001$")
            lines.append(f"{names[fam] if k == 0 else ''} & {VARIABLES[var]} & {len(rows)} & {sign}{abs(100*diffs.mean()):.0f} & {J['chi2_per_dof']:.1f} ({pv}) \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / args.table).write_text("\n".join(lines) + "\n")
    print("wrote", args.out, "and", args.table)


if __name__ == "__main__":
    main()
