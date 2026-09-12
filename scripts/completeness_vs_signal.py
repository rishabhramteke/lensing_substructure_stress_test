"""Referee round 3, point 6: is the concentration collapse a property of the SIGNAL?

For every subhalo in the matched c=60 / c=15 test populations compute a concentration-independent
signal proxy -- the TNFW projected mass inside a fixed aperture (default 0.1") around the subhalo
-- and plot completeness against it for both populations and all three families. If the two
concentrations collapse onto one curve, completeness is a function of the projected signal and
the c=60 vs c=15 comparison is a statement about how much signal a diffuse perturber puts inside
the resolution element, not about two arbitrary c values.

    python scripts/completeness_vs_signal.py [--a-root results/baseline_a] [--b-root results/baseline_b/fitted]
-> results/completeness_vs_signal.json (+ the paper figure is drawn by make_paper_figures.py::fig13_signal)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.LensModel.Profiles.tnfw import TNFW  # noqa: E402
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

LC = LensCosmo(z_lens=0.5, z_source=1.0)
PROF = TNFW()


def projected_mass_msun(sub, R_arcsec):
    """TNFW projected mass inside R (arcsec) in solar masses, from the population's own conversion."""
    Rs, alpha_Rs = LC.nfw_physical2angle(M=10 ** sub["log10_M200"], c=sub["concentration"])
    rho0 = PROF.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
    m2d_angular = PROF.mass_2d(R_arcsec, Rs, rho0, sub["tau"] * Rs)      # in units of Sigma_crit * arcsec^2
    return float(m2d_angular * LC.sigma_crit_angle)                       # M_sun


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-root", default="results/baseline_a"); ap.add_argument("--b-root", default="results/baseline_b/fitted")
    ap.add_argument("--aperture", type=float, default=0.1)
    ap.add_argument("--out", default="results/completeness_vs_signal.json")
    args = ap.parse_args()
    res = {"aperture_arcsec": args.aperture, "families": {}}
    truths = {c: [json.loads(l) for l in open(ROOT / f"data/test_fixed{c}/truth.jsonl")] for c in (60, 15)}
    for fam, getter in (("A", lambda pop: family_a_like(args.a_root, pop)), ("B", lambda pop: family_a_like(args.b_root, pop)), ("C", unet_detections)):
        res["families"][fam] = {}
        for c in (60, 15):
            det = getter(f"test_fixed{c}")
            if det is None:
                continue
            rows = [(projected_mass_msun(truths[c][i]["subhalo"], args.aperture), truths[c][i]["subhalo"]["log10_M200"], bool(d)) for i, d in det.items()]
            res["families"][fam][f"c{c}"] = {"log10_Mproj": [float(np.log10(m)) for m, _, _ in rows], "log10_M200": [x for _, x, _ in rows], "detected": [d for _, _, d in rows]}
    (ROOT / args.out).write_text(json.dumps(res))
    # quick text summary: completeness in bins of log10 M_proj, both concentrations
    edges = np.arange(7.0, 10.01, 0.5)
    for fam, d in res["families"].items():
        print(f"== Family {fam}")
        for c in ("c60", "c15"):
            if c not in d: continue
            mp = np.array(d[c]["log10_Mproj"]); det = np.array(d[c]["detected"])
            cells = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (mp >= lo) & (mp < hi)
                cells.append(f"{lo:.1f}-{hi:.1f}: {100*det[sel].mean():.0f}% (n={sel.sum()})" if sel.sum() else f"{lo:.1f}-{hi:.1f}: --")
            print(f"  {c}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
