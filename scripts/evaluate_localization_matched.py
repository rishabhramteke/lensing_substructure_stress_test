"""Localization compared on the SAME lenses, binned by subhalo mass and by perturbation S/N (referee round 21, M5).

The three families' headline localization fractions come from different lens sets (the free-position scan on the
300-lens seed-0 subsample, Family B and the U-Net on the full population) and from different detected mass
distributions. Here every family is scored on the c=60 lenses of that 300-lens subsample which both physical
statistics retain, each at its own 10% calibration, and the localized fraction (reported position within 2 pixels,
0.16") is given per bin of log10 M200 and of log10 S/N_pert, with the number of detections in each bin.

    python scripts/evaluate_localization_matched.py      -> results/localization_matched.json, paper/tables/loc_matched.tex
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from paper_config import B_FULL, unet_seeds  # noqa: E402
from completeness_vs_signal import perturbation_snr  # noqa: E402
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

A_FREE = ROOT / "results/baseline_a_joint_freepos_full"
LOC = 0.16
M_BINS = [(8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5)]
S_BINS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5)]


def thr90(root):
    ns = [json.loads(l) for l in open(root / "no_subhalo/scan_results.jsonl")]
    return float(np.quantile([r["delta_chi2"] for r in ns if r["reliable_fit"]], 0.90))


def physical(root):
    thr = thr90(root)
    recs = {r["index"]: r for r in (json.loads(l) for l in open(root / "test_fixed60/scan_results.jsonl"))}
    kept = {i for i, r in recs.items() if r["reliable_fit"] and r["has_subhalo"]}
    det = {i: (recs[i]["best_x"], recs[i]["best_y"]) for i in kept if recs[i]["delta_chi2"] >= thr}
    return kept, det


def unet(indices):
    run, ck = next(iter(unet_seeds().items()))
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(ROOT / ck / "model_best.pt", map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev); model.load_state_dict(ckpt["model_state"]); model.eval()
    thr = json.loads((ROOT / "results" / run / "results.json").read_text())["threshold_at_10pct_fpr"]
    ds = LensPatchDataset(ROOT / "data/test_fixed60", scale=ckpt["scale"]); c = (ds.num_pix - 1) / 2
    det = {}
    with torch.no_grad():
        for i in indices:
            prob = torch.sigmoid(model(ds[i][0].unsqueeze(0).to(dev)))[0, 0].cpu().numpy()
            if prob.max() >= thr:
                py, px = np.unravel_index(np.argmax(prob), prob.shape)
                det[i] = ((px - c) * ds.pixel_scale, (py - c) * ds.pixel_scale)
    return det, run


def main():
    truths = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    keptA, detA = physical(A_FREE)
    keptB, detB = physical(ROOT / B_FULL)
    common = sorted(keptA & keptB)
    detC, run = unet(common)
    snr = perturbation_snr(ROOT / "data/test_fixed60")
    out = {"n_common": len(common), "unet_run": run, "families": {}}
    for fam, det in (("A", detA), ("B", detB), ("C", detC)):
        ids = [i for i in common if i in det]
        m = np.array([truths[i]["subhalo"]["log10_M200"] for i in ids]); s = np.log10(snr[ids])
        loc = np.array([np.hypot(det[i][0] - truths[i]["subhalo"]["x"], det[i][1] - truths[i]["subhalo"]["y"]) < LOC for i in ids])
        row = {"n_det": len(ids), "frac_localized": float(loc.mean()) if len(ids) else None, "mass": {}, "snr": {}}
        for key, arr, bins in (("mass", m, M_BINS), ("snr", s, S_BINS)):
            for a, b in bins:
                sel = (arr >= a) & (arr < b)
                row[key][f"{a:g}-{b:g}"] = {"n": int(sel.sum()), "frac_localized": float(loc[sel].mean()) if sel.any() else None}
        out["families"][fam] = row
    (ROOT / "results/localization_matched.json").write_text(json.dumps(out, indent=2))

    def cell(d):
        return "--" if d["frac_localized"] is None else f"{100*d['frac_localized']:.0f} ({d['n']})"
    names = {"A": "A (position free)", "B": "B", "C": "C (U-Net)"}
    lines = [r"\begin{tabular}{lrcccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{$\log_{10}(M_{200}/\Msun)$} \\",
             r"family & all & " + " & ".join(f"{a:g}--{b:g}" for a, b in M_BINS) + r" \\", r"\midrule"]
    for f in "ABC":
        r = out["families"][f]
        lines.append(f"{names[f]} & {100*r['frac_localized']:.0f} ({r['n_det']}) & " + " & ".join(cell(r["mass"][f"{a:g}-{b:g}"]) for a, b in M_BINS) + r" \\")
    lines += [r"\midrule", r" & & \multicolumn{4}{c}{$\log_{10}{\rm S/N}_{\rm pert}$} \\",
              r" & & " + " & ".join(f"{a:g}--{b:g}" for a, b in S_BINS) + r" \\", r"\midrule"]
    for f in "ABC":
        r = out["families"][f]
        lines.append(f"{names[f]} & & " + " & ".join(cell(r["snr"][f"{a:g}-{b:g}"]) for a, b in S_BINS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/loc_matched.tex").write_text("\n".join(lines) + "\n")
    for f in "ABC":
        r = out["families"][f]
        print(f, r["n_det"], f"{100*r['frac_localized']:.1f}%", {k: cell(v) for k, v in r["mass"].items()}, {k: cell(v) for k, v in r["snr"].items()})
    print("common lenses:", len(common))


if __name__ == "__main__":
    main()
