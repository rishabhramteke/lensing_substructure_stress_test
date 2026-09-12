"""Referee round 6, point 2: an intermediate, tidally-plausible concentration.

Dutton & Maccio (2014) is a *field*-halo relation; surviving subhalos are tidally stripped and
denser at fixed M200, by factors of ~2-3 near the host centre (Moline et al. 2017). So the
literature's c=15 ablation is not "the LCDM value for a subhalo" -- c~20-40 is. This script scores
the matched c=30 population (same seed as test_fixed60/test_fixed15: identical lens, source,
subhalo mass and position, concentration the only difference) for Family A and the U-Net, at each
family's existing threshold, and reports the three-point trend and the paired flips.

    python scripts/evaluate_intermediate_concentration.py
-> results/intermediate_concentration.json, paper/tables/c30.tex
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
POPS = ("test_fixed60", "test_fixed30", "test_fixed15")
SEEDS = {"detector_v0_best": "checkpoints/unet_v0", "detector_v1_seed1": "checkpoints/unet_v1_seed1",
         "detector_v2_seed2": "checkpoints/unet_v2_seed2", "detector_v3_seed3": "checkpoints/unet_v3_seed3"}


def wilson(k, n, z=1.0):
    if not n: return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def scan(pop, root="results/baseline_a"):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def family_a():
    clean = [r for r in scan("no_subhalo") if r["reliable_fit"]]
    thr = float(np.quantile([r["delta_chi2"] for r in clean], 0.90))
    out = {"threshold": thr, "clean_fpr_floor": float(np.mean([r["delta_chi2"] > 0 for r in clean])), "populations": {}}
    det = {}
    for pop in POPS:
        recs = scan(pop)
        if not recs: continue
        rel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
        d = {"n_excluded": sum(1 for r in recs if not r["reliable_fit"]), "n": len(rel), "bins": {}, "bins_floor": {}}
        for lo, hi in BINS:
            sel = [r for r in rel if lo <= r["log10_M200_true"] < hi]
            for key, f in (("bins", lambda r: r["delta_chi2"] >= thr), ("bins_floor", lambda r: r["delta_chi2"] > 0)):
                k = sum(1 for r in sel if f(r)); p_, l_, h_ = wilson(k, len(sel))
                d[key][f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
        out["populations"][pop] = d
        det[pop] = {r["index"]: (r["delta_chi2"] >= thr, r["delta_chi2"] > 0) for r in rel}
    out["flips"] = {}
    for hi_pop, lo_pop in (("test_fixed60", "test_fixed30"), ("test_fixed30", "test_fixed15"), ("test_fixed60", "test_fixed15")):
        if hi_pop not in det or lo_pop not in det: continue
        common = sorted(set(det[hi_pop]) & set(det[lo_pop]))
        for j, key in ((0, "at_threshold"), (1, "floored")):
            lost = sum(1 for i in common if det[hi_pop][i][j] and not det[lo_pop][i][j])
            gained = sum(1 for i in common if det[lo_pop][i][j] and not det[hi_pop][i][j])
            out["flips"].setdefault(f"{hi_pop}_to_{lo_pop}", {})[key] = {"n_pairs": len(common), "lost": lost, "gained": gained}
    return out


def family_c():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    per_seed = []
    for run, ckdir in SEEDS.items():
        ckpt = ROOT / ckdir / "model_best.pt"
        res = ROOT / "results" / run / "results.json"
        if not ckpt.exists() or not res.exists():
            continue
        ck = torch.load(ckpt, map_location=dev)
        model = UNet(in_ch=1, base=ck.get("base", 16)).to(dev); model.load_state_dict(ck["model_state"]); model.eval()
        thr = json.loads(res.read_text())["threshold_at_10pct_fpr"]
        row = {"run": run, "threshold": thr, "populations": {}, "scores": {}}
        for pop in POPS:
            ds = LensPatchDataset(ROOT / "data" / pop, scale=ck["scale"])
            sc, masses, idxs = [], [], []
            with torch.no_grad():
                for i in range(len(ds)):
                    if not ds.truths[i].get("subhalo"): continue
                    img = ds[i][0]
                    sc.append(float(torch.sigmoid(model(img.unsqueeze(0).to(dev))).max()))
                    masses.append(ds.truths[i]["subhalo"]["log10_M200"]); idxs.append(i)
            sc = np.array(sc); masses = np.array(masses)
            row["scores"][pop] = {int(i): bool(s >= thr) for i, s in zip(idxs, sc)}
            b = {}
            for lo, hi in BINS:
                sel = (masses >= lo) & (masses < hi); k = int((sc[sel] >= thr).sum())
                p_, l_, h_ = wilson(k, int(sel.sum())); b[f"{lo}-{hi}"] = {"n": int(sel.sum()), "k": k, "p": p_, "lo": l_, "hi": h_}
            row["populations"][pop] = {"n": len(sc), "bins": b}
        per_seed.append(row)
    out = {"n_seeds": len(per_seed), "per_seed": [{k: v for k, v in r.items() if k != "scores"} for r in per_seed], "bins_mean": {}, "flips": {}}
    for lo, hi in BINS:
        key = f"{lo}-{hi}"
        out["bins_mean"][key] = {pop: {"mean": float(np.mean([r["populations"][pop]["bins"][key]["p"] for r in per_seed])),
                                       "std": float(np.std([r["populations"][pop]["bins"][key]["p"] for r in per_seed]))}
                                 for pop in POPS if all(pop in r["populations"] for r in per_seed)}
    for hi_pop, lo_pop in (("test_fixed60", "test_fixed30"), ("test_fixed30", "test_fixed15"), ("test_fixed60", "test_fixed15")):
        lost, gained = [], []
        for r in per_seed:
            if hi_pop not in r["scores"] or lo_pop not in r["scores"]: continue
            common = sorted(set(r["scores"][hi_pop]) & set(r["scores"][lo_pop]))
            lost.append(sum(1 for i in common if r["scores"][hi_pop][i] and not r["scores"][lo_pop][i]))
            gained.append(sum(1 for i in common if r["scores"][lo_pop][i] and not r["scores"][hi_pop][i]))
        if lost:
            out["flips"][f"{hi_pop}_to_{lo_pop}"] = {"lost_mean": float(np.mean(lost)), "gained_mean": float(np.mean(gained)),
                                                     "lost": lost, "gained": gained}
    return out


def main():
    res = {"A": family_a(), "C": family_c()}
    (ROOT / "results/intermediate_concentration.json").write_text(json.dumps(res, indent=2))
    A, C = res["A"], res["C"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"completeness at 10\% FPR & $c=60$ & $c=30$ & $c=15$ \\", r"\midrule",
             r"\multicolumn{4}{@{}l}{Family A, parametric scan} \\"]
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " +
                     " & ".join(f"{100*A['populations'][p]['bins'][k]['p']:.0f}\\%" for p in POPS) + r" \\")
    f = A["flips"]
    lines.append(r"\quad paired flip vs.\ $c{=}60$ & -- & " +
                 f"{f['test_fixed60_to_test_fixed30']['at_threshold']['lost']}\\,:\\,{f['test_fixed60_to_test_fixed30']['at_threshold']['gained']} & " +
                 f"{f['test_fixed60_to_test_fixed15']['at_threshold']['lost']}\\,:\\,{f['test_fixed60_to_test_fixed15']['at_threshold']['gained']}" + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{4}{@{}l}{Family C, U-Net (mean of 4 seeds)} \\")
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " +
                     " & ".join(f"{100*C['bins_mean'][k][p]['mean']:.0f}\\%" for p in POPS) + r" \\")
    cf = C["flips"]
    lines.append(r"\quad paired flip vs.\ $c{=}60$ & -- & " +
                 f"{cf['test_fixed60_to_test_fixed30']['lost_mean']:.0f}\\,:\\,{cf['test_fixed60_to_test_fixed30']['gained_mean']:.0f} & " +
                 f"{cf['test_fixed60_to_test_fixed15']['lost_mean']:.0f}\\,:\\,{cf['test_fixed60_to_test_fixed15']['gained_mean']:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/c30.tex").write_text("\n".join(lines) + "\n")
    print("A thr %.1f | excluded %s" % (A["threshold"], {p: A["populations"][p]["n_excluded"] for p in POPS}))
    for p in POPS:
        print("  A", p, " ".join(f"{100*A['populations'][p]['bins'][f'{lo}-{hi}']['p']:.0f}" for lo, hi in BINS),
              "| floored", " ".join(f"{100*A['populations'][p]['bins_floor'][f'{lo}-{hi}']['p']:.0f}" for lo, hi in BINS))
    print("  A flips:", json.dumps(A["flips"]))
    print("C seeds", C["n_seeds"])
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        print("  C", k, {p: f"{100*C['bins_mean'][k][p]['mean']:.0f}±{100*C['bins_mean'][k][p]['std']:.0f}" for p in POPS})
    print("  C flips:", json.dumps(C["flips"]))


if __name__ == "__main__":
    main()
