"""Score all three families on the SAME lenses (referee round 7, point M2).

Families A and B exclude macro-fit failures (chi2/dof >= 10) from every denominator; the
U-Net has no such exclusion. Those failures concentrate in the top two mass bins, which is
where the cross-family comparison lives, so the three families are not scored on the same
sample and the ordering in the top bin depends on which accounting is used.

This script removes the objection the cheap way: it intersects the lenses A and B both
retain, and scores all three families on that common set. It reports, per 0.5-dex bin and
per family, completeness on

  * the common retained set (the matched comparison used in the paper),
  * each family's own retained set (the old, unmatched accounting), and
  * every subhalo-bearing lens with exclusions counted as misses (conservative).

Family A is scored both at the 10%-FPR threshold and with the null-floored statistic
(Delta chi^2 > 0; see evaluate_null_floor.py), each calibrated on the clean lenses that
survive the same intersection.

    python scripts/evaluate_matched_denominators.py
-> results/matched_denominators.json, paper/tables/matched_denominators.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
POPS = ("no_subhalo", "test_fixed60", "test_fixed15")
A_ROOT, B_ROOT = "results/baseline_a_full", "results/baseline_b_full/fitted"
SEEDS = {"detector_v0_best": "checkpoints/unet_v0", "detector_v1_seed1": "checkpoints/unet_v1_seed1",
         "detector_v2_seed2": "checkpoints/unet_v2_seed2", "detector_v3_seed3": "checkpoints/unet_v3_seed3"}


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def scans(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return {r["index"]: r for r in (json.loads(l) for l in open(p))} if p.exists() else None


def unet_scores(pop):
    """Per-lens max-sigmoid score for every trained seed, plus each seed's own 10%-FPR threshold."""
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    out = {}
    for run, ckdir in SEEDS.items():
        ck_p, res_p = ROOT / ckdir / "model_best.pt", ROOT / "results" / run / "results.json"
        if not ck_p.exists() or not res_p.exists():
            continue
        ck = torch.load(ck_p, map_location=dev)
        model = UNet(in_ch=1, base=ck.get("base", 16)).to(dev)
        model.load_state_dict(ck["model_state"]); model.eval()
        ds = LensPatchDataset(ROOT / "data" / pop, scale=ck["scale"])
        sc = {}
        with torch.no_grad():
            for i in range(len(ds)):
                sc[i] = float(torch.sigmoid(model(ds[i][0].unsqueeze(0).to(dev))).max())
        out[run] = {"scores": sc, "threshold": json.loads(res_p.read_text())["threshold_at_10pct_fpr"]}
    return out


def completeness(hit, truths, idx):
    """hit: index -> bool. Returns per-bin k/n over the subhalo-bearing lenses in `idx`."""
    d = {}
    for lo, hi in BINS:
        sel = [i for i in idx if truths[i].get("subhalo") and lo <= truths[i]["subhalo"]["log10_M200"] < hi]
        k = sum(1 for i in sel if hit.get(i, False))
        p_, l_, h_ = wilson(k, len(sel))
        d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
    return d


def main():
    truths = {p: [json.loads(l) for l in open(ROOT / "data" / p / "truth.jsonl")] for p in POPS}
    A = {p: scans(A_ROOT, p) for p in POPS}
    B = {p: scans(B_ROOT, p) for p in POPS}
    if not all(A.values()) or not all(B.values()):
        print("missing runs"); return

    # ---- the common retained set, per population
    keep = {}
    for p in POPS:
        both = set(A[p]) & set(B[p])
        keep[p] = sorted(i for i in both if A[p][i]["reliable_fit"] and B[p][i]["reliable_fit"])
    res = {"n_total": {p: len(truths[p]) for p in POPS},
           "n_common_retained": {p: len(keep[p]) for p in POPS},
           "frac_retained": {p: len(keep[p]) / len(truths[p]) for p in POPS}, "families": {}}

    # ---- thresholds recalibrated on the clean lenses of the common set
    clean = keep["no_subhalo"]
    thrA = float(np.quantile([A["no_subhalo"][i]["delta_chi2"] for i in clean], 0.90))
    thrB = float(np.quantile([B["no_subhalo"][i]["delta_chi2"] for i in clean], 0.90))
    res["thresholds"] = {"A_10pct": thrA, "A_floor": 0.0, "B_10pct": thrB}
    res["clean_fpr"] = {"A_10pct": float(np.mean([A["no_subhalo"][i]["delta_chi2"] >= thrA for i in clean])),
                        "A_floor": float(np.mean([A["no_subhalo"][i]["delta_chi2"] > 0 for i in clean])),
                        "B_10pct": float(np.mean([B["no_subhalo"][i]["delta_chi2"] >= thrB for i in clean]))}

    U = {p: unet_scores(p) for p in ("test_fixed60", "test_fixed15")}
    res["n_unet_seeds"] = len(U["test_fixed60"])

    for fam in ("A", "A_floor", "B", "C"):
        res["families"][fam] = {}
        for pop in ("test_fixed60", "test_fixed15"):
            if fam == "C":
                per_seed = []
                for run, d in U[pop].items():
                    hit = {i: (s >= d["threshold"]) for i, s in d["scores"].items()}
                    per_seed.append(completeness(hit, truths[pop], keep[pop]))
                mean = {}
                for lo, hi in BINS:
                    k = f"{lo}-{hi}"
                    ks = [c[k]["k"] for c in per_seed]; n = per_seed[0][k]["n"]
                    p_, l_, h_ = wilson(int(round(np.mean(ks))), n)
                    mean[k] = {"n": n, "k": float(np.mean(ks)), "p": p_, "lo": l_, "hi": h_,
                               "seed_sd": float(np.std([c[k]["p"] for c in per_seed]))}
                res["families"][fam][pop] = {"matched": mean}
                # unmatched: every lens, the paper's old U-Net accounting
                allidx = list(range(len(truths[pop])))
                per_seed_all = []
                for run, d in U[pop].items():
                    hit = {i: (s >= d["threshold"]) for i, s in d["scores"].items()}
                    per_seed_all.append(completeness(hit, truths[pop], allidx))
                res["families"][fam][pop]["own"] = {f"{lo}-{hi}": {
                    "n": per_seed_all[0][f"{lo}-{hi}"]["n"],
                    "p": float(np.mean([c[f"{lo}-{hi}"]["p"] for c in per_seed_all]))} for lo, hi in BINS}
                continue
            S = A[pop] if fam.startswith("A") else B[pop]
            thr = thrA if fam == "A" else (0.0 if fam == "A_floor" else thrB)
            strict = fam == "A_floor"
            hit = {i: ((S[i]["delta_chi2"] > thr) if strict else (S[i]["delta_chi2"] >= thr)) for i in S}
            # conservative: a macro-fit failure can never be a detection, so it counts as a miss
            hit_cons = {i: (hit[i] and S[i]["reliable_fit"]) for i in S}
            res["families"][fam][pop] = {
                "matched": completeness(hit, truths[pop], keep[pop]),
                "own": completeness(hit, truths[pop], [i for i in S if S[i]["reliable_fit"]]),
                "conservative": completeness(hit_cons, truths[pop], list(S.keys()))}
    (ROOT / "results/matched_denominators.json").write_text(json.dumps(res, indent=2))

    def fmt(d):
        return f"{100*d['p']:.0f}$^{{+{100*(d['hi']-d['p']):.0f}}}_{{-{100*(d['p']-d['lo']):.0f}}}$"

    # ---- ONE completeness table, replacing the three that carried overlapping numbers
    # (full populations, common-sample, and the accounting comparison were identical for A and B)
    rows3 = [r"\begin{tabular}{@{}lll" + "c" * 6 + "@{}}", r"\toprule",
             r"family & population & sample & " + " & ".join(f"{lo}--{hi}" for lo, hi in BINS) + r" \\", r"\midrule"]
    blocks = [("A", "A scan"), ("A_floor", r"A scan, $\dchi>0$"), ("B", r"B linear $\delta\psi$"), ("C", "C U-Net")]
    for fam, label in blocks:
        first = True
        for pop, cl in (("test_fixed60", "$c{=}60$"), ("test_fixed15", "$c{=}15$")):
            d = res["families"][fam][pop]
            variants = ([("own", "retained"), ("conservative", "excl.\\ as misses")] if fam != "C"
                        else [("own", "all lenses"), ("matched", "A$\\cap$B lenses")])
            for k, (key, vlab) in enumerate(variants):
                if key not in d:
                    continue
                row = d[key]
                cells = " & ".join(fmt(row[f"{lo}-{hi}"]) if isinstance(row[f"{lo}-{hi}"], dict) and "lo" in row[f"{lo}-{hi}"]
                                   else f"{100*row[f'{lo}-{hi}']['p']:.0f}" for lo, hi in BINS)
                rows3.append(f"{label if first else ''} & {cl if k == 0 else ''} & {vlab} & {cells} \\\\")
                first = False
        rows3.append(r"\addlinespace[2pt]")
    rows3 += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/completeness.tex").write_text("\n".join(rows3) + "\n")
    lines = [r"\begin{tabular}{@{}ll" + "c" * 6 + "@{}}", r"\toprule",
             "family & population & " + " & ".join(f"{lo}--{hi}" for lo, hi in BINS) + r" \\", r"\midrule"]
    for fam, label in (("A", "A scan"), ("A_floor", r"A scan, $\dchi>0$"), ("B", r"B linear $\delta\psi$"), ("C", "C U-Net")):
        first = True
        for pop, cl in (("test_fixed60", "$c{=}60$"), ("test_fixed15", "$c{=}15$")):
            row = res["families"][fam][pop]["matched"]
            lines.append(f"{label if first else ''} & {cl} & " + " & ".join(fmt(row[f'{lo}-{hi}']) for lo, hi in BINS) + r" \\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/matched_denominators.tex").write_text("\n".join(lines) + "\n")

    # ---- second table: how the three accountings order the families, per bin
    rows2 = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"bin ($\log_{10}M_{200}$) & accounting & A scan & B linear $\delta\psi$ & C U-Net \\", r"\midrule"]
    for lo, hi in BINS[2:]:
        k = f"{lo}-{hi}"
        first = True
        for acc, lab in (("matched", "same lenses"), ("own", "each family's own"), ("conservative", "exclusions as misses")):
            cells = []
            for fam in ("A", "B", "C"):
                d = res["families"][fam]["test_fixed60"]
                key = acc if acc in d else ("own" if acc == "conservative" else "matched")   # C has no exclusions: own == conservative
                cells.append(f"{100*d[key][k]['p']:.0f}\%")
            rows2.append(f"{('$10^{'+str(lo)+'}$--$10^{'+str(hi)+'}$') if first else ''} & {lab} & " + " & ".join(cells) + r" \\")
            first = False
        rows2.append(r"\addlinespace[2pt]")
    rows2 += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/accounting.tex").write_text("\n".join(rows2) + "\n")

    print("common retained:", {p: f"{len(keep[p])}/{len(truths[p])} ({100*len(keep[p])/len(truths[p]):.0f}%)" for p in POPS})
    print("thresholds:", {k: round(v, 2) for k, v in res["thresholds"].items()}, "clean FPR:", {k: round(v, 4) for k, v in res["clean_fpr"].items()})
    for fam in ("A", "A_floor", "B", "C"):
        for pop in ("test_fixed60", "test_fixed15"):
            m = res["families"][fam][pop]["matched"]
            own = res["families"][fam][pop]["own"]
            print(f"  {fam:8s} {pop:14s} matched " + " ".join(f"{100*m[f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS)
                  + "   own " + " ".join(f"{100*own[f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS))


if __name__ == "__main__":
    main()
