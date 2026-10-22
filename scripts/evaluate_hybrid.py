"""The hybrid pipeline of Tsang+2024, measured: a learned flag confirmed by the joint parametric scan on the same lenses.

A lens is a hybrid detection if the learned detector scores it at or above its threshold (10% or 1% FPR on the arm's own
clean control) AND the joint scan gives Delta chi^2 >= 20 (or >= 48, the evidence-equivalent threshold). The scan then
runs only on the flagged lenses, so the flag rate is also the fraction of the scan's cost that is spent.

Detectors: the U-Net trained on 5e5 images and the neural ratio estimator trained on 1e5 (Tier 0); at the combined-realism
operating point the ratio estimator retrained there (results/nre_30k_realistic_seed0).

    python scripts/evaluate_hybrid.py      -> results/hybrid.json, paper/tables/hybrid.tex
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
T0 = {"no_subhalo": "full", "test_fixed60": "full", "test_fixed15": "full", "multipole_m4_a3": "full", "multipole_m4_a1": "full",
      "multipole_m3_a3": "full", "ellmultipole_m4_a3": "ellmp", "decoy_gaussian_s42_a10": "decoy", "decoy_gaussian_s43_a10": "decoy",
      "decoy_dipole_s42_a10": "decoy"}
REAL = {"no_subhalo": "realistic_no_subhalo", "test_fixed60": "realistic_fixed60", "test_fixed15": "realistic_fixed15",
        "multipole_m4_a3": "realistic_multipole_m4_a3"}


def scan(root, pop):
    d = {}
    for l in open(ROOT / root / pop / "scan_results.jsonl"):
        r = json.loads(l)
        if r["reliable_fit"]:
            d[r["index"]] = r
    return d


def scores(kind, pop, data_pop):
    cache = ROOT / "results/hybrid_scores" / f"{kind}_{data_pop}.npy"
    if kind == "nre":
        return np.load(ROOT / f"results/nre_100k_c60_seed0/scores_{pop}.npy")
    if kind == "nre_realistic":
        return np.load(ROOT / f"results/nre_30k_realistic_seed0/scores_{pop}.npy")
    if not cache.exists():
        import evaluate_strong_detectors as ESD
        global _U
        if "_U" not in globals():
            _U = ESD.Det("unet500k")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, _U.score(data_pop)[0])
    return np.load(cache)


def evaluate(kind, pops, scan_root, data_name):
    clean_s = scores(kind, "no_subhalo", data_name("no_subhalo"))
    out = {}
    for fpr in (0.10, 0.01):
        thr = float(np.quantile(clean_s, 1 - fpr))
        for T in (20, 48):
            key = f"net{round(100 * fpr)}_scan{T}"; R = {}
            for pop, sub in pops.items():
                sc = scan(scan_root(sub), pop); ii = np.array(sorted(sc))
                s = scores(kind, pop, data_name(pop))[ii]
                a = np.array([sc[i]["delta_chi2"] >= T for i in ii]); n = s >= thr; h = a & n
                rec = {"n": len(ii), "scan": float(a.mean()), "net": float(n.mean()), "hybrid": float(h.mean())}
                if pop.startswith("test_"):
                    m = np.array([sc[i]["log10_M200_true"] or 0 for i in ii]); hs = np.array([bool(sc[i]["has_subhalo"]) for i in ii])
                    for k, f in (("scan", a), ("net", n), ("hybrid", h)):
                        rec[k + "_bins"] = [float(f[hs & (m >= lo) & (m < hi)].mean()) if (hs & (m >= lo) & (m < hi)).any() else None for lo, hi in BINS]
                R[pop] = rec
            out[key] = R
    return out


def main():
    res = {}
    for kind in ("unet500k", "nre"):
        res[kind] = evaluate(kind, T0, lambda sub: f"results/baseline_a_joint_{sub}", lambda p: p)
    res["nre_realistic"] = evaluate("nre_realistic", REAL, lambda sub: "results/baseline_a_joint_realistic_full", lambda p: REAL[p])
    (ROOT / "results/hybrid.json").write_text(json.dumps(res, indent=2))
    table(res)


def table(res=None):
    res = res or json.loads((ROOT / "results/hybrid.json").read_text())
    pc = lambda v: "--" if v is None else f"{100 * v:.0f}"  # noqa: E731
    pc1 = lambda v: f"{100 * v:.1f}" if v < 0.095 else f"{100 * v:.0f}"  # noqa: E731
    L = [r"\begin{tabular}{@{}llcccccccccc@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{$c=60$ completeness (\%)} & clean & \multicolumn{4}{c}{multipole (\%)} & decoy \\ \cmidrule(lr){3-6}\cmidrule(lr){8-11}",
         r"flag & statistic & $10^{9}$ & $10^{9.5}$ & $10^{10}$ & $10^{10.5}$ & (\%) & $m{=}4$, 0.03 & $m{=}4$, 0.01 & $m{=}3$ & ell. & (\%) \\", r"\midrule"]
    rows = [("U-Net $5\\times10^{5}$, 10\\%", "unet500k", "net10_scan20"), ("ratio est., 10\\%", "nre", "net10_scan20"), ("ratio est., 1\\%", "nre", "net1_scan20")]
    first = True
    for lab, kind, key in rows:
        R = res[kind][key]
        stats = ([("scan, $\\dchi\\ge20$", "scan")] if first else []) + [("network", "net"), ("hybrid", "hybrid")]
        first = False
        for j, (sl, k) in enumerate(stats):
            dec = np.mean([R[p][k] for p in R if p.startswith("decoy")])
            L.append(f"{lab if j == 0 else ''} & {sl} & "
                     + " & ".join(pc(v) for v in R["test_fixed60"][k + "_bins"][2:]) + f" & {pc1(R['no_subhalo'][k])} & "
                     + " & ".join(pc(R[p][k]) for p in ("multipole_m4_a3", "multipole_m4_a1", "multipole_m3_a3", "ellmultipole_m4_a3")) + f" & {pc(dec)} \\\\")
        L.append(r"\addlinespace")
    R = res["nre_realistic"]["net10_scan20"]
    L.append(r"\midrule \multicolumn{12}{@{}l}{\emph{combined realism, ratio estimator retrained there, 10\%}} \\")
    for sl, k in (("scan, $\\dchi\\ge20$", "scan"), ("network", "net"), ("hybrid", "hybrid")):
        L.append(f" & {sl} & " + " & ".join(pc(v) for v in R["test_fixed60"][k + "_bins"][2:]) + f" & {pc1(R['no_subhalo'][k])} & "
                 + pc(R["multipole_m4_a3"][k]) + r" & -- & -- & -- & -- \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/hybrid.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
