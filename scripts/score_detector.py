"""Score any detector on the released suite from its per-image outputs, with the paper's metric and controls.

Write one file per population into a directory, aligned with the population's image index:
    scores_<population>.npy      (n,)    the image-level statistic, larger = more subhalo-like
    positions_<population>.npy   (n, 2)  optional, the reported (x, y) in arcsec, for localization
for at least no_subhalo and test_fixed60 (data/<population>/), and any of test_fixed15, test_fixed30, the multipole
populations, the decoys and the line-of-sight halos. The script reports, at 10% and 1% false-positive rate on the clean
control: completeness per 0.5-dex mass bin, the paired concentration flip c=60 -> c=15 (lost : gained, the same lenses),
the flag rate of every confounder, the AUC of c=60 against the clean control, the fraction of detections within 2 pixels
of the subhalo, and the hybrid with the released joint scan (the detector's flag confirmed at Delta chi^2 >= 20).

    python scripts/score_detector.py --scores results/nre_100k_c60_seed0 --out results/score_nre.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
CONFOUNDERS = ["multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3", "ellmultipole_m4_a3", "decoy_gaussian_s42_a10",
               "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10", "los_halo_z025", "los_halo_z075"]
SCAN = {"no_subhalo": "full", "test_fixed60": "full", "test_fixed15": "full", "multipole_m4_a1": "full", "multipole_m4_a3": "full",
        "multipole_m3_a3": "full", "ellmultipole_m4_a3": "ellmp", "decoy_gaussian_s42_a10": "decoy", "decoy_gaussian_s43_a10": "decoy",
        "decoy_dipole_s42_a10": "decoy"}
LOC_RADIUS = 2 * 0.08   # 2 pixels at the Tier-0 pixel scale


def truths(pop):
    return [json.loads(l) for l in open(ROOT / "data" / pop / "truth.jsonl")]


def scan_flags(pop, n):
    """Joint-scan decision at Delta chi^2 >= 20 per image index (None where the scan gates the lens)."""
    f = ROOT / f"results/baseline_a_joint_{SCAN.get(pop, '')}" / pop / "scan_results.jsonl"
    if pop not in SCAN or not f.exists():
        return None
    out = np.full(n, np.nan)
    for l in open(f):
        r = json.loads(l)
        if r["reliable_fit"] and r["index"] < n:
            out[r["index"]] = r["delta_chi2"] >= 20
    return out


def auc(pos, neg):
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True, help="directory of scores_<population>.npy (and positions_<population>.npy)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    load = lambda kind, pop: np.load(a.scores / f"{kind}_{pop}.npy") if (a.scores / f"{kind}_{pop}.npy").exists() else None  # noqa: E731
    clean = load("scores", "no_subhalo")
    if clean is None or load("scores", "test_fixed60") is None:
        raise SystemExit("need scores_no_subhalo.npy and scores_test_fixed60.npy")
    res = {"n_clean": len(clean), "auc_c60": None, "operating_points": {}}
    T60 = truths("test_fixed60"); s60 = load("scores", "test_fixed60")
    sub = np.array([t.get("subhalo") is not None for t in T60])
    res["auc_c60"] = auc(s60[sub], clean)
    clean_scan = scan_flags("no_subhalo", len(clean))
    for fpr in (0.10, 0.01):
        thr = float(np.quantile(clean, 1 - fpr)); R = {"threshold": thr, "realized_fpr": float(np.mean(clean >= thr)), "completeness": {}, "confounders": {}, "hybrid": {}}
        for pop in ("test_fixed60", "test_fixed30", "test_fixed15"):
            s = load("scores", pop)
            if s is None:
                continue
            T = truths(pop); m = np.array([(t["subhalo"] or {}).get("log10_M200", 0) if t.get("subhalo") else 0 for t in T]); has = m > 0
            det = s >= thr
            R["completeness"][pop] = {f"{lo}-{hi}": {"n": int((has & (m >= lo) & (m < hi)).sum()), "p": float(det[has & (m >= lo) & (m < hi)].mean())}
                                      for lo, hi in BINS if (has & (m >= lo) & (m < hi)).any()}
            pos = load("positions", pop)
            if pos is not None:
                xy = np.array([[t["subhalo"]["x"], t["subhalo"]["y"]] if t.get("subhalo") else [np.nan, np.nan] for t in T])
                ok = det & has
                R["completeness"][pop]["localized_fraction"] = float(np.mean(np.hypot(*(pos[ok] - xy[ok]).T) <= LOC_RADIUS)) if ok.any() else None
            sc = scan_flags(pop, len(s))
            if sc is not None:
                keep = ~np.isnan(sc); h = det & (sc == 1)
                R["hybrid"][pop] = {f"{lo}-{hi}": float(h[keep & has & (m >= lo) & (m < hi)].mean()) for lo, hi in BINS if (keep & has & (m >= lo) & (m < hi)).any()}
        s15 = load("scores", "test_fixed15")
        if s15 is not None:
            d60, d15 = s60[sub] >= thr, s15[sub] >= thr
            R["paired_flip_c60_to_c15"] = {"lost": int((d60 & ~d15).sum()), "gained": int((d15 & ~d60).sum()), "n_pairs": int(sub.sum())}
        for pop in CONFOUNDERS:
            s = load("scores", pop)
            if s is None:
                continue
            R["confounders"][pop] = float(np.mean(s >= thr))
            sc = scan_flags(pop, len(s))
            if sc is not None:
                keep = ~np.isnan(sc); R["hybrid"][pop] = float(np.mean(((s >= thr) & (sc == 1))[keep]))
        if clean_scan is not None:
            keep = ~np.isnan(clean_scan); R["hybrid"]["no_subhalo"] = float(np.mean(((clean >= thr) & (clean_scan == 1))[keep]))
        res["operating_points"][f"fpr_{round(100 * fpr)}"] = R
    a.out.write_text(json.dumps(res, indent=2))
    R = res["operating_points"]["fpr_10"]
    print(f"AUC c=60 vs clean {res['auc_c60']:.3f}")
    for pop, c in R["completeness"].items():
        print(pop, " ".join(f"{k}:{100 * v['p']:.0f}" for k, v in c.items() if isinstance(v, dict)), f"localized {c.get('localized_fraction')}")
    print("flip", R.get("paired_flip_c60_to_c15")); print("confounders", {k: round(100 * v, 1) for k, v in R["confounders"].items()})


if __name__ == "__main__":
    main()
