"""The joint scan at a survey-like 0.1% false-positive rate (referee round 26, M6): the threshold is the 99.9th
percentile of Delta chi^2 over the large clean control (the 10 000-lens clean population scanned in part, plus the
full 1 000-lens control), with a bootstrap over clean lenses for the threshold and the c=60 completeness it implies,
under the standard gate (chi^2/dof < 10) and a strict one (chi^2/dof < 2, applied to clean and subhalo lenses alike).

    python scripts/evaluate_fpr_0p1.py      -> results/fpr_0p1.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
NB = 2000


def load(root, pop):
    p = ROOT / "results" / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def gate_dof(r):
    """The gate statistic of the scan's own rule: the better of the smooth fit and the best joint fit, per pixel."""
    npix = 64 * 64
    j = min((c["chi2_joint"] for c in r.get("cells", [])), default=np.inf) / npix
    return min(r["chi2_smooth_per_dof"], j)


def big_clean():
    """The 10 000-lens clean population: the single-start scans, with each lens re-fitted by the seven-start rescue
    (results/baseline_a_joint_bigclean_rescue) replaced by its rescued record when that fits better."""
    big = {r["index"]: r for r in load("baseline_a_joint_bigclean", "no_subhalo") + load("baseline_a_joint_bigclean_rest", "no_subhalo")}
    for r in load("baseline_a_joint_bigclean_rescue", "no_subhalo"):
        if r["index"] not in big or gate_dof(r) < gate_dof(big[r["index"]]):
            big[r["index"]] = dict(r, rescued_record=True)
    return list(big.values())


def main():
    clean = big_clean() + load("baseline_a_joint_full", "no_subhalo")
    c60 = load("baseline_a_joint_full", "test_fixed60")
    rng = np.random.default_rng(0)
    out = {}
    for gate, gmax in (("standard", 10.0), ("strict", 2.0)):
        C = np.array([r["delta_chi2"] for r in clean if r["reliable_fit"] and gate_dof(r) < gmax])
        S = [r for r in c60 if r["reliable_fit"] and r["has_subhalo"] and gate_dof(r) < gmax]
        t = float(np.quantile(C, 0.999))
        comp = lambda thr: [float(np.mean([r["delta_chi2"] >= thr for r in S if a <= r["log10_M200_true"] < b])) for a, b in BINS]  # noqa: E731
        boot_t = np.array([np.quantile(rng.choice(C, len(C)), 0.999) for _ in range(NB)])
        boot_c = np.array([comp(x) for x in boot_t[:400]])
        out[gate] = {"n_clean": int(len(C)), "n_above_threshold": int((C > t).sum()), "threshold": t,
                     "threshold_68": [float(np.percentile(boot_t, 16)), float(np.percentile(boot_t, 84))],
                     "n_subhalo": len(S), "completeness": comp(t),
                     "completeness_68_lo": np.percentile(boot_c, 16, axis=0).tolist(), "completeness_68_hi": np.percentile(boot_c, 84, axis=0).tolist(),
                     "clean_chi2dof_of_tail": sorted(float(gate_dof(r)) for r in clean if r["reliable_fit"] and gate_dof(r) < gmax and r["delta_chi2"] > t),
                     "n_subhalo_all_retained": sum(1 for r in c60 if r["reliable_fit"] and r["has_subhalo"])}
    (ROOT / "results/fpr_0p1.json").write_text(json.dumps(out, indent=2))
    for g, d in out.items():
        print(g, "n_sub", d["n_subhalo"], "of", d["n_subhalo_all_retained"], "n_clean", d["n_clean"], "thr", round(d["threshold"], 1), "68%", [round(x, 1) for x in d["threshold_68"]], "tail n", d["n_above_threshold"],
              "comp", [round(100 * x) for x in d["completeness"]], "lo", [round(100 * x) for x in d["completeness_68_lo"]], "hi", [round(100 * x) for x in d["completeness_68_hi"]],
              "tail chi2/dof", [round(x, 2) for x in d["clean_chi2dof_of_tail"]])


if __name__ == "__main__":
    main()
