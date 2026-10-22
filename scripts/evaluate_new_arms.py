"""Family-A summaries of the arms added in referee rounds 21-25, in one file the paper's text, tables and Fig. 3 read.

Each arm is calibrated at 10% FPR on its own clean control and also reported at Delta chi^2 >= 20 (and >= 48, the
evidence-matched threshold of Sect. 4.3, where it matters). Every arm uses regated records (regate_joint.py). Arms whose
runs are missing are skipped, so the script can be re-run as runs land.

    python scripts/evaluate_new_arms.py        -> results/new_arms.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC = 0.16


def load(root, pop):
    f = R / root / pop / "scan_results.jsonl"
    recs = [json.loads(l) for l in open(f)] if (f.exists() and (R / root / pop / "manifest.json").exists()) else None
    rest = R / (root + "_rest") / pop / "scan_results.jsonl"   # the completion of an arm scanned in two passes (same seeds)
    if recs is not None and rest.exists():
        have = {r["index"] for r in recs}
        recs += [r for r in (json.loads(l) for l in open(rest)) if r["index"] not in have]
    return recs


def rel(recs):
    return [r for r in recs if r["reliable_fit"]]


def thr90(recs):
    return float(np.quantile([r["delta_chi2"] for r in rel(recs)], 0.90))


def wilson(k, n, z=1.0):
    if not n:
        return None, None, None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def comp(recs, t, idx=None):
    out = {}
    for lo, hi in BINS:
        S = [r for r in rel(recs) if r["has_subhalo"] and lo <= r["log10_M200_true"] < hi and (idx is None or r["index"] in idx)]
        k = sum(r["delta_chi2"] >= t for r in S); p, l, h = wilson(k, len(S))
        out[f"{lo}-{hi}"] = {"n": len(S), "k": int(k), "p": p, "lo": l, "hi": h}
    return out


def fpr(recs, t):
    rr = rel(recs)
    return {"n": len(rr), "p": float(np.mean([r["delta_chi2"] >= t for r in rr])) if rr else None}


def gated(recs):
    return {"n": len(recs), "gated": sum(not r["reliable_fit"] for r in recs)}


def arm(root, clean="no_subhalo", c60="test_fixed60", c15="test_fixed15", mps=("multipole_m4_a3",), decoys=(), clean_root=None):
    C = load(clean_root or root, clean)
    if C is None:
        return None
    t = thr90(C); out = {"root": root, "threshold": t, "clean": gated(C), "clean_fpr_20": fpr(C, 20)["p"], "clean_fpr_48": fpr(C, 48)["p"]}
    for key, pop in (("c60", c60), ("c15", c15)):
        X = load(root, pop)
        if X:
            out[key] = {"gate": gated(X), "cal": comp(X, t), "abs": comp(X, 20), "abs48": comp(X, 48)}
    for pop in mps + decoys:
        X = load(root, pop)
        if X:
            out[pop] = {"gate": gated(X), "cal": fpr(X, t)["p"], "abs": fpr(X, 20)["p"], "abs48": fpr(X, 48)["p"], "n": fpr(X, t)["n"]}
    return out


def paired_baseline(stress, base_root="baseline_a_joint_full", base_clean="no_subhalo"):
    """The Tier-0 full-population scan restricted to the stress arm's own lenses (same indices), at its own calibration."""
    if not stress or "c60" not in stress:
        return None
    X = load(stress["root"], "test_fixed60"); idx = {r["index"] for r in X}
    B = load(base_root, "test_fixed60"); tb = thr90(load(base_root, base_clean))
    return {"cal": comp(B, tb, idx), "abs": comp(B, 20, idx)}


def freepos_flags(root="baseline_a_joint_freepos_full", box=0.5):
    """Position-convergence flag as a classifier: distance of the fitted position from its starting node, and whether it
    sits on the box edge or at the mass bound, for real detections versus multipole false alarms."""
    C = load(root, "no_subhalo")
    if C is None:
        return None
    t = thr90(C); out = {}
    for lab, pop, want_sub in (("real", "test_fixed60", True), ("multipole", "multipole_m4_a3", False)):
        X = load(root, pop)
        if X is None:
            continue
        det = [r for r in rel(X) if r["delta_chi2"] >= 20 and r["has_subhalo"] == want_sub]
        d, edge, mbound, centre = [], 0, 0, 0
        for r in det:
            b = min(r["cells"], key=lambda c: c["chi2_joint"])
            dd = max(abs(b["x"] - b["node_x"]), abs(b["y"] - b["node_y"])); d.append(np.hypot(b["x"] - b["node_x"], b["y"] - b["node_y"]))
            edge += dd > box - 0.02; mbound += b["log10_m"] > 11.49; centre += np.hypot(b["x"], b["y"]) < 0.2
        out[lab] = {"n": len(det), "median_node_offset": float(np.median(d)) if d else None,
                    "frac_box_edge": edge / max(len(det), 1), "frac_mass_bound": mbound / max(len(det), 1), "frac_centre": centre / max(len(det), 1),
                    "frac_flagged": sum(1 for r in det if (lambda b: max(abs(b["x"] - b["node_x"]), abs(b["y"] - b["node_y"])) > box - 0.02 or b["log10_m"] > 11.49)(min(r["cells"], key=lambda c: c["chi2_joint"]))) / max(len(det), 1)}
        if want_sub:
            tr = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
            loc = [np.hypot(r["best_x"] - tr[r["index"]]["subhalo"]["x"], r["best_y"] - tr[r["index"]]["subhalo"]["y"]) < LOC for r in det]
            out[lab]["frac_localized"] = float(np.mean(loc)) if loc else None
    return out


def cap_agreement(root_hi, n_label):
    out = {}
    for pop in ("no_subhalo", "test_fixed60"):
        H = load(root_hi, pop)
        if H is None:
            continue
        L = {r["index"]: r for r in load("baseline_a_joint_full", pop)}
        pairs = [(r, L[r["index"]]) for r in H if r["index"] in L and r["reliable_fit"] and L[r["index"]]["reliable_fit"]]
        d = np.array([a["delta_chi2"] - b["delta_chi2"] for a, b in pairs])
        out[pop] = {"n": len(pairs), "median_diff": float(np.median(d)), "frac_absdiff_gt1": float(np.mean(np.abs(d) > 1)),
                    "decision_agree_20": float(np.mean([(a["delta_chi2"] >= 20) == (b["delta_chi2"] >= 20) for a, b in pairs])),
                    "gated": gated(H)["gated"], "gated_base": sum(not L[r["index"]]["reliable_fit"] for r in H)}
    return out or None


def main():
    A = {}
    A["realistic_full"] = arm("baseline_a_joint_realistic_full", decoys=("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"))
    A["realistic_300"] = arm("baseline_a_joint_realistic")
    A["freepos_full"] = arm("baseline_a_joint_freepos_full")
    A["invcrime"] = arm("baseline_a_joint_invcrime", c15="none")
    A["clumpy"] = arm("baseline_a_joint_clumpy", c15="none")
    A["tier1"] = arm("baseline_a_joint_tier1", c15="none", mps=())
    A["mpsub_bare"] = arm("baseline_a_joint_mpsub", c15="none", mps=())
    A["mpsub_mp4"] = arm("baseline_a_joint_mpsub_mp4", c15="none", mps=())
    A["mmax105"] = arm("baseline_a_joint_mmax105", c15="none")
    A["super"] = arm("baseline_a_joint_super", c15="none")
    A["box1"] = arm("baseline_a_joint_box1", c15="none")
    # referee round 26: sub-threshold population (M7), misspecification at 600 s and the PSF error alone (M2, M8), free centre (M5)
    A["pop"] = arm("baseline_a_joint_pop", c15="none", mps=())
    A["clumpy600"] = arm("baseline_a_joint_clumpy600", c15="none")
    A["super600"] = arm("baseline_a_joint_super600", c15="none")
    A["psferr"] = arm("baseline_a_joint_psferr", c15="none")
    A["freecentre"] = arm("baseline_a_joint_freecentre", c15="none")
    A["shallow600_ref"] = arm("baseline_a_joint_shallow600", c15="none", mps=())
    A["tmpl60"] = arm("baseline_a_joint_tmpl60", mps=())
    A["tmplcm"] = arm("baseline_a_joint_tmplcm", mps=())
    E = load("baseline_a_joint_ellmp", "ellmultipole_m4_a3"); F = load("baseline_a_joint_mp4", "ellmultipole_m4_a3")
    if E and F:
        tb = thr90(load("baseline_a_joint_full", "no_subhalo")); tm = thr90(load("baseline_a_joint_mp4", "no_subhalo"))
        A["ellmp"] = {"bare": {"cal": fpr(E, tb)["p"], "abs": fpr(E, 20)["p"], "n": fpr(E, tb)["n"]},
                      "free_circular_m4": {"cal": fpr(F, tm)["p"], "abs": fpr(F, 20)["p"], "n": fpr(F, tm)["n"]}}
    A["paired_baselines"] = {k: paired_baseline(A[k]) for k in ("realistic_full", "realistic_300", "invcrime", "clumpy", "mmax105", "box1", "super", "pop", "freecentre") if A.get(k)}
    for k in ("clumpy600", "super600"):
        if A.get(k):
            A["paired_baselines"][k + "_vs_600s"] = paired_baseline(A[k], base_root="baseline_a_joint_shallow600")
    A["freecentre_agreement"] = cap_agreement("baseline_a_joint_freecentre", 0)
    for k, (root, clean) in {"psf015": ("baseline_a_joint_psf015", "no_subhalo"), "shallow": ("baseline_a_joint_shallow", "no_subhalo"),
                             "shallow600": ("baseline_a_joint_shallow600", "no_subhalo"), "lenslight2": ("baseline_a_joint_lenslight2", "lenslight_no_subhalo")}.items():
        c60 = "test_fixed60" if k != "lenslight2" else "lenslight_fixed60"
        X = load(root, c60)
        if X:
            idx = {r["index"] for r in X}; B = load("baseline_a_joint_full", "test_fixed60"); tb = thr90(load("baseline_a_joint_full", "no_subhalo"))
            tX = thr90(load(root, clean))
            A["paired_baselines"][k] = {"arm_cal": comp(X, tX), "base_cal": comp(B, tb, idx), "arm_abs": comp(X, 20), "base_abs": comp(B, 20, idx)}
    A["freepos_flags"] = freepos_flags()
    A["freepos_flags_box1"] = freepos_flags("baseline_a_joint_box1", box=1.0)
    A["cap80"] = cap_agreement("baseline_a_joint_cap80", 80)
    A["cap200"] = cap_agreement("baseline_a_joint_cap200", 200)
    BC = load("baseline_a_joint_bigclean", "no_subhalo")
    if BC:
        allc = rel(BC) + rel(load("baseline_a_joint_full", "no_subhalo"))
        t01 = float(np.quantile([r["delta_chi2"] for r in allc], 0.999)); c60 = load("baseline_a_joint_full", "test_fixed60")
        A["fpr_0p1"] = {"n_clean": len(allc), "threshold": t01, "completeness_c60": comp(c60, t01),
                        "fpr_multipole_m4_a3": fpr(load("baseline_a_joint_full", "multipole_m4_a3"), t01)["p"],
                        "clean_fpr_20": float(np.mean([r["delta_chi2"] >= 20 for r in allc]))}
    (R / "new_arms.json").write_text(json.dumps(A, indent=2, default=float))
    f = lambda c: " ".join("--" if v["p"] is None else f"{100*v['p']:.0f}" for v in c.values())  # noqa: E731
    for k, v in A.items():
        if not v or k in ("paired_baselines",):
            continue
        if isinstance(v, dict) and "threshold" in v and "clean" in v:
            s = f"{k:16s} thr {v['threshold']:.1f} clean gated {v['clean']['gated']}/{v['clean']['n']} FPR20 {100*v['clean_fpr_20']:.1f}%"
            if "c60" in v:
                s += f" | c60 cal {f(v['c60']['cal'])} >=20 {f(v['c60']['abs'])}"
            if "c15" in v:
                s += f" | c15 cal {f(v['c15']['cal'])}"
            for pop in ("multipole_m4_a3", "decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"):
                if pop in v:
                    s += f" | {pop[:10]} {100*v[pop]['cal']:.0f}/{100*v[pop]['abs']:.0f}%"
            print(s)
        else:
            print(k, json.dumps(v, default=float)[:400])
    for k, v in A["paired_baselines"].items():
        if v and "arm_cal" in v:
            print(f"paired {k:10s} arm cal {f(v['arm_cal'])} | base cal {f(v['base_cal'])} | arm >=20 {f(v['arm_abs'])} | base >=20 {f(v['base_abs'])}")
        elif v:
            print(f"paired {k:10s} base-on-same-lenses cal {f(v['cal'])} >=20 {f(v['abs'])}")


if __name__ == "__main__":
    main()
