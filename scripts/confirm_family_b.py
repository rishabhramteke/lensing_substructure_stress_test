"""A confirmation stage for Family B (referee round 23): every lens the free-form statistic flags at its 10% calibration
is re-fitted as a parametric perturber seeded at the delta-kappa peak, the practice gravitational imaging uses to confirm
a candidate. The macro-model is re-fitted jointly with a TNFW (c = 15 template, as Family A) whose mass is free and whose
position is free within 0.3" of the peak; the smooth reference is the best of the single-start smooth fit and a smooth
re-fit from the joint macro-model with the perturber removed (Family A's basin control). A candidate is confirmed at
Delta chi^2 >= 20 and, separately, at >= 48 (Delta ln Z > 10 at the median Occam penalty of Sect. 4.3).

    python scripts/confirm_family_b.py --workers 12        -> results/family_b_confirmation.json, <root>/confirmed.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import fit_smooth, _unpack3, BOUNDS, SCALE, BOUNDS_LL, SCALE_LL  # noqa: E402

B = ROOT / "results/baseline_b_full/fitted"
BD = ROOT / "results/baseline_b_decoy/fitted"
POPS = [(B, "no_subhalo"), (B, "test_fixed60"), (B, "test_fixed15"), (B, "multipole_m4_a3"),
        (BD, "decoy_gaussian_s42_a10"), (BD, "decoy_gaussian_s43_a10"), (BD, "decoy_dipole_s42_a10")]
BOX = 0.3
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def one(rec):
    G = J.G; i = rec["index"]; truth = G["truths"][i]; data = np.asarray(G["images"][i])
    vec, chi2_s, _, noise_std = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=np.random.default_rng(i), maxiter=60,
                                          lens_light_components=G["n_ll"])
    bnd = list(BOUNDS) + list(BOUNDS_LL) * G["n_ll"]; scl = list(SCALE) + list(SCALE_LL) * G["n_ll"]
    n_par = len(vec); lo_m = [b[0] for b in bnd][:n_par]; hi_m = [b[1] for b in bnd][:n_par]; sc = scl[:n_par]
    im_s, im_j = G["image_model_smooth"], G["image_model"]

    def resid_s(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    px, py = rec["best_x"], rec["best_y"]

    def resid_j(v):
        kl, ks, kll = _unpack3(v[:n_par])
        return ((data - im_j.image(kwargs_lens=kl + [J.sub_kwargs(v[n_par], v[n_par + 1], v[n_par + 2])], kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    lo = lo_m + [7.5, px - BOX, py - BOX]; hi = hi_m + [11.5, px + BOX, py + BOX]
    x0 = np.clip(np.concatenate([vec, [9.0, px, py]]), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    rj = least_squares(resid_j, x0, bounds=(lo, hi), x_scale=sc + [1.0, 0.1, 0.1], method="trf", max_nfev=40 * len(x0))
    chi2_j = float(np.sum(rj.fun ** 2))
    rb = least_squares(resid_s, np.clip(rj.x[:n_par], lo_m, hi_m), bounds=(lo_m, hi_m), x_scale=sc, method="trf", max_nfev=40 * (n_par + 1))
    chi2_ref = min(chi2_s, float(np.sum(rb.fun ** 2)))
    return {"index": i, "has_subhalo": rec["has_subhalo"], "log10_M200_true": rec.get("log10_M200_true"),
            "chi2_smooth_per_dof": chi2_s / G["num_pix"] ** 2, "reliable_fit": chi2_s / G["num_pix"] ** 2 < 10 or chi2_j / G["num_pix"] ** 2 < 10,
            "delta_chi2_confirm": chi2_ref - chi2_j, "log10_m_fit": float(rj.x[n_par]), "x_fit": float(rj.x[n_par + 1]), "y_fit": float(rj.x[n_par + 2])}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=12)
    p.add_argument("--root", type=Path, help="one root holding every population (e.g. results/baseline_bgi for the gravitational-imaging "
                   "stage 1); default: Family B's full and decoy roots")
    p.add_argument("--min-stat", type=float, help="candidates are the lenses at or above this statistic (e.g. 10 for Delta ln Z) "
                   "instead of the 10%% calibration")
    p.add_argument("--out", default="family_b_confirmation.json")
    p.add_argument("--data-prefix", default="", help="population data directory = data/<prefix><name>, 'test_' dropped from the name when "
                   "a prefix is given (e.g. 'pop_' maps test_fixed60 -> data/pop_fixed60); 'tier1/' keeps the name")
    p.add_argument("--lens-light-components", type=int, default=1, help="lens-light components fitted when the data carry lens light")
    p.add_argument("--pops", nargs="+", help="name=datadir pairs under --root, for arms outside the standard population names "
                   "(e.g. ellmultipole_m4_a3=ellmultipole_m4_a3); requires --min-stat")
    a = p.parse_args()
    import multiprocessing as mp
    pops = [(a.root, pop) for _, pop in POPS] if a.root else POPS
    dmap = dict(x.split("=") for x in a.pops) if a.pops else {}
    if dmap:
        assert a.root and a.min_stat is not None, "--pops needs --root and --min-stat"
        pops = [(a.root, k) for k in dmap]
    B0 = a.root or B
    if a.min_stat is not None:
        thr = a.min_stat
    else:
        ns = [r for r in (json.loads(l) for l in open(B0 / "no_subhalo/scan_results.jsonl")) if r["reliable_fit"]]
        thr = float(np.quantile([r["delta_chi2"] for r in ns], 0.90))
    out = {"b_threshold": thr, "threshold_rule": "fixed" if a.min_stat is not None else "10% calibration", "pops": {}}
    for root, pop in pops:
        f = root / pop / "scan_results.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in open(f)]
        rel = [r for r in recs if r["reliable_fit"]]
        flagged = [r for r in rel if r["delta_chi2"] >= thr]
        done = {}
        cf = root / pop / "confirmed.jsonl"
        if cf.exists():
            done = {json.loads(l)["index"]: json.loads(l) for l in open(cf)}
        todo = [r for r in flagged if r["index"] not in done]
        dpop = dmap[pop] if pop in dmap else pop if not a.data_prefix else (a.data_prefix + pop if a.data_prefix.endswith("/") else a.data_prefix + pop.replace("test_", ""))
        man = json.loads((ROOT / "data" / dpop / "manifest.json").read_text())
        src = json.loads(open(f).readline()) if f.exists() else {}
        J._init(ROOT / "data" / dpop, 15.0, 1e12, True, a.lens_light_components, True, BOX, False, 1, False)
        print(f"{pop}: data/{dpop}", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool, open(cf, "a") as fo:
            for r in pool.imap_unordered(one, todo):
                done[r["index"]] = r; fo.write(json.dumps(r) + "\n"); fo.flush()
        C = [done[r["index"]] for r in flagged]
        d = {"n_retained": len(rel), "n_flagged_b": len(flagged), "flag_rate_b": len(flagged) / max(len(rel), 1)}
        for thr_c in (20, 48):
            d[f"confirmed_{thr_c}"] = sum(c["delta_chi2_confirm"] >= thr_c for c in C) / max(len(rel), 1)
        if any(r["has_subhalo"] for r in rel):
            for thr_c in (None, 20, 48):
                comp = {}
                for lo, hi in BINS:
                    S = [r for r in rel if r["has_subhalo"] and lo <= r["log10_M200_true"] < hi]
                    k = sum(1 for r in S if r["delta_chi2"] >= thr and (thr_c is None or done[r["index"]]["delta_chi2_confirm"] >= thr_c))
                    comp[f"{lo}-{hi}"] = {"n": len(S), "p": k / len(S) if S else None}
                d[f"completeness_{'b' if thr_c is None else f'confirmed_{thr_c}'}"] = comp
        out["pops"][pop] = d
        print(f"{pop:24s} B flags {100*d['flag_rate_b']:.1f}%  confirmed >=20 {100*d['confirmed_20']:.1f}%  >=48 {100*d['confirmed_48']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        (ROOT / "results" / a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
