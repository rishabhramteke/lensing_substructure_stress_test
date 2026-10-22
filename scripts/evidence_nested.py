"""Nested-sampling evidence ratio for the joint scan with the perturber position free, a check of the Laplace
approximation of evidence_laplace.py (and so of the Delta chi^2 >= 48 <-> Delta ln Z > 10 mapping of Sect. 4.3).

For a subsample of lenses of the free-position runs (results/baseline_a_joint_freepos/<pop>), the evidence of the smooth
model and of the smooth-plus-perturber model is computed with Nautilus under the same priors as the Laplace estimate:
for the perturber log10 M200 uniform on [7.5, 11.5] and a position uniform over the explored annulus (the placement
annulus widened by the 0.5" position box, so every mode of the perturber's position is integrated, not only the
scan's); for the macro-model a uniform box shared by the two models, centred between their optima and 8 Fisher
standard deviations of the smooth fit wider than the gap between them, so that its volume cancels in the ratio. The
likelihood is the scan's, -chi^2/2 with the noise the simulator estimates.

    python scripts/evidence_nested.py --workers 4 --per-pop 10      -> results/evidence_nested.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import _sim, _unpack3, BOUNDS, SCALE  # noqa: E402

ROOTDIR = ROOT / "results/baseline_a_joint_freepos"
POPS = ("no_subhalo", "test_fixed60", "multipole_m4_a3")
BOX, LOGM, NSIG = 0.5, (7.5, 11.5), 8.0


def one(args):
    import nautilus
    pop, rec, n_live = args
    G = J.G
    i = rec["index"]; data = np.asarray(G["images"][i]); theta_E = G["truths"][i]["lens_macro"]["theta_E"]
    noise_std = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"]).estimate_noise(data)
    n_par = rec["n_params_macro"]
    lo_m = np.array([b[0] for b in BOUNDS][:n_par]); hi_m = np.array([b[1] for b in BOUNDS][:n_par]); sc = SCALE[:n_par]
    im_s, im_j = G["image_model_smooth"], G["image_model"]

    def resid_s(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    def resid_j(v):
        kl, ks, kll = _unpack3(v[:n_par])
        return ((data - im_j.image(kwargs_lens=kl + [J.sub_kwargs(v[n_par], v[n_par + 1], v[n_par + 2])], kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    t0 = time.time()
    best = min(rec["cells"], key=lambda c: c["chi2_joint"])
    rs = least_squares(resid_s, np.clip(np.array(best["vec_macro"]), lo_m, hi_m), bounds=(lo_m, hi_m), x_scale=sc, method="trf", max_nfev=200 * (n_par + 1))
    xs = np.clip(rs.x, lo_m + 1e-6, hi_m - 1e-6)
    xj = np.array(best["vec_macro"])
    Jm = approx_derivative(resid_s, xs, method="2-point", abs_step=1e-4 * np.asarray(sc), bounds=(lo_m, hi_m))
    sig = np.sqrt(np.clip(np.diag(np.linalg.pinv(Jm.T @ Jm)), 1e-20, None))
    c, half = 0.5 * (xs + xj), 0.5 * np.abs(xs - xj) + NSIG * sig
    blo, bhi = np.maximum(c - half, lo_m), np.minimum(c + half, hi_m)
    r_in, r_out = max(0.0, 0.5 * theta_E - BOX), 1.4 * theta_E + BOX

    def prior_s(u):
        return blo + u * (bhi - blo)

    def prior_j(u):
        m = LOGM[0] + u[n_par] * (LOGM[1] - LOGM[0])
        r = np.sqrt(r_in ** 2 + u[n_par + 1] * (r_out ** 2 - r_in ** 2)); ph = 2 * np.pi * u[n_par + 2]
        return np.concatenate([prior_s(u[:n_par]), [m, r * np.cos(ph), r * np.sin(ph)]])

    def lnl_s(x):
        r = resid_s(x); return -0.5 * float(r @ r)

    def lnl_j(x):
        r = resid_j(x); return -0.5 * float(r @ r)

    out = {"population": pop, "index": i, "has_subhalo": rec["has_subhalo"], "log10_M200_true": rec.get("log10_M200_true"),
           "delta_chi2_scan": rec["delta_chi2"], "n_live": n_live}
    for name, pr, ll, nd in (("smooth", prior_s, lnl_s, n_par), ("joint", prior_j, lnl_j, n_par + 3)):
        s = nautilus.Sampler(pr, ll, n_dim=nd, n_live=n_live, seed=i, pass_dict=False)
        s.run(f_live=0.01, n_eff=2000, verbose=False)
        out[f"lnZ_{name}"] = float(s.log_z); out[f"n_like_{name}"] = int(s.n_like)
        _, lw, lls = s.posterior()
        out[f"max_lnL_{name}"] = float(np.max(lls))
    out["delta_lnZ"] = out["lnZ_joint"] - out["lnZ_smooth"]
    out["delta_chi2_ns"] = 2 * (out["max_lnL_joint"] - out["max_lnL_smooth"])
    out["occam"] = 0.5 * out["delta_chi2_ns"] - out["delta_lnZ"]
    out["t_s"] = time.time() - t0
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=4); p.add_argument("--per-pop", type=int, default=10)
    p.add_argument("--n-live", type=int, default=1000); p.add_argument("--only", nargs="*", help="pop:index entries (tests)")
    a = p.parse_args()
    import multiprocessing as mp
    lap = {(r["population"], r["index"]): r for r in json.loads((ROOT / "results/evidence_laplace.json").read_text())["records"]}
    outf = ROOT / "results/evidence_nested.jsonl"
    done = {(r["population"], r["index"]) for r in map(json.loads, open(outf))} if outf.exists() else set()
    for pop in POPS:
        d = ROOTDIR / pop
        man = json.loads((d / "manifest.json").read_text())
        rel = [r for r in map(json.loads, open(d / "scan_results.jsonl")) if r["reliable_fit"]]
        if a.only:
            todo = [r for r in rel if f"{pop}:{r['index']}" in a.only]
        elif pop == "no_subhalo":  # the clean lenses closest to the threshold
            todo = sorted(rel, key=lambda r: -r["delta_chi2"])[: a.per_pop]
        else:  # detections spread over Delta chi^2 from 20 to ~10^3, perturber off its prior bounds
            ok = sorted([r for r in rel if 20 <= r["delta_chi2"] <= 2000 and (pop, r["index"]) in lap and not lap[(pop, r["index"])]["perturber_on_bound"]],
                        key=lambda r: r["delta_chi2"])
            todo = [ok[k] for k in np.unique(np.linspace(0, len(ok) - 1, a.per_pop).round().astype(int))] if ok else []
        todo = [r for r in todo if (pop, r["index"]) not in done]
        if not todo:
            continue
        J._init(ROOT / "data" / man["population"], float(man.get("concentration_assumed", 15)), 1e12, True, 0, True, BOX, False, 1, False)
        with mp.get_context("fork").Pool(a.workers) as pool, open(outf, "a") as fo:
            for r in pool.imap_unordered(one, [(pop, r, a.n_live) for r in todo]):
                L = lap.get((pop, r["index"]))
                r["delta_lnZ_laplace"] = L["delta_lnZ"] if L else None; r["occam_laplace"] = L["occam"] if L else None
                fo.write(json.dumps(r) + "\n"); fo.flush()
                print(f"{pop} {r['index']}: dchi2 {r['delta_chi2_ns']:.1f}  dlnZ NS {r['delta_lnZ']:.1f}  Laplace {r['delta_lnZ_laplace']}  "
                      f"Occam {r['occam']:.1f}  ({r['t_s']:.0f}s, {r['n_like_smooth']}+{r['n_like_joint']} calls)", flush=True)
    recs = [json.loads(l) for l in open(outf)]
    det = [r for r in recs if r["occam_laplace"] is not None]
    summ = {"n": len(recs), "median_occam_ns": float(np.median([r["occam"] for r in recs])),
            "median_occam_ns_detections": float(np.median([r["occam"] for r in det])) if det else None,
            "median_occam_laplace_same": float(np.median([r["occam_laplace"] for r in det])) if det else None,
            "median_abs_dlnZ_ns_minus_laplace": float(np.median([abs(r["delta_lnZ"] - r["delta_lnZ_laplace"]) for r in det])) if det else None,
            "agree_dlnZ10": sum((r["delta_lnZ"] > 10) == (r["delta_lnZ_laplace"] > 10) for r in det), "n_det": len(det),
            "agree_dchi2_48": sum((r["delta_lnZ"] > 10) == (r["delta_chi2_ns"] >= 48) for r in recs), "records": recs}
    (ROOT / "results/evidence_nested.json").write_text(json.dumps(summ, indent=2))
    print({k: v for k, v in summ.items() if k != "records"})


if __name__ == "__main__":
    main()
