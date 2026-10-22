"""Laplace-approximation evidence ratio for the joint scan with the perturber position free (referee round 21, M1).

For every lens of the free-position runs (results/baseline_a_joint_freepos/<pop>) that the scan flags at
Delta chi^2 >= 20, the evidence of the smooth model and of the smooth-plus-perturber model is approximated about each
model's optimum,

    ln Z ~= -chi2_min/2 + (k/2) ln(2 pi) - (1/2) ln det F - ln V_prior,     F = J^T J,

with J the Jacobian of the noise-normalized residuals (so F is the Fisher matrix of the Gaussian likelihood) and uniform
priors: the macro-model's bounds, which are shared by the two models and cancel, and for the perturber log10 M200
uniform on [7.5, 11.5] and a position uniform over the region the scan explores (the placement annulus widened by the
0.5" position box). Delta ln Z = ln Z_joint - ln Z_smooth then carries the Occam penalty that Delta chi^2 lacks. Because
that penalty is positive, Delta ln Z > 10 essentially requires Delta chi^2 > 20, so lenses below Delta chi^2 = 20 are
counted as not detected at Delta ln Z > 10 without an evidence computation; the script checks the claim on every lens it
does compute. A lens whose perturber sits on a prior bound (mass or position box) is marked, since a Gaussian about a
bound misstates the posterior volume.

The smooth optimum is re-reached from the best cell's joint macro solution with the perturber removed (the basin
control's own starting point) and a larger evaluation budget; the joint optimum is the stored best cell, polished.

    python scripts/evidence_laplace.py --workers 12        -> results/evidence_laplace.json
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
POPS = ("no_subhalo", "multipole_m4_a3", "test_fixed60", "test_fixed15")
BOX = 0.5
LOGM = (7.5, 11.5)


def lnz(resid, x, lo, hi, scale):
    r = resid(x)
    Jm = approx_derivative(resid, x, method="2-point", rel_step=None, abs_step=1e-4 * np.asarray(scale), bounds=(lo, hi))
    F = Jm.T @ Jm
    sign, logdet = np.linalg.slogdet(F)
    k = len(x)
    return -0.5 * float(r @ r) + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet, float(r @ r), sign


def one(args):
    pop, rec = args
    G = J.G
    i = rec["index"]; data = np.asarray(G["images"][i]); theta_E = G["truths"][i]["lens_macro"]["theta_E"]
    sim = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"])
    noise_std = sim.estimate_noise(data)
    n_par = rec["n_params_macro"]
    lo_m = [b[0] for b in BOUNDS][:n_par]; hi_m = [b[1] for b in BOUNDS][:n_par]; sc = SCALE[:n_par]
    im_s, im_j = G["image_model_smooth"], G["image_model"]

    def resid_s(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    best = min(rec["cells"], key=lambda c: c["chi2_joint"])
    nx, ny = best["node_x"], best["node_y"]
    lo_j = lo_m + [LOGM[0], max(nx - BOX, -2.4), max(ny - BOX, -2.4)]; hi_j = hi_m + [LOGM[1], min(nx + BOX, 2.4), min(ny + BOX, 2.4)]
    sc_j = sc + [1.0, 0.1, 0.1]

    def resid_j(v):
        kl, ks, kll = _unpack3(v[:n_par])
        return ((data - im_j.image(kwargs_lens=kl + [J.sub_kwargs(v[n_par], v[n_par + 1], v[n_par + 2])], kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    t0 = time.time()
    x0s = np.clip(np.array(best["vec_macro"]), lo_m, hi_m)
    rs = least_squares(resid_s, x0s, bounds=(lo_m, hi_m), x_scale=sc, method="trf", max_nfev=200 * (n_par + 1))
    x0j = np.clip(np.array(best["vec_macro"] + [best["log10_m"], best["x"], best["y"]]), lo_j, hi_j)
    rj = least_squares(resid_j, x0j, bounds=(lo_j, hi_j), x_scale=sc_j, method="trf", max_nfev=80 * (n_par + 4))
    xs = np.clip(rs.x, np.array(lo_m) + 1e-6, np.array(hi_m) - 1e-6); xj = np.clip(rj.x, np.array(lo_j) + 1e-6, np.array(hi_j) - 1e-6)
    lz_s, c2_s, sg_s = lnz(resid_s, xs, lo_m, hi_m, sc)
    lz_j, c2_j, sg_j = lnz(resid_j, xj, lo_j, hi_j, sc_j)
    # perturber prior volume: log-mass width times the explored area, the placement annulus widened by the position box
    r_in, r_out = max(0.0, 0.5 * theta_E - BOX), 1.4 * theta_E + BOX
    ln_vpert = np.log(LOGM[1] - LOGM[0]) + np.log(np.pi * (r_out ** 2 - r_in ** 2))
    dlnz = lz_j - lz_s - ln_vpert
    m, x, y = rj.x[n_par], rj.x[n_par + 1], rj.x[n_par + 2]
    on_bound = bool(m > LOGM[1] - 0.01 or m < LOGM[0] + 0.01 or abs(x - nx) > BOX - 0.01 or abs(y - ny) > BOX - 0.01)
    return {"population": pop, "index": i, "has_subhalo": rec["has_subhalo"], "log10_M200_true": rec.get("log10_M200_true"),
            "delta_chi2_scan": rec["delta_chi2"], "delta_chi2_laplace": c2_s - c2_j, "delta_lnZ": float(dlnz),
            "occam": float(0.5 * (c2_s - c2_j) - dlnz), "ln_prior_volume_perturber": float(ln_vpert),
            "fisher_pos_def": bool(sg_s > 0 and sg_j > 0), "perturber_on_bound": on_bound,
            "log10_m_fit": float(m), "x_fit": float(x), "y_fit": float(y), "t_s": time.time() - t0}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=12); p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    import multiprocessing as mp
    out = {"pops": {}, "records": []}
    for pop in POPS:
        d = ROOTDIR / pop
        if not (d / "manifest.json").exists():
            continue
        man = json.loads((d / "manifest.json").read_text())
        recs = [json.loads(l) for l in open(d / "scan_results.jsonl")]
        rel = [r for r in recs if r["reliable_fit"]]
        todo = [r for r in rel if r["delta_chi2"] >= 20][: a.limit]
        J._init(ROOT / "data" / man["population"], float(man.get("concentration_assumed", 15)), 1e12, True, 0, True, BOX, False, 1, False)
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = list(pool.imap_unordered(one, [(pop, r) for r in todo]))
        out["records"] += res
        ok = [r for r in res if r["fisher_pos_def"]]
        out["pops"][pop] = {"n_retained": len(rel), "n_dchi2_20": len(todo), "n_computed": len(ok),
                            "n_dlnz_10": sum(1 for r in ok if r["delta_lnZ"] > 10),
                            "n_on_bound": sum(1 for r in res if r["perturber_on_bound"]),
                            "median_occam": float(np.median([r["occam"] for r in ok])) if ok else None,
                            "min_dchi2_passing_dlnz10": float(min([r["delta_chi2_laplace"] for r in ok if r["delta_lnZ"] > 10], default=np.nan))}
        s = out["pops"][pop]
        print(f"{pop:18s} retained {s['n_retained']}  dchi2>=20 {s['n_dchi2_20']}  dlnZ>10 {s['n_dlnz_10']}  "
              f"median Occam {s['median_occam']}  on bound {s['n_on_bound']}  ({time.time()-t0:.0f}s)", flush=True)
    (ROOT / "results/evidence_laplace.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
