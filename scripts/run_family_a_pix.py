"""Family A-pix: the parametric subhalo search with a pixelized source and the Bayesian evidence as the likelihood, the
protocol of the PyAutoLens detections (Nightingale et al. 2024; He et al. 2023) on Family A's own grid.

Per lens:
  1. Family A's smooth EPL+shear+Sersic(+lens light) fit (`fit_smooth`, same idealization); lens light subtracted; the
     lensed Sersic model is the adapt image of a brightness-adaptive rectangular source mesh (50 x 50, constant
     regularization, the mesh of Family B-GI);
  2. null: the macro model (theta_E, gamma, two ellipticity and two shear components) re-optimized under the
     pixelized-source evidence, the source regularization set by maximizing the evidence -> lnE_0;
  3. scan: at each of Family A's 24 nodes (3 radii x 8 angles) the evidence is maximized over the mass of a truncated
     NFW (c = 15, tau = 20, as Family A) at the null macro model; the best two nodes are then refined with the macro
     model, the mass and the position (within 0.5" of the node) free -> lnE_1;
  4. statistic delta_lnE = lnE_1 - lnE_0 (written also as `delta_chi2`, the alias the evaluators read); position and
     mass are those of the refined optimum.

    python scripts/run_family_a_pix.py --population test_fixed60 --only 0 5 --out results/baseline_apix/test_fixed60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import logging
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import autolens as al  # noqa: E402
import autoarray as aa  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.Util import param_util  # noqa: E402
from baseline_a.fit import fit_smooth, _unpack3, lens_light_image  # noqa: E402
from run_family_b import to_pal, noise_map_for, build_imaging, arc_mask, CHI2_DOF_UNRELIABLE  # noqa: E402
from run_family_b_gi import pyautolens_lens, source_blocks, Evidence, build_imaging_exact, logdet_pd  # noqa: E402

LC = LensCosmo(z_lens=0.5, z_source=1.0)
CONC, TAU, MASS_RANGE, BOX = 15.0, 20.0, (7.5, 11.5), 0.5


def subhalo(log10_m, x, y):
    Rs, a = LC.nfw_physical2angle(M=10 ** log10_m, c=CONC)
    return al.mp.NFWTruncatedSph(centre=(float(y), float(x)), kappa_s=float(a / (4 * Rs * (1 - np.log(2)))),
                                 scale_radius=float(Rs), truncation_radius=float(TAU * Rs))


class FixedGridSource:
    """A pixelized source on a grid FIXED per lens: a 50 x 50 tensor grid whose node spacing follows the
    brightness-weighted distribution of the ray-traced data pixels at the null lens model (resolution where the
    source is bright and magnified, as in an adaptive mesh), bilinear interpolation of every ray-traced pixel onto it,
    the simulator's PSF as an explicit blurring matrix, and gradient regularization over grid neighbours. Because the
    grid does not move, the mapping matrix, and so the evidence, varies continuously with the lens model; a mesh
    rebuilt from every trial lens makes the evidence jump by ~10 for a 1e-4" change in theta_E."""
    def __init__(self, beta_yx, weights, psf_kernel, mask, n=50, pad=0.3):
        from scipy import sparse
        from autolens.potential_correction import util as pcu
        self.n = n
        self.nodes = []
        for ax in (1, 0):   # x then y
            v = beta_yx[:, ax]; o = np.argsort(v); cw = np.cumsum(weights[o]); cw = cw / cw[-1]
            q = np.interp(np.linspace(0, 1, n), np.concatenate([[0], cw]), np.concatenate([[v[o][0]], v[o]]))
            span = v.max() - v.min(); q[0] -= pad * span; q[-1] += pad * span
            q = np.maximum.accumulate(q + np.arange(n) * 1e-9 * span)   # strictly increasing
            self.nodes.append(q)
        self.psf = sparse.csr_matrix(pcu.psf_matrix_from(np.asarray(psf_kernel), np.asarray(mask)))
        Dx = sparse.lil_matrix((n * (n - 1), n * n)); Dy = sparse.lil_matrix((n * (n - 1), n * n)); k = 0
        for j in range(n):
            for i in range(n - 1):
                Dx[k, j * n + i] = -1; Dx[k, j * n + i + 1] = 1
                Dy[k, i * n + j] = -1; Dy[k, (i + 1) * n + j] = 1; k += 1
        self.R = (Dx.T @ Dx + Dy.T @ Dy + 1e-6 * sparse.identity(n * n)).tocsc()
        self.ld_R = float(np.linalg.slogdet(self.R.toarray())[1])

    def mapping(self, beta_yx):
        from scipy import sparse
        n = self.n; xs, ys = self.nodes
        x = np.clip(beta_yx[:, 1], xs[0], xs[-1]); y = np.clip(beta_yx[:, 0], ys[0], ys[-1])
        ix = np.clip(np.searchsorted(xs, x) - 1, 0, n - 2); iy = np.clip(np.searchsorted(ys, y) - 1, 0, n - 2)
        tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]); ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy])
        rows = np.repeat(np.arange(len(x)), 4)
        cols = np.stack([iy * n + ix, iy * n + ix + 1, (iy + 1) * n + ix, (iy + 1) * n + ix + 1], 1).ravel()
        vals = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1).ravel()
        return self.psf @ sparse.csr_matrix((vals, (rows, cols)), shape=(len(x), n * n))


class SparseEvidence:
    """Bayesian evidence of the linear source inversion (the terms of run_family_b_gi.Evidence) with the sparse
    normal matrix factorized by sparse LU: solution and log-determinant at ~0.1 s for a 2 500-pixel source."""
    def __init__(self, data, noise, F, R, ld_R):
        from scipy import sparse
        w = 1.0 / noise ** 2
        self.F, self.d, self.w = F, data, w
        self.N = (F.T @ sparse.diags(w) @ F).tocsc(); self.D = np.asarray(F.T @ (w * data)).ravel()
        self.R, self.ld_R, self.n = R, ld_R, F.shape[1]
        self.noise_term = -0.5 * float(np.sum(np.log(2 * np.pi * noise ** 2)))

    def __call__(self, log_lams, full=False):
        from scipy.sparse.linalg import splu
        lam = 10.0 ** float(np.atleast_1d(log_lams)[0])
        try:
            # symmetric positive-definite: no row pivoting, so diag(U) > 0 and log det = sum log diag(U)
            lu = splu((self.N + lam * self.R).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
            diag = lu.U.diagonal()
            if np.any(diag <= 0):
                raise np.linalg.LinAlgError
            ld_A = float(np.sum(np.log(diag))); sol = lu.solve(self.D)
        except (RuntimeError, np.linalg.LinAlgError):
            return (-1e30, None) if full else -1e30
        resid = self.d - self.F @ sol; chi2 = float(np.sum(self.w * resid ** 2))
        lnz = self.noise_term - 0.5 * chi2 - 0.5 * lam * float(sol @ (self.R @ sol)) - 0.5 * ld_A + 0.5 * (self.n * np.log(lam) + self.ld_R)
        return (lnz, {"sol": sol, "chi2": chi2}) if full else lnz


class Lens:
    """Evidence of the pixelized source for a macro vector p = (theta_E, gamma, ell_1, ell_2, gamma_1, gamma_2) and an
    optional subhalo, at a fixed source regularization (or maximized over it)."""
    def __init__(self, mi, z_lens, p0, adapt, shape):
        self.mi, self.z, self.p0, self.adapt = mi, z_lens, np.asarray(p0, float), adapt
        self.pix = al.Pixelization(mesh=al.mesh.RectangularBilinearAdaptImage(shape=shape, weight_power=1.0, weight_floor=0.1),
                                   regularization=al.reg.Constant(coefficient=1.0))
        self.d, self.n = np.asarray(mi.data), np.asarray(mi.noise_map)
        self.n_eval = 0
        self.src = None; self.shape = shape

    def galaxy(self, p, sub=None):
        kw = dict(mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=(float(p[2]), float(p[3])), einstein_radius=float(p[0]), slope=float(p[1])),
                  shear=al.mp.ExternalShear(gamma_1=float(p[4]), gamma_2=float(p[5])))
        if sub is not None:
            kw["subhalo"] = subhalo(*sub)
        return al.Galaxy(redshift=self.z, **kw)

    def beta(self, g):
        grid = self.mi.grid
        return np.asarray(grid) - np.asarray(g.deflections_yx_2d_from(grid=grid))

    def lnE(self, p, sub=None, log_lam=None):
        self.n_eval += 1
        try:
            g = self.galaxy(p, sub); b = self.beta(g)
            if self.src is None:   # the grid is laid out once, at the first (null) lens model
                w = np.clip(np.asarray(self.adapt), 0, None); w = w / w.max() + 0.1
                self.src = FixedGridSource(b, w, self.mi.psf.kernel.native, self.mi.mask, n=self.shape[0])
            ev = SparseEvidence(self.d, self.n, self.src.mapping(b), self.src.R, self.src.ld_R)
        except Exception:  # noqa: BLE001 -- an unphysical trial (e.g. q > 1) is a bad sample
            return -1e30, log_lam
        if log_lam is None:
            r = minimize_scalar(lambda t: -ev([t]), bounds=(-4.0, 6.0), method="bounded", options={"xatol": 0.02})
            return float(-r.fun), float(r.x)
        return float(ev([log_lam])), log_lam


STEP = np.array([0.005, 0.02, 0.01, 0.01, 0.005, 0.005])


def polish(L, p, sub, log_lam, maxiter=15, mode="macro", node=None):
    """L-BFGS-B, in variables scaled by STEP so one finite-difference step is a small physical change (the fixed
    source grid makes the evidence smooth enough for gradients), over
      mode="macro"  the macro vector, any subhalo held fixed;
      mode="mass"   the macro vector and log10 M, the subhalo at the node (Family A's joint scan);
      mode="free"   the macro vector, log10 M and the position within BOX of the node (Family A's position-free scan)."""
    nfree = {"macro": 0, "mass": 1, "free": 3}[mode]
    x0 = np.concatenate([p, np.asarray(sub, float)[:nfree]]) if nfree else np.asarray(p, float)
    step = np.concatenate([STEP, [0.3, 0.05, 0.05][:nfree]])
    bnds = [(None, None)] * 6
    if nfree:
        bnds.append(((MASS_RANGE[0] - sub[0]) / 0.3, (MASS_RANGE[1] - sub[0]) / 0.3))
    if nfree == 3:
        bnds += [((node[0] - BOX - sub[1]) / 0.05, (node[0] + BOX - sub[1]) / 0.05), ((node[1] - BOX - sub[2]) / 0.05, (node[1] + BOX - sub[2]) / 0.05)]

    def unpack(v):
        if mode == "macro":
            return v, sub
        if mode == "mass":
            return v[:6], (v[6], sub[1], sub[2])
        return v[:6], tuple(v[6:9])

    def f(u):
        q, sb = unpack(x0 + u * step)
        return -L.lnE(q, sb, log_lam)[0]
    r = minimize(f, np.zeros(len(x0)), method="L-BFGS-B", bounds=bnds, options={"maxiter": maxiter, "eps": 1e-3})
    q, sb = unpack(x0 + r.x * step)
    return q, sb, -r.fun


def one_lens(data, noise, truth, kwargs_band, num_pix, ps, seed, i, shape, lens_light_components, candidate=None):
    t0 = time.time()
    vec, chi2_smooth, image_model, _ = fit_smooth(data, truth, kwargs_band, num_pix, rng=np.random.default_rng([seed + 1000, int(i)]),
                                                  maxiter=60, lens_light_components=lens_light_components)
    kl, ks, kll = _unpack3(vec)
    ll = lens_light_image(vec, kwargs_band, num_pix) if kll else 0.0
    lensed = np.asarray(image_model.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll, lens_light_add=False))
    sub_img = data - ll
    mask = arc_mask(sub_img, noise, ps)
    mi = build_imaging_exact(sub_img, noise, kwargs_band, num_pix).apply_mask(mask=mask).apply_over_sampling(over_sample_size_lp=1, over_sample_size_pixelization=1)
    adapt = al.Array2D(values=np.asarray(to_pal(lensed))[~np.asarray(mask)], mask=mask)
    g, _ = pyautolens_lens(kl, truth["cosmology"]["z_lens"], np.asarray(mi.grid.slim))
    p0 = np.array([g.mass.einstein_radius, g.mass.slope, g.mass.ell_comps[0], g.mass.ell_comps[1], g.shear.gamma_1, g.shear.gamma_2])
    L = Lens(mi, truth["cosmology"]["z_lens"], p0, adapt, shape)
    t_fit = time.time() - t0

    # null: regularization, then the macro model from the smooth fit and from four perturbed starts, long L-BFGS runs
    _, lam0 = L.lnE(p0)
    starts = [p0] + [p0 + d for d in (np.array([0.01, 0, 0, 0, 0, 0]), np.array([-0.01, 0, 0, 0, 0, 0]),
                                      np.array([0, 0.05, 0, 0, 0, 0]), np.array([0, -0.05, 0, 0, 0, 0]))]
    p_null, lnE0 = None, -np.inf
    for st in starts:
        q, _, e = polish(L, st, None, lam0, maxiter=150)
        if e > lnE0:
            p_null, lnE0 = q, e
    _, lam0 = L.lnE(p_null)
    p_null, _, lnE0 = polish(L, p_null, None, lam0, maxiter=60)

    theta_E = kl[0]["theta_E"]
    if candidate is None:
        nodes = [(r * np.cos(a), r * np.sin(a)) for r in np.linspace(0.5, 1.4, 3) * theta_E for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    else:
        nodes = [tuple(candidate)]

    def scan(p_start):
        # joint scan: at every node the macro model and the mass are fitted together; the two best refined with the
        # position free
        cells = []
        for x, y in nodes:
            q, sb, e = polish(L, p_start, (9.0, x, y), lam0, maxiter=15, mode="mass")
            cells.append({"node_x": float(x), "node_y": float(y), "log10_m": float(sb[0]), "delta_lnE_node": float(e - lnE0), "macro": [float(t) for t in q]})
        best = None
        for c in sorted(cells, key=lambda c: -c["delta_lnE_node"])[:2]:
            q, sb, _ = polish(L, np.array(c["macro"]), (c["log10_m"], c["node_x"], c["node_y"]), lam0, maxiter=25, mode="free", node=(c["node_x"], c["node_y"]))
            e1, l1 = L.lnE(q, sb)
            c.update({"refined_log10_m": float(sb[0]), "refined_x": float(sb[1]), "refined_y": float(sb[2]), "refined_lnE": float(e1)})
            if best is None or e1 > best[0]:
                best = (e1, q, sb, l1)
        return cells, best

    def basin(cells, best):
        # the smooth model re-optimized from the winning solution's and the three best nodes' macro models
        e_best, p_best, l_best = -np.inf, None, lam0
        for start in [best[1]] + [np.array(c["macro"]) for c in sorted(cells, key=lambda c: -c["delta_lnE_node"])[:3]]:
            q, _, _ = polish(L, start, None, best[3], maxiter=60)
            e, l = L.lnE(q)
            if e > e_best:
                e_best, p_best, l_best = e, q, l
        return e_best, p_best, l_best

    cells, best = scan(p_null)
    lnE0b, p_b, lam0b = basin(cells, best)
    n_pass = 1
    if lnE0b > lnE0 + 1.0:   # a better smooth optimum: it becomes the null and the scan is repeated from it
        lnE0, p_null, lam0 = lnE0b, p_b, lam0b
        cells, best = scan(p_null)
        lnE0b, p_b, lam0b = basin(cells, best)
        n_pass = 2
    lnE1, q1, sb1, lam1 = best
    basin_gain = lnE0b - lnE0
    lnE0_final = max(lnE0, lnE0b)
    for c in cells:
        c["refined_delta_lnE"] = c.pop("refined_lnE") - lnE0_final if "refined_lnE" in c else None
    for c in cells:
        c.pop("macro")
    return {"delta_lnE": float(lnE1 - lnE0_final), "lnE_null": float(lnE0_final), "lnE_sub": float(lnE1), "basin_gain_lnE": float(basin_gain), "n_scan_passes": n_pass,
            "log10_lam_null": float(lam0 if lnE0 >= lnE0b else lam0b), "log10_lam_sub": float(lam1),
            "best_log10_m": float(sb1[0]), "best_x": float(sb1[1]), "best_y": float(sb1[2]), "cells": cells,
            "chi2_smooth_per_dof": float(chi2_smooth / num_pix ** 2), "n_unmasked": int(len(L.d)), "n_evidence_evals": L.n_eval,
            "t_fit_s": t_fit, "t_scan_s": time.time() - t0 - t_fit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--only", type=int, nargs="+", help="population indices (the stratified subset)")
    p.add_argument("--candidates-from", type=Path, help="confirmation mode: a Family B-GI scan_results.jsonl; each lens with delta_lnZ >= "
                   "--min-delta-lnz is re-fitted at its dkappa peak only")
    p.add_argument("--min-delta-lnz", type=float, default=10.0)
    p.add_argument("--shard", help="k/K: every K-th lens of the list starting at k")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src-shape", type=int, default=50)
    p.add_argument("--lens-light-components", type=int, default=1, choices=[1, 2])
    args = p.parse_args()
    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kb, num_pix = manifest["kwargs_band"], manifest["image_shape"][0]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    noisy = pop_dir / "images_full_noisy.npy"
    images = np.load(noisy if noisy.exists() else pop_dir / "images_full_noiseless.npy", mmap_mode="r")
    cand = {}
    if args.candidates_from:
        for l in open(args.candidates_from):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("reliable_fit") and r.get("delta_lnZ", 0) >= args.min_delta_lnz and r.get("best_x") is not None:
                cand[int(r["index"])] = (r["best_x"], r["best_y"])
        args.only = sorted(cand)
    if args.shard:
        k_, K_ = map(int, args.shard.split("/")); args.only = args.only[k_::K_]
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "scan_results.jsonl"
    done = set()
    if out_file.exists():
        for l in open(out_file):
            try:
                done.add(json.loads(l)["index"])
            except (json.JSONDecodeError, KeyError):
                pass
    with open(out_file, "a") as f:
        for i in args.only:
            if i in done:
                continue
            t = truths[i]
            rec = {"index": int(i), "has_subhalo": t.get("subhalo") is not None, "log10_M200_true": (t.get("subhalo") or {}).get("log10_M200"),
                   "concentration_true": (t.get("subhalo") or {}).get("concentration"), "has_multipole": t.get("multipole") is not None}
            try:
                data = np.asarray(images[i]); noise = noise_map_for(data, kb, num_pix)
                r = one_lens(data, noise, t, kb, num_pix, kb["pixel_scale"], args.seed, i, (args.src_shape, args.src_shape), args.lens_light_components,
                             candidate=cand.get(int(i)))
                rec.update(r); rec.update({"delta_chi2": r["delta_lnE"], "reliable_fit": bool(r["chi2_smooth_per_dof"] < CHI2_DOF_UNRELIABLE)})
            except Exception as e:  # noqa: BLE001
                rec.update({"error": f"{type(e).__name__}: {e}", "reliable_fit": False, "delta_chi2": 0.0, "delta_lnE": 0.0, "best_x": None, "best_y": None})
            f.write(json.dumps(rec, default=float) + "\n"); f.flush()
            print(f"{args.population} {i}: dlnE {rec.get('delta_lnE'):.1f}  {rec.get('t_scan_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
