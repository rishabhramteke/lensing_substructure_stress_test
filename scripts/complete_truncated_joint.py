"""Finish every joint-scan lens the 900 s wall-clock guard cut short (referee round 20, 2026-09-28).

run_baseline_a_joint.py stopped a lens's cell loop, and then its basin loop, once the lens had used 900 s of wall
time. Which cells a truncated lens reached therefore depended on the machine, its load and whether it slept, so the
statistic was not a function of the data alone. Every individual fit is already capped by a fixed
function-evaluation budget (40 (n+1) evaluations per cell fit and per basin re-fit), so removing the wall-clock guard
leaves a statistic fixed by the data, the seed and that budget. This script does exactly that for lenses already on
disk: for every record with `truncated: true` it

  1. recomputes the smooth fit from the same seed (the stored chi^2 must be reproduced to 1e-7, else the lens is
     reported and left untouched),
  2. runs the grid cells the guard skipped, in the original order, with the original code path,
  3. runs the basin re-fit for every cell that lacks one,
  4. recomputes delta_chi2 and the basin fields exactly as run_baseline_a_joint.py does.

The pre-completion values are kept under `completion` so any change is traceable; `truncated` becomes false and
`truncated_by_guard` records that it once was. Progress is written to <pop>/completion.jsonl as lenses finish, so an
interrupted run resumes; the scan file is rewritten once a population is done. Run regate_joint.py afterwards.

For about 5% of lenses step 1 fails: the stored smooth chi^2 is not reproduced (relative gaps 1e-7 to O(1), all on
ill-conditioned fits; single- and multi-threaded recomputation agree with each other, packages, data and src/ are
unchanged since the runs, so the original value depended on the worker process, not on the data). Those lenses are
left truncated by the default pass; `--rerun-mismatched` then re-runs each of them in full through
run_baseline_a_joint.one_lens (same seed, same settings, no wall-clock limit), keeping the old record's values under
`rerun_full.before`.

    OMP_NUM_THREADS=1 python scripts/complete_truncated_joint.py --roots results/baseline_a_joint_full --workers 8
    python scripts/complete_truncated_joint.py --roots results/baseline_a_joint_mp4 --pops test_fixed15 --limit 2 --dry
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import fit_smooth, _unpack3, BOUNDS, SCALE, BOUNDS_LL, SCALE_LL, BOUNDS_MP, SCALE_MP  # noqa: E402

NPIX2 = 4096


def settings(man):
    """run_baseline_a_joint.py's arguments for one run, from its manifest (older manifests lack the later keys)."""
    conc = man.get("concentration_assumed", "15")
    return {"population": man["population"], "seed": int(man["seed"]), "n_sub": int(man.get("subsample_n", 300)),
            "skip": set(man.get("skipped_indices") or []), "explicit": man.get("explicit_indices"),
            "conc": conc if conc == "cm" else float(conc), "n_ll": int(man.get("lens_light_components") or 0),
            "free_position": bool(man.get("free_position")), "box": man.get("position_box_arcsec") or 0.5,
            "mp_orders": tuple(man["macro_multipole_orders"]) if man.get("macro_multipole_orders") else False,
            "restarts": int(man.get("smooth_restarts") or 1), "blind": bool(man.get("blind_initialization"))}


def fit_seeds(st, n_pop):
    """index -> fit_seed, reproducing run_baseline_a_joint.main()'s job list."""
    if st["explicit"]:
        idx = np.array(sorted(int(i) for i in st["explicit"]))
    else:
        n_sub = min(st["n_sub"], n_pop)
        idx = np.sort(np.random.default_rng(st["seed"]).choice(n_pop, size=n_sub, replace=False)) if n_sub < n_pop else np.arange(n_pop)
    return {int(i): st["seed"] * 100000 + k for k, i in enumerate(idx)}


def smooth_fit(i, fit_seed):
    """The smooth fit exactly as one_lens() computes it (same seed, same starts)."""
    G = J.G
    truth = G["truths"][i]; data = np.asarray(G["images"][i])
    rng_fit = np.random.default_rng(fit_seed)
    vec, chi2, _, noise_std = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60, blind=G.get("blind", False),
                                         lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
    if G["restarts"] > 1:
        for k in range(1, G["restarts"]):
            rng_k = np.random.default_rng(fit_seed + 7919 * k)
            v_k, c_k, _, _ = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], jitter_frac=0.3, rng=rng_k, maxiter=60,
                                        lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
            if c_k < chi2:
                vec, chi2 = v_k, c_k
        v_b, c_b, _, _ = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60, blind=True,
                                    lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
        if c_b < chi2:
            vec, chi2 = v_b, c_b
    return data, vec, chi2, noise_std


def complete(args):
    rec, fit_seed = args
    G = J.G
    t0 = time.time()
    i = rec["index"]
    data, vec, chi2_smooth, noise_std = smooth_fit(i, fit_seed)
    stored = rec["chi2_smooth_per_dof"] * NPIX2
    if abs(chi2_smooth - stored) > 1e-7 * max(1.0, abs(stored)):
        return {"index": i, "mismatch": True, "chi2_recomputed": chi2_smooth, "chi2_stored": stored}
    n_par = len(vec); n_ll = rec.get("lens_light_components_fitted", 0) or 0; n_mp = len(G["mp_orders"]) if G["mp_orders"] else 0
    bounds = list(BOUNDS) + list(BOUNDS_LL) * n_ll + list(BOUNDS_MP) * n_mp; scale = list(SCALE) + list(SCALE_LL) * n_ll + list(SCALE_MP) * n_mp
    lo_m = [b[0] for b in bounds]; hi_m = [b[1] for b in bounds]
    im_s = G["image_model_smooth"]

    def resid_smooth(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    before = {"n_cells": rec["n_cells"], "n_basin_refits": rec["n_basin_refits"], "delta_chi2": rec["delta_chi2"],
              "best_x": rec["best_x"], "best_y": rec["best_y"], "best_log10_m": rec["best_log10_m"],
              "chi2_smooth_basin_per_dof": rec["chi2_smooth_basin_per_dof"]}
    cells = rec["cells"]
    fp, box = G["free_position"], G["position_box"]
    theta_E = G["truths"][i]["lens_macro"]["theta_E"]
    radii = np.linspace(0.5, 1.4, 3) * theta_E; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    nodes = [(r * np.cos(a), r * np.sin(a)) for r in radii for a in angles]
    t_cells = time.time()
    for x, y in nodes[len(cells):]:
        if fp:
            lo = lo_m + [7.5, max(x - box, -2.4), max(y - box, -2.4)]; hi = hi_m + [11.5, min(x + box, 2.4), min(y + box, 2.4)]
            x0 = np.concatenate([vec, [9.0, x, y]]); xs = scale + [1.0, 0.1, 0.1]
        else:
            lo = lo_m + [7.5]; hi = hi_m + [11.5]; x0 = np.concatenate([vec, [9.0]]); xs = scale + [1.0]

        def resid(v, x=x, y=y):
            kl, ks, kll = _unpack3(v[:n_par])
            px, py = (v[n_par + 1], v[n_par + 2]) if fp else (x, y)
            model = G["image_model"].image(kwargs_lens=kl + [J.sub_kwargs(v[n_par], px, py)], kwargs_source=ks, kwargs_lens_light=kll)
            return ((data - model) / noise_std).ravel()

        res = least_squares(resid, x0, bounds=(lo, hi), x_scale=xs, method="trf", max_nfev=40 * len(x0))
        cells.append({"x": float(res.x[n_par + 1]) if fp else float(x), "y": float(res.x[n_par + 2]) if fp else float(y),
                      "node_x": float(x), "node_y": float(y), "chi2_joint": float(np.sum(res.fun ** 2)), "log10_m": float(res.x[n_par]),
                      "vec_macro": [float(v) for v in res.x[:n_par]], "nfev": int(res.nfev), "completed": True})
    t_cells = time.time() - t_cells
    t_b = time.time()
    if G["basin"]:
        for c in cells:
            if "chi2_smooth_from_joint_macro" in c:
                continue
            x0 = np.clip(np.array(c["vec_macro"]), lo_m, hi_m)
            rb = least_squares(resid_smooth, x0, bounds=(lo_m, hi_m), x_scale=scale, method="trf", max_nfev=40 * (n_par + 1))
            c["chi2_smooth_from_joint_macro"] = float(np.sum(rb.fun ** 2)); c["basin_nfev"] = int(rb.nfev); c["basin_completed"] = True
    t_b = time.time() - t_b
    chi2_polished = chi2_smooth - rec["chi2_polish_gain"]
    refits = [c["chi2_smooth_from_joint_macro"] for c in cells if "chi2_smooth_from_joint_macro" in c]
    chi2_smooth_basin = min([chi2_polished] + refits)
    best = min(cells, key=lambda c: c["chi2_joint"])
    rec["delta_chi2"] = float(chi2_smooth_basin - best["chi2_joint"])
    rec["delta_chi2_vs_unpolished_smooth"] = float(chi2_smooth - best["chi2_joint"])
    rec["delta_chi2_vs_polished_smooth"] = float(chi2_polished - best["chi2_joint"])
    rec["basin_gain"] = float(chi2_smooth - chi2_smooth_basin)
    rec["basin_gain_from_best_cell"] = float(chi2_smooth - best.get("chi2_smooth_from_joint_macro", chi2_smooth))
    rec["chi2_smooth_basin_per_dof"] = chi2_smooth_basin / NPIX2
    rec["best_x"], rec["best_y"], rec["best_log10_m"] = best["x"], best["y"], best["log10_m"]
    rec["n_cells"], rec["n_basin_refits"] = len(cells), len(refits)
    rec["truncated"] = False; rec["truncated_by_guard"] = True
    rec["completion"] = {"before": before, "t_cells_s": t_cells, "t_basin_s": t_b, "t_total_s": time.time() - t0,
                         "date": time.strftime("%Y-%m-%d")}
    return rec


PROVENANCE = ("rescued_record", "prerescue")


def rerun(args):
    """Re-run a lens whose smooth fit is not reproducible, from scratch, exactly as run_baseline_a_joint.py would
    with no wall-clock guard."""
    old, fit_seed = args
    t0 = time.time()
    new = J.one_lens((old["index"], fit_seed))
    for k in PROVENANCE:
        if k in old:
            new[k] = old[k]
    new["truncated"] = False; new["truncated_by_guard"] = True
    new["rerun_full"] = {"before": {k: old.get(k) for k in ("n_cells", "n_basin_refits", "delta_chi2", "best_x", "best_y", "best_log10_m",
                                                             "chi2_smooth_per_dof", "chi2_smooth_basin_per_dof", "reliable_fit")},
                         "reason": "stored smooth chi2 not reproducible", "t_total_s": time.time() - t0, "date": time.strftime("%Y-%m-%d")}
    return new


def _lock(popdir):
    """One process per population: a second launcher skips a population another live process holds."""
    lp = popdir / "completion.lock"
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return lp
    except FileExistsError:
        try:
            pid = int(lp.read_text().strip()); os.kill(pid, 0); return None       # holder alive
        except (ValueError, ProcessLookupError, PermissionError):
            lp.unlink(missing_ok=True); return _lock(popdir)                   # stale lock


def run_pop(popdir, workers, limit=None, dry=False, rerun_mismatched=False):
    lp = _lock(popdir)
    if lp is None:
        print(f"{popdir}: held by another process, skipped", flush=True); return
    try:
        _run_pop(popdir, workers, limit, dry, rerun_mismatched)
    finally:
        lp.unlink(missing_ok=True)


def _run_pop(popdir, workers, limit=None, dry=False, rerun_mismatched=False):
    man_p = popdir / "manifest.json"
    if not man_p.exists():
        print(f"{popdir}: no manifest (run unfinished), skipped"); return
    recs = [json.loads(l) for l in open(popdir / "scan_results.jsonl")]
    todo = [r for r in recs if r.get("truncated")]
    if not todo:
        print(f"{popdir}: nothing truncated"); return
    man = json.loads(man_p.read_text())
    rescue_man = None
    if any(r.get("rescued_record") for r in todo):
        rescue_man = json.loads((ROOT / "results/baseline_a_joint_rescue" / popdir.name / "manifest.json").read_text())
    prog_p = popdir / "completion.jsonl"
    done = {}
    if prog_p.exists():
        for l in open(prog_p):
            try:
                r = json.loads(l); done[r["index"]] = r
            except Exception:
                pass
    import multiprocessing as mp
    for group_rescued in (False, True):
        grp = [r for r in todo if bool(r.get("rescued_record")) == group_rescued and r["index"] not in done]
        if not grp:
            continue
        st = settings(rescue_man if group_rescued else man)
        pop_dir = ROOT / "data" / st["population"]
        J._init(pop_dir, st["conc"], 1e12, True, st["n_ll"], st["free_position"], st["box"], st["mp_orders"], st["restarts"], st["blind"])
        seeds = fit_seeds(st, len(J.G["truths"]))
        if limit:
            grp = grp[:limit]
        jobs = [(r, seeds[r["index"]]) for r in grp]
        t0 = time.time()
        if rerun_mismatched:
            print(f"{popdir.relative_to(ROOT)}: re-running {len(jobs)} non-reproducible lenses in full{' [rescued records]' if group_rescued else ''}", flush=True)
            with mp.get_context("fork").Pool(workers) as pool, open(prog_p, "a") as f:
                for k, out in enumerate(pool.imap_unordered(rerun, jobs)):
                    done[out["index"]] = out
                    f.write(json.dumps(out) + "\n"); f.flush()
                    b = out["rerun_full"]["before"]
                    print(f"  {k+1}/{len(jobs)} lens {out['index']}: cells {b['n_cells']}->{out['n_cells']}, dchi2 {b['delta_chi2']:.1f} -> {out['delta_chi2']:.1f}  ({out['rerun_full']['t_total_s']:.0f}s; {time.time()-t0:.0f}s elapsed)", flush=True)
            continue
        print(f"{popdir.relative_to(ROOT)}: completing {len(jobs)} of {len(todo)} truncated lenses ({len(done)} already done){' [rescued records]' if group_rescued else ''}", flush=True)
        with mp.get_context("fork").Pool(workers) as pool, open(prog_p, "a") as f:
            for k, out in enumerate(pool.imap_unordered(complete, jobs)):
                if out.get("mismatch"):
                    print(f"  !! lens {out['index']}: smooth chi2 not reproduced ({out['chi2_recomputed']:.6f} vs {out['chi2_stored']:.6f}); left as is", flush=True)
                    continue
                done[out["index"]] = out
                f.write(json.dumps(out) + "\n"); f.flush()
                c = out["completion"]
                print(f"  {k+1}/{len(jobs)} lens {out['index']}: cells {c['before']['n_cells']}->24, dchi2 {c['before']['delta_chi2']:.1f} -> {out['delta_chi2']:.1f}  ({c['t_total_s']:.0f}s; {time.time()-t0:.0f}s elapsed)", flush=True)
    if dry or limit:
        print(f"{popdir.relative_to(ROOT)}: {'dry run' if dry else 'limited run'}, scan file not rewritten"); return
    remaining = [r["index"] for r in todo if r["index"] not in done]
    if remaining:
        print(f"{popdir.relative_to(ROOT)}: {len(remaining)} lenses not completed (mismatch), scan file rewritten with the rest")
    out = [done.get(r["index"], r) for r in recs]
    with open(popdir / "scan_results.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    before = lambda r: (r.get("completion") or r.get("rerun_full") or {}).get("before")
    changed20 = sum(1 for r in out if before(r) and (r["delta_chi2"] >= 20) != (before(r)["delta_chi2"] >= 20))
    man["n_truncated"] = sum(1 for r in out if r.get("truncated"))
    man["guard_completion"] = {"n_completed": sum(1 for r in out if r.get("completion")), "n_rerun_full": sum(1 for r in out if r.get("rerun_full")),
                               "n_crossing_20": changed20,
                               "script": "complete_truncated_joint.py", "date": time.strftime("%Y-%m-%d")}
    man_p.write_text(json.dumps(man, indent=2))
    print(f"{popdir.relative_to(ROOT)}: done, {man['guard_completion']['n_completed']} completed, {changed20} crossed dchi2=20", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--pops", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="complete at most this many lenses per population (timing test)")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--rerun-mismatched", action="store_true", help="re-run, in full, the lenses the completion pass left truncated (see docstring)")
    a = p.parse_args()
    for root in a.roots:
        for popdir in sorted((ROOT / root).glob("*/")):
            if a.pops and popdir.name not in a.pops:
                continue
            if (popdir / "scan_results.jsonl").exists():
                run_pop(popdir, a.workers, a.limit, a.dry, a.rerun_mismatched)


if __name__ == "__main__":
    main()
