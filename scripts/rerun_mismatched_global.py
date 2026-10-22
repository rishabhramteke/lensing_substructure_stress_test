"""Full re-run of every non-reproducible truncated lens (complete_truncated_joint.py --rerun-mismatched), scheduled as ONE
lens-level pool across all populations, slowest lenses first, so no worker waits for the slowest lens of a population
before the next one starts.

Takes the per-population completion.lock of every population it schedules (a population another live process holds is
left to that process), appends each finished lens to <pop>/completion.jsonl exactly as the per-population pass does, and
finalizes each population (scan file rewritten, manifest updated) once all its lenses are in. Resumable.

    OMP_NUM_THREADS=1 python scripts/rerun_mismatched_global.py --workers 14
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import complete_truncated_joint as C  # noqa: E402

J = C.J
SKIP_ROOTS = ("baseline_a_joint_rescue", "baseline_a_joint_basin_c15", "baseline_a_joint_blind")   # not read by the paper
_CUR = {"key": None}


def cost(rec):
    """Rank-only cost estimate from the truncated record (its timers include machine sleep, so only the order is used)."""
    nc = max(rec["n_cells"], 1); nb = rec.get("n_basin_refits") or 0
    pc = rec["t_scan_s"] / nc; pb = rec["t_basin_s"] / nb if nb else pc
    return rec["t_fit_s"] + 24 * (pc + pb)


def work(job):
    popdir, st, rec, seed = job
    key = json.dumps(st, sort_keys=True, default=list)
    if _CUR["key"] != key:
        J._init(ROOT / "data" / st["population"], st["conc"], 1e12, True, st["n_ll"], st["free_position"], st["box"],
                st["mp_orders"], st["restarts"], st["blind"])
        _CUR["key"] = key
    return popdir, C.rerun((rec, seed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=14)
    a = p.parse_args()
    jobs, locks, pending = [], {}, {}
    for popdir in sorted((ROOT / "results").glob("baseline_a_joint_*/*/")):
        if popdir.parent.name in SKIP_ROOTS or not (popdir / "manifest.json").exists() or not (popdir / "scan_results.jsonl").exists():
            continue
        recs = [json.loads(l) for l in open(popdir / "scan_results.jsonl")]
        done = set()
        if (popdir / "completion.jsonl").exists():
            for l in open(popdir / "completion.jsonl"):
                try:
                    done.add(json.loads(l)["index"])
                except Exception:
                    pass
        todo = [r for r in recs if r.get("truncated") and r["index"] not in done]
        if not todo:
            continue
        lp = C._lock(popdir)
        if lp is None:
            print(f"{popdir.relative_to(ROOT)}: held by another process, left to it", flush=True); continue
        locks[str(popdir)] = lp; pending[str(popdir)] = len(todo)
        man = json.loads((popdir / "manifest.json").read_text())
        rescue_man = None
        if any(r.get("rescued_record") for r in todo):
            rescue_man = json.loads((ROOT / "results/baseline_a_joint_rescue" / popdir.name / "manifest.json").read_text())
        for grp_rescued in (False, True):
            grp = [r for r in todo if bool(r.get("rescued_record")) == grp_rescued]
            if not grp:
                continue
            st = C.settings(rescue_man if grp_rescued else man)
            st = {k: (sorted(v) if isinstance(v, set) else v) for k, v in st.items()}
            n_pop = sum(1 for _ in open(ROOT / "data" / st["population"] / "truth.jsonl"))
            seeds = C.fit_seeds(st, n_pop)
            jobs += [(str(popdir), st, r, seeds[r["index"]]) for r in grp]
    jobs.sort(key=lambda j: -cost(j[2]))
    print(f"{len(jobs)} lenses in {len(pending)} populations, {a.workers} workers, slowest first  {time.strftime('%c')}", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    try:
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
            for k, (popdir, out) in enumerate(pool.imap_unordered(work, jobs)):
                pd = Path(popdir)
                with open(pd / "completion.jsonl", "a") as f:
                    f.write(json.dumps(out) + "\n")
                b = out["rerun_full"]["before"]
                print(f"  {k+1}/{len(jobs)} {pd.relative_to(ROOT)} lens {out['index']}: dchi2 {b['delta_chi2']:.1f} -> {out['delta_chi2']:.1f}"
                      f"  ({out['rerun_full']['t_total_s']:.0f}s; {time.time()-t0:.0f}s elapsed)", flush=True)
                pending[popdir] -= 1
                if pending[popdir] == 0:
                    C._run_pop(pd, 1, rerun_mismatched=True)          # nothing left to run: merges and rewrites the scan file
                    locks.pop(popdir).unlink(missing_ok=True)
    finally:
        for lp in locks.values():
            lp.unlink(missing_ok=True)
    print(f"ALL DONE {time.strftime('%c')}", flush=True)


if __name__ == "__main__":
    main()
