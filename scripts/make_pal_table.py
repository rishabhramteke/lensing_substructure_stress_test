"""The published PyAutoLens subhalo pipeline on individual lenses, at the reference depth (Tier 0) and on the same lenses
at 600 s: one row per lens with the pipeline's Delta ln Z (refine minus mass_total), the goodness of fit of its
no-subhalo model (chi^2 per masked pixel of mass_total[1]), its subhalo mass and the offset of a real subhalo, and on the
same lens and depth the joint scan's Delta chi^2. A row marked "free m=4" runs the pipeline with a free m=4
PowerLawMultipole in the total-mass model (run_pyautolens_slam_subhalo.py --multipole), and the joint scan with its free
m=4 term (--macro-multipole).

    python scripts/make_pal_table.py      -> results/pyautolens_slam/summary.json, paper/tables/pal.tex
"""
import glob, json
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parents[1] / "external/autolens_workspace/output/subhalo_detect"
POP = {"no_subhalo": "clean", "multipole_m4_a3": r"$m{=}4$, $a_4/\thetaE=0.03$", "test_fixed60": r"$c=60$"}
# depth: (pipeline results root, data prefix, joint-scan root)
DEPTH = {"t0": ("results/pyautolens_slam", "", "results/baseline_a_joint_full"),
         "600": ("results/pyautolens_slam_600", "test_shallow600_", "results/baseline_a_joint_pal600")}


def rec(root, pop, i):
    f = ROOT / root / pop / "scan_results.jsonl"
    if not f.exists():
        return None
    return next((json.loads(l) for l in open(f) if json.loads(l)["index"] == i), None)


def arc_snr(pop, i):
    """[sum (I_noiseless / sigma)^2]^(1/2) with the simulator's own per-pixel noise estimate."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    from run_family_b import noise_map_for
    m = json.loads((ROOT / "data" / pop / "manifest.json").read_text()); kb, n = m["kwargs_band"], m["image_shape"][0]
    img = np.asarray(np.load(ROOT / "data" / pop / "images_full_noiseless.npy", mmap_mode="r")[i])
    noisy = np.asarray(np.load(ROOT / "data" / pop / "images_full_noisy.npy", mmap_mode="r")[i])
    return float(np.sqrt(np.sum((img / noise_map_for(noisy, kb, n)) ** 2)))


def chi2pix(lens, stage, multipole):
    """chi^2 per masked pixel of a search's best fit; the free-multipole and bare runs share a dataset name and are told
    apart by their model."""
    for d in sorted(glob.glob(glob.escape(str(WS / lens / stage)) + "/*/")):
        info = Path(d) / "model.info"
        if not (Path(d) / "image/fit.fits").exists() or not info.exists():
            continue
        if ("multipole" in info.read_text().lower()) != multipole:
            continue
        c = [x for x in fits.open(Path(d) / "image/fit.fits") if x.name == "CHI_SQUARED_MAP"][0].data; m = np.isfinite(c) & (c != 0)
        return float(c[m].sum() / m.sum())
    return None


def pal_rows():
    rows = {}
    for dep, (root, pre, aroot) in DEPTH.items():
        for pop in POP:
            for sub, mp in (("", False), ("_mp4", True)):
                f = ROOT / root / (pop + sub) / "scan_results.jsonl"
                if not f.exists():
                    continue
                for l in open(f):
                    r = json.loads(l); i = r["index"]; dpop = pre + (pop.removeprefix("test_") if pre else pop); lens = f"{dpop}_{i:05d}"
                    a = rec(aroot, pop + sub, i)
                    d = {"pal_dlnZ": r["delta_lnZ"], "pal_log10m": r["log10_M200_fit"], "pal_xy": [r["best_x"], r["best_y"]],
                         "pal_chi2pix_no_subhalo": chi2pix(lens, "mass_total[1]", mp), "pal_chi2pix_subhalo": chi2pix(lens, "subhalo[2]_[single_plane_refine]", mp),
                         "a_dchi2": a["delta_chi2"] if a else None, "a_chi2dof": a["chi2_smooth_per_dof"] if a else None,
                         "arc_snr": arc_snr(dpop, i), "hours": r["seconds"] / 3600}
                    if r.get("x_true") is not None:
                        d["pal_offset"] = float(np.hypot(r["best_x"] - r["x_true"], r["best_y"] - r["y_true"]))
                    key = (pop, i, mp)
                    rows.setdefault(key, {"population": pop, "index": i, "free_m4": mp, "log10m_true": r.get("log10_M200_true")})[dep] = d
    return rows


def main():
    rows = pal_rows()
    (ROOT / "results/pyautolens_slam/summary.json").write_text(json.dumps([dict(v, key=list(k)) for k, v in rows.items()], indent=2))
    f1 = lambda v, fmt: "--" if v is None else ("${:.1f}$".format(v) if fmt == "{:.0f}" and abs(v) < 9.5 else fmt.format(v))  # noqa: E731
    pair = lambda r, k, fmt: " / ".join(f1((r.get(dep) or {}).get(k), fmt) for dep in DEPTH)  # noqa: E731
    L = [r"\begin{tabular}{@{}lrcccccc@{}}", r"\toprule",
         r" & & & \multicolumn{4}{c}{published pipeline} & A \\ \cmidrule(lr){4-7}",
         r"lens & & arc S/N & $\Delta\ln Z$ & $\chi^2$/pix, no sub. & $\log_{10}M_{200}$ & offset & $\dchi$ \\", r"\midrule"]
    for r in sorted(rows.values(), key=lambda r: (list(POP).index(r["population"]), r["index"], r["free_m4"])):
        lab = POP[r["population"]] + (f", $10^{{{r['log10m_true']:.1f}}}\\,\\Msun$" if r["log10m_true"] else "") + (", free $m{=}4$" if r["free_m4"] else "")
        def mass(dep):
            d = r.get(dep)
            return "--" if not d else f"{d['pal_log10m']:.1f}" + ("$^{b}$" if d["pal_log10m"] > 10.95 else "")
        L.append(f"{lab} & {r['index']} & {pair(r, 'arc_snr', '{:.0f}')} & {pair(r, 'pal_dlnZ', '{:.0f}')} & {pair(r, 'pal_chi2pix_no_subhalo', '{:.2f}')} & "
                 f"{' / '.join(mass(dep) for dep in DEPTH)} & {pair(r, 'pal_offset', '{:.2f}' + chr(39) * 2)} & {pair(r, 'a_dchi2', '{:.0f}')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/pal.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
