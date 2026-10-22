"""The published PyAutoLens subhalo detection pipeline run unchanged on our simulated lenses.

The SLaM pipeline and subhalo search are executed verbatim from the PyAutoLens workspace at the installed library's
release tag (scripts/imaging/features/advanced/subhalo/detect/start_here.py): SOURCE LP, SOURCE PIX 1 and 2, LIGHT LP,
MASS TOTAL, the 2x2 grid of Nautilus searches over the subhalo centre (NFWMCRLudlowSph, M200 log-uniform 1e6-1e11) and
the refine search. Only the dataset block is replaced: our 64x64 image at 0.08", its noise map and the simulator's PSF,
real-space PSF convolution, a circular mask of 2.4" (the field half-width is 2.56"), and the grid half-width set to the mask radius. The detection
statistic is the pipeline's own: ln Z(refine) - ln Z(mass total).

    python scripts/run_pyautolens_slam_subhalo.py --population test_fixed60 --only 12 --out results/pyautolens_slam/test_fixed60
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("AUTOLENS_WORKSPACE", ROOT.parents[1] / "external/autolens_workspace"))
PIPE = WS / "scripts/imaging/features/advanced/subhalo/detect/start_here.py"
MASK_RADIUS = 2.4


def pipeline_sections():
    """(header and pipeline functions, driver from settings to refine) of the published script, unmodified."""
    src = PIPE.read_text().splitlines(keepends=True)
    i_data = next(k for k, l in enumerate(src) if l.startswith('dataset_name = "dark_matter_subhalo"'))
    i_set = next(k for k, l in enumerate(src) if l.startswith("settings_search = af.SettingsSearch("))
    i_end = next(k for k, l in enumerate(src) if l.startswith("result_with_subhalo = subhalo_refine("))
    i_end = next(k for k in range(i_end, len(src)) if src[k].startswith(")")) + 1
    driver = "".join(src[i_set:i_end]).replace("grid_dimension_arcsec=3.0", f"grid_dimension_arcsec={MASK_RADIUS}")
    return "".join(src[:i_data]), driver


def separable_psf_convolution(al):
    """Real-space PSF convolution of the mapping matrix as two banded matrix products when the kernel is rank one
    (the simulator's Gaussian PSF: second singular value < 1e-18 of the first). Same zero-padded 'same' convolution
    as jax.scipy.signal.convolve, to 2e-15 relative, at a fifth of the cost; other kernels use the library's path."""
    import numpy as np_
    C = al.Convolver
    orig = C.convolved_mapping_matrix_via_real_space_from

    def toeplitz(w, n):
        c, T = len(w) // 2, np_.zeros((n, n))
        for i in range(n):
            for t in range(len(w)):
                if 0 <= i - t + c < n:
                    T[i, i - t + c] = w[t]
        return T

    def fast(self, mapping_matrix, mask, blurring_mapping_matrix=None, blurring_mask=None, jax_method="direct", xp=np_):
        k = np_.asarray(self.kernel.native.array)
        U, S, Vt = np_.linalg.svd(k)
        if self.convolve_over_sample_size > 1 or xp is np_ or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0 or S[1] > 1e-12 * S[0]:
            return orig(self, mapping_matrix, mask, blurring_mapping_matrix, blurring_mask, jax_method, xp)
        state = self.state_from(mask=mask)
        mm = self.mapping_matrix_native_from(mapping_matrix=mapping_matrix, mask=state.mask, blurring_mapping_matrix=blurring_mapping_matrix,
                                             blurring_mask=state.blurring_mask, xp=xp)
        ty, tx = toeplitz(U[:, 0] * np_.sqrt(S[0]), mm.shape[0]), toeplitz(Vt[0] * np_.sqrt(S[0]), mm.shape[1])
        out = xp.einsum("ij,jkc->ikc", ty, xp.einsum("kl,jlc->jkc", tx, mm))
        return out[state.mask.slim_to_native_tuple]

    C.convolved_mapping_matrix_via_real_space_from = fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--only", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multipole", action="store_true", help="add a free m=4 PowerLawMultipole, sharing the power law's centre, "
                    "Einstein radius and slope, to the total-mass model and every search after it (the post-Lange et al. practice)")
    ap.add_argument("--tag-suffix", default="", help="appended to the output tag, so a lens is re-run from scratch with fresh sampler draws")
    args = ap.parse_args()
    skip = ROOT / "results/pyautolens_slam/skip.txt"  # "population:index" lines dropped from a running queue
    if os.environ.get("PAL_QUEUE"):   # a named queue reads its own skip list
        skip = ROOT / f"results/pyautolens_slam/skip_{os.environ['PAL_QUEUE']}.txt"
    if skip.exists() and all(f"{args.population}:{i}" in skip.read_text().split() for i in args.only):
        print("skipped", args.population, args.only); return
    threads = ROOT / "results/pyautolens_slam/threads.txt"  # thread count for this lens, read before JAX/NumPy load
    if threads.exists():
        k = threads.read_text().strip()
        os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={k}"
        for v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = k
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(WS)  # the workspace's own config (priors, search settings); search output under its output/
    header, driver = pipeline_sections()
    if args.multipole:
        a_ = "                mass=mass,\n                shear=source_result_for_lens.model.galaxies.lens.shear,"
        assert header.count(a_) == 1, "mass_total model block not found"
        header = header.replace(a_, "                mass=mass,\n                multipole=_mp4(mass),\n                shear=source_result_for_lens.model.galaxies.lens.shear,")
    g = {"__name__": "pipeline"}
    def _mp4(mass):
        m_ = g["af"].Model(g["al"].mp.PowerLawMultipole); m_.m = 4
        m_.centre, m_.einstein_radius, m_.slope = mass.centre, mass.einstein_radius, mass.slope
        return m_
    g["_mp4"] = _mp4
    exec(compile(header, str(PIPE), "exec"), g)
    al, np = g["al"], g["np"]
    separable_psf_convolution(al)

    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
    from run_family_b import noise_map_for  # noqa: E402
    from run_family_b_gi import build_imaging_exact  # noqa: E402
    pop = args.data_root / args.population
    man = json.loads((pop / "manifest.json").read_text())
    kb, n = man["kwargs_band"], man["image_shape"][0]
    truths = [json.loads(l) for l in open(pop / "truth.jsonl")]
    noisy = pop / "images_full_noisy.npy"
    images = np.load(noisy if noisy.exists() else pop / "images_full_noiseless.npy", mmap_mode="r")
    out_file = args.out / "scan_results.jsonl"
    done = {json.loads(l)["index"] for l in open(out_file)} if out_file.exists() else set()

    for i in args.only:
        if i in done:
            continue
        t0, tr = time.time(), truths[i]
        data = np.asarray(images[i]); noise = noise_map_for(data, kb, n)
        dataset = build_imaging_exact(data, noise, kb, n)
        # real-space PSF convolution: XLA's multithreaded FFT thunk deadlocks in this jaxlib on macOS (same result)
        dataset = al.Imaging(data=dataset.data, noise_map=dataset.noise_map, psf=al.Convolver(kernel=dataset.psf.kernel, normalize=True, use_fft=False))
        mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=MASK_RADIUS)
        dataset = dataset.apply_mask(mask=mask)
        over = al.util.over_sample.over_sample_size_via_radial_bins_from(grid=dataset.grid, sub_size_list=[4, 2, 2], radial_list=[0.3, 0.6], centre_list=[(0.0, 0.0)])
        dataset = dataset.apply_over_sampling(over_sample_size_lp=over)
        ns = dict(g, dataset=dataset, dataset_name=f"{args.population}_{i:05d}{args.tag_suffix}", mask_radius=MASK_RADIUS)
        exec(compile(driver, str(PIPE), "exec"), ns)
        m, r = ns["mass_result"], ns["result_with_subhalo"]
        sub = r.instance.galaxies.subhalo.mass
        grid = ns["result_subhalo_grid_search"]
        s = tr.get("subhalo") or {}
        rec = {"index": int(i), "has_subhalo": bool(s), "log10_M200_true": s.get("log10_M200"), "x_true": s.get("x"), "y_true": s.get("y"),
               "has_multipole": tr.get("multipole") is not None,
               "lnZ_mass_total": float(m.samples.log_evidence), "lnZ_subhalo": float(r.samples.log_evidence),
               "delta_lnZ": float(r.samples.log_evidence - m.samples.log_evidence),
               "max_lnL_mass_total": float(m.samples.max_log_likelihood_sample.log_likelihood),
               "max_lnL_subhalo": float(r.samples.max_log_likelihood_sample.log_likelihood),
               "grid_lnZ": [float(v) for v in np.asarray(grid.log_evidences_native).ravel()] if hasattr(grid, "log_evidences_native") else None,
               # PyAutoLens centre is (y, x)
               "best_y": float(sub.centre[0]), "best_x": float(sub.centre[1]), "log10_M200_fit": float(np.log10(sub.mass_at_200)),
               "seconds": time.time() - t0}
        with open(out_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
