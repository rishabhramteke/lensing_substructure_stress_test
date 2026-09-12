"""Materialise the non-physical decoy images used by noise_decoy_control.py as populations on
disk, so that Family B (run_family_b.py) can be scored on exactly the same decoys as the U-Net
and Family A (referee round 5, 2026-09-12). Same lens selection (rng seed), same decoy RNG
stream and loop order as noise_decoy_control.py, so the images are bit-identical.

    python scripts/make_decoy_populations.py --shape gaussian --seed 42
    python scripts/make_decoy_populations.py --shape dipole   --seed 42
-> data/decoy_<shape>_s<seed>_a<amp>/ for amp in 3, 6, 10
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from noise_decoy_control import inject_decoy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["gaussian", "dipole"], default="gaussian")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--amps", type=float, nargs="+", default=[3.0, 6.0, 10.0])
    args = p.parse_args()
    src = ROOT / "data/no_subhalo"
    manifest = json.loads((src / "manifest.json").read_text())
    truths = [json.loads(l) for l in open(src / "truth.jsonl")]
    images = np.load(src / "images_full_noisy.npy"); control = np.load(src / "images_control_noiseless.npy")
    num_pix = manifest["image_shape"][0]; kwargs_band = manifest["kwargs_band"]; ps = kwargs_band["pixel_scale"]
    rng = np.random.default_rng(args.seed); idx = rng.choice(len(truths), size=args.n, replace=False)
    rng_decoy = np.random.default_rng(args.seed + 1)
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band, kwargs_model={"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    for amp in args.amps:
        out = ROOT / "data" / f"decoy_{args.shape}_s{args.seed}_a{int(amp)}"; out.mkdir(parents=True, exist_ok=True)
        imgs, trs = [], []
        for i in idx:
            truth = dict(truths[i]); theta_E = truth["lens_macro"]["theta_E"]
            noise_std = float(np.median(sim.estimate_noise(images[i])))
            dimg, meta = inject_decoy(images[i], ps, num_pix, theta_E, noise_std, amp, rng_decoy, args.shape)
            truth["decoy"] = meta; truth["_source_index"] = int(i)
            imgs.append(dimg.astype(np.float32)); trs.append(truth)
        np.save(out / "images_full_noisy.npy", np.stack(imgs))
        np.save(out / "images_control_noiseless.npy", control[idx].astype(np.float32))
        np.save(out / "images_full_noiseless.npy", control[idx].astype(np.float32))
        with open(out / "truth.jsonl", "w") as f:
            for t in trs:
                f.write(json.dumps(t) + "\n")
        m = dict(manifest); m.update({"n": len(imgs), "decoy": {"shape": args.shape, "amp_sigma": amp, "seed": args.seed, "source_population": "no_subhalo", "note": "bit-identical to noise_decoy_control.py's decoys"}})
        (out / "manifest.json").write_text(json.dumps(m, indent=2, default=str))
        print("wrote", out, len(imgs))


if __name__ == "__main__":
    main()
