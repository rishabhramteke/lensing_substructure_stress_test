"""One place for which runs the paper's tables and figures are built from (revision of 2026-09-15).

Family A is the JOINT scan (macro-model re-fitted with the perturber at every grid cell); the
frozen fit-then-scan of the earlier drafts is kept as the fast-shortcut appendix. The U-Net is the
30 000-image budget with four seeds; the 8 000-image budget is the scale comparison. Every
evaluator imports these names so a root or a seed set is changed in exactly one file.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Family A roots (scan_results.jsonl per population)
A_JOINT_FULL = "results/baseline_a_joint_full"          # 1 000-lens Tier-0 populations, joint scan
A_FROZEN_FULL = "results/baseline_a_full"               # same populations, frozen macro-model (appendix)
A_FROZEN_SUB = "results/baseline_a"                     # 300-lens seed-0 subsamples, frozen (appendix, template control)
A_JOINT_LOS, A_FROZEN_LOS = "results/baseline_a_joint_los", "results/baseline_a_los"
A_JOINT_SHALLOW, A_FROZEN_SHALLOW = "results/baseline_a_joint_shallow", "results/baseline_a_shallow"
A_JOINT_DECOY = "results/baseline_a_joint_decoy"
A_JOINT_PSF, A_FROZEN_PSF = "results/baseline_a_joint_psf015", "results/baseline_a_psf015"
A_JOINT_LL, A_JOINT_LL2 = "results/baseline_a_joint_lenslight", "results/baseline_a_joint_lenslight2"
A_FROZEN_LL, A_FROZEN_LL2 = "results/baseline_a_lenslight", "results/baseline_a_lenslight2"
A_JOINT_TIDAL, A_FROZEN_TIDAL = "results/baseline_a_joint_tidal", "results/baseline_a_tidal"
# referee round 19 (2026-09-18)
A_JOINT_MP4, A_JOINT_MP34 = "results/baseline_a_joint_mp4", "results/baseline_a_joint_mp34"   # joint scan, free m=4 / m=3+4 macro-model, 300-lens subsamples
A_JOINT_CM = "results/baseline_a_joint_cm"            # c-M template on the c-M-drawn population (mass bias, realistic pairing)
A_JOINT_SHALLOW600 = "results/baseline_a_joint_shallow600"   # intermediate depth, 600 s
A_JOINT_RESCUE = "results/baseline_a_joint_rescue"    # multi-start re-fits of the gated core lenses (merged into A_JOINT_FULL)
A_ABS = 20.0        # absolute Delta chi^2 threshold quoted for Family A: Delta ln L ~ 10, the Delta ln Z > 10 proxy
A_ABS_STRICT = 100.0

# ---- Family B
B_FULL = "results/baseline_b_full/fitted"

# ---- U-Net seed sets: run name (results/<run>/results.json) -> checkpoint dir
UNET_SETS = {
    "30k": {"detector_30k_seed0": "checkpoints/unet_30k_seed0", "detector_30k_seed1": "checkpoints/unet_30k_seed1",
            "detector_30k_seed2": "checkpoints/unet_30k_seed2", "detector_30k_seed3": "checkpoints/unet_30k_seed3"},
    "8k": {"detector_v0_best": "checkpoints/unet_v0", "detector_v1_seed1": "checkpoints/unet_v1_seed1",
           "detector_v2_seed2": "checkpoints/unet_v2_seed2", "detector_v3_seed3": "checkpoints/unet_v3_seed3"},
    # trained on the combined realism arm (0.15" PSF, 600 s, lens light), referee round 21
    "realistic": {"detector_30k_realistic_seed0": "checkpoints/unet_30k_realistic_seed0", "detector_30k_realistic_seed1": "checkpoints/unet_30k_realistic_seed1"},
}
UNET_SET = os.environ.get("UNET_SET", "30k")
UNET_AGG = {"30k": "results/aggregate_30k", "8k": "results/aggregate"}[UNET_SET]
UNET_SUFFIX = {"30k": "", "8k": "_8k"}[UNET_SET]     # appended to output files when the secondary set is evaluated
UNET_SHALLOW = {"detector_30k_shallow_seed0": "checkpoints/unet_30k_shallow_seed0", "detector_30k_shallow_seed1": "checkpoints/unet_30k_shallow_seed1"}
UNET_C15 = {"detector_30k_c15_seed0": "checkpoints/unet_30k_c15_seed0", "detector_30k_c15_seed1": "checkpoints/unet_30k_c15_seed1"}   # trained at c=15 (round 19)
UNET_CM = {"detector_30k_cm_seed0": "checkpoints/unet_30k_cm_seed0", "detector_30k_cm_seed1": "checkpoints/unet_30k_cm_seed1"}       # trained on the Dutton & Maccio relation
UNET_PSF = {"detector_30k_psf015_seed0": "checkpoints/unet_30k_psf015_seed0", "detector_30k_psf015_seed1": "checkpoints/unet_30k_psf015_seed1"}


def unet_seeds(available_only=True):
    """The primary seed set, restricted to checkpoints that exist and have been evaluated."""
    d = UNET_SETS[UNET_SET]
    if not available_only:
        return d
    return {run: ck for run, ck in d.items() if (ROOT / ck / "model_best.pt").exists() and (ROOT / "results" / run / "results.json").exists()}


BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC_ARCSEC = 0.16   # 2 pixels


def wilson(k, n, z=1.0):
    import numpy as np
    if not n:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_scan(root, pop, require_manifest=False):
    """Records of one population's scan, or None. A run still writing has no manifest yet; by default
    partial output is returned (records are flushed one per line) so evaluators can be tested early."""
    p = ROOT / root / pop / "scan_results.jsonl"
    if not p.exists() or (require_manifest and not (ROOT / root / pop / "manifest.json").exists()):
        return None
    recs = []
    for l in open(p):
        try:
            recs.append(json.loads(l))
        except Exception:
            continue
    return recs or None
