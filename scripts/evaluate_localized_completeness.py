"""Completeness counting only localized detections (reported position within 2 pixels, 0.16", of the true subhalo) for
every family on the full c=60 population at its own 10% calibration (referee round 28, M2/M3): the grid scan, the
position-free scan, the free-form statistic, and the U-Nets trained on 30 000 and 500 000 images.

    python scripts/evaluate_localized_completeness.py      -> results/localized_completeness.json, paper/tables/loccomp.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from detector.dataset import load_population, normalize  # noqa: E402
from detector.unet import build_model  # noqa: E402

BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC = 0.16
T = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]


def scan_family(root):
    L = lambda p: [json.loads(l) for l in open(ROOT / root / p / "scan_results.jsonl")]  # noqa: E731
    t = float(np.quantile([r["delta_chi2"] for r in L("no_subhalo") if r["reliable_fit"]], 0.9))
    out = []
    for r in L("test_fixed60"):
        if r["reliable_fit"] and r["has_subhalo"]:
            s = T[r["index"]]["subhalo"]; det = r["delta_chi2"] >= t
            loc = det and r.get("best_x") is not None and np.hypot(r["best_x"] - s["x"], r["best_y"] - s["y"]) < LOC
            out.append((s["log10_M200"], det, loc))
    return out


@torch.no_grad()
def unet_family(ck, run):
    d = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    c = torch.load(ROOT / ck / "model_best.pt", map_location=d); m = build_model(c).to(d); m.load_state_dict(c["model_state"]); m.eval()
    thr = json.loads((ROOT / "results" / run / "results.json").read_text())["threshold_at_10pct_fpr"]
    imgs, truths, ps, n, _ = load_population(ROOT / "data/test_fixed60")
    x = torch.from_numpy(normalize(imgs, c["scale"])).unsqueeze(1); ctr = (n - 1) / 2; out = []
    for a in range(0, len(x), 128):
        p = torch.sigmoid(m(x[a:a + 128].to(d)))[:, 0].cpu().numpy()
        for k, pk in enumerate(p):
            s = truths[a + k].get("subhalo")
            if not s:
                continue
            det = pk.max() >= thr; py, px = np.unravel_index(np.argmax(pk), pk.shape)
            loc = det and np.hypot((px - ctr) * ps - s["x"], (py - ctr) * ps - s["y"]) < LOC
            out.append((s["log10_M200"], det, loc))
    return out


def per_bin(rows, which):
    return [float(np.mean([r[which] for r in rows if a <= r[0] < b])) for a, b in BINS]


def main():
    fams = {"A grid scan": scan_family("results/baseline_a_joint_full"), "A position-free scan": scan_family("results/baseline_a_joint_freepos_full"),
            "B free-form": scan_family("results/baseline_b_full/fitted"),
            "C U-Net, 30\\,000": unet_family("checkpoints/unet_30k_seed0", "detector_30k_seed0"),
            "C U-Net, $5\\times10^{5}$": unet_family("checkpoints/unet_30k_c60pub_seed0", "detector_30k_c60pub_seed0")}
    res = {k: {"all": per_bin(v, 1), "localized": per_bin(v, 2)} for k, v in fams.items()}
    (ROOT / "results/localized_completeness.json").write_text(json.dumps(res, indent=2))
    lines = [r"\begin{tabular}{@{}l" + "c" * 6 + "@{}}", r"\toprule",
             r" & \multicolumn{6}{c}{$c=60$ completeness (\%): all detections / localized only} \\",
             r"family & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" \\", r"\midrule"]
    for k, v in res.items():
        lines.append(f"{k} & " + " & ".join(f"{100 * a:.0f} / {100 * b:.0f}" for a, b in zip(v["all"], v["localized"])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/loccomp.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
