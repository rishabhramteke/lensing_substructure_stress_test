"""Every family at a 1% false-positive rate (referee round 27, M6), on the two operating points with 1 000-lens clean
controls: Tier 0 and the combined-realism arm (600 s, 0.15" PSF, two-component lens light). The threshold is the 99th
percentile of each family's statistic on its own clean control; completeness per mass bin on the c=60 and c=15
populations and the false-positive rate of the a4/thetaE=0.03 multipole population at it. The U-Net is the mean over
its seeds (four at Tier 0, the two networks trained on the realistic arm there).

    python scripts/evaluate_fpr1.py      -> results/fpr1.json, paper/tables/fpr1.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import build_model  # noqa: E402

BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
ARMS = {
    "tier0": {"A": "results/baseline_a_joint_full", "B": "results/baseline_b_full/fitted",
              "C": ["checkpoints/unet_30k_seed0", "checkpoints/unet_30k_seed1", "checkpoints/unet_30k_seed2", "checkpoints/unet_30k_seed3"],
              "pops": {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"},
              "data": {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}},
    "realistic": {"A": "results/baseline_a_joint_realistic_full", "B": "results/baseline_b_realistic_full/fitted",
                  "C": ["checkpoints/unet_30k_realistic_seed0", "checkpoints/unet_30k_realistic_seed1"],
                  "pops": {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"},
                  "pops_b": {"clean": "realistic_no_subhalo", "c60": "realistic_fixed60", "c15": "realistic_fixed15", "mp": "realistic_multipole_m4_a3"},
                  "data": {"clean": "realistic_no_subhalo", "c60": "realistic_fixed60", "c15": "realistic_fixed15", "mp": "realistic_multipole_m4_a3"}},
}


def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def scan(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [r for r in (json.loads(l) for l in open(p)) if r["reliable_fit"]] if p.exists() else None


@torch.no_grad()
def unet_scores(ck, pop):
    d = dev(); c = torch.load(ROOT / ck / "model_best.pt", map_location=d)
    m = build_model(c).to(d); m.load_state_dict(c["model_state"]); m.eval()
    ds = LensPatchDataset(ROOT / "data" / pop, scale=c["scale"])
    x = torch.from_numpy(ds.images).unsqueeze(1)
    s = np.concatenate([torch.sigmoid(m(x[a:a + 128].to(d))).amax(dim=(1, 2, 3)).cpu().numpy() for a in range(0, len(x), 128)])
    return s, ds.truths


def summarize(clean, sub, mp):
    """clean, mp: arrays of statistics; sub: dict pop -> list of (stat, log10 M)."""
    t = float(np.quantile(clean, 0.99))
    out = {"threshold": t, "n_clean": int(len(clean)), "fpr_multipole": float(np.mean(np.asarray(mp) >= t)) if mp is not None else None}
    for k, v in sub.items():
        out[k] = [float(np.mean([s >= t for s, m in v if a <= m < b])) if any(a <= m < b for _, m in v) else None for a, b in BINS]
    return out


def main():
    res = {}
    for arm, cfg in ARMS.items():
        R = {}
        for fam in ("A", "B"):
            P = cfg.get("pops_b", cfg["pops"]) if fam == "B" else cfg["pops"]
            C = scan(cfg[fam], P["clean"])
            if not C:
                continue
            sub = {}
            for key in ("c60", "c15"):
                X = scan(cfg[fam], P[key])
                if X:
                    sub[key] = [(r["delta_chi2"], r["log10_M200_true"]) for r in X if r["has_subhalo"]]
            M = scan(cfg[fam], P["mp"])
            R[fam] = summarize(np.array([r["delta_chi2"] for r in C]), sub, [r["delta_chi2"] for r in M] if M else None)
        seeds = []
        for ck in cfg["C"]:
            if not (ROOT / ck / "model_best.pt").exists():
                continue
            sc, _ = unet_scores(ck, cfg["data"]["clean"])
            sub = {}
            for key in ("c60", "c15"):
                s, T = unet_scores(ck, cfg["data"][key])
                sub[key] = [(s[i], t["subhalo"]["log10_M200"]) for i, t in enumerate(T) if t.get("subhalo")]
            sm, _ = unet_scores(ck, cfg["data"]["mp"])
            seeds.append(summarize(sc, sub, sm))
        if seeds:
            R["C"] = {"n_seeds": len(seeds), "fpr_multipole": float(np.mean([s["fpr_multipole"] for s in seeds])),
                      **{k: [float(np.mean([s[k][j] for s in seeds])) for j in range(len(BINS))] for k in ("c60", "c15")}}
        res[arm] = R
    (ROOT / "results/fpr1.json").write_text(json.dumps(res, indent=2))

    def row(lab, d):
        f = lambda v: "--" if v is None else f"{100 * v:.0f}"  # noqa: E731
        return f"{lab} & " + " & ".join(f(x) for x in d["c60"]) + " & " + " & ".join(f(x) for x in d["c15"][2:5]) + f" & {f(d['fpr_multipole'])} \\\\"
    lines = [r"\begin{tabular}{@{}l" + "c" * 6 + "ccc" + "c@{}}", r"\toprule",
             r" & \multicolumn{6}{c}{$c=60$ completeness, $\log_{10}(M_{200}/\Msun)$} & \multicolumn{3}{c}{$c=15$} & mult. \\",
             r"family & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" & 9--9.5 & 9.5--10 & 10--10.5 & FPR \\", r"\midrule"]
    names = {"A": "A joint scan", "B": "B free-form", "C": "C U-Net"}
    for arm, lab in (("realistic", "combined realism"),):
        lines.append(r"\multicolumn{11}{@{}l}{\emph{" + lab + r"}} \\")
        for fam in ("A", "B", "C"):
            if fam in res[arm]:
                lines.append(row(r"\quad " + names[fam], res[arm][fam]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/fpr1.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
