"""Every family at 1% and 0.1% false-positive rate on large clean controls (referee round 27, M5): Family A, Family B and the U-Nets on the 1 000-lens control plus the full 10 000-lens population. The threshold is bootstrapped over
clean lenses and the c=60 completeness carries the threshold uncertainty (16th-84th percentiles).

    python scripts/evaluate_lowfpr.py      -> results/lowfpr.json, paper/tables/lowfpr.tex
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
RNG = np.random.default_rng(0)


def scan(root, pop):
    return [r for r in (json.loads(l) for l in open(ROOT / root / pop / "scan_results.jsonl")) if r["reliable_fit"]]


@torch.no_grad()
def unet(ck, pop):
    d = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    c = torch.load(ROOT / ck / "model_best.pt", map_location=d); m = build_model(c).to(d); m.load_state_dict(c["model_state"]); m.eval()
    imgs, truths, *_ = load_population(ROOT / "data" / pop)
    x = torch.from_numpy(normalize(imgs, c["scale"])).unsqueeze(1)
    return np.concatenate([torch.sigmoid(m(x[a:a + 256].to(d))).amax(dim=(1, 2, 3)).cpu().numpy() for a in range(0, len(x), 256)]), truths


def summarize(clean, sub, mp):
    clean = np.asarray(clean); out = {"n_clean": int(len(clean))}
    for lab, q in (("fpr1", 0.99), ("fpr0p1", 0.999)):
        t = float(np.quantile(clean, q)); bt = np.array([np.quantile(RNG.choice(clean, len(clean)), q) for _ in range(500)])
        comp = lambda thr: [float(np.mean([s >= thr for s, m in sub if a <= m < b])) for a, b in BINS]  # noqa: E731
        bc = np.array([comp(x) for x in bt[:200]])
        out[lab] = {"threshold": t, "threshold_68": [float(np.percentile(bt, 16)), float(np.percentile(bt, 84))], "n_tail": int((clean > t).sum()),
                    "completeness": comp(t), "lo": np.percentile(bc, 16, axis=0).tolist(), "hi": np.percentile(bc, 84, axis=0).tolist(),
                    "fpr_multipole": float(np.mean(np.asarray(mp) >= t))}
    return out


def main():
    res = {}
    from evaluate_fpr_0p1 import gate_dof, big_clean
    pool = [r for r in big_clean() if r["reliable_fit"]] + scan("results/baseline_a_joint_full", "no_subhalo")
    c60 = [r for r in scan("results/baseline_a_joint_full", "test_fixed60") if r["has_subhalo"]]
    mp = scan("results/baseline_a_joint_full", "multipole_m4_a3")
    for lab, gmax in (("A", 10.0), ("A_strict", 2.0)):  # the standard gate, and the strict one applied to every population alike
        res[lab] = summarize([r["delta_chi2"] for r in pool if gate_dof(r) < gmax], [(r["delta_chi2"], r["log10_M200_true"]) for r in c60 if gate_dof(r) < gmax],
                             [r["delta_chi2"] for r in mp if gate_dof(r) < gmax])
    B = scan("results/baseline_b_bigclean/fitted", "no_subhalo_10k") + scan("results/baseline_b_full/fitted", "no_subhalo")
    res["B"] = summarize([r["delta_chi2"] for r in B], [(r["delta_chi2"], r["log10_M200_true"]) for r in scan("results/baseline_b_full/fitted", "test_fixed60") if r["has_subhalo"]],
                         [r["delta_chi2"] for r in scan("results/baseline_b_full/fitted", "multipole_m4_a3")])
    for lab, ck in (("C30k", "checkpoints/unet_30k_seed0"), ("C500k", "checkpoints/unet_30k_c60pub_seed0")):
        c1, _ = unet(ck, "no_subhalo_10k"); c2, _ = unet(ck, "no_subhalo"); s6, T = unet(ck, "test_fixed60"); sm, _ = unet(ck, "multipole_m4_a3")
        res[lab] = summarize(np.concatenate([c1, c2]), [(s6[i], t["subhalo"]["log10_M200"]) for i, t in enumerate(T) if t.get("subhalo")], sm)
    (ROOT / "results/lowfpr.json").write_text(json.dumps(res, indent=2))
    names = {"A": "A joint scan", "A_strict": "A, strict cut", "B": "B free-form", "C30k": "C U-Net, 30\\,000", "C500k": "C U-Net, 500\\,000"}
    lines = [r"\begin{tabular}{@{}lr" + "c" * 6 + "c@{}}", r"\toprule",
             r" & & \multicolumn{6}{c}{$c=60$ completeness (\%), $\log_{10}(M_{200}/\Msun)$} & mult. \\",
             r"family & $n_{\rm clean}$ & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" & FPR \\", r"\midrule"]
    for lab, key in (("1\\% FPR", "fpr1"), ("0.1\\% FPR", "fpr0p1")):
        lines.append(r"\multicolumn{9}{@{}l}{\emph{" + lab + r"}} \\")
        for f in ("A", "A_strict", "B", "C30k", "C500k"):
            d = res[f][key]
            cells = [f"{100 * c:.0f}" + (f"$_{{{100 * l:.0f}}}^{{{100 * h:.0f}}}$" if abs(h - l) >= 0.02 else "") for c, l, h in zip(d["completeness"], d["lo"], d["hi"])]
            lines.append(f"\\quad {names[f]} & {res[f]['n_clean']} & " + " & ".join(cells) + f" & {100 * d['fpr_multipole']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/lowfpr.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for f in res:
        print(f, {k: (round(res[f][k]["threshold"], 4), [round(x, 4) for x in res[f][k]["threshold_68"]], res[f][k]["n_tail"]) for k in ("fpr1", "fpr0p1")})


if __name__ == "__main__":
    main()
