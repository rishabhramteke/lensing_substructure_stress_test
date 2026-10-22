"""The gate of Sect. 2.3 on the full 1 000-lens joint-scan populations, as a table (referee round 22): lenses whose
first smooth start fails chi^2/dof < 10, how many are retained because a smooth re-fit (from a joint solution with the
perturber removed, or from the seven-start re-fit) reaches the gate, how many because only the joint model does, and
how many remain excluded, with the excluded subhalo-bearing lenses below 10^10 Msun.

    python scripts/make_gate_table.py        -> paper/tables/gate.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POPS = [("no_subhalo", "clean control"), ("test_fixed60", "$c=60$"), ("test_fixed30", "$c=30$"), ("test_fixed15", "$c=15$"),
        ("multipole_m4_a1", "$m=4$, $a_4/\\thetaE=0.01$"), ("multipole_m4_a3", "$m=4$, $a_4/\\thetaE=0.03$"), ("multipole_m3_a3", "$m=3$, $a_3/\\thetaE=0.03$")]


def main():
    G = json.loads((ROOT / "results/joint_scan.json").read_text())["gate"]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"population & first start & smooth re-fit & joint only & excluded & excl.\ $<10^{10}\,\Msun$ \\", r"\midrule"]
    for pop, lab in POPS:
        g = G[pop]
        below = "--"
        recs = [json.loads(l) for l in open(ROOT / f"results/baseline_a_joint_full/{pop}/scan_results.jsonl")]
        if any(r["has_subhalo"] for r in recs):
            below = str(sum(1 for r in recs if not r["reliable_fit"] and r["has_subhalo"] and r["log10_M200_true"] < 10.0))
        lines.append(f"{lab} & {g['gated_initial']} & {g['rescued_basin']} & {g['rescued_joint']} & {g['gated_final']} & {below} \\\\")
        assert g["gated_initial"] - g["rescued_basin"] - g["rescued_joint"] == g["gated_final"], pop
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/gate.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
