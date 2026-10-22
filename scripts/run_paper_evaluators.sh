#!/bin/zsh
# Every evaluator the revised paper's tables and figures depend on, in dependency order, once the
# basin repair has rewritten the core joint populations. Idempotent; re-run after the last scans land.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
until grep -q "joint_full/test_fixed60" results/repair_joint_basin.log; do sleep 60; done
echo "=== core repaired $(date)"
python scripts/evaluate_joint_scan.py > results/eval_joint_scan.log 2>&1
python scripts/evaluate_roc.py > results/eval_roc.log 2>&1
python scripts/evaluate_matched_denominators.py > results/eval_matched.log 2>&1
python scripts/completeness_vs_signal.py > results/eval_signal.log 2>&1
python scripts/reweight_by_mass_function.py > results/eval_reweight.log 2>&1
python scripts/plot_localization_accuracy.py --a-root results/baseline_a_joint_full --unet-ckpt checkpoints/unet_30k_seed0/model_best.pt \
    --unet-results results/detector_30k_seed0/results.json --b-root results/baseline_b_full/fitted --a-label "Family A (joint parametric scan)" --tag abc > results/eval_localization.log 2>&1
until grep -q "joint_shallow/test_fixed60" results/repair_joint_basin.log; do sleep 60; done
python scripts/evaluate_depth.py > results/eval_depth.log 2>&1
python -c "import sys; sys.path.insert(0,'scripts'); import make_paper_tables as t; t.decoy(); t.joint(); t.ledger(); t.mass_mechanism()" > results/eval_tables.log 2>&1
python scripts/make_paper_figures.py fig2 fig3 fig4 fig5 fig6 fig12 fig13 > results/eval_figures.log 2>&1
echo "=== evaluators done $(date)"
