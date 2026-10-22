#!/bin/zsh
# Every evaluator whose table or figure depends on the re-gated joint scans (referee round 19), in dependency order,
# then the number printer. Idempotent; re-run after merge_rescue.py + regate_joint.py and after the last scan lands.
#   zsh scripts/reevaluate_round19.sh            (about 20-30 min; the U-Net evaluators use the GPU)
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
run() { # run <script> <logtag> [args...]
  tag=$2; echo "=== $1 $tag $(date)"; python scripts/$1 ${@:3} > results/eval_r19_$(basename $1 .py)$tag.log 2>&1 || echo "!!! $1 $tag failed (see results/eval_r19_$(basename $1 .py)$tag.log)"; }
echo "=== refresh B $(date)"; zsh scripts/refresh_family_b.sh > results/eval_r19_refresh_b.log 2>&1
run evaluate_joint_scan.py ''
run evaluate_full_populations.py ''
run evaluate_roc.py ''
run evaluate_matched_denominators.py ''
run evaluate_intermediate_concentration.py ''
run evaluate_concentration_selection.py ''
run completeness_vs_signal.py ''
run reweight_by_mass_function.py ''
run evaluate_lens_light.py ''
run evaluate_lens_light.py _s8 --suffix _s8
run evaluate_depth.py ''
run evaluate_multipole_orders_joint.py ''
run evaluate_unet_lowc.py ''
run evaluate_joint_blind.py ''
[ -f scripts/evaluate_guard_completion.py ] && run evaluate_guard_completion.py ''
run make_gate_table.py ''
run evaluate_family_b_arms.py ''
run make_unet_objectives_table.py ''
run evaluate_aperture_scan.py ''
run evaluate_localization_matched.py ''
run evidence_laplace.py '' --workers 4
run los_effective_mass.py '' --workers 4
run round23_analyses.py ''
run evaluate_new_arms.py ''
run evaluate_family_b_arms.py _r25
run evaluate_decoy_family_b.py ''
run make_template_table.py ''
run make_unet_objectives_table.py _r25
run make_gate_table.py _r25
run evaluate_family_bgi.py ''
run evaluate_fpr_0p1.py ''
echo "=== tables $(date)"; python -c "import sys; sys.path.insert(0,'scripts'); import make_paper_tables as t; t.joint(); t.lenslight(); t.ledger(); t.ledger_a(); t.mass_mechanism()" > results/eval_r19_tables.log 2>&1 || echo "!!! tables failed"
python scripts/make_simulator_table.py >> results/eval_r19_tables.log 2>&1
echo "=== localization $(date)"; (cd scripts && python plot_localization_accuracy.py --a-root ../results/baseline_a_joint_freepos_full --unet-ckpt ../checkpoints/unet_30k_seed0/model_best.pt --unet-results ../results/detector_30k_seed0/results.json --b-root ../results/baseline_b_full/fitted --a-label "Family A (joint scan, position free)" --tag abc > ../results/eval_r19_localization.log 2>&1) || echo "!!! localization failed"
echo "=== figures $(date)"; python scripts/make_paper_figures.py fig1 fig2 fig3 fig4 fig5 fig8 fig11 fig12 fig13 > results/eval_r19_figures.log 2>&1 || echo "!!! figures failed"
echo "=== numbers $(date)"; (cd scripts && python print_paper_numbers.py > ../results/paper_numbers_round19.txt 2>&1) || echo "!!! numbers failed"
echo "REEVALUATE DONE $(date)"
