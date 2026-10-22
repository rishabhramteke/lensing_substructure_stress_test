#!/bin/zsh
# Final pass once every joint scan has landed: complete the basin control on the late roots, re-evaluate,
# regenerate every table and figure, and compile.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1
python scripts/repair_joint_basin.py --roots results/baseline_a_joint_full results/baseline_a_joint_psf015 results/baseline_a_joint_lenslight results/baseline_a_joint_lenslight2 results/baseline_a_joint_tidal results/baseline_a_joint_freepos --workers 6 >> results/repair_joint_basin.log 2>&1
cd scripts
python evaluate_joint_scan.py > ../results/eval_joint_scan_final.log 2>&1
python evaluate_lens_light.py > ../results/eval_lenslight_s1.log 2>&1
python evaluate_lens_light.py --suffix _s8 > ../results/eval_lenslight_s8.log 2>&1
python evaluate_roc.py > ../results/eval_roc.log 2>&1
python evaluate_matched_denominators.py > ../results/eval_matched.log 2>&1
python evaluate_intermediate_concentration.py > ../results/eval_c30.log 2>&1
python completeness_vs_signal.py > ../results/eval_signal.log 2>&1
python reweight_by_mass_function.py > ../results/eval_reweight.log 2>&1
python evaluate_depth.py > ../results/eval_depth.log 2>&1
python plot_localization_accuracy.py --a-root ../results/baseline_a_joint_freepos --unet-ckpt ../checkpoints/unet_30k_seed0/model_best.pt --unet-results ../results/detector_30k_seed0/results.json --b-root ../results/baseline_b_full/fitted --a-label "Family A (joint scan, position free)" --tag abc > ../results/eval_localization.log 2>&1
python -c "import make_paper_tables as t; t.decoy(); t.joint(); t.lenslight(); t.ledger(); t.mass_mechanism()" > ../results/eval_tables.log 2>&1
python make_simulator_table.py >> ../results/eval_tables.log 2>&1
cd ..
python scripts/make_paper_figures.py fig2 fig3 fig4 fig5 fig6 fig12 fig13 > results/eval_figures.log 2>&1
cd paper && pdflatex -interaction=nonstopmode -halt-on-error -jobname=Ramteke_2026_substructure_stress_test main.tex > build.log 2>&1 && bibtex Ramteke_2026_substructure_stress_test > /dev/null 2>&1 && pdflatex -interaction=nonstopmode -halt-on-error -jobname=Ramteke_2026_substructure_stress_test main.tex > /dev/null 2>&1 && pdflatex -interaction=nonstopmode -halt-on-error -jobname=Ramteke_2026_substructure_stress_test main.tex > build.log 2>&1
echo "FINALIZE DONE $(date)"
