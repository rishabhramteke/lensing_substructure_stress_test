#!/bin/zsh
# Regenerate every Family-B summary the paper reads after a Family-B rerun (round 24: lenstronomy source gradient):
# the shared evaluator on each seed set and variant, the Family-B summaries, the seed aggregate and the decoy table.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
for root in baseline_b baseline_b_seed200 baseline_b_seed300; do
  for v in fitted oracle; do
    [ -d results/$root/$v ] || continue
    python scripts/evaluate_baseline_a.py --root results/$root/$v --out results/$root/$v/summary > results/$root/$v.summary.log 2>&1 || echo "!!! evaluate $root/$v"
  done
  python scripts/summarize_family_b.py --root results/$root --out results/$root/summary_family_b.json > results/$root/summary_family_b.log 2>&1 || echo "!!! summarize $root"
done
python scripts/aggregate_family_b_seeds.py --summaries results/baseline_b/summary_family_b.json results/baseline_b_seed200/summary_family_b.json results/baseline_b_seed300/summary_family_b.json --out results/baseline_b/aggregate_fitted_3seeds.json > results/baseline_b/aggregate.log 2>&1 || echo "!!! aggregate"
python scripts/evaluate_decoy_family_b.py > /dev/null 2>&1 || echo "!!! decoy"
echo "REFRESH B DONE"
