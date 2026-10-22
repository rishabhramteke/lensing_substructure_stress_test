#!/bin/zsh
# Rescue pass, narrowed (2026-09-18 15:50): only the still-gated core lenses BELOW 10^10 Msun (the bins every headline uses;
# the gated lenses above are companion-galaxy halos the paper keeps out of every headline statement). Six smooth starts, then
# the full joint scan and basin control. merge_rescue.py folds the results in; regate_joint.py re-gates. no_subhalo done earlier.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-3}
out=results/baseline_a_joint_rescue/test_fixed60; if [ ! -f $out/manifest.json ]; then mkdir -p $out; echo "=== $out $(date)"; python scripts/run_baseline_a_joint.py --population test_fixed60 --n-subsample 1000 --seed 0 --concentration 15 --workers $W --restarts 6 --indices 239,304,330,431,440,471,486,492,631,698,734,844,888,902,939 --out $out > $out.log 2>&1; fi
out=results/baseline_a_joint_rescue/test_fixed15; if [ ! -f $out/manifest.json ]; then mkdir -p $out; echo "=== $out $(date)"; python scripts/run_baseline_a_joint.py --population test_fixed15 --n-subsample 1000 --seed 0 --concentration 15 --workers $W --restarts 6 --indices 239,304,330,440,486,492,698,734,807,888,902,939 --out $out > $out.log 2>&1; fi
out=results/baseline_a_joint_rescue/test_fixed30; if [ ! -f $out/manifest.json ]; then mkdir -p $out; echo "=== $out $(date)"; python scripts/run_baseline_a_joint.py --population test_fixed30 --n-subsample 1000 --seed 0 --concentration 15 --workers $W --restarts 6 --indices 239,304,330,411,440,486,492,631,698,734,815,902,939 --out $out > $out.log 2>&1; fi
echo "ALL DONE $(date)"
