#!/bin/zsh
# Joint (macro re-fitted) Family A scan on the FULL 1000-lens Tier-0 populations, in order of
# importance for the paper. Resumable: a population with a manifest is skipped.
#   caffeinate -i zsh scripts/launch_joint_full.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-9}
for pop in no_subhalo test_fixed60 test_fixed15 multipole_m4_a3 multipole_m4_a1 test_fixed30 multipole_m3_a3; do
  out=results/baseline_a_joint_full/$pop
  if [ -f $out/manifest.json ]; then echo "skip $pop (done)"; continue; fi
  echo "=== $pop  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n 1000 --n-subsample 1000 --seed 0 --concentration 15 \
      --workers $W --out $out > results/baseline_a_joint_full/$pop.log 2>&1
done
echo "ALL DONE $(date)"
