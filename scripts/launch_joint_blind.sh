#!/bin/zsh
# Data-only initialisation for the JOINT scan (referee round 20): the single smooth start is fit._blind_init_vec
# instead of the near-truth one, on the 300-lens seed-0 subsamples of the clean, c=60 and a4/thetaE=0.03 populations.
# No wall-clock guard (every lens runs its full function-evaluation budget). Compare against the same lenses in
# results/baseline_a_joint_full with scripts/evaluate_joint_blind.py. Resumable.
#   caffeinate -i zsh scripts/launch_joint_blind.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-6}
R=results/baseline_a_joint_blind
for pop in no_subhalo test_fixed60 multipole_m4_a3; do
  out=$R/$pop
  if [ -f $out/manifest.json ]; then echo "skip $out (done)"; continue; fi
  mkdir -p $R; echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n 300 --n-subsample 300 --seed 0 --concentration 15 \
      --workers $W --blind --lens-timeout 1e9 --out $out > $out.log 2>&1
done
echo "ALL DONE $(date)"
