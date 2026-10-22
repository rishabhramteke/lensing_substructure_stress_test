#!/bin/zsh
# Joint scan with the perturber position free (referee, second round): the 24 grid nodes become starting points of a
# continuous (x, y) optimisation, so localization is a measurement. Same 300-lens subsamples as the frozen seed-0 scans.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-6}
run() { pop=$1; seed=$2; out=results/baseline_a_joint_freepos/$3
  if [ -f $out/manifest.json ]; then echo "skip $out"; return; fi
  mkdir -p results/baseline_a_joint_freepos; echo "=== $out $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n 300 --n-subsample 300 --seed $seed --concentration 15 --workers $W --free-position --out $out > $out.log 2>&1
}
run no_subhalo 1 no_subhalo
run test_fixed60 0 test_fixed60
run multipole_m4_a3 3 multipole_m4_a3
run test_fixed15 0 test_fixed15
echo "ALL DONE $(date)"
