#!/bin/zsh
# Round 29, M1d: a practical joint scan (data-only start, free lens centre, free perturber position), 300 lenses each.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=${W:-10}
j() { local out=$1; shift; mkdir -p $(dirname $out); [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py "$@" --concentration 15 --workers $W --lens-timeout 1e9 --out $out > $out.log 2>&1; echo "done $out exit $? $(date)"; }
j $R/baseline_a_joint_practical/no_subhalo --population no_subhalo --n 300 --n-subsample 300 --seed 1 --blind --free-centre --free-position
j $R/baseline_a_joint_practical/test_fixed60 --population test_fixed60 --n 300 --n-subsample 300 --seed 0 --blind --free-centre --free-position
j $R/baseline_a_joint_practical/multipole_m4_a3 --population multipole_m4_a3 --n 300 --n-subsample 300 --seed 3 --blind --free-centre --free-position
echo "PRACTICAL DONE $(date)"
