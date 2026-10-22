#!/bin/zsh
# Referee round 28, Family A (one job at a time, 12 workers): high-mass rescue of the 70 gated c=60 lenses; the render
# test with the fitter supersampled 3x; the PSF-error-only test on 200 clean lenses; the rest of the 10 000-lens clean
# population for the low-FPR calibration.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=12
j() { local out=$1; shift; mkdir -p $(dirname $out); [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py "$@" --concentration 15 --workers $W --lens-timeout 1e9 --out $out > $out.log 2>&1; echo "done $out exit $? $(date)"; }
j $R/baseline_a_joint_rescue_highmass/test_fixed60 --population test_fixed60 --seed 0 --indices $(cat $R/gated_c60_indices.txt) --mass-init 11 --restarts 7
FIT_SUPERSAMPLING=3 j $R/baseline_a_joint_super_fit3/no_subhalo --population super_no_subhalo --n 100 --n-subsample 300 --seed 1
FIT_SUPERSAMPLING=3 j $R/baseline_a_joint_super_fit3/test_fixed60 --population super_fixed60 --n 100 --n-subsample 300 --seed 0
j $R/baseline_a_joint_psferr200/no_subhalo --population psferr_no_subhalo --n 200 --n-subsample 300 --seed 1
j $R/baseline_a_joint_bigclean_rest/no_subhalo --population no_subhalo_10k --seed 11 --indices $(cat $R/bigclean_rest_indices.txt)
echo "ROUND28 CPU DONE $(date)"
