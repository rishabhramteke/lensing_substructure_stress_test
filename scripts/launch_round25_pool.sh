#!/bin/zsh
# Referee round 25 Family-A controls, as a pool (start several copies; mkdir locks): a tight perturber-mass prior
# (log10 M200 <= 10.5), a 1" position box for the position-free scan, and a 5x function-evaluation cap on a subsample.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-5}; R=results
JOBS=(
  "multipole_m4_a3 300 300 3 $R/baseline_a_joint_mmax105/multipole_m4_a3 --mass-max 10.5"
  "no_subhalo 300 300 1 $R/baseline_a_joint_mmax105/no_subhalo --mass-max 10.5"
  "test_fixed60 300 300 0 $R/baseline_a_joint_mmax105/test_fixed60 --mass-max 10.5"
  "multipole_m4_a3 300 300 3 $R/baseline_a_joint_box1/multipole_m4_a3 --free-position --position-box 1.0"
  "no_subhalo 300 300 1 $R/baseline_a_joint_box1/no_subhalo --free-position --position-box 1.0"
  "test_fixed60 300 300 0 $R/baseline_a_joint_box1/test_fixed60 --free-position --position-box 1.0"
  "super_no_subhalo 300 300 1 $R/baseline_a_joint_super/no_subhalo"
  "super_fixed60 300 300 0 $R/baseline_a_joint_super/test_fixed60"
  "super_multipole_m4_a3 300 300 3 $R/baseline_a_joint_super/multipole_m4_a3"
  "no_subhalo 100 300 1 $R/baseline_a_joint_cap200/no_subhalo --nfev-mult 200"
  "test_fixed60 100 300 0 $R/baseline_a_joint_cap200/test_fixed60 --nfev-mult 200"
)
for j in "${JOBS[@]}"; do
  set -- ${=j}; pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  [ -f $out/manifest.json ] && continue
  mkdir -p $(dirname $out); mkdir $out.lock 2>/dev/null || continue
  echo "=== $out  $(date)  (pid $$)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
done
echo "POOL MEMBER DONE $(date) (pid $$)"
