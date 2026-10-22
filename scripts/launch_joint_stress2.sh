#!/bin/zsh
# Second driver for the stress populations: waits for the full-population m=3 run to free its cores, then
# takes the remaining populations in REVERSE order of launch_joint_stress.sh (both skip finished ones), so the
# two drivers meet in the middle with at most one duplicated population.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
until [ -f results/baseline_a_joint_full/multipole_m3_a3/manifest.json ]; do sleep 120; done
W=${WORKERS:-8}
run() {  # run <population> <n> <n_subsample> <seed> <out> [extra args]
  pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  if [ -f $out/manifest.json ] || [ -f $out/scan_results.jsonl ]; then echo "skip $out (done or in progress)"; return; fi
  mkdir -p $(dirname $out)
  echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --out $out "$@" > $out.log 2>&1
}
run test_tidal_c30 200 300 0 results/baseline_a_joint_tidal/test_tidal_c30
run test_tidal_c15 200 300 0 results/baseline_a_joint_tidal/test_tidal_c15
run lenslight_multipole_m4_a3  200 300 3 results/baseline_a_joint_lenslight2/lenslight_multipole_m4_a3 --lens-light-components 2
run lenslight_fixed60          200 300 0 results/baseline_a_joint_lenslight2/lenslight_fixed60         --lens-light-components 2
run lenslight_no_subhalo       200 300 1 results/baseline_a_joint_lenslight2/lenslight_no_subhalo      --lens-light-components 2
run lenslight_multipole_m4_a3  200 300 3 results/baseline_a_joint_lenslight/lenslight_multipole_m4_a3  --lens-light-components 1
run lenslight_fixed60          200 300 0 results/baseline_a_joint_lenslight/lenslight_fixed60          --lens-light-components 1
echo "ALL DONE $(date)"
