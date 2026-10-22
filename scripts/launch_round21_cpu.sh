#!/bin/zsh
# Referee round 21 CPU experiments, started once the full re-runs have finished. No wall-clock limit anywhere.
#   1. joint scan on the combined realism arm (0.15" PSF + 600 s + two-component lens light), 300-lens subsamples
#   2. cap-doubling control: 80 (n+1) evaluations per cell and basin fit, 300 clean and 300 c=60 lenses
#   3. joint scan on Tier 1 (COSMOS sources), 200 clean and 200 c=60 lenses
# Resumable per population (a finished population has a manifest). After it: regate_joint.py on the new roots.
#   caffeinate -i zsh scripts/launch_round21_cpu.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "waiting for the re-runs $(date)"
while pgrep -f "complete_truncated_joint|rerun_mismatched_global" >/dev/null; do sleep 300; done
until [ -f data/realistic_multipole_m4_a3/manifest.json ]; do sleep 60; done
echo "starting $(date)"
W=${WORKERS:-13}
run() {  # run <population> <n> <n_subsample> <seed> <out> [extra args]
  pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  if [ -f $out/manifest.json ]; then echo "skip $out"; return; fi
  mkdir -p $(dirname $out); echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
}
run realistic_no_subhalo      300 300 1 results/baseline_a_joint_realistic/no_subhalo      --lens-light-components 2
run realistic_fixed60         300 300 0 results/baseline_a_joint_realistic/test_fixed60    --lens-light-components 2
run realistic_multipole_m4_a3 300 300 3 results/baseline_a_joint_realistic/multipole_m4_a3 --lens-light-components 2
run realistic_fixed15         300 300 0 results/baseline_a_joint_realistic/test_fixed15    --lens-light-components 2
run no_subhalo   300 300 1 results/baseline_a_joint_cap80/no_subhalo   --nfev-mult 80
run test_fixed60 300 300 0 results/baseline_a_joint_cap80/test_fixed60 --nfev-mult 80
run tier1/no_subhalo   200 300 1 results/baseline_a_joint_tier1/no_subhalo
run tier1/test_fixed60 200 300 0 results/baseline_a_joint_tier1/test_fixed60
python scripts/regate_joint.py --roots results/baseline_a_joint_realistic results/baseline_a_joint_cap80 results/baseline_a_joint_tier1 --report
echo "ALL DONE $(date)"
