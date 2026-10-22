#!/bin/zsh
# Joint scan with a FREE m=4 term in the macro-model (referee round 19, 2026-09-18): the post-Lange+2025 practice,
# previously tested only on the frozen shortcut (Table B.4). Same 300-lens seed subsamples as results/baseline_a_mp34.
# Then the c=15 population (paired flip under the freer model) and, last, the line-of-sight arms with the
# perturber position free. Resumable.
#   caffeinate -i zsh scripts/launch_joint_multipole.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-7}
run() {  # run <population> <n> <n_subsample> <seed> <out> [extra args]
  pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  if [ -f $out/manifest.json ]; then echo "skip $out (done)"; return; fi
  mkdir -p $(dirname $out); echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --out $out "$@" > $out.log 2>&1
}
R=results/baseline_a_joint_mp4
run no_subhalo      300 300 1 $R/no_subhalo      --macro-multipole --macro-multipole-orders 4
run multipole_m4_a3 300 300 3 $R/multipole_m4_a3 --macro-multipole --macro-multipole-orders 4
run test_fixed60    300 300 0 $R/test_fixed60    --macro-multipole --macro-multipole-orders 4
run multipole_m4_a1 300 300 2 $R/multipole_m4_a1 --macro-multipole --macro-multipole-orders 4
run multipole_m3_a3 300 300 3 $R/multipole_m3_a3 --macro-multipole --macro-multipole-orders 4
run test_fixed15    300 300 0 $R/test_fixed15    --macro-multipole --macro-multipole-orders 4
# line-of-sight arms with the perturber position free (Sect. 4.5 localization measured, not grid-limited)
run los_halo_z025 300 300 5 results/baseline_a_joint_freepos/los_z025 --free-position
run los_halo_z075 300 300 5 results/baseline_a_joint_freepos/los_z075 --free-position
echo "ALL DONE $(date)"
