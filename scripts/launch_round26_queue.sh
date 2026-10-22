#!/bin/zsh
# Referee round 26 joint-scan arms, one job at a time with 8 workers (no oversubscription), each waiting for its data:
# free lens centre (M5), sub-threshold subhalo + line-of-sight population (M7), misspecification at 600 s and the PSF
# error alone (M2, M8). Same subsample seeds as the Tier-0 runs (clean 1, c=60/c=15 0, multipole 3).
# Two lanes may run at once (mkdir locks), so a lane's slow last lenses do not leave the machine idle:
#   WORKERS=6 zsh scripts/launch_round26_queue.sh &   (twice)
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=${WORKERS:-8}
JOBS=(
  "pop_no_subhalo 1 $R/baseline_a_joint_pop/no_subhalo"
  "pop_fixed60 0 $R/baseline_a_joint_pop/test_fixed60"
  "clumpy600_no_subhalo 1 $R/baseline_a_joint_clumpy600/no_subhalo"
  "clumpy600_fixed60 0 $R/baseline_a_joint_clumpy600/test_fixed60"
  "clumpy600_multipole_m4_a3 3 $R/baseline_a_joint_clumpy600/multipole_m4_a3"
  "no_subhalo 1 $R/baseline_a_joint_freecentre/no_subhalo --free-centre"
  "test_fixed60 0 $R/baseline_a_joint_freecentre/test_fixed60 --free-centre"
  "multipole_m4_a3 3 $R/baseline_a_joint_freecentre/multipole_m4_a3 --free-centre"
  "pop_fixed15 0 $R/baseline_a_joint_pop/test_fixed15"
  "super600_no_subhalo 1 $R/baseline_a_joint_super600/no_subhalo"
  "super600_fixed60 0 $R/baseline_a_joint_super600/test_fixed60"
  "super600_multipole_m4_a3 3 $R/baseline_a_joint_super600/multipole_m4_a3"
  "psferr_no_subhalo 1 $R/baseline_a_joint_psferr/no_subhalo"
  "psferr_fixed60 0 $R/baseline_a_joint_psferr/test_fixed60"
  "psferr_multipole_m4_a3 3 $R/baseline_a_joint_psferr/multipole_m4_a3"
)
remaining=(${JOBS[@]})
while (( ${#remaining} )); do
  next=(); ran=0
  for j in "${remaining[@]}"; do
    set -- ${=j}; pop=$1; seed=$2; out=$3; shift 3
    if [ -f $out/manifest.json ]; then continue; fi
    if [ ! -f data/$pop/manifest.json ] || (( ran )); then next+=("$j"); continue; fi
    mkdir -p $(dirname $out)
    if ! mkdir $out.lock 2>/dev/null; then next+=("$j"); continue; fi
    echo "=== $out  $(date)"
    python scripts/run_baseline_a_joint.py --population $pop --n 300 --n-subsample 300 --seed $seed --concentration 15 \
        --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
    echo "    done $out exit $? $(date)"; ran=1
  done
  remaining=(${next[@]})
  # jobs held by the other lane stay in the list until their manifest appears
  (( ran )) || sleep 120
done
echo "QUEUE DONE $(date)"
