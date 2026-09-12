#!/bin/zsh
# Family B seed-set launcher: 2 variants x 5 populations, <=2 concurrent processes, CPU only.
#   scripts/launch_family_b.sh 0 "fitted oracle"     # seed-set 0 (matches Family A's subsamples)
#   scripts/launch_family_b.sh 200 fitted            # seed-200 replicate set, fitted only
set -u
cd "$(dirname "$0")/.."
source ~/myenv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
BASE=${1:-0}; VARIANTS=${2:-"fitted oracle"}; N=${N:-300}
OUTROOT=results/baseline_b; [[ $BASE -ne 0 ]] && OUTROOT=results/baseline_b_seed${BASE}
mkdir -p $OUTROOT/logs
JOBS=()
for v in ${=VARIANTS}; do
  for spec in "test_fixed60:0" "test_fixed15:0" "no_subhalo:1" "multipole_m4_a1:2" "multipole_m4_a3:3"; do
    pop=${spec%%:*}; off=${spec##*:}; seed=$((BASE+off))
    JOBS+=("$pop $v $seed")
  done
done
printf '%s\n' "${JOBS[@]}" | xargs -P 2 -L 1 zsh -c '
  pop=$0; v=$1; seed=$2
  python scripts/run_family_b.py --population $pop --variant $v --seed $seed --n '"$N"' \
      --out '"$OUTROOT"'/$v/$pop > '"$OUTROOT"'/logs/${v}_${pop}.log 2>&1
  echo "done $v/$pop (exit $?)"'
echo "ALL DONE $(date)" >> $OUTROOT/logs/launcher.log
