#!/bin/zsh
# Referee round 24: rerun every Family B population with the lenstronomy source gradient (run_family_b.py default), in
# priority order (the full populations first), three at a time at low priority. The pre-fix results were moved to
# results/_b_pre_round24/ for comparison. Then the confirmation stage is rerun on the new peaks.
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
cat results/family_b_rerun_jobs.txt | xargs -P 3 -L 1 zsh -c '
  pop=$0; variant=$1; seed=$2; n=$3; nll=$4; out=$5
  if [ -f $out/manifest.json ]; then echo "skip $out"; exit 0; fi
  mkdir -p $out $(dirname $(dirname $out))/logs
  nice -n 10 python scripts/run_family_b.py --population $pop --variant $variant --seed $seed --n $n --lens-light-components $nll \
      --out $out > $(dirname $(dirname $out))/logs/${variant}_${pop//\//_}.log 2>&1
  echo "done $out (exit $?) $(date)"'
rm -f results/baseline_b_full/fitted/*/confirmed.jsonl results/baseline_b_decoy/fitted/*/confirmed.jsonl
nice -n 10 python scripts/confirm_family_b.py --workers 3 > results/family_b_confirmation.log 2>&1
echo "ALL DONE $(date)"
