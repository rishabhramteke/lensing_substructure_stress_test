#!/bin/zsh
# Round 29, after the misspecification grid: (1) the sub-threshold-population arm completed to its seed-defined 300 lenses
# (the lenses already scanned are skipped, the rest keep their per-lens seeds); (2) the seven-start rescue (original plus
# six restarts, as on the 1 000-lens control) of the 557 lenses of the 10 000-lens clean population whose best fit has
# chi^2/dof >= 1.5 or that fail the gate.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=${W:-10}
until grep -q "MISSPEC DONE" $R/launch_round29_misspec.log 2>/dev/null; do sleep 60; done
for spec in "pop_no_subhalo 1 no_subhalo" "pop_fixed60 0 test_fixed60"; do
  set -- ${=spec}; out=$R/baseline_a_joint_pop_rest/$3; mkdir -p $R/baseline_a_joint_pop_rest
  [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py --population $1 --n 300 --n-subsample 300 --seed $2 --concentration 15 \
      --workers $W --lens-timeout 1e9 --skip-indices $(cat $R/pop_done_$3.txt) --out $out > $out.log 2>&1
  echo "done $out exit $? $(date)"
done
out=$R/baseline_a_joint_bigclean_rescue/no_subhalo; mkdir -p $(dirname $out)
[ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py --population no_subhalo_10k --seed 11 --concentration 15 --workers $W \
    --lens-timeout 1e9 --restarts 6 --indices $(cat $R/bigclean_rescue_indices.txt) --out $out > $out.log 2>&1
echo "done $out exit $? $(date)"
echo "ROUND29 MORE DONE $(date)"
