#!/bin/zsh
# Misspecification-tolerance grid (referee round 29, M6): 20 clean populations (PSF too wide by 2/5/10%, two source
# clumps of 0.5/1/2/4% of the source flux each; Tier 0, 600 s, Euclid-like; seed 1, the lenses of every clean control),
# generated four at a time, then the joint scan on the standard 200-lens subsample of each. The Tier-0 10% PSF point is
# results/baseline_a_joint_psferr200 (identical configuration).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=${W:-10}
POPS=()
for b in t0 600 euclid; do
  for a in 2 5 10; do [ "$b$a" = "t010" ] || POPS+=(misspec_${b}_psf${a}_no_subhalo); done
  for a in 0p5 1 2 4; do POPS+=(misspec_${b}_clump${a}_no_subhalo); done
done
echo "gen start $(date)"
for p in $POPS; do
  [ -f data/$p/manifest.json ] || { python scripts/generate_tier0.py --config $p --n 1000 --seed 1 --out data/$p > data/$p.gen.log 2>&1 & }
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done
done
wait; echo "gen done $(date)"
for p in $POPS; do
  out=$R/baseline_a_joint_misspec/$p
  mkdir -p $R/baseline_a_joint_misspec
  [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py --population $p --n 200 --n-subsample 300 --seed 1 --concentration 15 --workers $W --lens-timeout 1e9 --out $out > $out.log 2>&1
  echo "done $p exit $? $(date)"
done
echo "MISSPEC DONE $(date)"
