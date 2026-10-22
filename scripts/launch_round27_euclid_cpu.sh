#!/bin/zsh
# Families A and B on the Euclid-VIS-like arm: B on the full 1 000-lens populations, A (joint scan) on the seed
# subsamples of 300 (clean seed 1, c=60/c=15 seed 0, multipole seed 3), as at Tier 0.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results/baseline_b_euclid/fitted
for spec in "euclid_no_subhalo 1 no_subhalo" "euclid_fixed60 0 test_fixed60" "euclid_fixed15 0 test_fixed15" "euclid_multipole_m4_a3 3 multipole_m4_a3"; do
  set -- ${=spec}
  [ -f results/baseline_b_euclid/fitted/$3/manifest.json ] || python scripts/run_family_b.py --population $1 --variant fitted --seed $2 --n 1000 --out results/baseline_b_euclid/fitted/$3 > results/baseline_b_euclid_$3.log 2>&1 &
done
wait
echo "B DONE $(date)"
for spec in "euclid_no_subhalo 1 no_subhalo" "euclid_fixed60 0 test_fixed60" "euclid_multipole_m4_a3 3 multipole_m4_a3" "euclid_fixed15 0 test_fixed15"; do
  set -- ${=spec}; out=results/baseline_a_joint_euclid/$3
  mkdir -p results/baseline_a_joint_euclid
  [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py --population $1 --n 300 --n-subsample 300 --seed $2 --concentration 15 --workers 10 --lens-timeout 1e9 --out $out > $out.log 2>&1
  echo "A $3 done $(date)"
done
echo "EUCLID CPU DONE $(date)"
