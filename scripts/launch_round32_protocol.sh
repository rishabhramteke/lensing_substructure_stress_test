#!/bin/zsh
# The paper's recommended protocol scored end to end, and the Euclid-depth 1% multipole arm for the survey-purity estimate.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=6; R=results
until [ -f data/gen32.done ]; do sleep 60; done
echo "start euclid a1 $(date)"
python scripts/run_baseline_a_joint.py --population euclid_multipole_m4_a1 --n 300 --n-subsample 1000 --seed 3 --workers $W --out $R/baseline_a_joint_euclid/multipole_m4_a1 > $R/baseline_a_joint_euclid/multipole_m4_a1.log 2>&1
echo "done euclid a1 $(date)"
P="--blind --restarts 7 --free-centre --free-position --macro-multipole --macro-multipole-orders 3,4"
O=$R/baseline_a_joint_protocol; mkdir -p $O
for p in no_subhalo test_fixed60 test_fixed15 multipole_m4_a3 multipole_m4_a1 multipole_m3_a3 ellmultipole_m4_a3; do
  echo "start $p $(date)"; python scripts/run_baseline_a_joint.py --population $p --n 200 --n-subsample 300 --seed 0 --workers $W ${=P} --out $O/$p > $O/$p.log 2>&1
done
for p in decoy_gaussian_s42_a10 decoy_gaussian_s43_a10 decoy_dipole_s42_a10; do
  echo "start $p $(date)"; python scripts/run_baseline_a_joint.py --population $p --n 100 --n-subsample 100 --seed 0 --workers $W ${=P} --out $O/$p > $O/$p.log 2>&1
done
for p in realistic_no_subhalo realistic_fixed60 realistic_multipole_m4_a3; do
  echo "start $p $(date)"; python scripts/run_baseline_a_joint.py --population $p --n 200 --n-subsample 300 --seed 0 --workers $W --lens-light-components 2 ${=P} --out $O/$p > $O/$p.log 2>&1
done
echo "ROUND32 PROTOCOL DONE $(date)"
