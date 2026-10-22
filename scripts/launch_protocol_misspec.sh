#!/bin/zsh
# The recommended protocol (Sect. 4.12) on the forward-model arms (Sect. 4.8: PSF 2% and 5% too wide, source clumps of
# 0.5-4% of the flux) and on real (COSMOS) sources, on the same lenses the bare joint scan was scored on.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
W=6; O=results/baseline_a_joint_protocol; mkdir -p $O
P="--blind --restarts 7 --free-centre --free-position --macro-multipole --macro-multipole-orders 3,4"
for a in psf2 psf5 clump0p5 clump1 clump2 clump4; do
  p=misspec_t0_${a}_no_subhalo
  echo "start $p $(date)"; python scripts/run_baseline_a_joint.py --population $p --n 200 --n-subsample 300 --seed 1 --workers $W ${=P} --out $O/$p > $O/$p.log 2>&1
done
echo "start tier1 $(date)"; python scripts/run_baseline_a_joint.py --population tier1/no_subhalo --n 200 --n-subsample 300 --seed 1 --workers $W ${=P} --out $O/tier1_no_subhalo > $O/tier1_no_subhalo.log 2>&1
echo "PROTOCOL MISSPEC DONE $(date)"
