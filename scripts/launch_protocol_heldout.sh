#!/bin/zsh
# The recommended protocol on 200 held-out lenses per population: lenses never used when the protocol was scored
# (results/baseline_a_joint_protocol), drawn with a fixed seed from the rest of each 1 000-lens population.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
P="--blind --restarts 7 --free-centre --free-position --macro-multipole --macro-multipole-orders 3,4"
O=results/baseline_a_joint_protocol_heldout; mkdir -p $O
for p in no_subhalo test_fixed60 multipole_m4_a3; do
  echo "start $p $(date)"; python scripts/run_baseline_a_joint.py --population $p --indices $(cat scratchpad/heldout_$p.txt) --seed 0 --workers 6 ${=P} --out $O/$p > $O/$p.log 2>&1
done
echo "HELDOUT DONE $(date)"
