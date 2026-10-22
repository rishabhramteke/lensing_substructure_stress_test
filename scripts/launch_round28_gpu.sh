#!/bin/zsh
# The ratio estimator retrained on the combined-realism and 135 s training sets (30 000 images each, the budget of the
# U-Nets retrained there), each scored on its own arm.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=2
python scripts/train_image_detector.py --arch nre --train data/train_realistic_30k --out checkpoints/nre_30k_realistic_seed0 --epochs 25 --seed 0 \
  --eval-pops realistic_no_subhalo realistic_fixed60 realistic_fixed15 realistic_multipole_m4_a3 > checkpoints/nre_30k_realistic_seed0.train.log 2>&1
python scripts/train_image_detector.py --arch nre --train data/train_shallow_30k --out checkpoints/nre_30k_shallow_seed0 --epochs 25 --seed 0 \
  --eval-pops test_shallow_no_subhalo test_shallow_fixed60 test_shallow_fixed15 test_shallow_multipole_m4_a3 > checkpoints/nre_30k_shallow_seed0.train.log 2>&1
echo "ROUND28 GPU DONE $(date)"
