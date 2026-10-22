#!/bin/zsh
# A minimal hyperparameter sweep for the U-Net trained at c=15 (30 000 images, the population of launch_unet_lowc.sh):
# learning rate x0.3 and x3, BCE class weight 20 and 1000, and augmentation with a cosine schedule over 40 epochs.
# Tests whether the c=15 failure belongs to the recipe's settings.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
run() { # run <name> <extra args>
  local out=checkpoints/unet_30k_c15_sweep_$1
  if [ ! -f $out/model_best.pt ]; then
    echo "=== train $1 $(date)"; python scripts/train_detector.py --train data/train_fixed15_30k --out $out --seed 0 ${=2} > $out.train.log 2>&1
  fi
  if [ ! -f results/detector_30k_c15_sweep_$1/results.json ]; then
    echo "=== eval $1 $(date)"; python scripts/evaluate_detector.py --model $out/model_best.pt --out results/detector_30k_c15_sweep_$1 > results/detector_30k_c15_sweep_$1.eval.log 2>&1
  fi
}
run lr3e-4 "--epochs 25 --lr 3e-4"
run lr3e-3 "--epochs 25 --lr 3e-3"
run pw20 "--epochs 25 --pos-weight 20"
run pw1000 "--epochs 25 --pos-weight 1000"
run augcos "--epochs 40 --augment --cosine"
echo "C15 SWEEP DONE $(date)"
