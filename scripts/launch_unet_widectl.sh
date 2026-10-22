#!/bin/zsh
# c=60 control for the 5-pixel disc objective (referee round 21): does the wide-disc recipe learn where the standard one does?
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
out=checkpoints/unet_30k_c60wide_seed0
[ -f $out/model_best.pt ] || python scripts/train_detector.py --train data/train_fixed60_30k --out $out --epochs 25 --seed 0 --objective seg --mask-radius 5 --pos-weight 24 > $out.train.log 2>&1
[ -f results/detector_30k_c60wide_seed0/results.json ] || python scripts/evaluate_detector.py --model $out/model_best.pt --out results/detector_30k_c60wide_seed0 > results/detector_30k_c60wide_seed0.eval.log 2>&1
echo "ALL DONE $(date)"
