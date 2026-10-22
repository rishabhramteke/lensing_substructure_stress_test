#!/bin/zsh
# The neural ratio estimator and the ConvNeXt-encoder U-Net, trained on the same 100 000 c=60 images as the 100k U-Net.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2
out=checkpoints/nre_100k_c60_seed0
[ -f results/nre_100k_c60_seed0/results.json ] || python scripts/train_image_detector.py --arch nre --train data/train_fixed60_100k --out $out --epochs 25 --seed 0 > $out.train.log 2>&1 &
out=checkpoints/unet_100k_c60convnext_seed0
( [ -f $out/model_best.pt ] || python scripts/train_detector.py --arch convnext_unet --train data/train_fixed60_100k --out $out --epochs 25 --seed 0 > $out.train.log 2>&1
  [ -f results/detector_100k_c60convnext_seed0/results.json ] || python scripts/evaluate_detector.py --model $out/model_best.pt --out results/detector_100k_c60convnext_seed0 > results/detector_100k_c60convnext_seed0.eval.log 2>&1 ) &
wait
echo "ALL DONE $(date)"
