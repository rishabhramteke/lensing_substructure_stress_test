#!/bin/zsh
# Alternative training objectives at c=15 (30 000 images, the c=15 training population of launch_unet_lowc.sh):
# an image-level objective on the frame's maximum logit, and a 5-pixel disc target with the class weight rescaled by area.
# Two seeds each, evaluated on the standard Tier-0 test populations.
#   caffeinate -i zsh scripts/launch_unet_altobj.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
train() { # train <out> <seed> [args...]
  if [ -f $1/model_best.pt ]; then echo "skip $1"; return; fi
  echo "=== train $1 $(date)"; python scripts/train_detector.py --train data/train_fixed15_30k --out $1 --epochs 25 --seed $2 ${@:3} > $1.train.log 2>&1
}
for s in 0 1; do
  train checkpoints/unet_30k_c15img_seed$s $s --objective image
  train checkpoints/unet_30k_c15wide_seed$s $s --objective seg --mask-radius 5 --pos-weight 24
done
for name in c15img_seed0 c15img_seed1 c15wide_seed0 c15wide_seed1; do
  if [ -f results/detector_30k_$name/results.json ]; then echo "skip eval $name"; continue; fi
  echo "=== evaluate $name $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_$name/model_best.pt --out results/detector_30k_$name > results/detector_30k_$name.eval.log 2>&1
done
python scripts/aggregate_unet_seeds.py --runs detector_30k_c15img_seed0 detector_30k_c15img_seed1 --out results/aggregate_30k_c15img > results/aggregate_30k_c15img.log 2>&1
python scripts/aggregate_unet_seeds.py --runs detector_30k_c15wide_seed0 detector_30k_c15wide_seed1 --out results/aggregate_30k_c15wide > results/aggregate_30k_c15wide.log 2>&1
echo "ALL DONE $(date)"
