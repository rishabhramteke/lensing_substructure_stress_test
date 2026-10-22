#!/bin/zsh
# Controls for the alternative objectives (referee round 21): the image-level objectives trained at c=60, where the
# segmentation recipe does learn, so a failure at c=15 can be told apart from a failure of the objective itself; and the
# smooth-maximum (log-sum-exp) image-level objective at both concentrations. Runs after launch_unet_altobj.sh.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
until grep -a -q "ALL DONE" results/launch_unet_altobj.log; do sleep 60; done
train() { # train <train_pop> <out> <seed> [args...]
  if [ -f $2/model_best.pt ]; then echo "skip $2"; return; fi
  echo "=== train $2 $(date)"; python scripts/train_detector.py --train $1 --out $2 --epochs 25 --seed $3 ${@:4} > $2.train.log 2>&1
}
train data/train_fixed60_30k checkpoints/unet_30k_c60img_seed0 0 --objective image
train data/train_fixed60_30k checkpoints/unet_30k_c60lse_seed0 0 --objective image_lse
train data/train_fixed15_30k checkpoints/unet_30k_c15lse_seed0 0 --objective image_lse
train data/train_fixed15_30k checkpoints/unet_30k_c15lse_seed1 1 --objective image_lse
for name in c60img_seed0 c60lse_seed0 c15lse_seed0 c15lse_seed1; do
  if [ -f results/detector_30k_$name/results.json ]; then echo "skip eval $name"; continue; fi
  echo "=== evaluate $name $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_$name/model_best.pt --out results/detector_30k_$name > results/detector_30k_$name.eval.log 2>&1
done
python scripts/aggregate_unet_seeds.py --runs detector_30k_c15lse_seed0 detector_30k_c15lse_seed1 --out results/aggregate_30k_c15lse > results/aggregate_30k_c15lse.log 2>&1
echo "ALL DONE $(date)"
