#!/bin/zsh
# U-Net controls for referee round 23: (1) fine-tuning the c=60 networks on the c=15 training set, (2) a 50-epoch cosine
# schedule at c=15, (3) 100 000-image training sets at c=15 and c=60 (seed 600, generated here), (4) the four standard
# networks on a 10 000-lens clean control (0.1% FPR calibration). Evaluated on the standard Tier-0 test populations.
#   caffeinate -i zsh scripts/launch_unet_round23.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
train() { # train <train_pop> <out> <seed> [args...]
  if [ -f $2/model_best.pt ]; then echo "skip $2"; return; fi
  echo "=== train $2 $(date)"; python scripts/train_detector.py --train $1 --out $2 --seed $3 ${@:4} > $2.train.log 2>&1
}
ev() { # ev <name>
  if [ -f results/detector_30k_$1/results.json ]; then echo "skip eval $1"; return; fi
  echo "=== evaluate $1 $(date)"; python scripts/evaluate_detector.py --model checkpoints/unet_30k_$1/model_best.pt --out results/detector_30k_$1 > results/detector_30k_$1.eval.log 2>&1
}
for s in 0 1; do
  train data/train_fixed15_30k checkpoints/unet_30k_c15ft_seed$s $s --epochs 25 --lr 3e-4 --init checkpoints/unet_30k_seed$s/model_best.pt
  ev c15ft_seed$s
done
train data/train_fixed15_30k checkpoints/unet_30k_c15cos_seed0 0 --epochs 50 --cosine
ev c15cos_seed0
[ -f data/train_fixed15_100k/manifest.json ] || python scripts/generate_tier0.py --config tsang_fixed15 --n 100000 --seed 600 --out data/train_fixed15_100k > data/train_fixed15_100k.gen.log 2>&1
[ -f data/train_fixed60_100k/manifest.json ] || python scripts/generate_tier0.py --config tsang_fixed60 --n 100000 --seed 600 --out data/train_fixed60_100k > data/train_fixed60_100k.gen.log 2>&1
train data/train_fixed15_100k checkpoints/unet_30k_c15big_seed0 0 --epochs 25
ev c15big_seed0
train data/train_fixed60_100k checkpoints/unet_30k_c60big_seed0 0 --epochs 25
ev c60big_seed0
python scripts/make_unet_objectives_table.py > /dev/null 2>&1
echo "ALL DONE $(date)"
