#!/bin/zsh
# U-Net trained at LOW concentration (referee round 19, 2026-09-18): the missing control for the c=15 collapse.
# Two 30 000-image training populations matched to the c=60 one (same seed 500: same lenses, sources, subhalo masses
# and positions; only the concentration differs), one at fixed c=15 and one drawn from the Dutton & Maccio (2014)
# relation, two seeds each, evaluated on the standard Tier-0 test populations.
#   caffeinate -i zsh scripts/launch_unet_lowc.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
gen() { # gen <config> <out_name>
  if [ -f data/$2/manifest.json ]; then echo "skip gen $2"; return; fi
  echo "=== gen $2 $(date)"; python scripts/generate_tier0.py --config $1 --n 30000 --out data/$2 --seed 500 > data/$2.gen.log 2>&1
}
train() { # train <train_pop> <seed> <out>
  if [ -f $3/model_best.pt ]; then echo "skip $3"; return; fi
  echo "=== train $3 $(date)"; python scripts/train_detector.py --train $1 --out $3 --epochs 25 --seed $2 > $3.train.log 2>&1
}
gen tsang_fixed15 train_fixed15_30k
train data/train_fixed15_30k 0 checkpoints/unet_30k_c15_seed0
train data/train_fixed15_30k 1 checkpoints/unet_30k_c15_seed1
gen tsang_cdm train_cdm_30k
train data/train_cdm_30k 0 checkpoints/unet_30k_cm_seed0
train data/train_cdm_30k 1 checkpoints/unet_30k_cm_seed1
for name in c15_seed0 c15_seed1 cm_seed0 cm_seed1; do
  if [ -f results/detector_30k_$name/results.json ]; then echo "skip eval $name"; continue; fi
  echo "=== evaluate $name $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_$name/model_best.pt --out results/detector_30k_$name > results/detector_30k_$name.eval.log 2>&1
done
python scripts/aggregate_unet_seeds.py --runs detector_30k_c15_seed0 detector_30k_c15_seed1 --out results/aggregate_30k_c15 > results/aggregate_30k_c15.log 2>&1
python scripts/aggregate_unet_seeds.py --runs detector_30k_cm_seed0 detector_30k_cm_seed1 --out results/aggregate_30k_cm > results/aggregate_30k_cm.log 2>&1
echo "ALL DONE $(date)"
