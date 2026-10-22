#!/bin/zsh
# U-Net chain (MPS): two more 30k seeds so the 30k budget has four seeds like the 8k one; then the
# same recipe trained at the PSF-widened and at the shallow operating point, evaluated on their own tiers.
#   caffeinate -i zsh scripts/launch_unet.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
train() { # train <train_pop> <seed> <out>
  if [ -f $3/model_best.pt ]; then echo "skip $3"; return; fi
  echo "=== train $3 $(date)"; python scripts/train_detector.py --train $1 --out $3 --epochs 25 --seed $2 > $3.train.log 2>&1
}
train data/train_fixed60_30k 2 checkpoints/unet_30k_seed2
train data/train_fixed60_30k 3 checkpoints/unet_30k_seed3
for s in 0 1 2 3; do
  [ -f results/detector_30k_seed$s/results.json ] && [ $s -lt 2 ] && { echo "skip eval seed $s"; continue; }
  echo "=== evaluate 30k seed $s $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_seed$s/model_best.pt --out results/detector_30k_seed$s > results/detector_30k_seed$s.eval.log 2>&1
done
python scripts/aggregate_unet_seeds.py --runs detector_30k_seed0 detector_30k_seed1 detector_30k_seed2 detector_30k_seed3 --out results/aggregate_30k > results/aggregate_30k.log 2>&1
# wait for the re-rendered training populations from launch_psf.sh
for pop in train_psf015_30k train_shallow_30k; do
  while [ ! -f data/$pop/manifest.json ]; do sleep 60; done
done
train data/train_psf015_30k  0 checkpoints/unet_30k_psf015_seed0
train data/train_psf015_30k  1 checkpoints/unet_30k_psf015_seed1
train data/train_shallow_30k 0 checkpoints/unet_30k_shallow_seed0
train data/train_shallow_30k 1 checkpoints/unet_30k_shallow_seed1
for pop in psf015_no_subhalo psf015_fixed60 psf015_fixed15 psf015_multipole_m4_a3; do
  while [ ! -f data/$pop/manifest.json ]; do sleep 60; done
done
for s in 0 1; do
  echo "=== evaluate psf015 seed $s $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_psf015_seed$s/model_best.pt --out results/detector_30k_psf015_seed$s \
      --test60 psf015_fixed60 --test15 psf015_fixed15 --clean psf015_no_subhalo --mp-a1 none --mp-a3 psf015_multipole_m4_a3 > results/detector_30k_psf015_seed$s.eval.log 2>&1
  echo "=== evaluate shallow seed $s $(date)"
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_shallow_seed$s/model_best.pt --out results/detector_30k_shallow_seed$s \
      --test60 test_shallow_fixed60 --test15 test_shallow_fixed15 --clean test_shallow_no_subhalo --mp-a1 none --mp-a3 test_shallow_multipole_m4_a3 > results/detector_30k_shallow_seed$s.eval.log 2>&1
done
# the deep-trained 30k network applied to the PSF-widened tier (transfer, no retraining)
for s in 0 1; do
  python scripts/evaluate_detector.py --model checkpoints/unet_30k_seed$s/model_best.pt --out results/detector_30k_seed${s}_on_psf015 \
      --test60 psf015_fixed60 --test15 psf015_fixed15 --clean psf015_no_subhalo --mp-a1 none --mp-a3 psf015_multipole_m4_a3 > results/detector_30k_seed${s}_on_psf015.eval.log 2>&1
done
echo "ALL DONE $(date)"
