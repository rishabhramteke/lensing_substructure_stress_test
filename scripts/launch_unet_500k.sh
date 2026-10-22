#!/bin/zsh
# Referee rounds 22-25: the U-Net at the published training scale (5x10^5 c=60 images, Tsang et al. 2024), as five
# 100 000-image chunks (seeds 700-704), one seed, then evaluated on the standard Tier-0 test populations.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1
for k in 0 1 2 3 4; do
  d=data/train_fixed60_500k_part$k
  [ -f $d/manifest.json ] || python scripts/generate_tier0.py --config tsang_fixed60 --n 100000 --seed $((700+k)) --out $d > $d.gen.log 2>&1 &
done
wait
export OMP_NUM_THREADS=2
out=checkpoints/unet_30k_c60pub_seed0
[ -f $out/model_best.pt ] || python scripts/train_detector.py --train data/train_fixed60_500k_part{0,1,2,3,4} --out $out --epochs 25 --seed 0 > $out.train.log 2>&1
[ -f results/detector_30k_c60pub_seed0/results.json ] || python scripts/evaluate_detector.py --model $out/model_best.pt --out results/detector_30k_c60pub_seed0 > results/detector_30k_c60pub_seed0.eval.log 2>&1
echo "ALL DONE $(date)"
