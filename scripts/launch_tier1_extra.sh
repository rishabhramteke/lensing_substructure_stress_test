#!/bin/zsh
# Tier-1 (COSMOS source) additions: 30 000 training images for a U-Net at the Tier-0 budget, then the network, and
# gravitational-imaging stage 1 on the seed-0 300-lens subsamples of the Tier-1 clean, c=60 and multipole populations.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
for k in 0 1 2; do
  d=data/tier1/train_fixed60_30k_part$k
  [ -f $d/manifest.json ] || python scripts/generate_tier0.py --config cosmos_fixed60 --n 10000 --seed $((110+k)) --out $d > $d.gen.log 2>&1 &
done
for pop in no_subhalo test_fixed60 multipole_m4_a3; do
  for k in 0 1; do
    python scripts/run_family_b_gi.py --population $pop --data-root data/tier1 --n 300 --seed 0 --shard $k/2 --out results/baseline_bgi_tier1/$pop > results/baseline_bgi_tier1_${pop}_$k.log 2>&1 &
  done
done
wait
echo "TIER1 DATA + GI STAGE 1 DONE $(date)"
export OMP_NUM_THREADS=2
out=checkpoints/unet_tier1_30k_seed0
[ -f $out/model_best.pt ] || python scripts/train_detector.py --train data/tier1/train_fixed60_30k_part{0,1,2} --out $out --epochs 25 --seed 0 > $out.train.log 2>&1
[ -f results/detector_tier1_30k_seed0/results.json ] || python scripts/evaluate_detector.py --model $out/model_best.pt --data-root data/tier1 --out results/detector_tier1_30k_seed0 > results/detector_tier1_30k_seed0.eval.log 2>&1
echo "ALL DONE $(date)"
