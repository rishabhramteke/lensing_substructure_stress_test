#!/bin/zsh
# U-Net on the combined realism arm (referee round 21): two seeds trained on 30 000 realistic c=60 images (0.15" PSF,
# 600 s, two-component lens light, not subtracted), scored on the realistic test populations; the Tier-0-trained seed 0
# is scored there too, for the transfer. Runs after launch_unet_altobj2.sh.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
until grep -a -q "ALL DONE" results/launch_unet_altobj2.log; do sleep 60; done
for s in 0 1; do
  out=checkpoints/unet_30k_realistic_seed$s
  if [ ! -f $out/model_best.pt ]; then echo "=== train $out $(date)"; python scripts/train_detector.py --train data/train_realistic_30k --out $out --epochs 25 --seed $s > $out.train.log 2>&1; fi
done
ev() { # ev <model> <out>
  if [ -f $2/results.json ]; then echo "skip $2"; return; fi
  echo "=== evaluate $2 $(date)"
  python scripts/evaluate_detector.py --model $1 --out $2 --test60 realistic_fixed60 --test15 realistic_fixed15 --clean realistic_no_subhalo \
      --mp-a1 realistic_multipole_m4_a3 --mp-a3 realistic_multipole_m4_a3 > $2.eval.log 2>&1
}
ev checkpoints/unet_30k_realistic_seed0/model_best.pt results/detector_30k_realistic_seed0
ev checkpoints/unet_30k_realistic_seed1/model_best.pt results/detector_30k_realistic_seed1
ev checkpoints/unet_30k_seed0/model_best.pt results/detector_30k_seed0_on_realistic
python scripts/aggregate_unet_seeds.py --runs detector_30k_realistic_seed0 detector_30k_realistic_seed1 --out results/aggregate_30k_realistic > results/aggregate_30k_realistic.log 2>&1
echo "ALL DONE $(date)"
