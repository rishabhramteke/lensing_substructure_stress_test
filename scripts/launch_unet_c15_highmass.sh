#!/bin/zsh
# Positive control for the c=15 collapse: the identical U-Net recipe trained on 30 000 c=15 images whose subhalos all lie
# in 10^10-10^11 Msun, where the c=15 imprint is strong (S/N_pert > 100). If it learns, the recipe can learn a c=15
# subhalo and the collapse at lower mass is the weaker signal; if it does not, the recipe is the limit.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
D=data/train_fixed15_highmass_30k; C=checkpoints/unet_30k_c15_highmass_seed0; R=results/detector_30k_c15_highmass_seed0
[ -f $D/manifest.json ] || { echo "gen $(date)"; python scripts/generate_tier0.py --config tsang_fixed15_highmass --n 30000 --out $D --seed 500 > $D.gen.log 2>&1; }
[ -f $C/model_best.pt ] || { echo "train $(date)"; python scripts/train_detector.py --train $D --out $C --epochs 25 --seed 0 > $C.train.log 2>&1; }
[ -f $R/results.json ] || { echo "eval $(date)"; python scripts/evaluate_detector.py --model $C/model_best.pt --out $R > $R.eval.log 2>&1; }
echo "C15 HIGHMASS DONE $(date)"
