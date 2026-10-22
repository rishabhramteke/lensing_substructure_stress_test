#!/bin/zsh
# Referee round 27 networks, one after another on the GPU, each evaluated afterwards: a U-Net trained on 100 000 images
# whose lenses carry m=4 multipoles (|a4| ~ |N(0, 0.01)| theta_E), one trained on 100 000 images at c=30, and one trained
# on 30 000 Euclid-VIS-like images. The recipe is the paper's (25 epochs, constant lr) for comparability.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=2
run() { # run <name> <eval args> -- <train sets...>
  local name=$1; shift; local ev=$1; shift
  local out=checkpoints/unet_$name
  [ -f $out/model_best.pt ] || python scripts/train_detector.py --train "$@" --out $out --epochs 25 --seed 0 > $out.train.log 2>&1
  [ -f results/detector_$name/results.json ] || python scripts/evaluate_detector.py --model $out/model_best.pt --out results/detector_$name ${=ev} > results/detector_$name.eval.log 2>&1
  echo "done $name $(date)"
}
run 100k_mpsub_seed0 "" data/train_mpsub_100k_part{0,1,2,3}
run 100k_c30_seed0 "" data/train_fixed30_100k_part{0,1,2,3}
run 30k_euclid_seed0 "--test60 euclid_fixed60 --test15 euclid_fixed15 --clean euclid_no_subhalo --mp-a3 euclid_multipole_m4_a3 --mp-a1 none" data/euclid_train_fixed60_part{0,1,2}
echo "TRAIN DONE $(date)"
