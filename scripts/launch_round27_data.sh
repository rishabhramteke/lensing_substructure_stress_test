#!/bin/zsh
# Referee round 27 data: the Euclid-VIS-like arm (clean, c=60, c=15, multipole; 1 000 each, Tier-0 seeds) and its
# 30 000-image training set; 100 000 training images with lens multipoles (|a4| ~ |N(0, 0.01)| theta_E, the mpsub
# population) and 100 000 at c=30, each in four 25 000-image chunks on new seeds.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
{
  echo "euclid_no_subhalo 1000 1 data/euclid_no_subhalo"
  echo "euclid_fixed60 1000 101 data/euclid_fixed60"
  echo "euclid_fixed15 1000 101 data/euclid_fixed15"
  echo "euclid_multipole_m4_a3 1000 2 data/euclid_multipole_m4_a3"
  for k in 0 1 2; do echo "euclid_fixed60 10000 $((150+k)) data/euclid_train_fixed60_part$k"; done
  for k in 0 1 2 3; do echo "mpsub_fixed60 25000 $((130+k)) data/train_mpsub_100k_part$k"; done
  for k in 0 1 2 3; do echo "tsang_fixed30 25000 $((140+k)) data/train_fixed30_100k_part$k"; done
} | xargs -P 6 -L 1 zsh -c '[ -f $3/manifest.json ] || python scripts/generate_tier0.py --config $0 --n $1 --seed $2 --out $3 > $3.gen.log 2>&1; echo "gen $3 done exit $? $(date)"'
echo "DATA DONE $(date)"
