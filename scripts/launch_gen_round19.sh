#!/bin/zsh
# Populations added in referee round 19: a c-M-drawn test population matched to test_fixed60/15 (seed 101) and the
# Tier-0 populations re-rendered at 600 s (between the 5400 s preset and the 135 s shallow arm), same seeds as deep.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
gen() { # gen <config> <out_name> <seed>
  if [ -f data/$2/manifest.json ]; then echo "skip gen $2"; return; fi
  echo "=== gen $2 $(date)"; python scripts/generate_tier0.py --config $1 --n 1000 --out data/$2 --seed $3 > data/$2.gen.log 2>&1
}
gen tsang_cdm                  test_cdm                         101
gen shallow600_no_subhalo      test_shallow600_no_subhalo       1
gen shallow600_fixed60         test_shallow600_fixed60          101
gen shallow600_fixed15         test_shallow600_fixed15          101
gen shallow600_multipole_m4_a3 test_shallow600_multipole_m4_a3  2
echo "ALL DONE $(date)"
