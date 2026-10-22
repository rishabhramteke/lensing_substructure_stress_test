#!/bin/zsh
# PSF-widened (0.15") and shallow re-renderings: test populations, 30k training populations, and the
# frozen scan on the PSF-widened test populations (same 300-lens subsamples as results/baseline_a).
#   caffeinate -i nice -n 5 zsh scripts/launch_psf.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1
gen() { # gen <config> <n> <seed> <out>
  if [ -f $4/manifest.json ]; then echo "skip $4"; return; fi
  echo "=== generate $4 $(date)"; python scripts/generate_tier0.py --config $1 --n $2 --seed $3 --out $4 > $4.gen.log 2>&1
}
gen psf015_no_subhalo      1000 1   data/psf015_no_subhalo
gen psf015_fixed60         1000 101 data/psf015_fixed60
gen psf015_fixed15         1000 101 data/psf015_fixed15
gen psf015_multipole_m4_a3 1000 2   data/psf015_multipole_m4_a3
gen psf015_fixed60         30000 500 data/train_psf015_30k
gen shallow_fixed60        30000 500 data/train_shallow_30k
# frozen scan on the PSF-widened populations, same subsample seeds as results/baseline_a
scan() { # scan <pop> <seed> <out>
  if [ -f $3/manifest.json ]; then echo "skip $3"; return; fi
  mkdir -p $(dirname $3); echo "=== frozen scan $3 $(date)"
  python scripts/run_baseline_a.py --population $1 --n 300 --seed $2 --out $3 > $3.log 2>&1
}
scan psf015_no_subhalo      1 results/baseline_a_psf015/no_subhalo
scan psf015_fixed60         0 results/baseline_a_psf015/test_fixed60
scan psf015_fixed15         0 results/baseline_a_psf015/test_fixed15
scan psf015_multipole_m4_a3 3 results/baseline_a_psf015/multipole_m4_a3
echo "ALL DONE $(date)"
