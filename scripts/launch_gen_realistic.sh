#!/bin/zsh
# Combined realism arm (referee round 21): 0.15" PSF + 600 s + two-component lens light, test populations with the Tier-0
# seeds (so the lenses, sources and subhalos match) and a 30 000-image c=60 training population (seed 500).
#   caffeinate -i zsh scripts/launch_gen_realistic.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1
gen() { # gen <config> <n> <seed> <out>
  if [ -f $4/manifest.json ]; then echo "skip $4"; return; fi
  echo "=== generate $4 $(date)"; python scripts/generate_tier0.py --config $1 --n $2 --seed $3 --out $4 > $4.gen.log 2>&1
}
gen realistic_no_subhalo      1000 1   data/realistic_no_subhalo
gen realistic_fixed60         1000 101 data/realistic_fixed60
gen realistic_fixed15         1000 101 data/realistic_fixed15
gen realistic_multipole_m4_a3 1000 2   data/realistic_multipole_m4_a3
gen realistic_fixed60         30000 500 data/train_realistic_30k
echo "ALL DONE $(date)"
