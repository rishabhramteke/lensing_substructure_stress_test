#!/bin/zsh
# Referee round 26 populations: the source and render misspecification arms at 600 s, and the PSF error alone.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for arm in clumpy600 super600 psferr; do
  echo "${arm}_no_subhalo 1"; echo "${arm}_fixed60 101"; echo "${arm}_multipole_m4_a3 2"
done | xargs -P 3 -L 1 zsh -c '[ -f data/$0/manifest.json ] || python scripts/generate_tier0.py --config $0 --n 1000 --seed $1 --out data/$0 > data/$0.gen.log 2>&1; echo "gen $0 done $(date)"'
