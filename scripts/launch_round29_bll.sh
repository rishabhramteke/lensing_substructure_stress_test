#!/bin/zsh
# Family B on the 1 000-lens double-Sersic lens-light clean and c=60 populations (each into its own directory).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for p in no_subhalo fixed60; do
  out=results/baseline_b_lenslight1k/fitted/lenslight1k_$p
  [ -f $out/manifest.json ] || python scripts/run_family_b.py --population lenslight1k_$p --n 1000 --seed 0 --lens-light-components 2 --out $out > $out.log 2>&1
  echo "done $p exit $? $(date)"
done
echo "BLL DONE $(date)"
