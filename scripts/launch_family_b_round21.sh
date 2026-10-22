#!/bin/zsh
# Family B on the arms it had not been run on (referee round 21): the combined realism arm, the 0.15" PSF, the 600 s and
# 135 s depths and the two line-of-sight populations. 300-lens subsamples with Family A's subsample seeds. Two at a time.
#   caffeinate -i zsh scripts/launch_family_b_round21.sh
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
JOBS=(
  "realistic_no_subhalo realistic 1 2" "realistic_fixed60 realistic 0 2" "realistic_fixed15 realistic 0 2" "realistic_multipole_m4_a3 realistic 3 2"
  "psf015_no_subhalo psf015 1 1" "psf015_fixed60 psf015 0 1" "psf015_fixed15 psf015 0 1" "psf015_multipole_m4_a3 psf015 3 1"
  "test_shallow600_no_subhalo shallow600 1 1" "test_shallow600_fixed60 shallow600 0 1" "test_shallow600_fixed15 shallow600 0 1" "test_shallow600_multipole_m4_a3 shallow600 3 1"
  "test_shallow_no_subhalo shallow 1 1" "test_shallow_fixed60 shallow 0 1" "test_shallow_fixed15 shallow 0 1" "test_shallow_multipole_m4_a3 shallow 3 1"
  "los_halo_z025 los 0 1" "los_halo_z075 los 0 1"
)
printf '%s\n' "${JOBS[@]}" | xargs -P 2 -L 1 zsh -c '
  pop=$0; arm=$1; seed=$2; nll=$3; out=results/baseline_b_$arm/fitted/$pop
  if [ -f $out/summary.json ] || [ -f $out/results.jsonl ] && [ -f $out/manifest.json ]; then echo "skip $out"; exit 0; fi
  mkdir -p results/baseline_b_$arm/logs
  python scripts/run_family_b.py --population $pop --variant fitted --seed $seed --n 300 --lens-light-components $nll \
      --out $out > results/baseline_b_$arm/logs/fitted_$pop.log 2>&1
  echo "done $out (exit $?) $(date)"'
echo "ALL DONE $(date)"
