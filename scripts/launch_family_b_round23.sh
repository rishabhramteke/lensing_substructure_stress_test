#!/bin/zsh
# Family B for referee round 23: the confirmation stage (parametric re-fit at the delta-kappa peak), then B on the
# inverse-crime and multipole-plus-subhalo arms, the full 1 000-lens realistic arm and its decoys, and the 10 000-lens
# clean control. Three at a time.
#   caffeinate -i zsh scripts/launch_family_b_round23.sh
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python scripts/confirm_family_b.py --workers 3 > results/family_b_confirmation.log 2>&1
JOBS=(
  "invcrime_no_subhalo invcrime 1 1 300" "invcrime_fixed60 invcrime 0 1 300" "invcrime_multipole_m4_a3 invcrime 3 1 300"
  "mpsub_no_subhalo mpsub 1 1 300" "mpsub_fixed60 mpsub 0 1 300"
  "realistic_no_subhalo realistic_full 0 2 1000" "realistic_fixed60 realistic_full 0 2 1000" "realistic_fixed15 realistic_full 0 2 1000" "realistic_multipole_m4_a3 realistic_full 0 2 1000"
  "realistic_decoy_gaussian_s42_a10 realistic_full 0 2 100" "realistic_decoy_gaussian_s43_a10 realistic_full 0 2 100" "realistic_decoy_dipole_s42_a10 realistic_full 0 2 100"
  "no_subhalo_10k bigclean 0 1 10000"
)
printf '%s\n' "${JOBS[@]}" | xargs -P 3 -L 1 zsh -c '
  pop=$0; arm=$1; seed=$2; nll=$3; n=$4; out=results/baseline_b_$arm/fitted/$pop
  if [ -f $out/manifest.json ]; then echo "skip $out"; exit 0; fi
  mkdir -p results/baseline_b_$arm/logs
  python scripts/run_family_b.py --population $pop --variant fitted --seed $seed --n $n --lens-light-components $nll \
      --out $out > results/baseline_b_$arm/logs/fitted_$pop.log 2>&1
  echo "done $out (exit $?) $(date)"'
echo "ALL DONE $(date)"
