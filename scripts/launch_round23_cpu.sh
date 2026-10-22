#!/bin/zsh
# Referee round 23 Family-A experiments, after launch_round22_cpu.sh. No wall-clock limit.
#   inverse crime (5x supersampled, 10%-wider-PSF renders, fitted with the nominal model), 300-lens subsamples
#   multipole + subhalo in the same lens (|a4| ~ |N(0, 0.01)| theta_E), bare and free-m=4 scans, with their own clean nulls
#   the combined realism arm on the FULL 1 000-lens populations, and its 10-sigma decoys
#   the position-free scan on the FULL 1 000-lens populations
#   3 000 extra clean lenses (seed 11) for a 0.1% FPR calibration
#   caffeinate -i zsh scripts/launch_round23_cpu.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "waiting for round 22 $(date)"
# (runs alongside launch_round22_cpu.sh; disjoint roots)
echo "starting $(date)"
W=${WORKERS:-13}
run() {  # run <population> <n> <n_subsample> <seed> <out> [extra args]
  pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  if [ -f $out/manifest.json ]; then echo "skip $out"; return; fi
  mkdir -p $(dirname $out); echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
}
R=results
run invcrime_no_subhalo      300 300 1 $R/baseline_a_joint_invcrime/no_subhalo
run invcrime_fixed60         300 300 0 $R/baseline_a_joint_invcrime/test_fixed60
run invcrime_multipole_m4_a3 300 300 3 $R/baseline_a_joint_invcrime/multipole_m4_a3
run mpsub_no_subhalo 300 300 1 $R/baseline_a_joint_mpsub/no_subhalo
run mpsub_fixed60    300 300 0 $R/baseline_a_joint_mpsub/test_fixed60
run mpsub_no_subhalo 300 300 1 $R/baseline_a_joint_mpsub_mp4/no_subhalo   --macro-multipole
run mpsub_fixed60    300 300 0 $R/baseline_a_joint_mpsub_mp4/test_fixed60 --macro-multipole
for pop in no_subhalo fixed60 multipole_m4_a3 fixed15; do
  tgt=$pop; [ $pop = fixed60 ] && tgt=test_fixed60; [ $pop = fixed15 ] && tgt=test_fixed15
  run realistic_$pop 1000 1000 0 $R/baseline_a_joint_realistic_full/$tgt --lens-light-components 2
done
for d in gaussian_s42_a10 gaussian_s43_a10 dipole_s42_a10; do
  run realistic_decoy_$d 100 100 0 $R/baseline_a_joint_realistic_full/decoy_$d --lens-light-components 2
done
for pop in no_subhalo test_fixed60 multipole_m4_a3 test_fixed15; do
  run $pop 1000 1000 0 $R/baseline_a_joint_freepos_full/$pop --free-position
done
run no_subhalo_10k 3000 3000 11 $R/baseline_a_joint_bigclean/no_subhalo
python scripts/regate_joint.py --roots $R/baseline_a_joint_invcrime $R/baseline_a_joint_mpsub $R/baseline_a_joint_mpsub_mp4 $R/baseline_a_joint_realistic_full $R/baseline_a_joint_freepos_full $R/baseline_a_joint_bigclean --report
echo "ALL DONE $(date)"
