#!/bin/zsh
# Round-23 Family-A runs as a pool: start several copies of this script; each claims the next unclaimed population
# (atomic mkdir lock <out>.lock) from ONE list ordered longest first, so the slow tails of different populations overlap
# instead of queuing. A population with a manifest is done; one whose lock exists is someone else's.
#   WORKERS=5 caffeinate -i zsh scripts/launch_round23_pool.sh     (x3)
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-5}; R=results
JOBS=(
  "no_subhalo 1000 1000 0 $R/baseline_a_joint_freepos_full/no_subhalo --free-position"
  "test_fixed60 1000 1000 0 $R/baseline_a_joint_freepos_full/test_fixed60 --free-position"
  "realistic_no_subhalo 1000 1000 0 $R/baseline_a_joint_realistic_full/no_subhalo --lens-light-components 2"
  "realistic_fixed60 1000 1000 0 $R/baseline_a_joint_realistic_full/test_fixed60 --lens-light-components 2"
  "multipole_m4_a3 1000 1000 0 $R/baseline_a_joint_freepos_full/multipole_m4_a3 --free-position"
  "test_fixed15 1000 1000 0 $R/baseline_a_joint_freepos_full/test_fixed15 --free-position"
  "realistic_multipole_m4_a3 1000 1000 0 $R/baseline_a_joint_realistic_full/multipole_m4_a3 --lens-light-components 2"
  "realistic_fixed15 1000 1000 0 $R/baseline_a_joint_realistic_full/test_fixed15 --lens-light-components 2"
  "no_subhalo_10k 3000 3000 11 $R/baseline_a_joint_bigclean/no_subhalo"
  "invcrime_fixed60 300 300 0 $R/baseline_a_joint_invcrime/test_fixed60"
  "invcrime_multipole_m4_a3 300 300 3 $R/baseline_a_joint_invcrime/multipole_m4_a3"
  "mpsub_no_subhalo 300 300 1 $R/baseline_a_joint_mpsub/no_subhalo"
  "mpsub_fixed60 300 300 0 $R/baseline_a_joint_mpsub/test_fixed60"
  "mpsub_no_subhalo 300 300 1 $R/baseline_a_joint_mpsub_mp4/no_subhalo --macro-multipole"
  "mpsub_fixed60 300 300 0 $R/baseline_a_joint_mpsub_mp4/test_fixed60 --macro-multipole"
  "realistic_decoy_gaussian_s42_a10 100 100 0 $R/baseline_a_joint_realistic_full/decoy_gaussian_s42_a10 --lens-light-components 2"
  "realistic_decoy_gaussian_s43_a10 100 100 0 $R/baseline_a_joint_realistic_full/decoy_gaussian_s43_a10 --lens-light-components 2"
  "realistic_decoy_dipole_s42_a10 100 100 0 $R/baseline_a_joint_realistic_full/decoy_dipole_s42_a10 --lens-light-components 2"
)
for j in "${JOBS[@]}"; do
  set -- ${=j}; pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  [ -f $out/manifest.json ] && continue
  mkdir -p $(dirname $out)
  mkdir $out.lock 2>/dev/null || continue          # claimed by another pool member
  echo "=== $out  $(date)  (pid $$)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed --concentration 15 \
      --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
done
echo "POOL MEMBER DONE $(date) (pid $$)"
