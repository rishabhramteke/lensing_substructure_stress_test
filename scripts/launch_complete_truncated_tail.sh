#!/bin/zsh
# Second worker pool for complete_truncated_joint.py, walking the root list from the END (per-population locks keep
# the two launchers off each other's populations). Start once the blind-init run has freed its cores.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-5}
for root in baseline_a_joint_cm baseline_a_joint_shallow600 baseline_a_joint_tidal baseline_a_joint_shallow baseline_a_joint_psf015 \
            baseline_a_joint_lenslight2 baseline_a_joint_lenslight baseline_a_joint_los baseline_a_joint_decoy baseline_a_joint_freepos; do
  echo "##### $root $(date)"
  python scripts/complete_truncated_joint.py --roots results/$root --workers $W
done
echo "ALL DONE $(date)"
