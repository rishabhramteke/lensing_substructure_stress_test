#!/bin/zsh
# Referee round 20: remove the 900 s wall-clock guard from every joint-scan number by finishing each truncated lens
# (complete_truncated_joint.py), headline populations first. Resumable (per-population completion.jsonl).
#   caffeinate -i zsh scripts/launch_complete_truncated.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-8}
for root in baseline_a_joint_full baseline_a_joint_mp4 baseline_a_joint_mp34 baseline_a_joint_freepos baseline_a_joint_decoy \
            baseline_a_joint_los baseline_a_joint_lenslight baseline_a_joint_lenslight2 baseline_a_joint_psf015 \
            baseline_a_joint_shallow baseline_a_joint_tidal baseline_a_joint_shallow600 baseline_a_joint_cm; do
  echo "##### $root $(date)"
  python scripts/complete_truncated_joint.py --roots results/$root --workers $W
done
echo "ALL DONE $(date)"
