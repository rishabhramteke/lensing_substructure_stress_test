#!/bin/zsh
# Referee round 20: after launch_complete_truncated.sh, re-run in full every lens whose stored smooth chi^2 could not be
# reproduced (complete_truncated_joint.py --rerun-mismatched). Resumable (<pop>/completion.jsonl). Pass roots as args.
#   caffeinate -i zsh scripts/launch_rerun_mismatched.sh baseline_a_joint_full baseline_a_joint_mp4 ...
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
W=${WORKERS:-5}
for root in "$@"; do
  echo "##### $root $(date)"
  python scripts/complete_truncated_joint.py --roots results/$root --workers $W --rerun-mismatched
done
echo "ALL DONE $(date)"
