#!/bin/zsh
# Completes the Tier-0 / 600 s pairs of the published-pipeline case study: multipole 57 and c=60 lens 8 at 600 s, one at a
# time after the lens-20 resume queue ($1).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PAL_QUEUE=q600
R=results/pyautolens_slam_600
while kill -0 $1 2>/dev/null; do sleep 120; done
run() { echo "start $* $(date)"; python scripts/run_pyautolens_slam_subhalo.py "$@" >> $R/log.txt 2>&1; echo "done $* exit $? $(date)"; }
run --population test_shallow600_multipole_m4_a3 --only 57 --out $R/multipole_m4_a3
run --population test_shallow600_fixed60 --only 8 --out $R/test_fixed60
echo "Q600 PAIRS DONE $(date)"
