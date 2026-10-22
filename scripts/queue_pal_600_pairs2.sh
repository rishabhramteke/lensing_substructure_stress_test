#!/bin/zsh
# The last Tier-0 / 600 s pair of the published-pipeline case study: c=60 lens 156 at 600 s, after the pairs queue ($1).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PAL_QUEUE=q600
R=results/pyautolens_slam_600
while kill -0 $1 2>/dev/null; do sleep 120; done
echo "start test_fixed60 156 $(date)"; python scripts/run_pyautolens_slam_subhalo.py --population test_shallow600_fixed60 --only 156 --out $R/test_fixed60 >> $R/log.txt 2>&1
echo "done test_fixed60 156 exit $? $(date)"
