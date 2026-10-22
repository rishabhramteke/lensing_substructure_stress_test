#!/bin/zsh
# Finishes the 600 s clean lens 20 after the q600 queue ($1): its search resumes from the sampler checkpoints.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PAL_QUEUE=q600
R=results/pyautolens_slam_600
while kill -0 $1 2>/dev/null; do sleep 120; done
echo "start no_subhalo 20 $(date)"; python scripts/run_pyautolens_slam_subhalo.py --population test_shallow600_no_subhalo --only 20 --out $R/no_subhalo >> $R/log.txt 2>&1
echo "done no_subhalo 20 exit $? $(date)"
