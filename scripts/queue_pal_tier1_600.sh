#!/bin/zsh
# The published pipeline on real (COSMOS) sources at 600 s: one chain of lenses ($2 = "pop:index pop:index"), started
# after process $1 exits (0 = now).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
R=results/pyautolens_slam_tier1_600
mkdir -p $R
while [ "$1" != 0 ] && kill -0 $1 2>/dev/null; do sleep 120; done
for pi in ${=2}; do
  p=${pi%%:*}; i=${pi##*:}
  echo "start $p $i $(date)"; python scripts/run_pyautolens_slam_subhalo.py --population test_cosmos600_$p --only $i --out $R/$p >> $R/log_$p$i.txt 2>&1
  echo "done $p $i exit $? $(date)"
done
