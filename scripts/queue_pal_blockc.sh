#!/bin/zsh
# Round 30, Block C: after the PyAutoLens queue ($1 = its launcher PID), one lens at a time: multipole lens 45 with a free m=4
# multipole in the published mass model; clean lens 4 re-run from scratch (fresh sampler draws); multipole lens 57 with the m=4 term.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PAL_QUEUE=blockc
R=results/pyautolens_slam
while kill -0 $1 2>/dev/null; do sleep 120; done
run() { echo "start $* $(date)"; python scripts/run_pyautolens_slam_subhalo.py "$@" >> $R/log_blockc.txt 2>&1; echo "done $* exit $? $(date)"; }
run --population multipole_m4_a3 --only 45 --multipole --out results/pyautolens_slam_mp4/multipole_m4_a3
run --population no_subhalo --only 4 --tag-suffix _rep --out results/pyautolens_slam_rep/no_subhalo
run --population multipole_m4_a3 --only 57 --multipole --out results/pyautolens_slam_mp4/multipole_m4_a3
echo "BLOCKC DONE $(date)"
