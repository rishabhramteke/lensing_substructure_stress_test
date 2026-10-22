#!/bin/zsh
# Two lanes of the core PyAutoLens subset. Lane A starts when the lens-8 launcher (PID $1) exits, lane B when the
# Family A clean-control run (PID $2) exits; lane B is paused while system free memory is below 15% and resumed above 30%.
cd "$(dirname $0)/.."
R=results/pyautolens_slam
mon() { echo "$(date +%T) free=$(memory_pressure | awk '/free percentage/{print $5}') $(ps -axo rss=,command= | grep '[r]un_pyautolens_slam_subhalo.py --pop' | awk '{printf "%s ", $1}')" >> $R/mem_monitor.log; }
( while kill -0 $1 2>/dev/null; do sleep 60; done; PAL_THREADS=2 scripts/launch_pyautolens_slam.sh $(cat $R/lane_a.txt) > $R/lane_a.log 2>&1 ) &
( while kill -0 $2 2>/dev/null; do sleep 60; done; PAL_THREADS=2 scripts/launch_pyautolens_slam.sh $(cat $R/lane_b.txt) > $R/lane_b.log 2>&1 ) &
while true; do
  mon; f=$(memory_pressure | awk '/free percentage/{print $5+0}')
  B=$(pgrep -f "launch_pyautolens_slam.sh $(head -1 $R/lane_b.txt)")
  if [ -n "$B" ]; then for p in $(pgrep -P $B); do
    if [ $f -lt 15 ]; then kill -STOP $p; echo "$(date +%T) paused lane B $p (free $f%)" >> $R/mem_monitor.log; fi
    if [ $f -gt 30 ]; then kill -CONT $p 2>/dev/null; fi; done; fi
  pgrep -f "launch_pyautolens_slam.sh" >/dev/null || [ -n "$(jobs -r)" ] || break
  sleep 60
done
