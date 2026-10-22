#!/bin/zsh
# One PyAutoLens lens at a time (two concurrent JAX processes slow each other more than 20-fold on this machine):
# wait for the lens-8 launcher ($1), resume the frozen lane-B launcher ($2) and its running lens ($3), wait for lane B,
# then run lane A.
cd "$(dirname $0)/.."
R=results/pyautolens_slam
while kill -0 $1 2>/dev/null; do sleep 60; done
kill -CONT $3 $2
while kill -0 $2 2>/dev/null; do sleep 60; done
PAL_THREADS=2 scripts/launch_pyautolens_slam.sh $(cat $R/lane_a.txt) > $R/lane_a.log 2>&1
