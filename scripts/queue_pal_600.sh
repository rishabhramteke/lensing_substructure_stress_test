#!/bin/zsh
# Round 30, Block C revised: the published pipeline at the 600 s operating point (arc S/N ~5e2, where its default
# pixelized source can fit the arc), on the same lenses as its Tier-0 runs, one at a time after the Tier-0 queue ($1):
# clean 4, multipole 11, clean 20, multipole 45, multipole 45 with a free m=4 multipole, c=60 291.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PAL_QUEUE=q600
R=results/pyautolens_slam_600
mkdir -p $R
while kill -0 $1 2>/dev/null; do sleep 120; done
run() { echo "start $* $(date)"; python scripts/run_pyautolens_slam_subhalo.py "$@" >> $R/log.txt 2>&1; echo "done $* exit $? $(date)"; }
run --population test_shallow600_no_subhalo --only 4 --out $R/no_subhalo
run --population test_shallow600_multipole_m4_a3 --only 11 --out $R/multipole_m4_a3
run --population test_shallow600_no_subhalo --only 20 --out $R/no_subhalo
run --population test_shallow600_multipole_m4_a3 --only 45 --out $R/multipole_m4_a3
run --population test_shallow600_multipole_m4_a3 --only 45 --multipole --out $R/multipole_m4_a3_mp4
run --population test_shallow600_fixed60 --only 291 --out $R/test_fixed60
echo "Q600 DONE $(date)"
