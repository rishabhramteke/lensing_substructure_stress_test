#!/bin/zsh
# Referee round 22 CPU experiments, after launch_round21_cpu.sh. No wall-clock limit. 300-lens seed subsamples.
#   M3  out-of-family multipole: elliptical m=4 (Paugnat & Gilman 2025) truth, scanned bare and with a free CIRCULAR m=4
#   M1b source misspecification: two faint clumps in the source, fitted with a single Sersic (clean, c=60, multipole)
#   M6  template control on the joint scan: c=60 template and the c-M relation, on clean, c=60 and c=15
#   caffeinate -i zsh scripts/launch_round22_cpu.sh
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "waiting for round 21 $(date)"
# (runs alongside launch_round21_cpu.sh; the two write disjoint roots)
echo "starting $(date)"
W=${WORKERS:-13}
run() {  # run <population> <n> <n_subsample> <seed> <out> [extra args]
  pop=$1; n=$2; nsub=$3; seed=$4; out=$5; shift 5
  if [ -f $out/manifest.json ]; then echo "skip $out"; return; fi
  mkdir -p $(dirname $out); echo "=== $out  $(date)"
  python scripts/run_baseline_a_joint.py --population $pop --n $n --n-subsample $nsub --seed $seed \
      --workers $W --lens-timeout 1e9 --out $out "$@" > $out.log 2>&1
}
run ellmultipole_m4_a3     300 300 3 results/baseline_a_joint_mp4/ellmultipole_m4_a3   --concentration 15 --macro-multipole
run ellmultipole_m4_a3     300 300 3 results/baseline_a_joint_ellmp/ellmultipole_m4_a3 --concentration 15
run clumpy_no_subhalo      300 300 1 results/baseline_a_joint_clumpy/no_subhalo        --concentration 15
run clumpy_multipole_m4_a3 300 300 3 results/baseline_a_joint_clumpy/multipole_m4_a3   --concentration 15
run clumpy_fixed60         300 300 0 results/baseline_a_joint_clumpy/test_fixed60      --concentration 15
run no_subhalo   300 300 1 results/baseline_a_joint_tmpl60/no_subhalo   --concentration 60
run test_fixed60 300 300 0 results/baseline_a_joint_tmpl60/test_fixed60 --concentration 60
run test_fixed15 300 300 0 results/baseline_a_joint_tmpl60/test_fixed15 --concentration 60
run no_subhalo   300 300 1 results/baseline_a_joint_tmplcm/no_subhalo   --concentration cm
run test_fixed60 300 300 0 results/baseline_a_joint_tmplcm/test_fixed60 --concentration cm
run test_fixed15 300 300 0 results/baseline_a_joint_tmplcm/test_fixed15 --concentration cm
python scripts/regate_joint.py --roots results/baseline_a_joint_ellmp results/baseline_a_joint_clumpy results/baseline_a_joint_tmpl60 results/baseline_a_joint_tmplcm --report
python scripts/regate_joint.py --roots results/baseline_a_joint_mp4 --report
echo "ALL DONE $(date)"
