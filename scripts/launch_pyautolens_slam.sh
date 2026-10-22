#!/bin/zsh
# The published PyAutoLens subhalo pipeline on the lenses given as "population:index" arguments, one after another.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
# few threads: on a shared, oversubscribed machine XLA's fine-grained parallel kernels stall on descheduled workers
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${PAL_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS=${PAL_THREADS:-2} OMP_NUM_THREADS=${PAL_THREADS:-2} OPENBLAS_NUM_THREADS=${PAL_THREADS:-2}
for spec in "$@"; do
  pop=${spec%%:*}; i=${spec##*:}
  echo "start $pop $i $(date)"
  python scripts/run_pyautolens_slam_subhalo.py --population $pop --only $i --out results/pyautolens_slam/$pop >> results/pyautolens_slam/log_${pop}_${i}.txt 2>&1
  echo "done $pop $i exit $? $(date)"
done
