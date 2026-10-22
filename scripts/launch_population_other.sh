#!/bin/zsh
# The sub-threshold population arm for Families B, C and gravitational imaging (Family A runs in launch_round26_queue.sh).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
mkdir -p results/baseline_b_pop/fitted results/baseline_bgi_pop
for spec in "pop_no_subhalo 1 no_subhalo" "pop_fixed60 0 test_fixed60" "pop_fixed15 0 test_fixed15"; do
  set -- ${=spec}
  [ -f results/baseline_b_pop/fitted/$3/manifest.json ] || python scripts/run_family_b.py --population $1 --variant fitted --seed $2 --n 300 --out results/baseline_b_pop/fitted/$3 > results/baseline_b_pop_$3.log 2>&1 &
done
wait
for spec in "pop_no_subhalo no_subhalo" "pop_fixed60 test_fixed60" "pop_fixed15 test_fixed15"; do
  set -- ${=spec}
  ids=$(python -c "import numpy as np;n=sum(1 for _ in open('data/$1/truth.jsonl'));r=np.random.default_rng(0);i=np.sort(r.choice(n,300,replace=False));print(' '.join(map(str,i[:100])))")
  for k in 0 1; do
    python scripts/run_family_b_gi.py --population $1 --only ${=ids} --shard $k/2 --out results/baseline_bgi_pop/$2 > results/baseline_bgi_pop_$2_$k.log 2>&1 &
  done
done
wait
python scripts/confirm_family_b.py --root results/baseline_bgi_pop --min-stat 10 --out family_bgi_pop_confirmation.json --workers 4 --data-prefix pop_ > results/family_bgi_pop_confirmation.log 2>&1
echo "POPULATION OTHER DONE $(date)"
