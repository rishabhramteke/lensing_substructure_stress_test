#!/bin/zsh
# Family B-GI pilot: the first 60 lenses of the seed-0 300-lens subsample of the clean, c=60 and multipole populations,
# 4 single-threaded workers per population (resumable; the full run later skips what is done).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for pop in no_subhalo test_fixed60 multipole_m4_a3; do
  N=$(python -c "import json,numpy as np;n=sum(1 for _ in open('data/$pop/truth.jsonl'));r=np.random.default_rng(0);i=np.sort(r.choice(n,size=min(300,n),replace=False)) if n>300 else np.arange(n);print(' '.join(map(str,i[:60])))")
  arr=(${=N})
  for k in 0 1 2 3; do
    sub=(); for ((j=k+1; j<=${#arr}; j+=4)); do sub+=${arr[j]}; done
    python scripts/run_family_b_gi.py --population $pop --only ${sub[@]} --out results/baseline_bgi/$pop > results/baseline_bgi_pilot_${pop}_$k.log 2>&1 &
  done
  wait
  echo "done $pop $(date)"
done
