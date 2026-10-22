#!/bin/zsh
# Pool for the gravitational-imaging runs: stage 1 (joint pixelized source + dpsi, run_family_b_gi.py), stage 2 (the
# parametric pixelized-source confirmation of every stage-1 candidate, run_family_a_pix.py --candidates-from) and the
# stand-alone pixelized-source evidence scan on the stratified subset (results/apix_subset.json).
#   zsh scripts/launch_sota_pool.sh NW      starts NW workers; running it again adds workers (tasks are mkdir-locked)
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
NW=${1:-4}; L=results/_sota_locks; mkdir -p $L results/baseline_bgi results/baseline_bgi_confirm results/baseline_apix
BIG=(no_subhalo test_fixed60 test_fixed15 multipole_m4_a3); DEC=(decoy_gaussian_s42_a10 decoy_gaussian_s43_a10 decoy_dipole_s42_a10)
tasks=()
for p in no_subhalo test_fixed60; do for k in 0 1 2 3 4 5; do tasks+="s1:$p:$k/6:300"; done; done
for p in no_subhalo test_fixed60; do for k in 0 1 2 3; do tasks+="s2:$p:$k/4"; done; done
for p in test_fixed15 multipole_m4_a3; do for k in 0 1 2 3 4 5; do tasks+="s1:$p:$k/6:300"; done; done
for p in no_subhalo test_fixed60 multipole_m4_a3; do for k in 0 1 2 3 4 5; do tasks+="ap:$p:$k/6"; done; done
for p in test_fixed15 multipole_m4_a3; do for k in 0 1 2 3; do tasks+="s2:$p:$k/4"; done; done
for p in $DEC; do for k in 0 1; do tasks+="s1:$p:$k/2:100"; done; done
for p in $DEC; do for k in 0 1; do tasks+="s2:$p:$k/2"; done; done

s1_done() {   # every stage-1 shard of population $1 finished
  local n=$(ls $L | grep -c "^s1_$1_.*\.done$"); [[ $1 == decoy_* ]] && (( n >= 2 )) || (( n >= 6 ))
}
worker() {
  while true; do
    local left=0
    for t in $tasks; do
      local kind=${t%%:*} rest=${t#*:}; local pop=${rest%%:*}; rest=${rest#*:}; local shard=${rest%%:*}; local n=${rest#*:}
      local tag=${kind}_${pop}_${shard/\//of}
      [ -f $L/$tag.done ] && continue
      left=1
      [[ $kind == s2 ]] && ! s1_done $pop && continue
      mkdir $L/$tag.lock 2>/dev/null || continue
      echo "=== $tag  $(date)  (worker $$)"
      case $kind in
        s1) python scripts/run_family_b_gi.py --population $pop --n $n --seed 0 --shard $shard --out results/baseline_bgi/$pop > $L/$tag.log 2>&1 ;;
        s2) python scripts/run_family_a_pix.py --population $pop --candidates-from results/baseline_bgi/$pop/scan_results.jsonl --shard $shard --out results/baseline_bgi_confirm/$pop > $L/$tag.log 2>&1 ;;
        ap) python scripts/run_family_a_pix.py --population $pop --only $(python -c "import json;print(' '.join(map(str,json.load(open('results/apix_subset.json'))['$pop'])))") --shard $shard --out results/baseline_apix/$pop > $L/$tag.log 2>&1 ;;
      esac
      [ $? -eq 0 ] && touch $L/$tag.done
      rmdir $L/$tag.lock
    done
    (( left )) || break
    sleep 60
  done
}
for w in $(seq 1 $NW); do worker > results/launch_sota_pool_$$_$w.log 2>&1 & done
wait
echo "ALL DONE $(date)"
