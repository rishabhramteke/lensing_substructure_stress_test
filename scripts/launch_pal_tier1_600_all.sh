#!/bin/zsh
# The published PyAutoLens subhalo pipeline on every Tier-1 (COSMOS-source) lens at 600 s: 50 clean and 50 c=60.
# Clean and c=60 lenses alternate in each chain so that a partial run already gives both rates; $1 chains run in
# parallel. A lens that already has a result is skipped, so the script can be restarted at any time.
cd "$(dirname $0)/.."
N=${1:-3}; R=results/pyautolens_slam_tier1_600; mkdir -p $R/no_subhalo $R/fixed60
done_idx() { [ -f $R/$1/scan_results.jsonl ] && python3 -c "import json,sys;print(' '.join(str(json.loads(l)['index']) for l in open(sys.argv[1])))" $R/$1/scan_results.jsonl; }
D0=" $(done_idx no_subhalo) "; D6=" $(done_idx fixed60) "
L=()
for i in $(seq 0 49); do
  [[ "$D0" == *" $i "* ]] || L+=("no_subhalo:$i")
  [[ "$D6" == *" $i "* ]] || L+=("fixed60:$i")
done
echo "$(date) ${#L} lenses to run in $N chains"
for c in $(seq 1 $N); do
  chain=(); k=0
  for x in $L; do (( k % N == c - 1 )) && chain+=("$x"); (( k++ )); done
  screen -dmS pal1_$c zsh -c "cd $PWD; caffeinate -i scripts/queue_pal_tier1_600.sh 0 '${chain[*]}' > results/queue_pal_tier1_600_all_$c.log 2>&1"
  echo "chain $c: ${#chain} lenses, first ${chain[1]}"
done
