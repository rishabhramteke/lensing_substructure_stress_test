#!/bin/zsh
# Stop runs at a reduced sample (first use: the population arm (Family A at 150 of its 300 lenses, gravitational imaging at 60 of 100
# per population) so the remaining queue can use the CPU; each stopped run gets a manifest recording the trim.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
trim_a() { # trim_a <out> <n>
  local out=$1 n=$2
  while [ ! -f $out/manifest.json ]; do
    local k=$(cat $out/scan_results.jsonl 2>/dev/null | wc -l)
    if (( k >= n )); then
      pkill -f -- "--out $out\$" ; sleep 2; pkill -9 -f -- "--out $out\$"
      python - <<PY
import json
from pathlib import Path
out = Path("$out"); recs = []
for l in open(out / "scan_results.jsonl"):
    try: recs.append(json.loads(l))
    except json.JSONDecodeError: pass
recs.sort(key=lambda r: r["index"])
with open(out / "scan_results.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
(out / "manifest.json").write_text(json.dumps({"population": out.name, "n": len(recs), "trimmed": True,
    "note": "stopped at a reduced sample to free CPU (referee round 26); the records are the first lenses of the seed subsample to finish"}, indent=2))
print("trimmed", out, len(recs))
PY
      return
    fi
    sleep 60
  done
}
trim_gi() { # trim_gi <out-dir> <n>  (both shards append to the same file)
  local out=$1 n=$2
  while true; do
    local k=$(cat $out/scan_results.jsonl 2>/dev/null | wc -l)
    if (( k >= n )); then pkill -f -- "--out $out\$"; echo "trimmed GI $out at $k"; return; fi
    pgrep -f -- "--out $out\$" >/dev/null || { echo "GI $out finished on its own ($k)"; return; }
    sleep 60
  done
}
trim_a ${A1:-results/baseline_a_joint_super600/test_fixed60} ${N1:-299} &
trim_a ${A2:-results/baseline_a_joint_psferr/no_subhalo} ${N2:-30} &
wait
echo "TRIM DONE $(date)"
