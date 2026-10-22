#!/bin/zsh
# The pixelized-source evidence search (run_family_a_pix.py, fixed-grid source) on the stratified subset
# results/apix_subset.json: 60 c=60 lenses in four mass bins, 60 clean, 30 multipole and 10 per decoy set.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
mkdir -p results/baseline_apix
go() { # go <pop> <K>
  local ids=$(python -c "import json;print(' '.join(map(str,json.load(open('results/apix_subset.json'))['$1'])))")
  for k in $(seq 0 $(($2-1))); do
    python scripts/run_family_a_pix.py --population $1 --only ${=ids} --shard $k/$2 --out results/baseline_apix/$1 > results/baseline_apix/$1_$k.log 2>&1 &
  done
}
go test_fixed60 5; go no_subhalo 4; go multipole_m4_a3 2; go decoy_gaussian_s42_a10 1; go decoy_gaussian_s43_a10 1; go decoy_dipole_s42_a10 1
wait
echo "APIX DONE $(date)"
