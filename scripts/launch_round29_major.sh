#!/bin/zsh
# Round 29 major revisions, CPU part (after launch_round29_more.sh):
#  M1b gravitational imaging (stage 1 + parametric confirmation) on the double-Sersic lens-light arm (300 lenses x 3) and
#      the 600 s arm (200 x 2);
#  M1d a practical joint scan: data-only start, free lens centre, free perturber position (300 lenses x 3);
#  M4  the 0.15" PSF arm on the full 1 000-lens populations (joint scan); the lens-light arm at 1 000 lenses for Family B.
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R=results; W=${W:-10}
until grep -q "ROUND29 MORE DONE" $R/launch_round29_more.log 2>/dev/null; do sleep 60; done
gi() {  # population, n, out-root, out-name, lens-light components
  local out=$3/$4; mkdir -p $out
  [ -f $out/.done ] && return
  for k in $(seq 0 $((W-1))); do python scripts/run_family_b_gi.py --population $1 --n $2 --seed 0 --shard $k/$W --lens-light-components $5 --out $out > $out.shard$k.log 2>&1 & done
  wait; touch $out/.done; echo "gi done $out $(date)"
}
for p in no_subhalo fixed60 multipole_m4_a3; do gi lenslight_$p 300 $R/baseline_bgi_lenslight ${p/fixed60/test_fixed60} 2; done
for p in no_subhalo fixed60; do gi test_shallow600_$p 200 $R/baseline_bgi_600 ${p/fixed60/test_fixed60} 1; done
python scripts/confirm_family_b.py --workers $W --root $R/baseline_bgi_lenslight --min-stat 10 --data-prefix lenslight_ --lens-light-components 2 --out family_bgi_lenslight_confirmation.json > $R/confirm_bgi_lenslight.log 2>&1
python scripts/confirm_family_b.py --workers $W --root $R/baseline_bgi_600 --min-stat 10 --data-prefix test_shallow600_ --out family_bgi_600_confirmation.json > $R/confirm_bgi_600.log 2>&1
echo "gi confirm done $(date)"
j() { local out=$1; shift; mkdir -p $(dirname $out); [ -f $out/manifest.json ] || python scripts/run_baseline_a_joint.py "$@" --concentration 15 --workers $W --lens-timeout 1e9 --out $out > $out.log 2>&1; echo "done $out exit $? $(date)"; }
j $R/baseline_a_joint_practical/no_subhalo --population no_subhalo --n 300 --n-subsample 300 --seed 1 --blind --free-centre --free-position
j $R/baseline_a_joint_practical/test_fixed60 --population test_fixed60 --n 300 --n-subsample 300 --seed 0 --blind --free-centre --free-position
j $R/baseline_a_joint_practical/multipole_m4_a3 --population multipole_m4_a3 --n 300 --n-subsample 300 --seed 3 --blind --free-centre --free-position
j $R/baseline_a_joint_psf015_full/no_subhalo --population psf015_no_subhalo --n 1000 --n-subsample 1000 --seed 1
j $R/baseline_a_joint_psf015_full/test_fixed60 --population psf015_fixed60 --n 1000 --n-subsample 1000 --seed 0
for p in no_subhalo fixed60 multipole_m4_a3; do
  [ -f data/lenslight1k_$p/manifest.json ] || python scripts/generate_tier0.py --config lenslight_$p --n 1000 --seed 7 --out data/lenslight1k_$p > data/lenslight1k_$p.gen.log 2>&1
  python scripts/run_family_b.py --population lenslight1k_$p --n 1000 --seed 0 --lens-light-components 2 --out $R/baseline_b_lenslight1k > $R/baseline_b_lenslight1k_$p.log 2>&1
  echo "B lenslight1k $p exit $? $(date)"
done
echo "ROUND29 MAJOR DONE $(date)"
