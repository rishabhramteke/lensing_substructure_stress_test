#!/bin/zsh
# Gravitational imaging (stage 1 + parametric confirmation at Delta ln Z >= 10, Delta chi^2 >= 20) on the arms the summary
# map had not run it on: the elliptical multipole and the foreground line-of-sight halo (300 lenses, Tier-0 calibration),
# the 600 s multipole lenses (200), and 200-lens subsamples of the 0.15" PSF, combined-realism, single-Sersic lens-light,
# Euclid-like, clumpy-source and supersampled-render arms; and Family B on the elliptical multipole (1 000 lenses).
cd "$(dirname $0)/.."
source /Users/rishabh.ramteke/myenv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
R=results; W=${W:-6}
python scripts/run_family_b.py --population ellmultipole_m4_a3 --n 1000 --seed 0 --out $R/baseline_b_ell/fitted/ellmultipole_m4_a3 > $R/baseline_b_ell.log 2>&1
echo "B ell exit $? $(date)"
gi() {  # population, n, out-root, out-name, lens-light components
  local out=$3/$4; mkdir -p $out
  [ -f $out/.done ] && return
  for k in $(seq 0 $((W-1))); do python scripts/run_family_b_gi.py --population $1 --n $2 --seed 0 --shard $k/$W --lens-light-components $5 --out $out > $out.shard$k.log 2>&1 & done
  wait; touch $out/.done; echo "gi done $out $(date)"
}
conf() {  # root, extra args
  local root=$1; shift
  python scripts/confirm_family_b.py --workers $W --root $root --min-stat 10 "$@" > $root.confirm.log 2>&1; echo "confirm $root exit $? $(date)"
}
gi ellmultipole_m4_a3 300 $R/baseline_bgi_ell ellmultipole_m4_a3 1
conf $R/baseline_bgi_ell --pops ellmultipole_m4_a3=ellmultipole_m4_a3 --out family_bgi_ell_confirmation.json
gi los_halo_z025 300 $R/baseline_bgi_los los_halo_z025 1
conf $R/baseline_bgi_los --pops los_halo_z025=los_halo_z025 --out family_bgi_los_confirmation.json
gi test_shallow600_multipole_m4_a3 200 $R/baseline_bgi_600 multipole_m4_a3 1
conf $R/baseline_bgi_600 --data-prefix test_shallow600_ --out family_bgi_600_confirmation.json
for p in no_subhalo fixed60; do gi psf015_$p 200 $R/baseline_bgi_psf015 ${p/fixed60/test_fixed60} 1; done
conf $R/baseline_bgi_psf015 --data-prefix psf015_ --out family_bgi_psf015_confirmation.json
for p in no_subhalo fixed60 multipole_m4_a3; do gi realistic_$p 200 $R/baseline_bgi_realistic ${p/fixed60/test_fixed60} 2; done
conf $R/baseline_bgi_realistic --data-prefix realistic_ --lens-light-components 2 --out family_bgi_realistic_confirmation.json
for p in no_subhalo fixed60 multipole_m4_a3; do gi lenslight_$p 200 $R/baseline_bgi_lenslight1 ${p/fixed60/test_fixed60} 1; done
conf $R/baseline_bgi_lenslight1 --data-prefix lenslight_ --lens-light-components 1 --out family_bgi_lenslight1_confirmation.json
for p in no_subhalo fixed60 multipole_m4_a3; do gi euclid_$p 200 $R/baseline_bgi_euclid ${p/fixed60/test_fixed60} 1; done
conf $R/baseline_bgi_euclid --data-prefix euclid_ --out family_bgi_euclid_confirmation.json
gi clumpy_no_subhalo 200 $R/baseline_bgi_clumpy no_subhalo 1
conf $R/baseline_bgi_clumpy --data-prefix clumpy_ --out family_bgi_clumpy_confirmation.json
gi invcrime_no_subhalo 200 $R/baseline_bgi_invcrime no_subhalo 1
conf $R/baseline_bgi_invcrime --data-prefix invcrime_ --out family_bgi_invcrime_confirmation.json
echo "ROUND31 GI DONE $(date)"
