#!/bin/zsh
# After launch_unet_round23.sh: the four standard networks on the 10 000-lens clean control (0.1% FPR), and the
# realistic-arm networks on the realistic decoys and the full realistic test populations.
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
until grep -a -q "ALL DONE" results/launch_unet_round23.log; do sleep 120; done
python scripts/evaluate_unet_populations.py --set 30k --pops no_subhalo_10k test_fixed60 test_fixed15 --tag _bigclean > results/unet_bigclean.log 2>&1
python scripts/evaluate_unet_populations.py --set realistic --clean realistic_no_subhalo --tag _realistic \
    --pops realistic_fixed60 realistic_fixed15 realistic_multipole_m4_a3 realistic_decoy_gaussian_s42_a10 realistic_decoy_gaussian_s43_a10 realistic_decoy_dipole_s42_a10 > results/unet_realistic_pops.log 2>&1
echo "ALL DONE $(date)"
