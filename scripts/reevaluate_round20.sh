#!/bin/zsh
# Referee round 20: once both completion launchers and the blind-init run have finished, re-gate every joint root,
# evaluate the two new controls, then regenerate every table, figure and number (reevaluate_round19.sh).
source /Users/rishabh.ramteke/myenv/bin/activate
cd /Users/rishabh.ramteke/Documents/space_research/projects/05_dark_matter_strong_lensing
until grep -q "ALL DONE" results/launch_complete_truncated.log 2>/dev/null && grep -q "ALL DONE" results/launch_joint_blind.log 2>/dev/null \
      && { [ ! -f results/launch_complete_truncated_tail.log ] || grep -q "ALL DONE" results/launch_complete_truncated_tail.log; }; do sleep 300; done
# a population a launcher skipped because the other held it is finished by the other; one last sweep catches anything left
python scripts/complete_truncated_joint.py --roots $(ls -d results/baseline_a_joint_{full,mp4,mp34,freepos,decoy,los,lenslight,lenslight2,psf015,shallow,tidal,shallow600,cm}) --workers 12 > results/eval_r20_final_sweep.log 2>&1
python scripts/regate_joint.py > results/eval_r20_regate.log 2>&1
python scripts/evaluate_joint_blind.py > results/eval_r20_blind.log 2>&1
python scripts/evaluate_guard_completion.py > results/eval_r20_guard.log 2>&1
zsh scripts/reevaluate_round19.sh > results/eval_r20_reevaluate.log 2>&1
(cd scripts && python print_paper_numbers.py > ../results/paper_numbers_round20.txt 2>&1)
echo "ROUND20 DONE $(date)" >> results/eval_r20_reevaluate.log
