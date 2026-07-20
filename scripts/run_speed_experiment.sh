#!/bin/bash
# ============================================================================
# R1 efficacy / action-scaling speed experiment (speed_vs_nactions figure).
# Multi-target centre-out reach; embedding (sl) vs standard actor-critic vs N.
#
# Learning rates TUNED by grid search at N=384 (150k eps), 2026-07-14:
#   standard : actor_lr = 1e-3   (best of {1e-4:43deg, 5e-4:47, 1e-3:22.7, 3e-3:26})
#   sl       : actor_lr = 1e-4   (best final of {1e-4:10.3, 5e-4:13.7, 1e-3:14.3})
#   critic_lr = 5 * actor_lr (set inside multitarget_bottleneck.py)
# 6 seeds x 8 action-counts x 300k episodes. Run from repo root with PYTHONPATH set.
# Overwrites models/paper/multitarget_{sl,standard}_seed{S}_nact{N}.pth (no tag).
# ============================================================================
set -u
NS="8 16 24 48 96 192 384 768"
SEEDS="0 1 2 3 4 5"
EPISODES=300000
STD_LR=1e-3
SL_LR=1e-4
MAXJOBS=12
mkdir -p speed_logs

jobs_list=speed_logs/jobs.txt
: > "$jobs_list"
for seed in $SEEDS; do
  for N in $NS; do
    echo "standard $seed $N $STD_LR" >> "$jobs_list"
    echo "sl $seed $N $SL_LR" >> "$jobs_list"
  done
done

run_one() {
  read agent seed N lr <<< "$1"
  OMP_NUM_THREADS=3 python3 scripts/multitarget_bottleneck.py --agent "$agent" --seed "$seed" \
    --num_actions "$N" --episodes "$EPISODES" --actor_lr "$lr" \
    > "speed_logs/mt_${agent}_s${seed}_N${N}.log" 2>&1
}
export -f run_one
export EPISODES

cat "$jobs_list" | xargs -P "$MAXJOBS" -I {} bash -c 'run_one "$@"' _ {}
echo "ALL_SPEED_RUNS_DONE"
