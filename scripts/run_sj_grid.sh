#!/bin/bash
# Comprehensive single-joint scaling sweep with the FIXED task+loop:
#   continuous constant-difficulty targets (no N=16 spacing-vs-TOL artifact) + dense reward +
#   episode-annealed temp. Per-agent lr is chosen from this grid so baselines are fairly
#   optimized (must tie SL at low N). agents x N x lr x 2 seeds, 300k eps.
#   -> multitarget_{agent}_cont_lr{lr}_seed{s}_nact{N}.pth  (final_success stored inside)
cd "$HOME/action-embedding" || exit 1
source .venv/bin/activate
export PYTHONPATH="$HOME/action-embedding"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
pkill -f "_long" 2>/dev/null        # drop the now-subsumed sl_long budget test
sleep 2
: > /tmp/sjgrid_jobs.txt
for ag in standard bottleneck sl; do
  for N in 8 16 24 48 96 192 384 768; do
    for lr in 3e-4 1e-3 3e-3; do
      for s in 0 1; do echo "$ag $N $lr $s" >> /tmp/sjgrid_jobs.txt; done
    done
  done
done
run_one() {
  read ag N lr s <<< "$1"
  python3 scripts/multitarget_bottleneck.py --agent "$ag" --num_actions "$N" --seed "$s" \
      --episodes 300000 --actor_lr "$lr" --continuous_targets --eval_every 10000 \
      --tag "_cont_lr${lr}" > "/tmp/sjgrid_${ag}_N${N}_lr${lr}_s${s}.log" 2>&1
}
export -f run_one
cat /tmp/sjgrid_jobs.txt | xargs -P 12 -I {} bash -c 'run_one "$@"' _ {}
echo "SJGRID DONE $(date)" > /tmp/sjgrid_done.txt
