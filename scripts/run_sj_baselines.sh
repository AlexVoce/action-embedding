#!/bin/bash
# Re-run the single-joint BASELINES (standard, bottleneck) properly optimized: lower lr (1e-4)
# + longer budget (600k), all N, 2 seeds. The 300k grid under-tuned them (lr too high), so they
# missed the low-N ceiling. SL is already optimal (ceiling in ~15k eps) so it is NOT re-run.
# -> multitarget_{agent}_cont_lr1e-4_seed{s}_nact{N}.pth ; make_sj_analysis then picks best lr.
cd "$HOME/action-embedding" || exit 1
source .venv/bin/activate
export PYTHONPATH="$HOME/action-embedding"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: > /tmp/sjbase_jobs.txt
for ag in standard bottleneck; do
  for N in 8 16 24 48 96 192 384 768; do
    for s in 0 1; do echo "$ag $N $s" >> /tmp/sjbase_jobs.txt; done
  done
done
run_one() {
  read ag N s <<< "$1"
  python3 scripts/multitarget_bottleneck.py --agent "$ag" --num_actions "$N" --seed "$s" \
      --episodes 600000 --actor_lr 1e-4 --continuous_targets --eval_every 20000 \
      --tag "_cont_lr1e-4" > "/tmp/sjbase_${ag}_N${N}_s${s}.log" 2>&1
}
export -f run_one
cat /tmp/sjbase_jobs.txt | xargs -P 12 -I {} bash -c 'run_one "$@"' _ {}
echo "SJBASE DONE $(date)" > /tmp/sjbase_done.txt
