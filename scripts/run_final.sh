#!/bin/bash
# Final two-joint scaling run at the best lr PER agent (from the lr grid):
#   embedding @ lr=3e-4, standard @ lr=1e-4.  4 seeds, per-seed embeddings, all k.
# One job per (agent,k) -> prrfp_final_{agent}_k{k}.json ; 12 jobs, -P 12 (one wave).
cd "$HOME/action-embedding" || exit 1
source .venv/bin/activate
export PYTHONPATH="$HOME/action-embedding"
pkill -f proprio_reacher_rl 2>/dev/null
sleep 3
: > /tmp/final_jobs.txt
for k in 8 16 24 32 48 64; do
  echo "embedding $k 3e-4" >> /tmp/final_jobs.txt
  echo "standard $k 1e-4"  >> /tmp/final_jobs.txt
done
run_one() {
  read agent k lr <<< "$1"
  python3 scripts/proprio_reacher_rl.py --ks "$k" --seeds 0 1 2 3 --episodes 400000 \
      --emb_steps 300000 --emb_state fingertip --eval_every 10000 --agents "$agent" --actor_lr "$lr" \
      --out "figures/paper/prrfp_final_${agent}_k${k}.json" > "/tmp/final_${agent}_k${k}.log" 2>&1
}
export -f run_one
cat /tmp/final_jobs.txt | xargs -P 12 -I {} bash -c 'run_one "$@"' _ {}
echo "FINAL DONE $(date)" > /tmp/final_done.txt
