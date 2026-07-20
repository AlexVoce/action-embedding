"""A5 (Park 10deg): extend the dual-adaptation interference to small separations (10deg = Park's
actual condition, and 20deg), for SL vs random embeddings. Reuses fig4_interference.run and writes
per-(condition,sep) JSONs so make_fig4_figure picks them up.

usage: a5_smallsep.py <condition> <sep> [seeds...]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pathlib import Path
from scripts.fig4_interference import run, BASE
from definitions import paper_fig_dir

cond = sys.argv[1]
sep = int(sys.argv[2])
seeds = [int(x) for x in sys.argv[3:]] or [0, 1, 2]
torch.set_num_threads(4)

rows = []
for seed in seeds:
    r = run(cond, seed, sep, {**BASE, "seed": seed}, adapt_episodes=100000)
    rows.append(r)
    print("[a5 %s sep=%d s%d] adapt_mean=%.1f" % (cond, sep, seed, r["adapt_mean"]), flush=True)
json.dump(rows, open(Path(paper_fig_dir) / f"fig4_{cond}_sep{sep}.json", "w"))
print("A5 %s sep=%d DONE" % (cond, sep), flush=True)
