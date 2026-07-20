"""Corrected random-embedding control (R1 part-b bottleneck). Rotation-generalization vs
angle-from-target, SL-learned per-seed embeddings vs random per-seed embeddings, both through the
corrected adaptation pipeline (per-seed embeddings + early-stop). Shaded SEM bands (paper style)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from definitions import paper_fig_dir, revision_fig_dir

SL_SEEDS = [2, 3, 4, 5, 6, 7, 8, 9, 35]      # corrected per-seed learned embeddings (seed 10 dropped: bad)
RAND_SEEDS = [50, 51, 52, 53, 54]            # per-seed random embeddings
COLOR = {"sl": "#2E8B57", "random": "#A94850"}
LABEL = {"sl": "SL-learned embedding (ours)", "random": "Random 2-D embedding"}
P = Path(paper_fig_dir)


def profile(seeds):
    gs = []
    for s in seeds:
        fp = P / f"generalization_stats_seed_{s}_target_135_rotation_-30.csv"
        if fp.exists():
            gs.append(pd.read_csv(fp).groupby("angle from target")["angular error"].mean())
    if not gs:
        return None, None, None
    idx = sorted(set.intersection(*[set(g.index) for g in gs]))
    arr = np.array([[g.loc[a] for a in idx] for g in gs])
    return np.array(idx), arr.mean(0), arr.std(0) / max(1, np.sqrt(arr.shape[0]))


fig, ax = plt.subplots(figsize=(5.4, 4))
for cond, seeds in [("sl", SL_SEEDS), ("random", RAND_SEEDS)]:
    x, m, s = profile(seeds)
    if x is None:
        print("no data for", cond); continue
    ax.plot(x, m, color=COLOR[cond], lw=2, label=LABEL[cond])
    ax.fill_between(x, m - s, m + s, color=COLOR[cond], alpha=0.2, lw=0)
    loc = m[(x >= -45) & (x <= 45)].mean(); gl = m[(x < -45) | (x > 45)].mean()
    print("%s: local err~%.0f global err~%.0f (n=%d)" % (cond, loc, gl, len(seeds)), flush=True)
ax.set(xlabel="angle from adapted target (°)", ylabel="angular error (°)",
       title="Adaptation generalization: learned vs random embedding", xlim=[-90, 180])
ax.invert_yaxis()  # low error (better) up, matching the Fig 3I angular-error panel
ax.legend(frameon=False, fontsize=9); ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "random_control_generalization.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
