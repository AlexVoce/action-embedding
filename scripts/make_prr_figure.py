"""Proprioceptive reacher scaling figure: episodes-to-criterion and final success vs N
(=k^2) for embedding vs standard. Tests whether the embedding's learning speed stays flat
while standard degrades as the action space grows."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import paper_fig_dir, revision_fig_dir

LABEL = {"embedding": "Embedding (ours)", "standard": "Standard AC"}
COLOR = {"embedding": "#2E8B57", "standard": "#A94850"}
CAP = 150000


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "prr2"
    rows = []
    for f in sorted(glob.glob(str(Path(paper_fig_dir) / f"{tag}_k*.json"))):
        rows += json.load(open(f))
    if not rows:
        print(f"no {tag}_k*.json"); return

    def agg(key, none_val):
        d = defaultdict(list)
        for r in rows:
            v = r[key]
            v = none_val if v is None else v
            d[(r["agent"], r["N"])].append(v)
        return {k: (np.mean(v), np.std(v) / max(1, np.sqrt(len(v)))) for k, v in d.items()}

    succ = agg("final_success", 0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    for agent in ["standard", "embedding"]:
        pts = sorted([(N, m, s) for (a, N), (m, s) in succ.items() if a == agent])
        if pts:
            Ns, ms, ss = zip(*pts)
            ax.errorbar(Ns, ms, yerr=ss, marker="o", color=COLOR[agent], label=LABEL[agent], capsize=3, lw=2)
    ax.set(xscale="log", xlabel="number of actions  N = k²", ylabel="final success rate",
           title="Two-joint reacher: reaching success vs action-space size")
    ax.legend(frameon=False, fontsize=9); ax.spines[["right", "top"]].set_visible(False)
    hit = succ  # keep name for the summary print below
    fig.tight_layout()
    out = Path(revision_fig_dir) / f"proprio_reacher_scaling_{tag}"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png")
    for (a, N), (m, s) in sorted(hit.items()):
        print(f"  {a:<10} N={N:<5} hit_ep={m:.0f}  success={succ.get((a,N),(np.nan,))[0]:.2f}")


if __name__ == "__main__":
    main()
