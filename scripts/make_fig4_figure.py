"""Fig 4 interference figure: mean adaptation amount (Fig-3I quantity) of two
opposite-rotation targets vs their angular separation, SL-ring vs random embedding.
Interference => low adaptation at small separation, recovering as targets separate."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import paper_fig_dir, revision_fig_dir

COLOR = {"sl": "#2E8B57", "random": "#A94850"}
LABEL = {"sl": "SL-learned ring (ours)", "random": "Random 2-D embedding"}


def main():
    rows = []
    for f in glob.glob(str(Path(paper_fig_dir) / "fig4_*_sep*.json")):
        rows += json.load(open(f))
    if not rows:
        print("no fig4_*_sep*.json"); return
    agg = defaultdict(list)
    for r in rows:
        agg[(r["condition"], r["sep"])].append(r["adapt_mean"])

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for cond in ["sl", "random"]:
        pts = sorted([(sep, np.mean(v), np.std(v) / max(1, np.sqrt(len(v))))
                      for (c, sep), v in agg.items() if c == cond])
        if pts:
            xs, ms, es = zip(*pts)
            ax.errorbar(xs, ms, yerr=es, marker="o", color=COLOR[cond], label=LABEL[cond], capsize=3, lw=2)
    ax.set(xlabel="angular separation between the two targets (deg)",
           ylabel="adaptation amount (%, mean of both targets)",
           title="Dual-adaptation interference vs target separation")
    ax.legend(frameon=False, fontsize=9); ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "fig4_interference"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png")
    for cond in ["sl", "random"]:
        for (c, sep), v in sorted(agg.items()):
            if c == cond:
                print(f"  {cond} sep={sep}: adapt={np.mean(v):.1f} +/- {np.std(v)/max(1,np.sqrt(len(v))):.1f}")


if __name__ == "__main__":
    main()
