"""Random-embedding control figure: rotation-generalization vs angle-from-target for the
SL-learned ring vs a random 2-D embedding (mean +/- sem over seeds). SL should generalize
locally (peak near the trained target, decaying with angle); random should be flat/absent."""
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import paper_fig_dir, revision_fig_dir

COLOR = {"sl": "#2E8B57", "random": "#A94850"}
LABEL = {"sl": "SL-learned ring (ours)", "random": "Random 2-D embedding"}
ANGLES = [-135, -90, -45, 0, 45, 90, 135, 180]


def main():
    data = {"sl": defaultdict(list), "random": defaultdict(list)}
    for fp in glob.glob(str(Path(paper_fig_dir) / "generalization_trackA_*_target135_rot-30.csv")):
        m = re.search(r"trackA_(sl|random)_seed(\d+)_", fp)
        if not m:
            continue
        cond = m.group(1)
        df = pd.read_csv(fp)
        g = df.groupby("angle from target")["rotation generalization"].mean()
        for a in ANGLES:
            if a in g.index:
                data[cond][a].append(g.loc[a])

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for cond in ["sl", "random"]:
        xs = [a for a in ANGLES if data[cond][a]]
        ms = [np.mean(data[cond][a]) for a in xs]
        es = [np.std(data[cond][a]) / max(1, np.sqrt(len(data[cond][a]))) for a in xs]
        ax.errorbar(xs, ms, yerr=es, marker="o", color=COLOR[cond], label=LABEL[cond], capsize=3, lw=2)
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set(xlabel="angle from adapted target (deg)", ylabel="rotation generalization (%)",
           title="Adaptation generalization requires the SL-learned structure")
    ax.legend(frameon=False, fontsize=9); ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "random_control_generalization"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png")
    for cond in ["sl", "random"]:
        loc = np.mean([np.mean(data[cond][a]) for a in [-45, 0, 45] if data[cond][a]])
        glob_ = np.mean([np.mean(data[cond][a]) for a in [-135, -90, 90, 135, 180] if data[cond][a]])
        print(f"  {cond}: local~{loc:.0f}%  global~{glob_:.0f}%")


if __name__ == "__main__":
    main()
