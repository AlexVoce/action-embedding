"""Multi-joint (d-dimensional reach) speed figure: success rate & episodes-to-criterion
vs action dimensionality d (and vs N = k**d). Reads figures/paper/mdmt_d*.json."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import paper_fig_dir, revision_fig_dir

LABEL = {"embedding": "Embedding model (ours)", "standard": "Standard actor-critic"}
COLOR = {"embedding": "#2E8B57", "standard": "#A94850"}


def main():
    rows = []
    for f in sorted(glob.glob(str(Path(paper_fig_dir) / "mj_d*.json"))):
        rows += json.load(open(f))
    if not rows:
        print("no mdmt_d*.json found"); return

    def agg(key, none_val):
        d = defaultdict(list)
        for r in rows:
            v = r[key]
            v = none_val if v is None else v
            d[(r["agent"], r["d"], r["N"])].append(v)
        return {k: (np.mean(v), np.std(v) / max(1, np.sqrt(len(v)))) for k, v in d.items()}

    succ = agg("final_success", 0.0)
    hit = agg("hit_ep", 150000)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for agent in ["standard", "embedding"]:
        pts = sorted([(d, N, m, s) for (a, d, N), (m, s) in succ.items() if a == agent])
        if pts:
            ds, Ns, ms, ss = zip(*pts)
            axes[0].errorbar(ds, ms, yerr=ss, marker="o", color=COLOR[agent], label=LABEL[agent], capsize=3)
        pts = sorted([(d, N, m, s) for (a, d, N), (m, s) in hit.items() if a == agent])
        if pts:
            ds, Ns, ms, ss = zip(*pts)
            axes[1].errorbar(ds, ms, yerr=ss, marker="o", color=COLOR[agent], label=LABEL[agent], capsize=3)
    Ns_by_d = {d: N for (_, d, N) in succ}
    xt = sorted(Ns_by_d)
    xtl = [f"{d}\n(N={Ns_by_d[d]})" for d in xt]
    axes[0].set(xlabel="Action dimensionality d", ylabel="Greedy success rate",
                title="Multi-joint reach: performance vs d"); axes[0].set_xticks(xt); axes[0].set_xticklabels(xtl)
    axes[1].set(xlabel="Action dimensionality d", ylabel="Episodes to criterion",
                title="Multi-joint reach: learning speed vs d"); axes[1].set_xticks(xt); axes[1].set_xticklabels(xtl)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8); ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "speed_vs_dimensionality"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".pdf/.png")
    for (a, d, N), (m, s) in sorted(succ.items()):
        print(f"  {a:<10} d={d} N={N:<4} success={m:.2f}+/-{s:.2f}  hit_ep={hit.get((a,d,N),(float('nan'),))[0]:.0f}")


if __name__ == "__main__":
    main()
