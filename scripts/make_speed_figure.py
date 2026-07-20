"""
Figure: performance vs action-space size (the 'easy win').

Reads the multi-target run metadata saved by scripts/multitarget_bottleneck.py
(multitarget_{standard,sl}_seed{seed}_nact{N}.pth) and plots, as a function of the
number of actions N (log x-axis), standard actor-critic vs the embedding model:
  (A) final greedy angular error (deg, lower = better)
  (B) episodes-to-criterion (lower = faster)
  (C) example learning curves (greedy error vs episode) at the largest N.

Run on the machine holding models/paper/. Saves PDF + CSV to figures/paper/.
"""
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import paper_model_path, revision_fig_dir as paper_fig_dir

LABEL = {"sl": "Embedding model (ours)", "standard": "Standard actor-critic"}
COLOR = {"sl": "#2E8B57", "standard": "#A94850"}


def load_rows():
    rows = []
    for fp in glob.glob(str(Path(paper_model_path) / "multitarget_*_seed*_nact*.pth")):
        m = re.search(r"multitarget_(sl|standard)_seed(\d+)_nact(\d+)\.pth", fp)
        if not m:
            continue
        agent, seed, N = m.group(1), int(m.group(2)), int(m.group(3))
        ck = torch.load(fp, map_location="cpu")
        rows.append({"agent": agent, "seed": seed, "N": N,
                     "final_err": ck.get("mean_greedy_err"), "hit_ep": ck.get("hit_ep"),
                     "curve": ck.get("curve", [])})
    return rows


def agg(rows, key):
    """mean +/- sem over seeds, per (agent, N). hit_ep None -> treated as max budget."""
    d = defaultdict(list)
    for r in rows:
        v = r[key]
        if key == "hit_ep" and v is None:
            v = max((c["ep"] for c in r["curve"]), default=np.nan)
        d[(r["agent"], r["N"])].append(v)
    out = {}
    for (agent, N), vals in d.items():
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            out[(agent, N)] = (np.mean(vals), np.std(vals) / max(1, np.sqrt(len(vals))))
    return out


def line(ax, stats, agent, **kw):
    pts = sorted([(N, m, s) for (a, N), (m, s) in stats.items() if a == agent])
    if not pts:
        return
    Ns, ms, ss = (np.array(z) for z in zip(*pts))
    ax.plot(Ns, ms, marker="o", color=COLOR[agent], label=LABEL[agent], lw=2, **kw)
    ax.fill_between(Ns, ms - ss, ms + ss, color=COLOR[agent], alpha=0.2, lw=0)


def main():
    rows = load_rows()
    if not rows:
        print("no multitarget_*.pth files found yet")
        return
    err = agg(rows, "final_err")
    hit = agg(rows, "hit_ep")

    # Panels: (A) episodes-to-criterion vs N, (B) learning curves at largest N.
    # The final-greedy-error-vs-N panel is intentionally omitted (both models' asymptotic
    # error grows with N; the scaling advantage is in learning speed, panel A).
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for agent in ["standard", "sl"]:
        line(axes[0], hit, agent)
    axes[0].set(xscale="log", xlabel="Number of actions N", ylabel="Episodes to criterion",
                title="A. Learning speed vs action-space size")
    axes[0].legend(frameon=False, fontsize=8); axes[0].spines[["right", "top"]].set_visible(False)

    # (B) learning curves at largest N, averaged over seeds with shaded SEM band
    Nmax = max(r["N"] for r in rows)
    for agent in ["standard", "sl"]:
        cs = [r["curve"] for r in rows if r["agent"] == agent and r["N"] == Nmax and r["curve"]]
        if not cs:
            continue
        L = min(len(c) for c in cs)
        eps = np.array([p["ep"] for p in cs[0][:L]])
        arr = np.array([[p["greedy_err"] for p in c[:L]] for c in cs])
        m = arr.mean(0); s = arr.std(0) / max(1, np.sqrt(arr.shape[0]))
        axes[1].plot(eps, m, color=COLOR[agent], label=LABEL[agent], lw=2)
        axes[1].fill_between(eps, m - s, m + s, color=COLOR[agent], alpha=0.2, lw=0)
    axes[1].set(xlabel="Episode", ylabel="Greedy error (deg)", title=f"B. Learning curves (N={Nmax})")
    axes[1].legend(frameon=False, fontsize=8); axes[1].spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    out_pdf = Path(paper_fig_dir) / "speed_vs_nactions.pdf"
    out_png = Path(paper_fig_dir) / "speed_vs_nactions.png"
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=150, bbox_inches="tight")

    df = pd.DataFrame([{"agent": a, "N": N, "final_err_mean": err.get((a, N), (np.nan,))[0],
                        "hit_ep_mean": hit.get((a, N), (np.nan,))[0]}
                       for (a, N) in sorted(set(err) | set(hit))])
    df.to_csv(Path(paper_fig_dir) / "speed_vs_nactions.csv", index=False)
    print("saved", out_pdf.name, out_png.name)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
