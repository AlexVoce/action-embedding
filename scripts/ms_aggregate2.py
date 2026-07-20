"""Aggregate the EARLY-STOPPED multi-seed data (ms2_*) into clean final panels: Fig 3I
(generalization + angular error, mean +/- SEM), Fig 3E (SL re-learning to the early stop),
Fig 3F (pre/post distribution), SL-vs-RL race."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from definitions import paper_fig_dir, revision_fig_dir

P = Path(paper_fig_dir)
seeds = sorted(int(f.split("seed")[-1].split(".")[0]) for f in glob.glob(str(P / "ms2_gen_seed*.csv")))
print("aggregating seeds:", seeds, flush=True)
sem = lambda a, ax=0: np.array(a).std(ax) / max(1, np.sqrt(np.array(a).shape[ax]))

gens = []
for s in seeds:
    df = pd.read_csv(P / f"ms2_gen_seed{s}.csv")
    gens.append(df.groupby("angle from target")[["rotation generalization", "angular error"]].mean())
idx = sorted(set.intersection(*[set(g.index) for g in gens]))
rot_gen = np.array([[g.loc[a, "rotation generalization"] for a in idx] for g in gens])
ang_err = np.array([[g.loc[a, "angular error"] for a in idx] for g in gens])

# SL curves (to early stop) and RL curves; align on common length
sl = [json.load(open(P / f"ms2_fig3e_seed{s}.json")) for s in seeds]
rl = [json.load(open(P / f"ms2_race_seed{s}.json")) for s in seeds]
Lsl = min(len(d["errs"]) for d in sl)
sl_x = np.array(sl[0]["eps"][:Lsl]); sl_y = np.array([d["errs"][:Lsl] for d in sl])
Lrl = min(len(d["rl_errs"]) for d in rl)
rl_x = np.array(rl[0]["rl_eps"][:Lrl]); rl_y = np.array([d["rl_errs"][:Lrl] for d in rl])
stop_eps = [d["stop_ep"] for d in sl]

pre = np.array([json.load(open(P / f"ms2_dist_seed{s}.json"))["pre"] for s in seeds], float)
post = np.array([json.load(open(P / f"ms2_dist_seed{s}.json"))["post"] for s in seeds], float)
pre = (pre / pre.sum(1, keepdims=True)).mean(0); post = (post / post.sum(1, keepdims=True)).mean(0)
n_acts = len(pre); opt_pre = round(135 / 360 * n_acts) % n_acts; opt_post = round(165 / 360 * n_acts) % n_acts

fig, ax = plt.subplots(2, 2, figsize=(9, 7))
m, e = rot_gen.mean(0), sem(rot_gen)
ax[0, 0].plot(idx, m, "k", lw=2); ax[0, 0].fill_between(idx, m - e, m + e, color="k", alpha=0.2)
ax[0, 0].axhline(100, color="gray", ls=":", lw=.8)
ax[0, 0].set(xlabel="angle from target (°)", ylabel="rotation generalization (%)", title=f"Fig 3I: local generalization (n={len(seeds)}, early-stopped)", xlim=[-90, 180]); ax[0, 0].spines[["right", "top"]].set_visible(False)
m, e = ang_err.mean(0), sem(ang_err)
ax[0, 1].plot(idx, m, color="#44123F", lw=2); ax[0, 1].fill_between(idx, m - e, m + e, color="#44123F", alpha=0.2)
ax[0, 1].axhline(0, color="gray", ls=":", lw=.8)
ax[0, 1].set(xlabel="angle from target (°)", ylabel="angular error (°)", title="Fig 3I: local angular error", xlim=[-90, 180]); ax[0, 1].invert_yaxis(); ax[0, 1].spines[["right", "top"]].set_visible(False)
for x, y, c, lab in [(sl_x, sl_y, "#2E8B57", "SL decoder (error-driven)"), (rl_x, rl_y, "#A94850", "RL policy (reward-driven)")]:
    m, e = y.mean(0), sem(y)
    ax[1, 0].plot(x, m, color=c, lw=2, label=lab); ax[1, 0].fill_between(x, m - e, m + e, color=c, alpha=0.2)
ax[1, 0].axhline(0, color="k", lw=.5, ls=":"); ax[1, 0].set(xlabel="adaptation episode", ylabel="achieved angular error (°)", title="Fig 3E / race: SL fast vs RL slow"); ax[1, 0].legend(frameon=False, fontsize=8); ax[1, 0].spines[["right", "top"]].set_visible(False)
b = np.arange(n_acts)
ax[1, 1].bar(b, pre, alpha=0.5, color="#888", label="pre"); ax[1, 1].bar(b, post, alpha=0.5, color="#2E8B57", label="post")
ax[1, 1].axvline(opt_pre, color="k", ls="--", lw=1); ax[1, 1].axvline(opt_post, color="#A94850", lw=1)
ax[1, 1].set(xlabel="action index", ylabel="frequency", title="Fig 3F: distribution swap (135°→165°)"); ax[1, 1].legend(frameon=False, fontsize=8); ax[1, 1].spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "fig3_adaptation_multiseed_earlystop.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
loc = rot_gen[:, [i for i, a in enumerate(idx) if -45 <= a <= 45]].mean()
glob = rot_gen[:, [i for i, a in enumerate(idx) if a < -45 or a > 45]].mean()
tgt = ang_err[:, np.argmin(np.abs(idx))].mean()
print("stop_eps=%s  locality=%.0f  target angular-error=%.1f  SL end=%.0f RL end=%.0f" % (stop_eps, loc - glob, tgt, sl_y.mean(0)[-1], rl_y.mean(0)[-1]), flush=True)
