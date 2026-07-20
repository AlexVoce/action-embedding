"""Aggregate multi-seed adaptation data into clean final panels: Fig 3I (generalization profile,
mean +/- SEM over seeds), Fig 3E (re-learning curve), Fig 3F (pre/post action distribution), and
the SL-vs-RL race. Reads figures/paper/ms_*_seed{S}.* written by ms_worker.py."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from definitions import paper_fig_dir, revision_fig_dir

P = Path(paper_fig_dir)
seeds = sorted(int(f.split("seed")[-1].split(".")[0]) for f in glob.glob(str(P / "ms_gen_seed*.csv")))
print("aggregating seeds:", seeds, flush=True)


def sem(a, axis=0):
    a = np.array(a)
    return a.std(axis) / max(1, np.sqrt(a.shape[axis]))


# ---- Fig 3I: generalization profile, averaged over seeds ----
gens = []
for s in seeds:
    df = pd.read_csv(P / f"ms_gen_seed{s}.csv")
    g = df.groupby("angle from target")[["rotation generalization", "angular error"]].mean()
    gens.append(g)
idx = sorted(set.intersection(*[set(g.index) for g in gens]))
rot_gen = np.array([[g.loc[a, "rotation generalization"] for a in idx] for g in gens])
ang_err = np.array([[g.loc[a, "angular error"] for a in idx] for g in gens])

# ---- Fig 3E + race curves ----
def load_curves(key_e, key_v):
    xs, ys = None, []
    for s in seeds:
        d = json.load(open(P / f"ms_race_seed{s}.json"))
        xs = d[key_e]
        ys.append(d[key_v])
    L = min(len(y) for y in ys)
    return np.array(xs[:L]), np.array([y[:L] for y in ys])

sl_x, sl_y = load_curves("sl_eps", "sl_errs")
rl_x, rl_y = load_curves("rl_eps", "rl_errs")

# ---- Fig 3F distributions ----
pre = np.array([json.load(open(P / f"ms_dist_seed{s}.json"))["pre"] for s in seeds], dtype=float)
post = np.array([json.load(open(P / f"ms_dist_seed{s}.json"))["post"] for s in seeds], dtype=float)
pre = (pre / pre.sum(1, keepdims=True)).mean(0)
post = (post / post.sum(1, keepdims=True)).mean(0)

n_acts = len(pre)
opt_pre = int(round(135 / 360 * n_acts)) % n_acts
opt_post = int(round(165 / 360 * n_acts)) % n_acts

fig, ax = plt.subplots(2, 2, figsize=(9, 7))
# 3I generalization
m, e = rot_gen.mean(0), sem(rot_gen)
ax[0, 0].plot(idx, m, color="k", lw=2); ax[0, 0].fill_between(idx, m - e, m + e, color="k", alpha=0.2)
ax[0, 0].set(xlabel="angle from target (°)", ylabel="rotation generalization (%)", title=f"Fig 3I: local reorganization (n={len(seeds)})", xlim=[-90, 180])
ax[0, 0].spines[["right", "top"]].set_visible(False)
# 3I angular error
m, e = ang_err.mean(0), sem(ang_err)
ax[0, 1].plot(idx, m, color="#44123F", lw=2); ax[0, 1].fill_between(idx, m - e, m + e, color="#44123F", alpha=0.2)
ax[0, 1].set(xlabel="angle from target (°)", ylabel="angular error (°)", title="Fig 3I: local angular error", xlim=[-90, 180]); ax[0, 1].invert_yaxis()
ax[0, 1].spines[["right", "top"]].set_visible(False)
# 3E / race
for x, y, c, lab in [(sl_x, sl_y, "#2E8B57", "SL decoder (error-driven)"), (rl_x, rl_y, "#A94850", "RL policy (reward-driven)")]:
    m, e = y.mean(0), sem(y)
    ax[1, 0].plot(x, m, color=c, lw=2, label=lab); ax[1, 0].fill_between(x, m - e, m + e, color=c, alpha=0.2)
ax[1, 0].set(xlabel="adaptation episode", ylabel="achieved angular error (°)", title="Fig 3E / race: SL fast vs RL slow")
ax[1, 0].axhline(0, color="k", lw=.5, ls=":"); ax[1, 0].legend(frameon=False, fontsize=8); ax[1, 0].spines[["right", "top"]].set_visible(False)
# 3F distributions
b = np.arange(n_acts)
ax[1, 1].bar(b, pre, alpha=0.5, color="#888", label="pre")
ax[1, 1].bar(b, post, alpha=0.5, color="#2E8B57", label="post")
ax[1, 1].axvline(opt_pre, color="k", ls="--", lw=1); ax[1, 1].axvline(opt_post, color="#A94850", ls="-", lw=1)
ax[1, 1].set(xlabel="action index", ylabel="frequency", title="Fig 3F: distribution swap (135°→165°)"); ax[1, 1].legend(frameon=False, fontsize=8)
ax[1, 1].spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "fig3_adaptation_multiseed.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
loc = rot_gen[:, [i for i, a in enumerate(idx) if -45 <= a <= 45]].mean()
glob = rot_gen[:, [i for i, a in enumerate(idx) if a < -45 or a > 45]].mean()
print("locality(local-global)=%.0f  SL end=%.0f  RL end=%.0f" % (loc - glob, sl_y.mean(0)[-1], rl_y.mean(0)[-1]), flush=True)
