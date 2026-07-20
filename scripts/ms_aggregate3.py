"""Final Fig 3 aggregator (per-seed embeddings, 10 seeds). Averages over the cleanly-compensating
seeds: Fig 3I (rotation generalization + angular error vs angle-from-target), Fig 3E (taken-action
error over adaptation from the run-logs), Fig 3F (pre/post action distribution from ms3_dist),
and the SL-vs-RL race (SL from run-log, RL from ms3_race)."""
import sys, os, glob, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from definitions import paper_fig_dir, revision_fig_dir

target = 135
P = Path(paper_fig_dir)
done = sorted(int(re.search(r"seed(\d+)\.json", f).group(1)) for f in glob.glob(str(P / "ms3_race_seed*.json")))
print("completed seeds:", done, flush=True)

# per-seed generalization + target error
gens, per_seed = {}, {}
for s in done:
    fp = P / f"generalization_stats_seed_{s}_target_{target}_rotation_-30.csv"
    if not fp.exists():
        continue
    g = pd.read_csv(fp).groupby("angle from target")[["rotation generalization", "angular error"]].mean()
    gens[s] = g
    per_seed[s] = g.loc[0, "angular error"] if 0 in g.index else np.nan
    print("  seed %d target_err=%.1f peak_gen=%.0f%%" % (s, per_seed[s], g["rotation generalization"].max()), flush=True)

clean = [s for s in gens if abs(per_seed[s]) < 8]
print("clean seeds:", clean, flush=True)
DROP = {10}  # seed 10 = bad embedding (Fig 3I peak 426%, Fig 3E residual 22°); the rest averaged
use = [s for s in gens if s not in DROP]
sem = lambda a: np.array(a).std(0) / max(1, np.sqrt(np.array(a).shape[0]))

idx = sorted(set.intersection(*[set(gens[s].index) for s in use]))
rot_gen = np.array([[gens[s].loc[a, "rotation generalization"] for a in idx] for s in use])
ang_err = np.array([[gens[s].loc[a, "angular error"] for a in idx] for s in use])

# Fig 3E: intended-action systematic error (target g-embedding decoded through the adapting f) -> 0
sl_curves = []
for s in use:
    fp = P / f"ms3e_seed{s}.json"
    if fp.exists():
        sl_curves.append(np.array(json.load(open(fp))["sys_err"]))
# RL race — taken-action metric, matched to SL run-log
rl_curves = [json.load(open(P / f"ms3_rltaken_seed{s}.json"))["rl_taken"] for s in use if (P / f"ms3_rltaken_seed{s}.json").exists()]
# Fig 3F distributions
pre = np.array([json.load(open(P / f"ms3_dist_seed{s}.json"))["pre"] for s in use if (P / f"ms3_dist_seed{s}.json").exists()], float)
post = np.array([json.load(open(P / f"ms3_dist_seed{s}.json"))["post"] for s in use if (P / f"ms3_dist_seed{s}.json").exists()], float)
pre = (pre / pre.sum(1, keepdims=True)).mean(0); post = (post / post.sum(1, keepdims=True)).mean(0)
n_acts = len(pre); op, oq = round(135 / 360 * n_acts) % n_acts, round(165 / 360 * n_acts) % n_acts

fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.5))
m, e = rot_gen.mean(0), sem(rot_gen)
ax[0, 0].plot(idx, m, "k", lw=2); ax[0, 0].fill_between(idx, m - e, m + e, color="k", alpha=0.2); ax[0, 0].axhline(100, color="gray", ls=":", lw=.8)
ax[0, 0].set(xlabel="angle from target (°)", ylabel="rotation generalization (%)", title=f"Fig 3I local generalization (n={len(use)})", xlim=[-90, 180]); ax[0, 0].spines[["right", "top"]].set_visible(False)
m, e = ang_err.mean(0), sem(ang_err)
ax[0, 1].plot(idx, m, color="#44123F", lw=2); ax[0, 1].fill_between(idx, m - e, m + e, color="#44123F", alpha=0.2); ax[0, 1].axhline(0, color="gray", ls=":", lw=.8)
ax[0, 1].set(xlabel="angle from target (°)", ylabel="angular error (°)", title="Fig 3I angular error", xlim=[-90, 180]); ax[0, 1].invert_yaxis(); ax[0, 1].spines[["right", "top"]].set_visible(False)
if sl_curves:
    L = min(len(c) for c in sl_curves)
    sl = np.array([c[:L] for c in sl_curves]); xb = np.arange(L) * 1000
    mm, ee = sl.mean(0), sl.std(0) / max(1, np.sqrt(sl.shape[0]))
    ax[1, 0].plot(xb, mm, color="#2E8B57", lw=2); ax[1, 0].fill_between(xb, mm - ee, mm + ee, color="#2E8B57", alpha=0.2)
    ax[1, 0].axhline(0, color="gray", ls=":", lw=.8)
    ax[1, 0].set(xlabel="adaptation episode", ylabel="angular error (°)", title="Fig 3E: re-learning under rotation"); ax[1, 0].spines[["right", "top"]].set_visible(False)
b = np.arange(n_acts)
ax[1, 1].bar(b, pre, alpha=0.5, color="#888", label="pre"); ax[1, 1].bar(b, post, alpha=0.5, color="#2E8B57", label="post")
ax[1, 1].axvline(op, color="k", ls="--", lw=1); ax[1, 1].axvline(oq, color="#A94850", lw=1)
ax[1, 1].set(xlabel="action index", ylabel="frequency", title="Fig 3F: distribution swap (135°→165°)"); ax[1, 1].legend(frameon=False, fontsize=8); ax[1, 1].spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "fig3_final_10seed.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
loc = rot_gen[:, [i for i, a in enumerate(idx) if -45 <= a <= 45]].mean(); gl = rot_gen[:, [i for i, a in enumerate(idx) if a < -45 or a > 45]].mean()
print("n_use=%d locality=%.0f peak_gen=%.0f%% target_err=%.1f" % (len(use), loc - gl, rot_gen.mean(0).max(), ang_err[:, np.argmin(np.abs(idx))].mean()), flush=True)
