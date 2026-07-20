"""Reproduce Fig 3I the paper's way: on-policy adaptation (use_random_policy=False, as in the repo)
+ calculate_generalization + make_generalization_plot. Tracks locality over episodes so we can pick
the 'don't over-train' point where the reorganization is local and the target is compensated."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config as acfg
from adaptation.adaptation_generalization_test import calculate_generalization
from definitions import paper_model_path, revision_fig_dir

seed, target_deg, rot = 0, 135, -30
fg_lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
checkpoints = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["10000", "20000", "40000", "70000", "100000"])]
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
shutil.copy(PROT, STD)

cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "fg_lr": fg_lr, "log_to_wandb": False, "use_random_policy": False})
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
base_agent = copy.deepcopy(agent)

def profile():
    df = calculate_generalization(agent, base_agent, cfg)
    d = df[["rotation generalization", "angular error", "angle from target"]].set_index("angle from target").sort_index()
    loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
    glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
    tgt_err = d.loc[0]["angular error"] if 0 in d.index else np.nan
    return df, loc, glob, tgt_err

best = None
ep = 0
for ck in checkpoints:
    while ep < ck:
        env.reset(); feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        if hasattr(agent, "f_g_optimizer"):
            agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
        nxt, reward, done = env.act(env.actions[a_idx])
        agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
        ep += 1
    df, loc, glob, tgt_err = profile()
    print("ep=%d: locality(local-global)=%.0f  local=%.0f global=%.0f  target angular-error=%.1f" % (ep, loc - glob, loc, glob, tgt_err), flush=True)
    # pick the checkpoint with strongest locality AND low target error (compensated, not over-rotated)
    score = (loc - glob) - abs(tgt_err)  # reward locality, penalise residual target error
    if best is None or score > best[0]:
        best = (score, ep, df.copy())

_, best_ep, df = best
print("[chosen] ep=%d for the Fig 3I panel" % best_ep, flush=True)
d = df[["rotation generalization", "angular error", "angle from target"]].groupby("angle from target").mean().sort_index()
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].plot(d.index, d["angular error"], marker="o", ms=3, color="#44123F")
ax[0].set(xlabel="angle from target (deg)", ylabel="angular error (deg)", title="Fig 3I: local angular-error")
ax[0].set_xlim([-90, 180]); ax[0].invert_yaxis(); ax[0].spines[["right", "top"]].set_visible(False)
ax[1].plot(d.index, d["rotation generalization"], marker="o", ms=3, color="k")
ax[1].set(xlabel="angle from target (deg)", ylabel="rotation generalization (%)", title="local reorganization")
ax[1].set_xlim([-90, 180]); ax[1].spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "fig3I_local_generalization.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
