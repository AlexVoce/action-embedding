"""Test whether the SL adaptation produces LOCAL reorganization (Fig 3I): angular error drops near
the trained target but NOT far away. Sweep the exploration mix (fraction of RANDOM vs on-policy
sampling). On-policy over-samples locally -> local reorganization; pure random -> global (flat).
Per-action greedy achieved error vs angle-from-target IS the Fig 3I profile."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config as acfg
from definitions import paper_model_path, revision_fig_dir

seed, target_deg, rot = 0, 135, -30
adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
fg_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"


def gembs_norot(agent, env):
    """desired-transition embedding for every action (no rotation)."""
    out = []
    for a in env.actions:
        env.reset(); s = env.get_features(env.current_xy); nxt, _, _ = env.act(a)
        with torch.no_grad():
            out.append(agent.g(s, env.get_features(nxt)))
    return out


def run(p_random):
    shutil.copy(PROT, STD)
    cfg = dict(acfg)
    cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "fg_lr": fg_lr, "log_to_wandb": False})
    env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
    agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
    env0 = ReachTask(cfg)
    e_all = gembs_norot(agent, env0)          # per-action desired-transition embeddings
    actions_deg = np.round(np.degrees(env.actions))
    rng = np.random.RandomState(seed)
    for ep in range(adapt_eps):
        env.reset(); feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        if hasattr(agent, "f_g_optimizer"):
            agent.f_g_optimizer.zero_grad()
        use_rand = rng.rand() < p_random
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=use_rand, import_policy_mean=False, policy_mean=None)
        nxt, reward, done = env.act(env.actions[a_idx])
        agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    # per-action greedy achieved error (how well the decoder compensates to reach each action's target)
    errs = []
    for i, e_i in enumerate(e_all):
        with torch.no_grad():
            d = float(np.degrees(env.actions[int(torch.argmax(agent.f(e_i)).item())]))
        achieved_err = abs(((d + rot - actions_deg[i] + 180) % 360) - 180)
        errs.append(achieved_err)
    errs = np.array(errs)
    afromt = ((actions_deg - target_deg + 180) % 360) - 180
    order = np.argsort(afromt)
    return afromt[order], errs[order]


fig, ax = plt.subplots(figsize=(5.6, 4))
for p_random, col, lab in [(0.0, "#2E8B57", "on-policy (local over-sampling)"),
                            (0.15, "#4477AA", "mostly on-policy + 15% random"),
                            (1.0, "#A94850", "pure random (global)")]:
    x, y = run(p_random)
    ax.plot(x, y, marker="o", ms=3, color=col, label=lab)
    tgt = y[np.argmin(np.abs(x))]
    far = y[np.abs(x) > 90].mean()
    print("p_random=%.2f: target-error=%.0f  far-field-error=%.0f  (local pattern => target low, far ~30)" % (p_random, tgt, far), flush=True)
ax.set(xlabel="angle from trained target (deg)", ylabel="achieved angular error (deg)",
       title="Fig 3I: local vs global reorganization by exploration")
ax.axhline(0, color="k", lw=.5, ls=":"); ax.set_xlim([-90, 180]); ax.legend(frameon=False, fontsize=8)
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "local_reorganization.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
