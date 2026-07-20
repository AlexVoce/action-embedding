"""SL-vs-RL adaptation race ('why multiple systems'). Same 30deg rotation, same protected base.
SL = error-driven decoder (f) adaptation (cerebellar); RL = reward-driven actor/critic policy
gradient (basal ganglia), f frozen. Track greedy achieved error over episodes for each. Thesis:
SL re-learns much faster than RL."""
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
adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"


def run(mode):
    shutil.copy(PROT, STD)
    cfg = dict(acfg)
    cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "log_to_wandb": False})
    env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
    if mode == "sl":
        cfg.update({"fg_lr": 1e-3, "actor_lr": 0.0, "critic_lr": 0.0})
        agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)  # f_plastic default
        rand = True  # decoupled exploration
    else:  # rl: actor/critic plastic, f frozen
        cfg.update({"fg_lr": 0.0, "actor_lr": 1e-4, "critic_lr": 5e-4})
        from core.agent import ACLearningAgentWithEmbedding
        agent = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=str(STD),
                                             f_plastic=False, g_plastic=False, actor_plastic=True, critic_plastic=True)
        rand = False
    env0 = ReachTask(cfg)
    ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
    env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
    with torch.no_grad():
        e_target = agent.g(s0, env0.get_features(nxt0))

    def greedy_err():
        feats = env.get_features(env.current_xy)
        with torch.no_grad():
            emb = e_target if mode == "sl" else torch.tanh(agent.actor(feats))
            g = float(np.degrees(env.actions[int(torch.argmax(agent.f(emb)).item())]))
        return abs(((g + rot - target_deg + 180) % 360) - 180)

    curve = []
    for ep in range(adapt_eps):
        env.reset(); feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        if hasattr(agent, "f_g_optimizer"):
            agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=(rand and mode == "sl"), import_policy_mean=False, policy_mean=None)
        nxt, reward, done = env.act(env.actions[a_idx])
        agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
        if ep % 250 == 0:
            curve.append((ep, greedy_err()))
    return curve


COL = {"sl": "#2E8B57", "rl": "#A94850"}
LAB = {"sl": "SL decoder adaptation (error-driven, cerebellar)", "rl": "RL policy adaptation (reward-driven, BG)"}
fig, ax = plt.subplots(figsize=(5.6, 4))
for mode in ["sl", "rl"]:
    c = run(mode)
    e = [x for x, _ in c]; v = [y for _, y in c]
    ax.plot(e, v, color=COL[mode], lw=2, label=LAB[mode])
    print("%s: start=%.0f end=%.0f min=%.0f" % (mode, v[1], v[-1], min(v)), flush=True)
ax.set(xlabel="adaptation episode", ylabel="achieved angular error (deg)",
       title="Re-learning after 30° rotation: SL fast vs RL slow")
ax.axhline(0, color="k", lw=.5, ls=":"); ax.legend(frameon=False, fontsize=8); ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "adapt_mechanism_relearn.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
