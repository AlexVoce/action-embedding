"""Fig 3E/3F: SL decoder-adaptation to a 30deg visuomotor rotation. Exploration decoupled
(random) so the inverse model relearns the global remap. Track the greedy (noise-free) decode of
the target's transition embedding over adaptation (Fig 3E: achieved error 30->0), and the action
distribution pre vs post (Fig 3F: swap from ~135 to ~165)."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from core.agent import softmax_with_temperature, add_noise_to_action_probs
from adaptation.config import config as acfg
from definitions import paper_model_path, revision_fig_dir

seed, target_deg, rot = 0, 135, -30
adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
fg_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
shutil.copy(PROT, STD)

cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "fg_lr": fg_lr, "log_to_wandb": False})
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
env0 = ReachTask(cfg)
ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
with torch.no_grad():
    e_target = agent.g(s0, env0.get_features(nxt0))

def greedy_achieved_err():
    with torch.no_grad():
        g = float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))
    return abs(((g + rot - target_deg + 180) % 360) - 180)  # achieved (g+rot) vs target

def sample_dist(n=3000):
    acts = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        acts.append(int(torch.multinomial(p, 1).item()))
    return np.array(acts)

dist_pre = sample_dist()
curve = []
for ep in range(adapt_eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=True, import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    if ep % 250 == 0:
        curve.append((ep, greedy_achieved_err()))
dist_post = sample_dist()
eps = [e for e, _ in curve]; errs = [v for _, v in curve]
print("Fig3E greedy achieved-error curve: start=%.0f end=%.0f (min=%.0f)" % (errs[1], errs[-1], min(errs)), flush=True)

n_acts = env.n_actions
opt_pre = int(round(135 / 360 * n_acts)) % n_acts
opt_post = int(round(165 / 360 * n_acts)) % n_acts
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].plot(eps, errs, color="#2E8B57", lw=2)
ax[0].set(xlabel="adaptation episode", ylabel="achieved angular error (deg)", title="Fig 3E: re-learning under 30° rotation")
ax[0].axhline(0, color="k", lw=.5, ls=":"); ax[0].spines[["right", "top"]].set_visible(False)
bins = np.arange(0, n_acts + 1)
ax[1].hist(dist_pre, bins=bins, density=True, alpha=0.55, color="#888", label="pre-adaptation")
ax[1].hist(dist_post, bins=bins, density=True, alpha=0.55, color="#2E8B57", label="post-adaptation")
ax[1].axvline(opt_pre + .5, color="k", ls="--", lw=1, label="unrotated optimal (135°)")
ax[1].axvline(opt_post + .5, color="#A94850", ls="-", lw=1, label="compensated optimal (165°)")
ax[1].set(xlabel="action index", ylabel="frequency", title="Fig 3F: action distribution swap")
ax[1].legend(frameon=False, fontsize=7); ax[1].spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "fig3_adaptation_relearning.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
print("pre median action=%.0f  post median action=%.0f deg" % (
    np.degrees(env.actions[int(np.median(dist_pre))]), np.degrees(env.actions[int(np.median(dist_post))])), flush=True)
