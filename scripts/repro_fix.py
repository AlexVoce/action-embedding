"""Try the adaptation fixes (toggleable). Retrain the base policy with the given flags, report
the coherence (policy radius vs ring), then run the paper's SL adaptation loop tracking the
per-episode GREEDY decode — success = it rotates through the compensated action (165) so Fig 3E
can land there.

usage: repro_fix.py <base_eps> <adapt_eps> <actor_tanh:0/1> <zero_f_bias:0/1> [fg_lr]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

base_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
adapt_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
actor_tanh = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
zero_f_bias = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
fg_lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
seed, target_deg, rot = 0, 135, -30
print("=== FIX base_eps=%d adapt_eps=%d actor_tanh=%s zero_f_bias=%s fg_lr=%.0e ===" %
      (base_eps, adapt_eps, actor_tanh, zero_f_bias, fg_lr), flush=True)

# 1) retrain base with flags
from core.config import config as bcfg
bcfg["max_episodes"] = base_eps
bcfg["log_to_wandb"] = False
bcfg["save_model"] = True
bcfg["actor_tanh"] = actor_tanh
bcfg["zero_f_bias"] = zero_f_bias
sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
from core.policy_learning import train_agent
train_agent(bcfg)

# 2) adaptation with per-episode greedy tracking (paper's agent setup)
from core.continuous_env import ReachTask
from adaptation.config import config as acfg
from core.model_loading_utils import load_trained_full_model_basetask
cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot,
            "fg_lr": fg_lr, "log_to_wandb": False, "actor_tanh": actor_tanh, "zero_f_bias": zero_f_bias})
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
comp = float(np.degrees(env.actions[agent.find_optimal_action_ind()]))

def greedy_deg():
    # select_action always applies one tanh to the (noised) mean; greedy = tanh(actor(feats))
    feats = env.get_features(env.current_xy)
    with torch.no_grad():
        a = int(torch.argmax(agent.f(torch.tanh(agent.actor(feats)))).item())
    return float(np.degrees(env.actions[a]))

feats0 = env.get_features(env.current_xy)
with torch.no_grad():
    pol_emb = torch.tanh(agent.actor(feats0))
print("[coherence] policy radius=%.3f  ring=%.3f  base greedy=%.0f  compensated=%.0f" %
      (float(torch.norm(pol_emb)), np.linalg.norm(agent.get_action_embeddings_via_g(), axis=1).mean(),
       greedy_deg(), comp), flush=True)

traj = []
for ep in range(adapt_eps):
    env.reset()
    feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    if ep % 200 == 0:
        traj.append((ep, greedy_deg()))
print("greedy-decode trajectory:", "  ".join("%d:%.0f" % (e, d) for e, d in traj), flush=True)
gg = [d for _, d in traj]
hit = any(abs(((d - comp + 180) % 360) - 180) < 8 for d in gg)
print("REACHED compensation (165) at some point: %s ; final greedy=%.0f" % (hit, gg[-1]), flush=True)
