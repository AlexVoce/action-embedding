"""Record the per-episode GREEDY action decoded through the actor path during the paper's SL
adaptation, at fine resolution, to see whether it rotates through the correct compensation
(compensated action) — in which case target-aware early stopping lands Fig 3E/3F correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from adaptation.config import config as acfg
from core.model_loading_utils import load_trained_full_model_basetask

seed, target_deg, rot = 0, 135, -30
cfg = dict(acfg)
cfg["seed"] = seed
cfg["reach_angle"] = float(np.radians(target_deg))
cfg["rotation_angle"] = rot
cfg["fg_lr"] = 1e-4
cfg["log_to_wandb"] = False

env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)   # f_plastic=True, actor frozen

comp = agent.find_optimal_action_ind()                       # compensated action index (under rotation)
comp_deg = float(np.degrees(env.actions[comp]))
print("compensated (target) action = %.0f deg (base greedy should start ~135)" % comp_deg)

def greedy_deg():
    feats = env.get_features(env.current_xy)
    with torch.no_grad():
        e = torch.tanh(agent.actor(feats))          # actor path, no noise (what the policy centres on)
        a = int(torch.argmax(agent.f(e)).item())
    return float(np.degrees(env.actions[a]))

traj = []
for ep in range(6000):
    env.reset()
    feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False,
                                                       import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    if ep % 100 == 0:
        traj.append((ep, greedy_deg()))
print("greedy-decode (deg) trajectory [ep:deg]:")
print("  " + "  ".join("%d:%.0f" % (ep, d) for ep, d in traj))
