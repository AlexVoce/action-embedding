"""Diagnostic: does the SL-adapted decoder rotate as a function of embedding radius?
The policy acts at ~0.85 (double-tanh cap) but the ring is at ~1.16. If the adapted decode is
compensated at the ring radius but not at the actor radius, that pinpoints why Fig 3E stalls."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from core.config import config as cfg
from definitions import paper_model_path

base_fn = os.path.join(paper_model_path, "fully_trained_policy_model_one_target_seed_0_weight_decay_0.0001_tanh_policy_mean_target_135_n_actions_24.pth")
post_fn = os.path.join(paper_model_path, "post_adaptation_model_seed_0_rotation_-30_temp_2.5_weight_decay_0.0001_tanh_policy_mean_n_actions_24_target_135.0.pth")
env = ReachTask(cfg)
base = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=base_fn, fg_load_path=None, g_plastic=False, f_plastic=False, actor_plastic=False, critic_plastic=False)
post = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=post_fn, fg_load_path=None, g_plastic=False, f_plastic=False, actor_plastic=False, critic_plastic=False)
acts = env.actions

def deg(i):
    return float(np.degrees(acts[i]))

g = base.get_action_embeddings_via_g()
idx = int(np.argmin(np.abs(((np.round(np.degrees(acts)) - 135 + 180) % 360) - 180)))
theta = float(np.arctan2(g[idx][1], g[idx][0]))
print("target g-emb angle=%.0fdeg  ring_radius=%.2f  (actor operates ~0.85; compensated action=165)" % (np.degrees(theta), np.linalg.norm(g[idx])))
print("radius | base-decode | post-decode")
for r in [0.6, 0.85, 1.0, 1.16, 1.3]:
    e = torch.tensor([r * np.cos(theta), r * np.sin(theta)], dtype=torch.float32)
    bd = deg(int(torch.argmax(base.f(e)).item()))
    pd = deg(int(torch.argmax(post.f(e)).item()))
    print("  %.2f  |   %3.0f       |   %3.0f" % (r, bd, pd))
