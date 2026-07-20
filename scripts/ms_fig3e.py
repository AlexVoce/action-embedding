"""Fig 3E re-learning curve as the systematic error of the INTENDED action: the target's
transition-embedding decoded through the adapting f. Signed-mean angular error -> 0 as the
decoder relearns the compensation (consistent with Fig 3I; not limited by exploration spread or
the actor-path lag). Reuses the per-seed base; adaptation only.

usage: ms_fig3e.py <seed> [eps]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from core.agent import softmax_with_temperature, add_noise_to_action_probs
from core.plotting import find_angle_difference
from adaptation.config import config as acfg
from definitions import paper_model_path, paper_fig_dir

seed = int(sys.argv[1]); eps = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
target_deg, rot = 135, -30
cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=1e-4, log_to_wandb=False, use_random_policy=False)
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
env0 = ReachTask({**cfg})
ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
with torch.no_grad():
    e_target = agent.g(s0, env0.get_features(nxt0))
# fixed goal = the target endpoint (for find_angle_difference's optimal-under-rotation)
goal = env0.current_xy  # not used; find_angle_difference uses env rotation + optimal action

def systematic_err(n=300):
    diffs = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        a = int(torch.multinomial(p, 1).item())
        diffs.append(find_angle_difference(env, env.actions[a]))  # signed error to compensated optimal
    return float(np.mean(diffs))  # signed mean -> 0 as distribution centres on compensated

curve = []  # signed systematic error (no early stop; fixed window)
for ep in range(eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
    if hasattr(agent, "f_g_optimizer"):
        agent.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
    if ep % 1000 == 0:
        curve.append([ep, systematic_err()])  # SIGNED
json.dump({"eps": [e for e, _ in curve], "sys_err": [v for _, v in curve]},
          open(Path(paper_fig_dir) / f"ms3e_seed{seed}.json", "w"))
sgn = [v for _, v in curve]
print("SEED %d FIG3E DONE (start=%.0f end=%.0f min|.|=%.1f)" % (seed, sgn[0], sgn[-1], min(abs(v) for v in sgn)), flush=True)
