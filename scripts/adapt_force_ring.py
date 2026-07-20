"""Force the (frozen) policy ONTO the ring for SL adaptation: set the actor to output the target
action's g-embedding (actor_tanh=False so it reaches ring radius), reuse the existing base f/g,
then run the paper's SL decoder-adaptation. The policy now sits where f adapts, so decoder
rotation should transfer to behaviour. Track greedy decode + taken-action error (Fig 3E).

usage: adapt_force_ring.py <adapt_eps> <fg_lr> [seed] [target_deg] [rot]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from adaptation.config import config as acfg
from definitions import paper_model_path
from core.plotting import find_angle_difference

adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
fg_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
target_deg = int(sys.argv[4]) if len(sys.argv) > 4 else 135
rot = int(sys.argv[5]) if len(sys.argv) > 5 else -30

cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot,
            "fg_lr": fg_lr, "log_to_wandb": False, "actor_tanh": False})  # unbounded actor -> reaches ring
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
# load the CLEAN embedding (f/g) directly (the base-model files got clobbered; we override the
# actor anyway). f_plastic=True (SL decoder adaptation), actor/critic/g frozen.
fg_path = Path(paper_model_path) / 'action_embedding_model_seed_0_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth'
agent = ACLearningAgentWithEmbedding(env, cfg, fg_load_path=str(fg_path),
                                     f_plastic=True, g_plastic=False, actor_plastic=False, critic_plastic=False)

# target action's g-embedding computed on a NO-ROTATION env (so the policy INTENDS the target,
# i.e. its embedding decodes to the target action under the base decoder). On the ring.
env0 = ReachTask(cfg)  # no adaptation_rotation
g = []
for a in env0.actions:
    env0.reset()
    s = env0.get_features(env0.current_xy)
    nxt, _, _ = env0.act(a)
    with torch.no_grad():
        g.append(agent.g(s, env0.get_features(nxt)).numpy())
g = np.array(g)
tidx = int(np.argmin(np.abs(((np.round(np.degrees(env.actions)) - target_deg + 180) % 360) - 180)))
gemb = torch.tensor(g[tidx], dtype=torch.float32)
with torch.no_grad():
    agent.actor.mean_head.weight.zero_()
    agent.actor.mean_head.bias.copy_(torch.atanh(torch.clamp(gemb, -0.999, 0.999)))

comp = float(np.degrees(env.actions[agent.find_optimal_action_ind()]))

def greedy_deg():
    feats = env.get_features(env.current_xy)
    with torch.no_grad():
        return float(np.degrees(env.actions[int(torch.argmax(agent.f(torch.tanh(agent.actor(feats)))).item())]))

with torch.no_grad():
    pol = torch.tanh(agent.actor(env.get_features(env.current_xy)))
print("[coherence] policy radius=%.3f  ring=%.3f  base greedy=%.0f  compensated=%.0f"
      % (float(torch.norm(pol)), np.linalg.norm(g, axis=1).mean(), greedy_deg(), comp), flush=True)

traj, errs = [], []
for ep in range(adapt_eps):
    env.reset()
    feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    errs.append(abs(find_angle_difference(env, env.actions[a_idx])))
    if ep % 200 == 0:
        traj.append((ep, greedy_deg()))
print("greedy-decode trajectory:", "  ".join("%d:%.0f" % (e, d) for e, d in traj), flush=True)
gg = [d for _, d in traj]
hit_ep = next((e for e, d in traj if abs(((d - comp + 180) % 360) - 180) < 8), None)
print("REACHED compensation (165): %s (first at ep %s) ; final greedy=%.0f"
      % (hit_ep is not None, hit_ep, gg[-1]), flush=True)
w = 500
roll = [round(float(np.mean(errs[max(0, i - w):i + 1])), 1) for i in range(0, len(errs), 1000)]
print("taken-action error (Fig 3E) rolling:", roll, flush=True)
