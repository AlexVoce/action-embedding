"""Definitive test of the adaptation story: SL on f recalibrates the inverse model so the desired
transition decodes to the COMPENSATED action. Regenerate a protected clean base, run the paper's
adaptation (learned actor explores, f_plastic), and track argmax f(g(s->target)) over episodes
(should rotate 135 -> ~165) plus calculate_generalization locality at the end.

usage: repro_clean.py <base_eps> <adapt_eps>
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from definitions import paper_model_path

base_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
adapt_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
seed, target_deg, rot = 0, 135, -30
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"

# 1) regenerate clean standard base (default config) unless protected copy exists
if not PROT.exists():
    from core.config import config as bcfg
    bcfg["max_episodes"] = base_eps
    bcfg["log_to_wandb"] = False
    bcfg["save_model"] = True
    sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
    from core.policy_learning import train_agent
    train_agent(bcfg)
    shutil.copy(STD, PROT)
    print("[base] regenerated + protected", flush=True)
else:
    print("[base] using existing protected copy", flush=True)
shutil.copy(PROT, STD)  # ensure the loader finds a clean copy

# 2) adaptation with the paper's setup, tracking the relearned inverse-model decode of the target
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config as acfg
from adaptation.adaptation_generalization_test import calculate_generalization
from core.plotting import find_angle_difference
import copy

cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "log_to_wandb": False})
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
base_agent = copy.deepcopy(agent)

# desired-transition embedding for the target (computed on a NO-rotation env), fixed
env0 = ReachTask(cfg)
ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
with torch.no_grad():
    e_target = agent.g(s0, env0.get_features(nxt0))
comp = float(np.degrees(env.actions[agent.find_optimal_action_ind()]))

def invmodel_decode():
    with torch.no_grad():
        return float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))

print("[setup] target=%d compensated=%.0f  base inv-model decode of desired transition=%.0f"
      % (target_deg, comp, invmodel_decode()), flush=True)

traj = []
for ep in range(adapt_eps):
    env.reset()
    feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
    if ep % 2000 == 0:
        traj.append((ep, invmodel_decode()))
print("inv-model decode of desired transition over adaptation:", flush=True)
print("  " + "  ".join("%d:%.0f" % (e, d) for e, d in traj), flush=True)
df = calculate_generalization(agent, base_agent, cfg)
d = df.set_index("angle from target")["adaptation amount"]
near = d.loc[[a for a in [-15, 0, 15] if a in d.index]].mean()
print("[result] calculate_generalization adaptation_amount near target = %.0f%% ; final inv-model decode = %.0f (compensated=%.0f)"
      % (near, traj[-1][1], comp), flush=True)
