"""Decisive: after SL adaptation, is the SAMPLED behavioral distribution (from the target's
transition embedding through the relearned f) centered on correct compensation (165, reaching the
target) or overshot (210)? This is what calculate_generalization actually scores and what Fig 3F
plots."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, copy
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from core.agent import softmax_with_temperature, add_noise_to_action_probs
from adaptation.config import config as acfg
from adaptation.adaptation_generalization_test import calculate_generalization
from definitions import paper_model_path

seed, target_deg, rot = 0, 135, -30
adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
fg_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
rand_pol = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False  # random exploration decouples the on-policy feedback loop
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
shutil.copy(PROT, STD)

cfg = dict(acfg)
cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot, "fg_lr": fg_lr, "log_to_wandb": False})
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
base_agent = copy.deepcopy(agent)
env0 = ReachTask(cfg)
ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
with torch.no_grad():
    e_target = agent.g(s0, env0.get_features(nxt0))
comp = float(np.degrees(env.actions[agent.find_optimal_action_ind()]))

def sampled_stats(n=2000):
    acts = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        acts.append(float(np.degrees(env.actions[int(torch.multinomial(p, 1).item())])))
    acts = np.array(acts)
    # circular mean and mean achieved error to target (achieved = action + rotation)
    achieved = (acts + rot) % 360
    err = np.abs(((achieved - target_deg + 180) % 360) - 180)
    return acts, np.mean(err)

print("compensated action=%.0f  (achieving target=%d)" % (comp, target_deg), flush=True)
a_pre, err_pre = sampled_stats(2000)
print("PRE-adapt sampled: mean achieved error=%.1f deg ; action mode~%.0f" % (err_pre, np.median(a_pre)), flush=True)
for ep in range(adapt_eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
    a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=rand_pol, import_policy_mean=False, policy_mean=None)
    nxt, reward, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
a_post, err_post = sampled_stats(2000)
with torch.no_grad():
    greedy_post = float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))
print("POST greedy (argmax, no-noise) decode of target embedding = %.0f (compensated=%.0f)" % (greedy_post, comp), flush=True)
hist = np.histogram(a_post, bins=np.arange(0, 361, 15))[0]
print("POST-adapt sampled: mean achieved error=%.1f deg ; action median~%.0f" % (err_post, np.median(a_post)), flush=True)
print("POST action histogram (15deg bins, deg:count>3%%):",
      "  ".join("%d:%d" % (b, c) for b, c in zip(range(0, 360, 15), hist) if c > 60), flush=True)
df = calculate_generalization(agent, base_agent, cfg)
d = df.set_index("angle from target")["adaptation amount"]
print("calc_generalization adaptation at target(0)=%.0f%%" % d.loc[0] if 0 in d.index else "n/a", flush=True)
