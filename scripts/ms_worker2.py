"""Multi-seed adaptation with EARLY STOPPING (how it was done originally: monitor, stop before
over-training). Reuses the already-trained per-seed base. Tracks the sampled mean achieved error
at the target; keeps the best (most-compensated) decoder state and stops once it starts to
over-rotate, then restores the best f. Saves Fig 3I (calculate_generalization at the stop),
Fig 3E curve (to the stop), pre/post distributions, and the RL race.

usage: ms_worker2.py <seed> [sl_cap] [rl_eps]
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from core.agent import ACLearningAgentWithEmbedding, softmax_with_temperature, add_noise_to_action_probs
from adaptation.config import config as acfg
from adaptation.adaptation_generalization_test import calculate_generalization
from definitions import paper_model_path, paper_fig_dir

seed = int(sys.argv[1])
sl_cap = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
rl_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
target_deg, rot = 135, -30
OUT = Path(paper_fig_dir)
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"


def target_embedding(agent):
    e0 = ReachTask({**acfg, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot})
    ti = int(np.argmin(np.abs(((np.round(np.degrees(e0.actions)) - target_deg + 180) % 360) - 180)))
    e0.reset(); s = e0.get_features(e0.current_xy); nxt, _, _ = e0.act(e0.actions[ti])
    with torch.no_grad():
        return agent.g(s, e0.get_features(nxt))


cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=1e-4, log_to_wandb=False, use_random_policy=False)
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
base_agent = copy.deepcopy(agent)
e_target = target_embedding(agent)


def sampled_achieved_err(n=400):
    a = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        a.append(float(np.degrees(env.actions[int(torch.multinomial(p, 1).item())])))
    a = np.array(a)
    return float(np.mean(np.abs(((a + rot - target_deg + 180) % 360) - 180)))


def dist_hist(n=3000):
    a = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        a.append(int(torch.multinomial(p, 1).item()))
    return np.histogram(a, bins=np.arange(env.n_actions + 1))[0].tolist()


def greedy_err():
    with torch.no_grad():
        g = float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))
    return abs(((g + rot - target_deg + 180) % 360) - 180)


dist_pre = dist_hist()
curve = []
best = (1e9, 0, copy.deepcopy(agent.f.state_dict()))
worse_count = 0
for ep in range(sl_cap):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
    if hasattr(agent, "f_g_optimizer"):
        agent.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
    if ep % 500 == 0:
        curve.append([ep, greedy_err()])
    if ep % 1000 == 0 and ep > 0:
        me = sampled_achieved_err()
        if me < best[0] - 0.5:
            best = (me, ep, copy.deepcopy(agent.f.state_dict())); worse_count = 0
        elif me > best[0] + 6:   # started to over-rotate past the best
            worse_count += 1
            if worse_count >= 2:
                break
stop_ep = best[1]
agent.f.load_state_dict(best[2])   # restore the best (compensated, not over-trained) decoder
dist_post = dist_hist()
calculate_generalization(agent, base_agent, cfg).to_csv(OUT / f"ms2_gen_seed{seed}.csv")
curve = [c for c in curve if c[0] <= stop_ep + 500]
json.dump({"eps": [e for e, _ in curve], "errs": [v for _, v in curve], "stop_ep": stop_ep, "best_err": best[0]},
          open(OUT / f"ms2_fig3e_seed{seed}.json", "w"))
json.dump({"pre": dist_pre, "post": dist_post}, open(OUT / f"ms2_dist_seed{seed}.json", "w"))
print("seed%d: SL early-stopped at ep=%d (achieved err=%.1f)" % (seed, stop_ep, best[0]), flush=True)

# RL race (actor plastic, f frozen) — full run (reward-driven, slow)
cfg2 = dict(acfg)
cfg2.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=0.0, actor_lr=1e-4, critic_lr=5e-4, log_to_wandb=False)
env2 = ReachTask(cfg2, adaptation_rotation=float(np.radians(rot)))
agent2 = ACLearningAgentWithEmbedding(env2, cfg2, full_model_load_path=str(STD),
                                      f_plastic=False, g_plastic=False, actor_plastic=True, critic_plastic=True)


def greedy_err_rl():
    feats = env2.get_features(env2.current_xy)
    with torch.no_grad():
        g = float(np.degrees(env2.actions[int(torch.argmax(agent2.f(torch.tanh(agent2.actor(feats)))).item())]))
    return abs(((g + rot - target_deg + 180) % 360) - 180)


rl_curve = []
for ep in range(rl_eps):
    env2.reset(); feats = env2.get_features(env2.current_xy)
    agent2.actor_optimizer.zero_grad(); agent2.critic_optimizer.zero_grad()
    if hasattr(agent2, "f_g_optimizer"):
        agent2.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent2.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env2.act(env2.actions[a_idx])
    agent2.update(feats, a_idx, emb, env2.get_features(nxt), r, done)
    if ep % 500 == 0:
        rl_curve.append([ep, greedy_err_rl()])
json.dump({"rl_eps": [e for e, _ in rl_curve], "rl_errs": [v for _, v in rl_curve]}, open(OUT / f"ms2_race_seed{seed}.json", "w"))
print("SEED %d DONE" % seed, flush=True)
