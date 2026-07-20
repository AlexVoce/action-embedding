"""Multi-seed adaptation worker: for one seed, (1) ensure a base policy, (2) run on-policy SL
decoder-adaptation saving the Fig 3I generalization df, the Fig 3E greedy re-learning curve and
pre/post action distributions (Fig 3F), and (3) run RL adaptation saving the race curve. All
per-seed outputs are written to figures/paper/ms_*_seed{S}.* for the aggregator.

usage: ms_worker.py <seed> [base_eps] [sl_eps] [rl_eps]
"""
import sys, os, shutil, json, copy
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
base_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
sl_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
rl_eps = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
target_deg, rot = 135, -30
OUT = Path(paper_fig_dir)
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
PROT = Path(paper_model_path) / f"PROTECTED_base_seed0_target{target_deg}.pth"

# 1) base policy (reuse protected seed-0; otherwise train fresh for this seed)
if seed == 0 and PROT.exists():
    shutil.copy(PROT, STD)
    print("seed0: reused protected base", flush=True)
else:
    from core.config import config as bcfg
    bcfg["max_episodes"] = base_eps
    bcfg["log_to_wandb"] = False
    bcfg["save_model"] = True
    sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
    from core.policy_learning import train_agent
    train_agent(bcfg)
    print("seed%d: base trained" % seed, flush=True)


def target_embedding(agent):
    env0 = ReachTask({**acfg, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot})
    ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
    env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
    with torch.no_grad():
        return agent.g(s0, env0.get_features(nxt0))


# 2) SL on-policy adaptation
cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=1e-4, log_to_wandb=False, use_random_policy=False)
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
base_agent = copy.deepcopy(agent)
e_target = target_embedding(agent)


def greedy_err_sl():
    with torch.no_grad():
        g = float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))
    return abs(((g + rot - target_deg + 180) % 360) - 180)


def sample_dist(n=3000):
    a = []
    for _ in range(n):
        e = torch.normal(e_target, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        a.append(int(torch.multinomial(p, 1).item()))
    return np.histogram(a, bins=np.arange(env.n_actions + 1))[0].tolist()


dist_pre = sample_dist()
curve = []
for ep in range(sl_eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
    if hasattr(agent, "f_g_optimizer"):
        agent.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
    if ep % 500 == 0:
        curve.append([ep, greedy_err_sl()])
dist_post = sample_dist()
calculate_generalization(agent, base_agent, cfg).to_csv(OUT / f"ms_gen_seed{seed}.csv")
json.dump({"eps": [e for e, _ in curve], "errs": [v for _, v in curve]}, open(OUT / f"ms_fig3e_seed{seed}.json", "w"))
json.dump({"pre": dist_pre, "post": dist_post}, open(OUT / f"ms_dist_seed{seed}.json", "w"))
print("seed%d: SL adaptation done" % seed, flush=True)

# 3) RL adaptation (actor/critic plastic, f frozen)
shutil.copy(PROT if (seed == 0 and PROT.exists()) else STD, STD)
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
json.dump({"sl_eps": [e for e, _ in curve], "sl_errs": [v for _, v in curve],
           "rl_eps": [e for e, _ in rl_curve], "rl_errs": [v for _, v in rl_curve]},
          open(OUT / f"ms_race_seed{seed}.json", "w"))
print("SEED %d DONE" % seed, flush=True)
