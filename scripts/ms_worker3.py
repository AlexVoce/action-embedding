"""Comprehensive per-seed worker for the FINAL Fig 3 panels with per-seed embeddings.
Per seed: (1) embedding (own seed), (2) base policy on it, (3) SL adaptation via the repo's
run_and_log_adaptation_experiment with the real early-stop (angle_diff<10) -> saves gen_stats
(Fig 3I) + run_log (Fig 3E) + post-adaptation model, (4) pre/post action distributions from the
target's transition embedding (Fig 3F), (5) RL adaptation race curve. Reuses existing
embedding/base if present.

usage: ms_worker3.py <seed> [base_eps]
"""
import sys, os, subprocess, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding, softmax_with_temperature, add_noise_to_action_probs
from adaptation.config import config as acfg
from definitions import paper_model_path, paper_fig_dir

seed = int(sys.argv[1])
base_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
target_deg, rot = 135, -30
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mp = Path(paper_model_path); OUT = Path(paper_fig_dir)
EMB = mp / f"action_embedding_model_seed_{seed}_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth"
STD = mp / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
POST = mp / f"post_adaptation_model_seed_{seed}_rotation_-30_temp_2.5_weight_decay_0.0001_tanh_policy_mean_n_actions_24_target_{target_deg}.0.pth"

# 1) embedding
if not EMB.exists():
    subprocess.run([sys.executable, "scripts/embedding_learning.py", "--seed", str(seed)], cwd=root,
                   env=dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="5"), check=True)
# 2) base
if not STD.exists():
    from core.config import config as bcfg
    bcfg["fg_load_path"] = str(EMB); bcfg["max_episodes"] = base_eps; bcfg["log_to_wandb"] = False; bcfg["save_model"] = True
    sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
    from core.policy_learning import train_agent
    train_agent(bcfg)
print("seed%d: embedding+base ready" % seed, flush=True)

# 3) SL adaptation via the repo function (real early-stop) -> gen_stats + run_log + post model
from adaptation.adaptation_exp import run_and_log_adaptation_experiment
cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), log_to_wandb=False, save_model=True,
           save_figs_locally=False, max_episodes=150000, angle_diff_criterion=10.0, early_stopping_criterion=30000)
run_and_log_adaptation_experiment(cfg)
print("seed%d: SL adaptation done" % seed, flush=True)

# 4) Fig 3F: pre/post action distributions from the target's transition embedding
def load(path):
    e = ReachTask({**cfg}, adaptation_rotation=float(np.radians(rot)))
    a = ACLearningAgentWithEmbedding(e, cfg, full_model_load_path=str(path), fg_load_path=None,
                                     g_plastic=False, f_plastic=False, actor_plastic=False, critic_plastic=False)
    return e, a

env0 = ReachTask({**cfg})
ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])

def dist_from(agent):
    with torch.no_grad():
        e_t = agent.g(s0, env0.get_features(nxt0))
    a = []
    for _ in range(4000):
        e = torch.normal(e_t, torch.tensor(agent.internal_policy_std))
        p = softmax_with_temperature(agent.f(e), temperature=agent.softmax_inv_temp)
        p = add_noise_to_action_probs(p, noise_level=0.008)
        a.append(int(torch.multinomial(p, 1).item()))
    return np.histogram(a, bins=np.arange(25))[0].tolist()

_, base_a = load(STD)
_, post_a = load(POST)
json.dump({"pre": dist_from(base_a), "post": dist_from(post_a)}, open(OUT / f"ms3_dist_seed{seed}.json", "w"))
print("seed%d: Fig3F distributions saved" % seed, flush=True)

# 5) RL race
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
rlc = []
for ep in range(60000):
    env2.reset(); feats = env2.get_features(env2.current_xy)
    agent2.actor_optimizer.zero_grad(); agent2.critic_optimizer.zero_grad()
    if hasattr(agent2, "f_g_optimizer"):
        agent2.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent2.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env2.act(env2.actions[a_idx])
    agent2.update(feats, a_idx, emb, env2.get_features(nxt), r, done)
    if ep % 500 == 0:
        rlc.append([ep, greedy_err_rl()])
json.dump({"rl_eps": [e for e, _ in rlc], "rl_errs": [v for _, v in rlc]}, open(OUT / f"ms3_race_seed{seed}.json", "w"))
print("SEED %d DONE" % seed, flush=True)
