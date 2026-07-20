"""RL-only stage for the multi-seed race (seeds 1-4 crashed on a self-copy before RL). Loads the
existing per-seed base, runs reward-driven RL adaptation (f frozen), and writes ms_race_seed{S}.json
combining the already-saved SL curve (from ms_fig3e_seed{S}.json) with the RL curve."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from adaptation.config import config as acfg
from definitions import paper_model_path, paper_fig_dir

seed = int(sys.argv[1])
rl_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
target_deg, rot = 135, -30
OUT = Path(paper_fig_dir)
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"

cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=0.0, actor_lr=1e-4, critic_lr=5e-4, log_to_wandb=False)
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=str(STD),
                                     f_plastic=False, g_plastic=False, actor_plastic=True, critic_plastic=True)


def greedy_err_rl():
    feats = env.get_features(env.current_xy)
    with torch.no_grad():
        g = float(np.degrees(env.actions[int(torch.argmax(agent.f(torch.tanh(agent.actor(feats)))).item())]))
    return abs(((g + rot - target_deg + 180) % 360) - 180)


rl_curve = []
for ep in range(rl_eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
    if hasattr(agent, "f_g_optimizer"):
        agent.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    nxt, r, done = env.act(env.actions[a_idx])
    agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
    if ep % 500 == 0:
        rl_curve.append([ep, greedy_err_rl()])

sl = json.load(open(OUT / f"ms_fig3e_seed{seed}.json"))
json.dump({"sl_eps": sl["eps"], "sl_errs": sl["errs"],
           "rl_eps": [e for e, _ in rl_curve], "rl_errs": [v for _, v in rl_curve]},
          open(OUT / f"ms_race_seed{seed}.json", "w"))
print("SEED %d RL DONE" % seed, flush=True)
