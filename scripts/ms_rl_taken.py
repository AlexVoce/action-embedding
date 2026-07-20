"""Fix the race metric: measure RL adaptation with the SAME taken-action angle_diff as the SL
run-log (find_angle_difference of the taken action), so SL and RL are directly comparable."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from core.plotting import find_angle_difference
from adaptation.config import config as acfg
from definitions import paper_model_path, paper_fig_dir

seed = int(sys.argv[1])
rl_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
target_deg, rot = 135, -30
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"

cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), rotation_angle=rot, fg_lr=0.0, actor_lr=1e-4, critic_lr=5e-4, log_to_wandb=False)
env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
agent = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=str(STD),
                                     f_plastic=False, g_plastic=False, actor_plastic=True, critic_plastic=True)
errs = []
for ep in range(rl_eps):
    env.reset(); feats = env.get_features(env.current_xy)
    agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
    if hasattr(agent, "f_g_optimizer"):
        agent.f_g_optimizer.zero_grad()
    a_idx, emb, m, ls = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
    action = env.actions[a_idx]
    nxt, r, done = env.act(action)
    agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
    errs.append(abs(find_angle_difference(env, action)))  # taken-action error, same as SL run-log
w = 5000
roll = [float(np.mean(errs[max(0, i - w):i + 1])) for i in range(0, len(errs), w)]
json.dump({"eps": list(range(0, len(errs), w)), "rl_taken": roll}, open(Path(paper_fig_dir) / f"ms3_rltaken_seed{seed}.json", "w"))
print("SEED %d RLTAKEN DONE (start=%.0f end=%.0f)" % (seed, roll[1] if len(roll) > 1 else roll[0], roll[-1]), flush=True)
