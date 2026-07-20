"""Produce a clean Fig 3E re-learning curve using the PROTECTED base (no retrain). Sweep fg_lr,
track the inverse-model decode of the target transition + the taken-action achieved error at fine
resolution, and record the episode of best compensation (early-stop point). Goal: a smooth
135 -> 165 relearning that we can stop at correct compensation before the over-rotation."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, copy
from pathlib import Path
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config as acfg
from core.plotting import find_angle_difference
from definitions import paper_model_path

seed, target_deg, rot = 0, 135, -30
PROT = Path(paper_model_path) / f"PROTECTED_base_seed{seed}_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
adapt_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000

for fg_lr in [1e-3, 3e-4, 1e-4]:
    shutil.copy(PROT, STD)
    cfg = dict(acfg)
    cfg.update({"seed": seed, "reach_angle": float(np.radians(target_deg)), "rotation_angle": rot,
                "fg_lr": fg_lr, "log_to_wandb": False})
    env = ReachTask(cfg, adaptation_rotation=float(np.radians(rot)))
    agent = load_trained_full_model_basetask(cfg, env, target_deg, seed)
    env0 = ReachTask(cfg)
    ti = int(np.argmin(np.abs(((np.round(np.degrees(env0.actions)) - target_deg + 180) % 360) - 180)))
    env0.reset(); s0 = env0.get_features(env0.current_xy); nxt0, _, _ = env0.act(env0.actions[ti])
    with torch.no_grad():
        e_target = agent.g(s0, env0.get_features(nxt0))
    comp = float(np.degrees(env.actions[agent.find_optimal_action_ind()]))

    def decode():
        with torch.no_grad():
            return float(np.degrees(env.actions[int(torch.argmax(agent.f(e_target)).item())]))

    traj, best = [], (1e9, None, None)  # (|decode-comp|, ep, decode)
    for ep in range(adapt_eps):
        env.reset()
        feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False, import_policy_mean=False, policy_mean=None)
        nxt, reward, done = env.act(env.actions[a_idx])
        agent.update(feats, a_idx, emb, env.get_features(nxt), reward, done)
        if ep % 200 == 0:
            d = decode()
            traj.append((ep, d))
            err = abs(((d - comp + 180) % 360) - 180)
            if err < best[0]:
                best = (err, ep, d)
    print("fg_lr=%.0e decode traj: %s" % (fg_lr, "  ".join("%d:%.0f" % (e, d) for e, d in traj[::3])), flush=True)
    print("           best compensation: ep=%s decode=%.0f (target compensated=%.0f)" % (best[1], best[2], comp), flush=True)
