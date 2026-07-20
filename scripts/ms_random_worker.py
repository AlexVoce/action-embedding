"""Random-embedding control for the corrected pipeline (R1 part-b bottleneck baseline).
Per seed: build a FRESH RANDOM (untrained) g/f of the same architecture, train a base policy on
it, then run the repo's adaptation with the real early-stop -> generalization_stats. Compared
against the SL per-seed generalization to show a generic low-D embedding does NOT reproduce the
local pattern. Uses seed IDs 50+ so filenames don't collide with the SL runs.

usage: ms_random_worker.py <seed> [base_eps]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ActionEmbeddingPredictor, ActionMapping
from adaptation.config import config as acfg
from definitions import paper_model_path

seed = int(sys.argv[1])
base_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
target_deg = 135
mp = Path(paper_model_path)
EMB = mp / f"action_embedding_model_seed_{seed}_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth"
STD = mp / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"

# 1) random (untrained) g/f, saved in the embedding-file format policy_learning expects
env = ReachTask({**acfg})
torch.manual_seed(seed); np.random.seed(seed)
g = ActionEmbeddingPredictor(env.n_features, 2)
f = ActionMapping(2, env.n_actions)
torch.save({"g_state_dict": g.state_dict(), "f_state_dict": f.state_dict(),
            "params": {"state_dim": env.n_features, "embedding_dim": 2, "n_actions": env.n_actions}}, EMB)
print("seed%d: random embedding saved" % seed, flush=True)

# 2) base policy on the random embedding
from core.config import config as bcfg
bcfg["fg_load_path"] = str(EMB); bcfg["max_episodes"] = base_eps; bcfg["log_to_wandb"] = False; bcfg["save_model"] = True
sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
from core.policy_learning import train_agent
train_agent(bcfg)
print("seed%d: base on random embedding trained" % seed, flush=True)

# 3) adaptation with real early-stop -> generalization_stats_seed_{seed}_target_135_rotation_-30.csv
from adaptation.adaptation_exp import run_and_log_adaptation_experiment
cfg = dict(acfg)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), log_to_wandb=False, save_model=False,
           save_figs_locally=False, max_episodes=150000, angle_diff_criterion=10.0, early_stopping_criterion=30000)
run_and_log_adaptation_experiment(cfg)
print("SEED %d RANDOM DONE" % seed, flush=True)
