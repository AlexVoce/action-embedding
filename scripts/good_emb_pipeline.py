"""Decisive test of the per-seed-embedding hypothesis: train a fresh embedding on a given seed,
build a base policy on THAT embedding, adapt with the repo's real early-stop, and report whether
the target fully compensates (error -> 0, peak generalization ~100%). If seed-0's embedding was a
'bad' (bimodal) one, a good-seed embedding should compensate cleanly.

usage: good_emb_pipeline.py <emb_seed> [base_eps]
"""
import sys, os, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pathlib import Path
from definitions import paper_model_path, paper_fig_dir

emb_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 35
base_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
target_deg = 135
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = Path(paper_model_path) / f"action_embedding_model_seed_{emb_seed}_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{emb_seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"

# 1) train embedding on this seed (subprocess -> clean config)
if not EMB.exists():
    env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="6")
    print("training embedding seed %d ..." % emb_seed, flush=True)
    subprocess.run([sys.executable, "scripts/embedding_learning.py", "--seed", str(emb_seed)], cwd=root, env=env, check=True)
print("embedding exists:", EMB.exists(), flush=True)

# 2) base policy on THIS embedding
from core.config import config as bcfg
bcfg["fg_load_path"] = str(EMB)
bcfg["max_episodes"] = base_eps
bcfg["log_to_wandb"] = False
bcfg["save_model"] = True
sys.argv = ["x", "--seed", str(emb_seed), "--target", repr(float(np.radians(target_deg)))]
from core.policy_learning import train_agent
train_agent(bcfg)
print("base policy trained on seed-%d embedding" % emb_seed, flush=True)

# 3) adaptation with the real early-stop
from adaptation.config import config as acfg
from adaptation.adaptation_exp import run_and_log_adaptation_experiment
cfg = dict(acfg)
cfg.update(seed=emb_seed, reach_angle=float(np.radians(target_deg)), log_to_wandb=False, save_model=False,
           save_figs_locally=False, max_episodes=150000, angle_diff_criterion=10.0, early_stopping_criterion=30000)
run_and_log_adaptation_experiment(cfg)

rl = pd.read_csv(Path(paper_fig_dir) / f"adaptation_run_log_seed_{emb_seed}_target_{target_deg}.csv")
gs = pd.read_csv(Path(paper_fig_dir) / f"generalization_stats_seed_{emb_seed}_target_{target_deg}_rotation_-30.csv")
d = gs[["rotation generalization", "angular error", "angle from target"]].set_index("angle from target").sort_index()
loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
print("RESULT emb_seed=%d: stop=%d locality=%.0f target_err=%.1f peak_gen=%.0f%% final_angle_diff=%.1f" % (
    emb_seed, int(rl["episode"].max()), loc - glob, d.loc[0]["angular error"] if 0 in d.index else -1,
    d["rotation generalization"].max(), rl["angle_diff"].abs().tail(5000).mean()), flush=True)
