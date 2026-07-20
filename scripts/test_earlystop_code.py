"""Run the repo's ACTUAL adaptation with its real early-stop (angle_diff<10 criterion, as in
second_adaptation_config) on the protected base, and report where it stops + the resulting
generalization profile. Tests whether the code's own early stop fires cleanly on my model."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pathlib import Path
from adaptation.config import config
from adaptation.adaptation_exp import run_and_log_adaptation_experiment
from definitions import paper_model_path, paper_fig_dir

seed, target_deg = 0, 135
PROT = Path(paper_model_path) / f"PROTECTED_base_seed0_target{target_deg}.pth"
STD = Path(paper_model_path) / f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{target_deg}_n_actions_24.pth"
shutil.copy(PROT, STD)

cfg = dict(config)
cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), log_to_wandb=False, save_model=False,
           save_figs_locally=False, max_episodes=150000,
           angle_diff_criterion=10.0, early_stopping_criterion=30000)  # the REAL early stop
run_and_log_adaptation_experiment(cfg)

rl = pd.read_csv(Path(paper_fig_dir) / f"adaptation_run_log_seed_{seed}_target_{target_deg}.csv")
gs = pd.read_csv(Path(paper_fig_dir) / f"generalization_stats_seed_{seed}_target_{target_deg}_rotation_-30.csv")
stop_ep = int(rl["episode"].max())
d = gs[["rotation generalization", "angular error", "angle from target"]].set_index("angle from target").sort_index()
loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
tgt = d.loc[0]["angular error"] if 0 in d.index else np.nan
print("EARLYSTOP: stopped at episode %d (max 150000)" % stop_ep, flush=True)
print("  locality(local-global)=%.0f  target angular-error=%.1f  peak gen=%.0f%%" % (loc - glob, tgt, d["rotation generalization"].max()), flush=True)
print("  final taken-action angle_diff (mean last 5000) = %.1f" % rl["angle_diff"].abs().tail(5000).mean(), flush=True)
