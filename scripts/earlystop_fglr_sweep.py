"""Does a gentler decoder LR converge to compensation (peak gen ~100%, angular error ~0) instead
of over-rotating (peak >>100%)? Run the repo's adaptation_exp with its real early-stop
(angle_diff<10) at several fg_lr on the protected base; report stop episode, locality, target
error, peak generalization."""
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
RL = Path(paper_fig_dir) / f"adaptation_run_log_seed_{seed}_target_{target_deg}.csv"
GS = Path(paper_fig_dir) / f"generalization_stats_seed_{seed}_target_{target_deg}_rotation_-30.csv"

for fg_lr in [3e-5, 1e-5, 3e-6]:
    shutil.copy(PROT, STD)
    cfg = dict(config)
    cfg.update(seed=seed, reach_angle=float(np.radians(target_deg)), log_to_wandb=False, save_model=False,
               save_figs_locally=False, max_episodes=200000, fg_lr=fg_lr,
               angle_diff_criterion=10.0, early_stopping_criterion=30000)
    run_and_log_adaptation_experiment(cfg)
    rl = pd.read_csv(RL); gs = pd.read_csv(GS)
    d = gs[["rotation generalization", "angular error", "angle from target"]].set_index("angle from target").sort_index()
    loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
    glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
    tgt = d.loc[0]["angular error"] if 0 in d.index else np.nan
    print("fg_lr=%.0e stop=%d locality=%.0f target_err=%.1f peak_gen=%.0f%% min_angerr=%.1f" % (
        fg_lr, int(rl["episode"].max()), loc - glob, tgt, d["rotation generalization"].max(), d["angular error"].min()), flush=True)
