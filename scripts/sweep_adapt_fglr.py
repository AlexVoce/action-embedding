"""Sweep decoder-adaptation lr in the PAPER's exact adaptation_exp loop. Report the run-log
angle_diff trajectory (Fig 3E quantity). Goal: find fg_lr where the taken-action error dips to
~0 (correct compensation) instead of over-rotating to the ~45 deg overshoot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from adaptation.config import config
from adaptation.adaptation_exp import run_and_log_adaptation_experiment
from definitions import paper_fig_dir

RUNLOG = os.path.join(paper_fig_dir, "adaptation_run_log_seed_0_target_135.csv")
for fg_lr in [3e-5, 1e-5, 3e-6, 1e-6]:
    cfg = dict(config)
    cfg["fg_lr"] = fg_lr
    cfg["max_episodes"] = 30000
    cfg["seed"] = 0
    cfg["reach_angle"] = float(np.radians(135))
    cfg["log_to_wandb"] = False
    cfg["save_model"] = False
    cfg["save_figs_locally"] = False
    run_and_log_adaptation_experiment(cfg)
    ad = pd.read_csv(RUNLOG)["angle_diff"].abs().values
    roll = [round(float(np.mean(ad[max(0, i - 500):i + 1])), 1) for i in range(0, len(ad), 1500)]
    allroll = np.array([np.mean(ad[max(0, i - 500):i + 1]) for i in range(len(ad))])
    print("fg_lr=%.0e  traj=%s  MIN=%.1f@ep%d  final=%.1f"
          % (fg_lr, roll, allroll.min(), int(allroll.argmin()), roll[-1]), flush=True)
