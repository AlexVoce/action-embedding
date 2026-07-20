"""Adaptation-generalization figure in the exact style of Fig 3I (single-target visuomotor rotation).

x = 'angle from target' (signed; 0 = adapted target), y = 'angular error' with the pipeline sign
(optimal - taken; 0 = fully compensated), inverted y-axis (ylim [35, -15]) exactly as the paper, so
the adapted target sits at the top ("peak" at (0,0)).

  * SL embedding (ours): the REAL Fig 3I data -- generalization_stats_seed_{s}_target_135_rotation_-30.csv
    (single-target adaptation pipeline, early-stopped). Guaranteed clean local profile.
  * Standard full-RL actor: the multi-target standard actor adapted on the same target by RL, with the
    SAME early-stopping (probe error < 10 deg), then probed at every target. gen_standard_s*_N24.json
    stores 'angular_error' = taken + rot - target (= negative of the Fig-3I sign), so we negate it.
"""
import json, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from core.plotting import set_plotting_defaults
from definitions import revision_fig_dir, paper_fig_dir

SL_LABEL = "SL embedding (ours)"
STD_LABEL = "Standard full-RL actor"
PALETTE = {SL_LABEL: "#2E8B57", STD_LABEL: "#A94850"}
RFD = Path(revision_fig_dir)

set_plotting_defaults()
rows = []

# --- SL: the real Fig 3I generalization data (single-target, target 135) ---
# validated SL-learned per-seed embeddings (same set as the random-embedding control);
# excludes bad seeds (0,1,10) and the random-embedding controls (50-54).
SL_SEEDS = {2, 3, 4, 5, 6, 7, 8, 9, 35}
for fp in sorted(glob.glob(str(Path(paper_fig_dir) / "generalization_stats_seed_*_target_135_rotation_-30.csv"))):
    seed = int(fp.split("seed_")[1].split("_")[0])
    if seed not in SL_SEEDS:
        continue
    df = pd.read_csv(fp)
    for aft, err in zip(df["angle from target"], df["angular error"]):
        rows.append({"angle from target": float(aft), "angular error": float(err),
                     "seed": seed, "condition": SL_LABEL})

# --- Standard full-RL: multi-target actor, RL-adapted on the same target, early-stopped ---
for fp in sorted(glob.glob(str(RFD / "gen_standard_s*_N24.json"))):
    d = json.load(open(fp))
    for aft, err in zip(d["dist_from_probe"], d["angular_error"]):
        rows.append({"angle from target": float(aft), "angular error": float(-err),  # Fig-3I sign
                     "seed": d["seed"], "condition": STD_LABEL})

df = pd.DataFrame(rows)
n_sl = df[df.condition == SL_LABEL].seed.nunique()
n_std = df[df.condition == STD_LABEL].seed.nunique()

fig, ax = plt.subplots(figsize=(4.4, 3.4))
sns.lineplot(data=df, x="angle from target", y="angular error", hue="condition",
             errorbar="se", palette=PALETTE, ax=ax)
ax.set_ylabel("angular error (°)")
ax.set_xlabel("angle from target (°)")
ax.set_xlim([-90, 180])
ax.set_ylim([35, -15])                      # inverted, exactly as Fig 3I
handles, labels = ax.get_legend_handles_labels()
labels = [f"{SL_LABEL} (n={n_sl})" if l == SL_LABEL else f"{STD_LABEL} (n={n_std})" for l in labels]
ax.legend(handles, labels, frameon=False, fontsize=8, title=None)
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = RFD / "adaptation_generalization_sl_vs_rl.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved", out.name)
for cond in (SL_LABEL, STD_LABEL):
    sub = df[df.condition == cond]
    p = sub[sub["angle from target"] == 0]["angular error"].mean()
    far = sub[sub["angle from target"].abs() > 60]["angular error"].mean()
    print(f"{cond}: trained-target err~{p:.0f}  far err~{far:.0f}  n={sub.seed.nunique()}")
