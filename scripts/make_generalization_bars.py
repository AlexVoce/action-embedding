"""Adaptation TRANSFER to untrained targets (bar chart), SL embedding vs standard full-RL.

After adapting ONE target to a 30 deg visuomotor rotation, every *other* (untrained) target starts at
the un-adapted baseline error = the rotation magnitude (30 deg). We measure how much the adaptation
changed each untrained target's error:

    transfer = 30 (baseline) - |angular error after adaptation|

    transfer > 0  -> adaptation GENERALISED to that target (error reduced)
    transfer ~ 0  -> no transfer (target stayed at the un-adapted baseline)
    transfer < 0  -> adaptation INTERFERED (error increased above baseline)

We split untrained targets into NEAR (0 < |angle from target| <= 45 deg) and FAR (> 45 deg), and
EXCLUDE the trained target itself (trivially compensated by both systems). Read-out:
  * SL embedding: strong positive transfer near, weaker far -> LOCAL generalisation (as in humans).
  * Standard full-RL: ~0 transfer near (no generalisation) and NEGATIVE far (interference).

We summarise as bars rather than the full curve because the standard-RL generalisation profile is too
disorganised across seeds to plot meaningfully.
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

SL_LABEL, STD_LABEL = "SL embedding (ours)", "Standard full-RL actor"
PALETTE = {SL_LABEL: "#2E8B57", STD_LABEL: "#A94850"}
SL_SEEDS = {2, 3, 4, 5, 6, 7, 8, 9, 35}
BASELINE = 30.0          # un-adapted error at any untrained target = rotation magnitude
NEAR = 45.0
NEAR_LBL, FAR_LBL = "near\n(≤45°)", "far\n(>45°)"
RFD = Path(revision_fig_dir)
set_plotting_defaults()


def transfer_near_far(aft, err):
    aft = np.abs(np.asarray(aft, float)); err = np.abs(np.asarray(err, float))
    untrained = aft > 1e-6                                   # drop the trained target itself
    t = BASELINE - err
    near = t[untrained & (aft <= NEAR)]
    far = t[untrained & (aft > NEAR)]
    return float(near.mean()), float(far.mean())


rows = []
for fp in sorted(glob.glob(str(Path(paper_fig_dir) / "generalization_stats_seed_*_target_135_rotation_-30.csv"))):
    seed = int(fp.split("seed_")[1].split("_")[0])
    if seed not in SL_SEEDS:
        continue
    d = pd.read_csv(fp)
    n, f = transfer_near_far(d["angle from target"], d["angular error"])
    rows += [{"system": SL_LABEL, "region": NEAR_LBL, "seed": seed, "transfer": n},
             {"system": SL_LABEL, "region": FAR_LBL, "seed": seed, "transfer": f}]
for fp in sorted(glob.glob(str(RFD / "gen_standard_s*_N24.json"))):
    d = json.load(open(fp))
    n, f = transfer_near_far(d["dist_from_probe"], d["angular_error"])
    rows += [{"system": STD_LABEL, "region": NEAR_LBL, "seed": d["seed"], "transfer": n},
             {"system": STD_LABEL, "region": FAR_LBL, "seed": d["seed"], "transfer": f}]
df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(4.2, 3.4))
sns.barplot(data=df, x="region", y="transfer", hue="system", palette=PALETTE, errorbar="se",
            capsize=0.12, err_kws={"linewidth": 1.2}, ax=ax)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel(""); ax.set_ylabel("adaptation transfer (° error reduced\nvs un-adapted baseline)")
ax.legend(frameon=False, fontsize=8, title=None, loc="lower left")
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = RFD / "adaptation_generalization_bars.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved", out.name)
for sysname in (SL_LABEL, STD_LABEL):
    s = df[df.system == sysname]
    for reg in (NEAR_LBL, FAR_LBL):
        v = s[s.region == reg]["transfer"]
        print(f"{sysname:24s} {reg.split(chr(10))[0]:5s}: transfer {v.mean():+5.1f} ± {v.sem():.1f} (n={len(v)})")
