"""Adaptation-mechanism comparison (why multiple systems): re-learning speed of the SL
decoder-adaptation vs reward-driven RL adaptation, on the SL-ring base policy, using the
validated adaptation loop and the paper's Fig-3E error (find_angle_difference of the taken
action). SL (error-driven) should re-learn faster than RL (reward-driven)."""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.track_a_control import train_adapt, BASE
from definitions import paper_fig_dir, revision_fig_dir

COLOR = {"sl": "#2E8B57", "rl": "#A94850"}
LABEL = {"sl": "SL adaptation (error-driven, cerebellar)", "rl": "RL adaptation (reward-driven, BG)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--adapt_episodes", type=int, default=100000)
    ap.add_argument("--make_fig_only", action="store_true")
    args = ap.parse_args()

    if not args.make_fig_only:
        for mode in ["sl", "rl"]:
            for seed in args.seeds:
                # paper's exact adaptation config: natural actor (import_policy_mean=False).
                # SL -> f plastic (error-driven); RL -> actor/critic plastic (reward-driven).
                train_adapt("sl", seed, 135, -30, {**BASE, "seed": seed},
                            adapt_episodes=args.adapt_episodes, adapt_mode=mode,
                            use_gemb_policy=False, align_actor_to_gemb=False)

    # build figure from saved relearn curves
    curves = defaultdict(list)
    for f in glob.glob(str(Path(paper_fig_dir) / "relearn_sl_*_seed*_rot-30.json")):
        d = json.load(open(f))
        curves[d["adapt_mode"]].append((d["roll_err"], d["step"]))
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for mode in ["sl", "rl"]:
        if not curves[mode]:
            continue
        step = curves[mode][0][1]
        L = min(len(c) for c, _ in curves[mode])
        arr = np.array([c[:L] for c, _ in curves[mode]])
        eps = np.arange(L) * step
        m = arr.mean(0); e = arr.std(0) / max(1, np.sqrt(arr.shape[0]))
        ax.plot(eps, m, color=COLOR[mode], label=LABEL[mode], lw=2)
        ax.fill_between(eps, m - e, m + e, color=COLOR[mode], alpha=0.2)
    ax.set(xlabel="adaptation episodes", ylabel="taken-action error (deg)",
           title="Re-learning after rotation: SL (fast) vs RL (slow)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "adapt_mechanism_relearn"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png")
    for mode in ["sl", "rl"]:
        if curves[mode]:
            arr = np.array([c[:min(len(x) for x, _ in curves[mode])] for c, _ in curves[mode]])
            print(f"  {mode}: start~{arr[:,1].mean():.0f}deg  end~{arr[:,-1].mean():.0f}deg")


if __name__ == "__main__":
    main()
