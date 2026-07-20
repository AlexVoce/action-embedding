"""
A4 (R2 #8): how sensitive is the adaptation-generalization SCALE to a tuned parameter?

Reuses the validated SL adaptation (scripts.track_a_control.train_adapt) to get an adapted
agent, then sweeps the policy exploration std used in the generalization readout and
measures locality (local minus global rotation-generalization). Shows whether the observed
spatial scale is a robust prediction or depends on the (hand-tuned) exploration width ->
lets us state honestly in the response which it is.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.track_a_control import train_adapt, BASE
from adaptation.adaptation_generalization_test import calculate_generalization
from definitions import revision_fig_dir


def locality(df):
    d = df[["rotation generalization", "angle from target"]].set_index("angle from target")
    loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
    glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
    return float(loc), float(glob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stds", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--target_deg", type=int, default=135)
    ap.add_argument("--rotation_deg", type=int, default=-30)
    ap.add_argument("--adapt_episodes", type=int, default=100000)
    ap.add_argument("--out", type=str, default="figures/paper/a4_sensitivity.json")
    args = ap.parse_args()

    per_std = defaultdict(list)
    rows = []
    for seed in args.seeds:
        agent, base, acfg = train_adapt("sl", seed, args.target_deg, args.rotation_deg,
                                        {**BASE}, adapt_episodes=args.adapt_episodes)
        for std in args.stds:
            agent.set_policy_std(std)
            df = calculate_generalization(agent, base, {**acfg, "seed": seed})
            loc, glob = locality(df)
            per_std[std].append(loc - glob)
            rows.append({"seed": seed, "std": std, "local": loc, "global": glob, "locality": loc - glob})
            print(f"[A4 s{seed} std={std}] locality={loc-glob:.1f} (local={loc:.1f} global={glob:.1f})", flush=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))

    stds = sorted(per_std)
    means = [np.mean(per_std[s]) for s in stds]
    sems = [np.std(per_std[s]) / max(1, np.sqrt(len(per_std[s]))) for s in stds]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(stds, means, yerr=sems, marker="o", color="#2E8B57", capsize=3, lw=2)
    ax.set(xlabel="policy exploration std (embedding space)", ylabel="generalization locality (local − global)",
           title="A4: sensitivity of adaptation-generalization scale\nto exploration width")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "a4_sensitivity"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png")
    for s, m, e in zip(stds, means, sems):
        print(f"  std={s}: locality={m:.1f} +/- {e:.1f}")


if __name__ == "__main__":
    main()
