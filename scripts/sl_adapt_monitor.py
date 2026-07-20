"""Run the validated SL decoder-adaptation with monitoring: greedy re-reach error + decode
heatmaps every N episodes, across a couple of decoder learning rates, to find the config
that rotates the decoder correctly without over-training (per Jesse's hand-tuning approach)."""
import argparse
import json
from pathlib import Path

from scripts.track_a_control import train_adapt, BASE
from definitions import revision_fig_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fg_lrs", type=float, nargs="+", default=[1e-4, 3e-5])
    ap.add_argument("--adapt_episodes", type=int, default=100000)
    ap.add_argument("--monitor_every", type=int, default=20000)
    args = ap.parse_args()

    summary = {}
    for fg_lr in args.fg_lrs:
        agent, base, acfg = train_adapt("sl", args.seed, 135, -30, {**BASE},
                                        adapt_episodes=args.adapt_episodes,
                                        monitor_every=args.monitor_every, fg_lr=fg_lr)
        # final greedy re-reach
        from scripts.track_a_control import _greedy_achieved_err
        gerr = _greedy_achieved_err(agent, agent.env, 135, -30)
        summary[str(fg_lr)] = gerr
        print(f"[DONE fg_lr={fg_lr}] final greedy_err={gerr:.1f}", flush=True)
    Path(revision_fig_dir).joinpath("sl_adapt_monitor_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", summary, flush=True)


if __name__ == "__main__":
    main()
