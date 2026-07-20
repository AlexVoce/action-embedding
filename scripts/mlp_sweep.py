"""Hyperparameter sweep for the expressive (MLP) encoder g on the multi-joint torus.
Goal: find settings where g learns a clean, decodable torus (high decode acc + high
per-joint ring recovery) at the scales where we want the speed advantage (d=2 dense,
d=3). Runs on Spark (CPU) in the background."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.multijoint import MultiJointReach, train_sl_embedding


def ring_r2(g, env):
    cfeat = env.center_features()
    with torch.no_grad():
        codes = np.array([g(cfeat, env.features(env.configs[c])).numpy() for c in range(env.n_actions)])
    phi = env.angles(env.configs.T).T
    r2s = []
    for j in range(env.d):
        target = np.column_stack([np.cos(phi[:, j]), np.sin(phi[:, j])])
        aug = np.column_stack([codes, np.ones(len(codes))])
        A, *_ = np.linalg.lstsq(aug, target, rcond=None)
        pred = aug @ A
        r2s.append(1 - ((target - pred) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())
    return float(np.mean(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="figures/paper/mlp_sweep.json")
    ap.add_argument("--only_idx", type=int, default=-1, help="run just this config index (for parallel launch)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    # (d, k) settings we care about
    settings = [(2, 16), (3, 6)]     # dense 2-torus (N=256) and 3-torus (N=216)
    lrs = [3e-4, 1e-3, 3e-3]
    hiddens = [128, 256]
    steps_list = [400000]
    weight_decays = [0.0, 1e-4]
    seeds = [0]

    t0 = time.time()
    results = []
    grid = list(itertools.product(settings, lrs, hiddens, steps_list, weight_decays, seeds))
    if args.only_idx >= 0:
        grid = [grid[args.only_idx]]
        args.out = args.out.replace(".json", f"_{args.only_idx}.json")
    print(f"sweep of {len(grid)} configs -> {args.out}", flush=True)
    for (d, k), lr, hidden, steps, wd, seed in grid:
        env = MultiJointReach(d, k)
        g, f, acc = train_sl_embedding(MultiJointReach(d, k), seed, 2 * d, steps=steps, lr=lr,
                                       nonlinear=True, hidden=hidden, weight_decay=wd,
                                       betas=(0.9, 0.999))
        r2 = ring_r2(g, env)
        row = {"d": d, "k": k, "N": env.n_actions, "lr": lr, "hidden": hidden, "steps": steps,
               "weight_decay": wd, "seed": seed, "decode_acc": acc, "ring_r2": r2}
        results.append(row)
        print(f"[d={d} k={k} N={env.n_actions} lr={lr} h={hidden} wd={wd}] "
              f"decode_acc={acc:.3f} ring_r2={r2:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))  # incremental save
    print(f"\nDONE {len(results)} configs in {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
