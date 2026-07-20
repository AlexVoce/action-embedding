"""Visualise the 2-joint learned manifold as its true structure: S^1 x S^1 (two circles).
For each joint j we recover the 2-D subspace of the learned code that encodes that joint's
angle (least-squares codes -> (cos phi_j, sin phi_j)) and scatter it, coloured by phi_j.
A clean product-of-circles => each panel is a clean ring; colouring a ring by the OTHER
joint should be uniformly mixed (that circle is 'collapsed' in this projection)."""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.multijoint import MultiJointReach, train_sl_embedding
from definitions import revision_fig_dir


def recover_ring(codes, phi_j):
    target = np.column_stack([np.cos(phi_j), np.sin(phi_j)])
    aug = np.column_stack([codes, np.ones(len(codes))])
    A, *_ = np.linalg.lstsq(aug, target, rcond=None)
    proj = aug @ A
    ss_res = ((target - proj) ** 2).sum(); ss_tot = ((target - target.mean(0)) ** 2).sum()
    return proj, 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--emb_steps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonlinear_g", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)

    env = MultiJointReach(2, args.k)
    g, f, acc = train_sl_embedding(MultiJointReach(2, args.k), args.seed, 4, steps=args.emb_steps,
                                   nonlinear=args.nonlinear_g)
    cfeat = env.center_features()
    with torch.no_grad():
        codes = np.array([g(cfeat, env.features(env.configs[c])).numpy() for c in range(env.n_actions)])
    phi = env.angles(env.configs.T).T  # (N,2)

    proj1, r1 = recover_ring(codes, phi[:, 0])
    proj2, r2 = recover_ring(codes, phi[:, 1])
    print(f"[d=2 k={args.k}] decode_acc={acc:.3f}  ring1_R2={r1:.3f}  ring2_R2={r2:.3f}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    s0 = axes[0].scatter(proj1[:, 0], proj1[:, 1], c=phi[:, 0], cmap="twilight", s=12)
    axes[0].set_title(f"Joint-1 circle (recovered)\ncoloured by joint 1  (R^2={r1:.2f})"); plt.colorbar(s0, ax=axes[0])
    s1 = axes[1].scatter(proj2[:, 0], proj2[:, 1], c=phi[:, 1], cmap="twilight", s=12)
    axes[1].set_title(f"Joint-2 circle (recovered)\ncoloured by joint 2  (R^2={r2:.2f})"); plt.colorbar(s1, ax=axes[1])
    # joint-1 ring coloured by joint 2: should be uniformly mixed if the circles are independent
    s2 = axes[2].scatter(proj1[:, 0], proj1[:, 1], c=phi[:, 1], cmap="twilight", s=12)
    axes[2].set_title("Joint-1 circle coloured by joint 2\n(mixed => joints independent)"); plt.colorbar(s2, ax=axes[2])
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlabel("recovered cos"); ax.set_ylabel("recovered sin")
    fig.suptitle(f"2-joint manifold as S^1 x S^1 (two circles), k={args.k}", fontsize=11)
    fig.tight_layout()
    suffix = "_mlpg" if args.nonlinear_g else ""
    out = Path(revision_fig_dir) / f"torus_two_rings{suffix}"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
