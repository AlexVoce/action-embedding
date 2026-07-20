"""Show the learned action manifold for the two-joint arm (Reacher) is a torus (two rings)
via dim reduction. Trains the SL embedding, embeds every joint config, and plots PCA
coloured by joint 1, joint 2, and fingertip position. Motor-equivalent configs (same
fingertip) should collapse together."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.two_joint_arm import TwoJointArm, train_sl_embedding
from definitions import revision_fig_dir


def recover_ring(codes, angle):
    target = np.column_stack([np.cos(angle), np.sin(angle)])
    aug = np.column_stack([codes, np.ones(len(codes))])
    A, *_ = np.linalg.lstsq(aug, target, rcond=None)
    proj = aug @ A
    r2 = 1 - ((target - proj) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()
    return proj, float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--emb_dim", type=int, default=4)
    ap.add_argument("--emb_steps", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)

    env = TwoJointArm(args.k)
    g, f, acc = train_sl_embedding(TwoJointArm(args.k), args.seed, args.emb_dim, steps=args.emb_steps)
    home = env.features(env.home_fp)
    with torch.no_grad():
        codes = np.array([torch.tanh(g(torch.cat([home, env.features(env.fingertips[c])], -1))).numpy()
                          for c in range(env.n_actions)])
    j1 = env.levels[env.configs[:, 0]]; j2 = env.levels[env.configs[:, 1]]
    proj = PCA(n_components=3).fit_transform(codes)
    _, r1 = recover_ring(codes, j1); _, r2 = recover_ring(codes, j2)
    print(f"[arm k={args.k} N={env.n_actions} emb_dim={args.emb_dim}] decode_acc={acc:.3f} "
          f"ring1_R2={r1:.3f} ring2_R2={r2:.3f}", flush=True)

    fig = plt.figure(figsize=(13, 4.2))
    for i, (col, lab) in enumerate([(j1, "joint 1"), (j2, "joint 2"), (env.fingertips[:, 0], "fingertip x")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        p = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=col, cmap="twilight", s=12)
        ax.set_title(f"Learned arm manifold, coloured by {lab}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle(f"Two-joint arm: learned action manifold (torus)  |  ring R^2 = {r1:.2f}, {r2:.2f}", fontsize=10)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "arm_manifold"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
