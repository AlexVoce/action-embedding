"""Visualise the action manifold the SL embedding learns for the multi-joint task.
Trains the d=2 (and d=3) embedding, gets the learned code for every joint config via g,
and plots it (PCA to 3D, coloured by each joint angle) to see whether it forms a clean
torus. Also reports PCA variance spectrum (a flat 2-torus lives in ~4 dims) and how well
the code recovers the ideal (cos/sin per joint) torus."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.multijoint import MultiJointReach, train_sl_embedding
from definitions import revision_fig_dir


def get_codes(g, env):
    cfeat = env.center_features()
    codes = []
    with torch.no_grad():
        for c in range(env.n_actions):
            codes.append(g(cfeat, env.features(env.configs[c])).numpy())
    return np.array(codes)


def ideal_recovery(codes, env):
    """R^2 of predicting learned code from ideal torus coords (cos/sin per joint)."""
    phi = env.angles(env.configs.T).T  # (N,d)
    ideal = np.concatenate([np.column_stack([np.cos(phi[:, j]), np.sin(phi[:, j])])
                            for j in range(env.d)], axis=1)  # (N, 2d)
    # least squares fit ideal -> codes
    A, *_ = np.linalg.lstsq(np.column_stack([ideal, np.ones(len(ideal))]), codes, rcond=None)
    pred = np.column_stack([ideal, np.ones(len(ideal))]) @ A
    ss_res = ((codes - pred) ** 2).sum(); ss_tot = ((codes - codes.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--emb_steps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)

    fig = plt.figure(figsize=(12, 7))
    for col, d in enumerate([2, 3]):
        env = MultiJointReach(d, args.k)
        g, f, acc = train_sl_embedding(MultiJointReach(d, args.k), args.seed, 2 * d, steps=args.emb_steps)
        codes = get_codes(g, env)
        pca = PCA(n_components=min(6, 2 * d))
        proj = pca.fit_transform(codes)
        evr = pca.explained_variance_ratio_
        r2 = ideal_recovery(codes, env)
        print(f"[d={d} k={args.k}] decode_acc={acc:.3f}  ideal_torus_R2={r2:.3f}  "
              f"PCA_evr={np.round(evr, 3).tolist()}", flush=True)

        phi = env.angles(env.configs.T).T
        for row, jcolor in enumerate([0, 1]):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            p = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2] if proj.shape[1] > 2 else np.zeros(len(proj)),
                           c=phi[:, jcolor], cmap="twilight", s=15)
            ax.set_title(f"d={d} (N={env.n_actions}, decode {acc:.2f})\nPCA of learned code, coloured by joint {jcolor+1}",
                         fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.suptitle("Learned action manifold (multi-joint): torus quality vs #joints", fontsize=11)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "multijoint_manifold"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
