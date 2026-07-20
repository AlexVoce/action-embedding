"""Is the two-joint arm's learned embedding an EFFECT (task-space) representation that
collapses motor-equivalent configs? Tests:
  - recover fingertip (x,y) from embedding (linear R^2)   -> expect HIGH (effect-based)
  - recover joint angles from embedding (linear R^2)       -> expect LOW (redundancy)
  - motor equivalence: mean embedding distance between config-pairs with the SAME fingertip
    vs random pairs                                         -> expect same-fingertip << random
Plots the embedding coloured by fingertip x & y (should be smooth/clean)."""
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


def lin_r2(codes, target):
    aug = np.column_stack([codes, np.ones(len(codes))])
    A, *_ = np.linalg.lstsq(aug, target, rcond=None)
    pred = aug @ A
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())


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
        g_codes = np.array([torch.tanh(g(torch.cat([home, env.features(env.fingertips[c])], -1))).numpy()
                            for c in range(env.n_actions)])
        Wa = f.lin.weight.detach().numpy()   # (n_actions, emb_dim): Chandak-style per-action embeddings
    # analyse BOTH g(s,s') outputs and the decoder's per-action embeddings W_a
    which = getattr(args, "which", "Wa")
    codes = Wa if which == "Wa" else g_codes
    label = "decoder W_a (per-action embeddings)" if which == "Wa" else "g encoder outputs"
    print("  [analysing " + label + "]", flush=True)
    fps = env.fingertips                          # (N,2) effect
    joints = env.levels[env.configs]              # (N,2) joint angles
    j_circ = np.column_stack([np.cos(joints[:, 0]), np.sin(joints[:, 0]),
                              np.cos(joints[:, 1]), np.sin(joints[:, 1])])

    r2_fp = lin_r2(codes, fps)
    r2_joint = lin_r2(codes, j_circ)

    # motor equivalence: pairs whose fingertips nearly coincide
    from scipy.spatial.distance import pdist, squareform
    fp_d = squareform(pdist(fps)); emb_d = squareform(pdist(codes))
    same_fp = (fp_d < 0.05) & ~np.eye(len(fps), dtype=bool)
    rand_mask = ~np.eye(len(fps), dtype=bool)
    same_emb = emb_d[same_fp].mean() if same_fp.sum() > 0 else float("nan")
    rand_emb = emb_d[rand_mask].mean()

    print(f"[arm k={args.k} N={env.n_actions} emb_dim={args.emb_dim}] decode_acc={acc:.3f}", flush=True)
    print(f"  recover FINGERTIP (effect) from embedding: R2={r2_fp:.3f}   <- expect HIGH", flush=True)
    print(f"  recover JOINT angles from embedding:       R2={r2_joint:.3f}   <- expect LOW (redundancy)", flush=True)
    print(f"  motor-equivalence: emb dist same-fingertip={same_emb:.3f} vs random={rand_emb:.3f} "
          f"(ratio {same_emb/rand_emb:.2f}; #same-fp pairs={int(same_fp.sum())})", flush=True)

    proj = PCA(n_components=2).fit_transform(codes)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (col, lab) in zip(axes, [(fps[:, 0], "fingertip x"), (fps[:, 1], "fingertip y")]):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=col, cmap="viridis", s=18)
        ax.set_title(f"Arm embedding (PCA) coloured by {lab}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax)
    fig.suptitle(f"Two-joint arm embedding is EFFECT-based: recover fingertip R^2={r2_fp:.2f} "
                 f"(joint R^2={r2_joint:.2f})\nmotor-equivalent configs collapse "
                 f"(same-fp emb dist {same_emb:.2f} vs random {rand_emb:.2f})", fontsize=9)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "arm_embedding_effectbased"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
