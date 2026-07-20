"""
Fix for the two-joint arm: proprioceptive (JOINT) state + REGRESS a continuous joint
command (keeps the inverse model; avoids kd discretisation).

Diagnosis being tested: the earlier failure was artificial redundancy from feeding the
encoder the FINGERTIP transition. With joint (proprioceptive) state the joint transition
uniquely determines the action, so the embedding should be a clean torus. Motor
equivalence then lives at the TASK level (many joint configs reach one fingertip),
handled by RL -- not a defect of the representation.

SL objective (regression, inverse-model): g(joint_feat, next_joint_feat) -> emb ->
regress the action's circular joint coords (cos/sin per joint). Reports how well the
embedding recovers joint angles (clean torus => high R^2), and compares against a
FINGERTIP-state encoder (should stay scrambled).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.two_joint_arm import TwoJointArm
from definitions import revision_fig_dir


def joint_feats(env, config, order=3):
    """Circular (proprioceptive) features of the joint configuration."""
    t = env.levels[np.asarray(config)]
    feats = []
    for j in range(2):
        for f in range(1, order + 1):
            feats.append(math.cos(f * t[j])); feats.append(math.sin(f * t[j]))
    return torch.tensor(feats, dtype=torch.float32)


def mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


def lin_r2(codes, target):
    aug = np.column_stack([codes, np.ones(len(codes))])
    A, *_ = np.linalg.lstsq(aug, target, rcond=None)
    pred = aug @ A
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())


def train(env, state_mode, seed, steps=200000, lr=1e-3, emb_dim=4, hidden=128):
    """state_mode: 'joint' (proprioceptive) or 'fingertip'. Regress circular joint command."""
    torch.manual_seed(seed); rng = np.random.RandomState(42)
    nfeat = 12 if state_mode == "joint" else env.n_features
    g = mlp(2 * nfeat, emb_dim, hidden)
    f = nn.Linear(emb_dim, 4)          # regress (cos t1, sin t1, cos t2, sin t2) -- continuous command
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr, weight_decay=1e-4)

    def feat(cfg):
        return joint_feats(env, cfg) if state_mode == "joint" else env.features(env.fk(cfg))
    home = feat(env.home)
    for it in range(steps):
        opt.zero_grad()
        a = rng.randint(env.n_actions)
        t = env.levels[env.configs[a]]
        target = torch.tensor([math.cos(t[0]), math.sin(t[0]), math.cos(t[1]), math.sin(t[1])], dtype=torch.float32)
        emb = torch.tanh(g(torch.cat([home, feat(env.configs[a])], -1)))
        loss = ((f(emb) - target) ** 2).mean()
        loss.backward(); opt.step()

    with torch.no_grad():
        codes = np.array([torch.tanh(g(torch.cat([home, feat(env.configs[c])], -1))).numpy()
                          for c in range(env.n_actions)])
    joints = env.levels[env.configs]
    j_circ = np.column_stack([np.cos(joints[:, 0]), np.sin(joints[:, 0]),
                              np.cos(joints[:, 1]), np.sin(joints[:, 1])])
    return codes, lin_r2(codes, j_circ), lin_r2(codes, env.fingertips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--emb_dim", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    env = TwoJointArm(args.k)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, mode in zip(axes, ["joint", "fingertip"]):
        codes, r2_joint, r2_fp = train(env, mode, args.seed, steps=args.steps, emb_dim=args.emb_dim)
        print(f"[{mode:9s} state] recover JOINT R2={r2_joint:.3f}  recover FINGERTIP R2={r2_fp:.3f}", flush=True)
        proj = PCA(n_components=2).fit_transform(codes)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=env.levels[env.configs[:, 0]], cmap="twilight", s=16)
        ax.set_title(f"{mode}-state encoder\nrecover joint R^2={r2_joint:.2f}, fingertip R^2={r2_fp:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); plt.colorbar(sc, ax=ax)
    fig.suptitle("Two-joint arm: proprioceptive (joint) state gives a clean action torus;\n"
                 "fingertip state does not (regressing continuous joint command)", fontsize=10)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "proprio_vs_fingertip_arm"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
