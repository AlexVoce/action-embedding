"""
Does local generalization survive in higher dimensions? (d=2 multi-joint torus)

The paper's 1-DOF result (adaptation at one reach direction generalizes locally to
nearby directions) is the 1-D instance of a manifold-alignment principle. Here we test
whether the SAME mechanism gives local generalization on a 2-torus.

Uses the multi-joint SL model (real g,f), applies a "rotation" on joint 1 (commanded
level offset by rho), adapts f via the paper's self-supervised update at one target
config T, then maps the adaptation amount over the whole (theta1,theta2) torus.
Prediction: a local blob around T (decaying with torus distance) = generalization holds.
Output: a 2-D heatmap saved to revision_figures/.
"""
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.multijoint import MultiJointReach, Encoder, Decoder, train_sl_embedding
from definitions import revision_fig_dir


def perturbed_achieved(env, action_idx, rho_levels, joint=0):
    """Apply a rotation of rho_levels on `joint`: achieved config = action with that joint shifted."""
    cfg = env.configs[action_idx].copy()
    cfg[joint] = (cfg[joint] + rho_levels) % env.k
    return cfg


def config_index(env, cfg):
    diffs = np.all(env.configs == cfg[None, :], axis=1)
    return int(np.argmax(diffs))


def greedy_action_for_config(g, f, env, cfg):
    cfeat = env.center_features()
    nfeat = env.features(cfg)
    with torch.no_grad():
        return torch.argmax(f(g(cfeat, nfeat))).item()


def adapt_f(g, f, env, target_cfg, rho_levels, joint, episodes=30000, lr=5e-3, temp=0.3):
    """Self-supervised inverse update of f under the joint-1 perturbation, at one target."""
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    nll = nn.NLLLoss()
    cfeat = env.center_features()
    for ep in range(episodes):
        # policy: pick action whose (perturbed) outcome is closest to target — but we let it
        # explore around the current greedy for the target's embedding
        z = g(cfeat, env.features(target_cfg)).detach()
        logits = f(z)
        probs = F.softmax(logits / temp, -1); probs = probs + 0.01; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        achieved = perturbed_achieved(env, a, rho_levels, joint)   # where action a actually lands
        # SL inverse update: embedding of the ACHIEVED config should decode to the taken action
        opt.zero_grad()
        z_ach = g(cfeat, env.features(achieved))
        pred = f(z_ach)
        loss = nll(torch.log_softmax(pred / temp, 0).unsqueeze(0), torch.tensor([a]))
        loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--rho_levels", type=int, default=2)   # perturbation size on joint 1
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--emb_steps", type=int, default=200000)
    ap.add_argument("--adapt_episodes", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)

    d = 2
    env = MultiJointReach(d, args.k)
    g, f, acc = train_sl_embedding(MultiJointReach(d, args.k), args.seed, 2 * d, steps=args.emb_steps)
    print(f"[d=2 k={args.k}] SL decode acc={acc:.3f}", flush=True)

    # target config near torus centre
    target_cfg = np.array([args.k // 2, args.k // 2])

    # base greedy action per config (no adaptation)
    base = np.array([greedy_action_for_config(g, f, env, env.configs[c]) for c in range(env.n_actions)])

    f_ad = copy.deepcopy(f)
    adapt_f(g, f_ad, env, target_cfg, args.rho_levels, args.joint, episodes=args.adapt_episodes)
    adapted = np.array([greedy_action_for_config(g, f_ad, env, env.configs[c]) for c in range(env.n_actions)])

    # adaptation amount per config: circular shift of greedy action's joint-`joint` level toward
    # compensating the perturbation, normalised by rho
    amt = np.zeros(env.n_actions)
    for c in range(env.n_actions):
        a0 = env.configs[base[c]]; a1 = env.configs[adapted[c]]
        dl = ((a1[args.joint] - a0[args.joint] + env.k // 2) % env.k) - env.k // 2  # signed circular level shift
        amt[c] = -dl / args.rho_levels * 100.0   # 100% = fully compensates the perturbation

    # reshape onto the (theta1,theta2) grid and plot heatmap
    grid = amt.reshape(args.k, args.k)  # configs enumerated in row-major (joint0 outer, joint1 inner)
    T_idx = config_index(env, target_cfg)
    # locality: mean adaptation within vs outside a torus radius of 2 levels from T
    def torus_dist(c):
        a = env.configs[c]
        dl = np.minimum(np.abs(a - target_cfg), env.k - np.abs(a - target_cfg))
        return np.sqrt((dl ** 2).sum())
    dists = np.array([torus_dist(c) for c in range(env.n_actions)])
    loc = amt[dists <= 2].mean(); glob = amt[dists > 2].mean()
    print(f"adaptation at T={target_cfg.tolist()}: local(<=2)={loc:.1f}%  global(>2)={glob:.1f}%  "
          f"locality={loc-glob:.1f}", flush=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(grid, origin="lower", cmap="viridis",
                   extent=[0, 360, 0, 360], aspect="auto", vmin=0, vmax=max(100, grid.max()))
    ax.scatter([target_cfg[1] / env.k * 360], [target_cfg[0] / env.k * 360], marker="*",
               s=200, color="red", edgecolor="white", label="adapted target")
    ax.set(xlabel="Joint 2 angle (deg)", ylabel="Joint 1 angle (deg)",
           title=f"Adaptation generalization on 2-joint torus\n(perturbed joint {args.joint+1}; locality={loc-glob:.0f})")
    plt.colorbar(im, ax=ax, label="adaptation amount (%)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "multijoint_generalization_torus"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
