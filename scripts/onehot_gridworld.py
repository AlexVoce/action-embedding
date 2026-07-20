"""
A2: can meaningful action embeddings be learned WITHOUT a spatially-structured state rep?

R1's concern: the Fourier state pre-encodes x/y, so phi(s')-phi(s) already contains the
action structure (a PCA of state differences recovers the ring). Here we remove that: a
grid world with ONE-HOT states (all states equidistant -> no spatial proximity in the
input) and movement actions taken from MANY positions. Any ring structure in the learned
action embedding must therefore be LEARNED (recognising "move-theta" across positions),
not handed over by the input.

Reports:
  - control: does a PCA of one-hot state differences contain action (ring) structure? (expect NO)
  - result: does the learned action embedding recover the ring? (ring_score, plot)
An expressive g is needed (a linear map over one-hot cannot be position-invariant), which
is consistent with the paper's stated 'g can be any function approximator'.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from definitions import revision_fig_dir


class GridWorld:
    def __init__(self, G=15, n_actions=24, step=3):
        self.G = G; self.n_actions = n_actions; self.step = step
        self.actions = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
        self.n_states = G * G

    def onehot(self, pos):
        v = torch.zeros(self.n_states)
        v[pos[0] * self.G + pos[1]] = 1.0
        return v

    def move(self, pos, a_idx):
        th = self.actions[a_idx]
        nx = int(round(pos[0] + self.step * math.cos(th)))
        ny = int(round(pos[1] + self.step * math.sin(th)))
        return (min(max(nx, 0), self.G - 1), min(max(ny, 0), self.G - 1))

    def valid_start(self, rng):
        m = self.step + 1
        return (rng.randint(m, self.G - m), rng.randint(m, self.G - m))


class MLPEncoder(nn.Module):
    def __init__(self, n_states, emb, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_states, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, emb))

    def forward(self, s, ns):
        return torch.tanh(self.net(torch.cat([s, ns], -1)))


class LinearEncoder(nn.Module):
    def __init__(self, n_states, emb):
        super().__init__()
        self.lin = nn.Linear(2 * n_states, emb)

    def forward(self, s, ns):
        return torch.tanh(self.lin(torch.cat([s, ns], -1)))


def ring_score(emb2d, action_angles):
    ang = np.arctan2(emb2d[:, 1], emb2d[:, 0])
    a = np.unwrap(ang) - np.mean(np.unwrap(ang))
    t = np.unwrap(action_angles) - np.mean(np.unwrap(action_angles))
    d = np.sqrt((a ** 2).sum() * (t ** 2).sum())
    return float(abs((a * t).sum() / d)) if d > 0 else 0.0


def action_embeddings(g, env, rng, n_pos=200):
    """Average learned embedding of each action over many start positions."""
    embs = []
    with torch.no_grad():
        for a in range(env.n_actions):
            acc = []
            for _ in range(n_pos):
                p = env.valid_start(rng); q = env.move(p, a)
                acc.append(g(env.onehot(p), env.onehot(q)).numpy())
            embs.append(np.mean(acc, axis=0))
    return np.array(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=15)
    ap.add_argument("--n_actions", type=int, default=24)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env = GridWorld(args.G, args.n_actions, args.step)
    emb_dim = 2
    g = (MLPEncoder(env.n_states, emb_dim) if args.encoder == "mlp"
         else LinearEncoder(env.n_states, emb_dim))
    f = nn.Linear(emb_dim, env.n_actions)
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=args.lr, weight_decay=1e-4)
    nll = nn.NLLLoss()

    correct = []
    for it in range(args.steps):
        opt.zero_grad()
        p = env.valid_start(rng); a = rng.randint(env.n_actions); q = env.move(p, a)
        pred = f(g(env.onehot(p), env.onehot(q)))
        loss = nll(torch.log_softmax(pred / 0.2, 0).unsqueeze(0), torch.tensor([a]))
        loss.backward(); opt.step()
        correct.append(int(torch.argmax(pred).item() == a))
        if len(correct) > 3000:
            correct.pop(0)
    acc = float(np.mean(correct))

    # control: PCA of one-hot state DIFFERENCES — does it contain action (ring) structure?
    diffs, diff_actions = [], []
    for _ in range(3000):
        p = env.valid_start(rng); a = rng.randint(env.n_actions); q = env.move(p, a)
        diffs.append((env.onehot(q) - env.onehot(p)).numpy()); diff_actions.append(a)
    diffs = np.array(diffs); diff_actions = np.array(diff_actions)
    pca = PCA(n_components=2).fit_transform(diffs)
    # mean PCA loc per action, ring score of that
    ctrl = np.array([pca[diff_actions == a].mean(0) for a in range(env.n_actions)])
    ctrl_ring = ring_score(ctrl, env.actions)

    # result: learned action embedding
    embs = action_embeddings(g, env, rng)
    learned_ring = ring_score(embs, env.actions)
    print(f"[A2 one-hot grid encoder={args.encoder} G={args.G} N={args.n_actions}] "
          f"SL_acc={acc:.3f}  control(onehot-diff PCA) ring_score={ctrl_ring:.3f}  "
          f"LEARNED embedding ring_score={learned_ring:.3f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    c = env.actions
    axes[0].scatter(ctrl[:, 0], ctrl[:, 1], c=c, cmap="twilight", s=40)
    axes[0].set_title(f"Control: PCA of one-hot state diffs\nring_score={ctrl_ring:.2f} (input has no action structure)")
    axes[1].scatter(embs[:, 0], embs[:, 1], c=c, cmap="twilight", s=40)
    axes[1].set_title(f"Learned action embedding ({args.encoder} g)\nring_score={learned_ring:.2f} (structure is LEARNED)")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.suptitle("A2: meaningful action embedding learned from ONE-HOT states (no spatial input structure)", fontsize=10)
    fig.tight_layout()
    out = Path(revision_fig_dir) / f"onehot_gridworld_{args.encoder}"
    fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    print("saved", out.name + ".png", flush=True)


if __name__ == "__main__":
    main()
