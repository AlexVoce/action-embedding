"""
Multi-joint reach (corrected): d circular joints, N = k**d configurations.

Fix vs the earlier grid version: each joint is a CIRCULAR DOF, so the action manifold is
a d-torus and the embedding uses 2 dims per joint -> embedding_dim = 2*d. A linear
decoder then computes logit_c = sum_j cos(phi_j^c - phi_j^target), whose argmax is the
nearest torus point -- so linear decoding works (unlike the grid, where dot-product
argmax != nearest cell). This keeps the shallow-linear architecture.

Actor params: standard ~ state_dim * k**d (exponential in #joints); embedding ~ state_dim * 2d
(linear). Sweep d and show the embedding's success/speed advantage grows with #joints.
"""
import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiJointReach:
    def __init__(self, d, k, order=3, tol_levels=1):
        self.d, self.k, self.order = d, k, order
        self.n_actions = k ** d
        self.configs = np.array(list(itertools.product(range(k), repeat=d)))  # (N,d) level indices
        self.center = np.zeros(d, dtype=int)
        self.tol_levels = tol_levels
        self.n_features = d * 2 * order
        self.target_idx = 0

    def angles(self, config):
        return 2 * np.pi * np.asarray(config) / self.k

    def features(self, config):
        phi = self.angles(config)
        feats = []
        for j in range(self.d):
            for f in range(1, self.order + 1):
                feats.append(np.cos(f * phi[j])); feats.append(np.sin(f * phi[j]))
        return torch.tensor(feats, dtype=torch.float32)

    def sample_target(self, idx=None):
        self.target_idx = np.random.randint(self.n_actions) if idx is None else idx
        return self.target_idx

    def target_features(self):
        return self.features(self.configs[self.target_idx])

    def center_features(self):
        return self.features(self.center)

    def hit(self, action_idx, target_idx=None):
        t = self.configs[self.target_idx if target_idx is None else target_idx]
        a = self.configs[action_idx]
        ld = np.minimum(np.abs(a - t), self.k - np.abs(a - t))  # circular level distance per joint
        return bool(np.all(ld <= self.tol_levels))

    def reward_of(self, action_idx):
        return 1.0 if self.hit(action_idx) else -0.1


class Encoder(nn.Module):
    def __init__(self, sd, emb):
        super().__init__()
        self.lin = nn.Linear(2 * sd, emb)
        nn.init.normal_(self.lin.weight, 0.0, 1.0 / math.sqrt(2 * sd)); nn.init.zeros_(self.lin.bias)

    def forward(self, s, ns):
        return torch.tanh(self.lin(torch.cat([s, ns], -1)))


class MLPEncoder(nn.Module):
    """Expressive (nonlinear) encoder g; f stays linear (Chandak-style division of labour)."""
    def __init__(self, sd, emb, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * sd, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, emb),
        )

    def forward(self, s, ns):
        return torch.tanh(self.net(torch.cat([s, ns], -1)))


class Decoder(nn.Module):
    def __init__(self, emb, n):
        super().__init__()
        self.lin = nn.Linear(emb, n)

    def forward(self, e):
        return self.lin(e)


def _head(sd, out, hidden):
    """Linear head (hidden=0) or MLP head (hidden>0). MLP = Chandak-style nonlinear actor."""
    if hidden and hidden > 0:
        return nn.Sequential(nn.Linear(sd, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                             nn.Linear(hidden, out))
    lin = nn.Linear(sd, out)
    nn.init.normal_(lin.weight, 0.0, 1e-2); nn.init.zeros_(lin.bias)
    return lin


class EmbActor(nn.Module):
    def __init__(self, sd, emb, n, decoder, hidden=0):
        super().__init__()
        self.enc = _head(sd, emb, hidden)
        self.dec = decoder
        for p in self.dec.parameters():
            p.requires_grad_(False)

    def bottleneck(self, s):
        return torch.tanh(self.enc(s))

    def forward(self, s):
        return self.dec(self.bottleneck(s))


class StdActor(nn.Module):
    def __init__(self, sd, n, hidden=0):
        super().__init__()
        self.head = _head(sd, n, hidden)

    def forward(self, s):
        return self.head(s)


class Critic(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.fc = nn.Linear(sd, 1)

    def forward(self, s):
        return self.fc(s)


def train_sl_embedding(env, seed, emb_dim, steps=120000, lr=0.01, temp=0.2, nonlinear=False,
                       hidden=128, weight_decay=1e-4, betas=(0.9, 0.999)):
    torch.manual_seed(seed)
    g = MLPEncoder(env.n_features, emb_dim, hidden=hidden) if nonlinear else Encoder(env.n_features, emb_dim)
    f = Decoder(emb_dim, env.n_actions)
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr,
                            weight_decay=weight_decay, betas=betas)
    loss_fn = nn.NLLLoss(); rng = np.random.RandomState(42)
    cfeat = env.center_features()
    corr = []
    for it in range(steps):
        opt.zero_grad()
        a = rng.randint(env.n_actions)
        nfeat = env.features(env.configs[a])
        pred = f(g(cfeat, nfeat))
        loss = loss_fn(torch.log_softmax(pred / temp, 0).unsqueeze(0), torch.tensor([a]))
        loss.backward(); opt.step()
        corr.append(int(torch.argmax(pred).item() == a))
        if len(corr) > 3000:
            corr.pop(0)
    return g, f, float(np.mean(corr))


def reward_decay(r, rmin, rmax, vmax, vmin):
    r = max(min(r, rmax), rmin)
    return vmax - (r - rmin) / (rmax - rmin) * (vmax - vmin)


def success_rate(actor, env):
    hits = 0
    with torch.no_grad():
        for idx in range(env.n_actions):
            env.sample_target(idx)
            a = torch.argmax(actor(env.target_features())).item()
            hits += env.hit(a, idx)
    return hits / env.n_actions


def run(kind, env, seed, episodes, emb_dim, decoder=None, actor_lr=3e-3, critic_lr=1.5e-2,
        crit=0.9, eval_every=5000, actor_hidden=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sd = env.n_features
    critic = Critic(sd)
    actor = (EmbActor(sd, emb_dim, env.n_actions, decoder, hidden=actor_hidden) if kind == "embedding"
             else StdActor(sd, env.n_actions, hidden=actor_hidden))
    a_opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    rh = []; window = 300; curve = []; hit_ep = None; sustained = 0
    for ep in range(episodes):
        env.sample_target()
        s = env.target_features()
        avg = np.mean(rh[-window:]) if len(rh) >= window else -0.1
        temp = reward_decay(avg, -0.1, 0.4, 3.0, 0.5)
        a_opt.zero_grad(); c_opt.zero_grad()
        probs = F.softmax(actor(s) / temp, -1); probs = probs + 0.008; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        r = env.reward_of(a)
        v = critic(s); adv = torch.tensor(float(r)) - v
        (-torch.log(probs[a]) * adv.detach()).backward(); adv.pow(2).backward()
        a_opt.step(); c_opt.step(); rh.append(r)
        if ep % eval_every == 0:
            sr = success_rate(actor, env)
            curve.append({"ep": ep, "success": sr})
            if sr >= crit:
                sustained += 1
                if hit_ep is None and sustained >= 2:
                    hit_ep = ep
            else:
                sustained = 0
    return {"hit_ep": hit_ep, "final_success": success_rate(actor, env), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=150000)
    ap.add_argument("--emb_steps", type=int, default=150000)
    ap.add_argument("--actor_lr", type=float, default=3e-3)
    ap.add_argument("--actor_hidden", type=int, default=0, help="MLP actor hidden units (0=linear; Chandak uses a NN actor)")
    ap.add_argument("--nonlinear_g", action="store_true")
    ap.add_argument("--emb_lr", type=float, default=0.01)
    ap.add_argument("--emb_hidden", type=int, default=128)
    ap.add_argument("--tol_levels", type=int, default=1, help="reward tolerance per joint (0=exact -> hard)")
    ap.add_argument("--out", type=str, default="figures/paper/multijoint.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time(); results = []
    for d in args.dims:
        emb_dim = 2 * d  # 2 dims per circular joint
        mk = lambda: MultiJointReach(d, args.k, tol_levels=args.tol_levels)
        env0 = mk(); N = env0.n_actions
        g, f, acc = train_sl_embedding(mk(), 0, emb_dim, steps=args.emb_steps, lr=args.emb_lr,
                                       nonlinear=args.nonlinear_g, hidden=args.emb_hidden)
        print(f"[d={d} N={N} emb_dim={emb_dim} nl_g={args.nonlinear_g} tol={args.tol_levels}] "
              f"SL decode acc={acc:.3f}", flush=True)
        for seed in args.seeds:
            r_emb = run("embedding", mk(), seed, args.episodes, emb_dim,
                        decoder=f, actor_lr=args.actor_lr, actor_hidden=args.actor_hidden)
            r_std = run("standard", mk(), seed, args.episodes, emb_dim,
                        actor_lr=args.actor_lr, actor_hidden=args.actor_hidden)
            for kind, res in [("embedding", r_emb), ("standard", r_std)]:
                results.append({"d": d, "N": N, "emb_dim": emb_dim, "k": args.k, "seed": seed,
                                "agent": kind, "emb_acc": acc, "hit_ep": res["hit_ep"],
                                "final_success": res["final_success"], "curve": res["curve"]})
                print(f"[d={d} N={N} s{seed} {kind:<9}] hit_ep={str(res['hit_ep']):<8} "
                      f"success={res['final_success']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
