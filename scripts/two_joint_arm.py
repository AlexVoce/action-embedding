"""
Realistic two-joint reach (Reacher-style), in the SL+RL action-embedding framework.

Two-link planar arm: joint angles (theta1, theta2) -> fingertip via forward kinematics.
Actions = discretised joint configs (k per joint, N=k^2). Reward is on the FINGERTIP
reaching a 2-D target (not exact joint match) -> tractable, and it naturally has motor
equivalence (different joint configs reach the same fingertip). State = the 2-D target.

Chandak-faithful architecture: nonlinear encoder g (MLP), LINEAR decoder f (dot-product +
softmax), and a nonlinear actor (MLP) over the embedding. Compares the embedding agent
against a standard actor-critic over the N discrete joint configs, as joint resolution k
(hence N=k^2) grows.
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


class TwoJointArm:
    def __init__(self, k, L1=1.0, L2=1.0, order=3, reward_radius=0.25):
        self.k = k; self.L1 = L1; self.L2 = L2; self.order = order
        self.reward_radius = reward_radius
        self.levels = np.linspace(0, 2 * np.pi, k, endpoint=False)
        self.configs = np.array(list(itertools.product(range(k), repeat=2)))  # (N,2) joint level idx
        self.n_actions = k * k
        self.home = np.array([0, 0])
        self.fingertips = np.array([self.fk(c) for c in self.configs])  # (N,2)
        self.home_fp = self.fk(self.home)
        self.n_features = 2 * 2 * order  # Fourier of 2-D target position
        self.target = self.fingertips[0]
        self.target_idx = 0

    def fk(self, config):
        t1 = self.levels[config[0]]; t2 = self.levels[config[1]]
        x = self.L1 * math.cos(t1) + self.L2 * math.cos(t1 + t2)
        y = self.L1 * math.sin(t1) + self.L2 * math.sin(t1 + t2)
        return np.array([x, y])

    def features(self, xy):
        # normalise task-space position to ~[-1,1] then Fourier
        p = np.asarray(xy) / (self.L1 + self.L2)
        feats = []
        for i in range(2):
            for f in range(1, self.order + 1):
                feats.append(math.cos(f * math.pi * p[i])); feats.append(math.sin(f * math.pi * p[i]))
        return torch.tensor(feats, dtype=torch.float32)

    def sample_target(self, idx=None):
        self.target_idx = np.random.randint(self.n_actions) if idx is None else idx
        self.target = self.fingertips[self.target_idx]
        return self.target_idx

    def target_features(self):
        return self.features(self.target)

    def hit(self, action_idx, target=None):
        t = self.target if target is None else target
        return bool(np.linalg.norm(self.fingertips[action_idx] - t) <= self.reward_radius)

    def reward_of(self, action_idx):
        return 1.0 if self.hit(action_idx) else -0.1

    def rewarding_fraction(self):
        # avg over targets of fraction of actions that reach within radius
        fr = []
        for ti in range(self.n_actions):
            t = self.fingertips[ti]
            fr.append(np.mean(np.linalg.norm(self.fingertips - t, axis=1) <= self.reward_radius))
        return float(np.mean(fr))


def mlp(inp, out, hidden):
    return nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class Decoder(nn.Module):           # LINEAR f (Chandak): dot-product + softmax
    def __init__(self, emb, n):
        super().__init__(); self.lin = nn.Linear(emb, n)

    def forward(self, e):
        return self.lin(e)


def train_sl_embedding(env, seed, emb_dim, steps=200000, lr=1e-3, temp=0.2, hidden=128):
    torch.manual_seed(seed)
    g = mlp(2 * env.n_features, emb_dim, hidden)                 # nonlinear encoder
    f = Decoder(emb_dim, env.n_actions)                         # linear decoder
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr, weight_decay=1e-4)
    loss_fn = nn.NLLLoss(); rng = np.random.RandomState(42)
    home_feat = env.features(env.home_fp)
    corr = []
    for it in range(steps):
        opt.zero_grad()
        a = rng.randint(env.n_actions)
        nfeat = env.features(env.fingertips[a])
        emb = torch.tanh(g(torch.cat([home_feat, nfeat], -1)))
        pred = f(emb)
        loss = loss_fn(torch.log_softmax(pred / temp, 0).unsqueeze(0), torch.tensor([a]))
        loss.backward(); opt.step()
        corr.append(int(torch.argmax(pred).item() == a))
        if len(corr) > 3000:
            corr.pop(0)
    return g, f, float(np.mean(corr))


class EmbActor(nn.Module):          # nonlinear actor -> embedding -> fixed linear decoder
    def __init__(self, sd, emb, decoder, hidden=128):
        super().__init__(); self.enc = mlp(sd, emb, hidden); self.dec = decoder
        for p in self.dec.parameters():
            p.requires_grad_(False)

    def forward(self, s):
        return self.dec(torch.tanh(self.enc(s)))


class StdActor(nn.Module):
    def __init__(self, sd, n, hidden=128):
        super().__init__(); self.net = mlp(sd, n, hidden)

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, sd):
        super().__init__(); self.fc = nn.Linear(sd, 1)

    def forward(self, s):
        return self.fc(s)


def reward_decay(r, rmin, rmax, vmax, vmin):
    r = max(min(r, rmax), rmin); return vmax - (r - rmin) / (rmax - rmin) * (vmax - vmin)


def success_rate(actor, env):
    hits = 0
    with torch.no_grad():
        for idx in range(env.n_actions):
            env.sample_target(idx)
            a = torch.argmax(actor(env.target_features())).item()
            hits += env.hit(a, env.fingertips[idx])
    return hits / env.n_actions


def run(kind, env, seed, episodes, emb_dim, decoder=None, actor_lr=1e-3, hidden=128,
        crit=0.9, eval_every=5000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sd = env.n_features
    critic = Critic(sd)
    actor = (EmbActor(sd, emb_dim, decoder, hidden) if kind == "embedding" else StdActor(sd, env.n_actions, hidden))
    a_opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=actor_lr * 5)
    rh = []; window = 300; curve = []; hit_ep = None; sustained = 0
    for ep in range(episodes):
        env.sample_target(); s = env.target_features()
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
            sr = success_rate(actor, env); curve.append({"ep": ep, "success": sr})
            if sr >= crit:
                sustained += 1
                if hit_ep is None and sustained >= 2:
                    hit_ep = ep
            else:
                sustained = 0
    return {"hit_ep": hit_ep, "final_success": success_rate(actor, env), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 24])   # N = k^2 = 64, 256, 576
    ap.add_argument("--emb_dim", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=120000)
    ap.add_argument("--emb_steps", type=int, default=200000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--reward_radius", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="figures/paper/two_joint_arm.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time(); results = []
    for k in args.ks:
        env0 = TwoJointArm(k, reward_radius=args.reward_radius); N = env0.n_actions
        frac = env0.rewarding_fraction()
        g, f, acc = train_sl_embedding(TwoJointArm(k, reward_radius=args.reward_radius), 0,
                                       args.emb_dim, steps=args.emb_steps, hidden=args.hidden)
        print(f"[k={k} N={N} emb_dim={args.emb_dim}] SL decode acc={acc:.3f} rewarding_frac={frac:.3g}", flush=True)
        for seed in args.seeds:
            r_emb = run("embedding", TwoJointArm(k, reward_radius=args.reward_radius), seed,
                        args.episodes, args.emb_dim, decoder=f, actor_lr=1e-3, hidden=args.hidden)
            r_std = run("standard", TwoJointArm(k, reward_radius=args.reward_radius), seed,
                        args.episodes, args.emb_dim, hidden=args.hidden)
            for kind, res in [("embedding", r_emb), ("standard", r_std)]:
                results.append({"k": k, "N": N, "emb_dim": args.emb_dim, "seed": seed, "agent": kind,
                                "emb_acc": acc, "rewarding_frac": frac, "hit_ep": res["hit_ep"],
                                "final_success": res["final_success"], "curve": res["curve"]})
                print(f"[k={k} N={N} s{seed} {kind:<9}] hit_ep={str(res['hit_ep']):<8} success={res['final_success']:.2f}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
