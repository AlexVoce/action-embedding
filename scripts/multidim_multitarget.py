"""
Speed demonstration, strongest / most reach-relevant version: multi-joint reach.

d-dimensional reach ("d joints"): action = d-D displacement discretised to k per axis
(N = k**d structured actions); target = any reachable d-D point; the actor observes the
Fourier code of the target. Multi-target, so the actor learns a nontrivial map and the
parameter-count advantage bites:
  - standard  : phi(target) -> Linear -> k**d logits         (params ~ state_dim * k**d)
  - embedding : phi(target) -> Linear -> d -> tanh -> fixed f -> k**d  (params ~ state_dim * d)

SL embedding (g,f) is pretrained reward-free on transitions, exactly as in the paper,
generalised to d dimensions. We sweep d = 1,2,3 and measure episodes-to-criterion
(greedy success rate >= 0.9). Prediction: standard's cost grows ~exponentially in d,
embedding stays ~flat -> a demonstrated speedup that scales with action dimensionality.
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


class MultiDimTargetReach:
    def __init__(self, d, k, order=3, step=0.35, reward_radius=None):
        self.d, self.k, self.order = d, k, order
        self.n_actions = k ** d
        self.center = np.full(d, 0.5)
        self.step = step
        self.levels = np.linspace(-step, step, k)
        self.action_vecs = np.array(list(itertools.product(self.levels, repeat=d)))  # (N,d)
        spacing = (2 * step) / (k - 1) if k > 1 else 2 * step
        self.reward_radius = reward_radius if reward_radius is not None else 0.5 * spacing
        self.n_features = d * 2 * order
        self.target_idx = 0
        self.target = self.center + self.action_vecs[0]

    def sample_target(self, idx=None):
        self.target_idx = np.random.randint(self.n_actions) if idx is None else idx
        self.target = self.center + self.action_vecs[self.target_idx]
        return self.target_idx

    def features(self, pos):
        feats = []
        for i in range(self.d):
            for f in range(1, self.order + 1):
                ph = np.pi * f * pos[i]
                feats.append(np.cos(ph)); feats.append(np.sin(ph))
        return torch.tensor(feats, dtype=torch.float32)

    def target_features(self):
        return self.features(self.target)

    def endpoint(self, a):
        return self.center + self.action_vecs[a]

    def reward_of(self, a):
        return 1.0 if np.linalg.norm(self.endpoint(a) - self.target) <= self.reward_radius else -0.1


class Encoder(nn.Module):
    def __init__(self, sd, d):
        super().__init__()
        self.lin = nn.Linear(2 * sd, d)
        nn.init.normal_(self.lin.weight, 0.0, 1.0 / math.sqrt(2 * sd)); nn.init.zeros_(self.lin.bias)

    def forward(self, s, ns):
        return torch.tanh(self.lin(torch.cat([s, ns], -1)))


class Decoder(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.lin = nn.Linear(d, n)

    def forward(self, e):
        return self.lin(e)


class EmbActor(nn.Module):
    def __init__(self, sd, d, n, decoder):
        super().__init__()
        self.enc = nn.Linear(sd, d)
        nn.init.normal_(self.enc.weight, 0.0, 1e-2); nn.init.zeros_(self.enc.bias)
        self.dec = decoder
        for p in self.dec.parameters():
            p.requires_grad_(False)

    def bottleneck(self, s):
        return torch.tanh(self.enc(s))

    def forward(self, s):
        return self.dec(self.bottleneck(s))


class StdActor(nn.Module):
    def __init__(self, sd, n):
        super().__init__()
        self.lin = nn.Linear(sd, n)
        nn.init.normal_(self.lin.weight, 0.0, 1e-2); nn.init.zeros_(self.lin.bias)

    def forward(self, s):
        return self.lin(s)


class Critic(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.fc = nn.Linear(sd, 1)

    def forward(self, s):
        return self.fc(s)


def train_sl_embedding(env, seed, steps=120000, lr=0.01, temp=0.2):
    torch.manual_seed(seed)
    g = Encoder(env.n_features, env.d); f = Decoder(env.d, env.n_actions)
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr,
                            weight_decay=1e-4, betas=(0.95, 0.999))
    loss_fn = nn.NLLLoss(); rng = np.random.RandomState(42)
    cfeat = env.features(env.center)
    corr = []
    for it in range(steps):
        opt.zero_grad()
        a = rng.randint(env.n_actions)
        nfeat = env.features(env.endpoint(a))
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
            hits += (np.linalg.norm(env.endpoint(a) - env.target) <= env.reward_radius)
    return hits / env.n_actions


def run(kind, env, seed, episodes, decoder=None, actor_lr=1e-3, critic_lr=5e-3,
        crit=0.9, eval_every=5000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sd = env.n_features
    critic = Critic(sd)
    if kind == "embedding":
        actor = EmbActor(sd, env.d, env.n_actions, decoder)
    else:
        actor = StdActor(sd, env.n_actions)
    a_opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    rh = []; window = 300; curve = []; hit_ep = None; sustained = 0
    for ep in range(episodes):
        env.sample_target()
        s = env.target_features()
        avg = np.mean(rh[-window:]) if len(rh) >= window else -0.1
        temp = reward_decay(avg, -0.1, 0.4, 3.0, 0.5)
        a_opt.zero_grad(); c_opt.zero_grad()
        logits = actor(s)
        probs = F.softmax(logits / temp, -1); probs = probs + 0.008; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        r = env.reward_of(a)
        v = critic(s); adv = torch.tensor(float(r)) - v
        (-torch.log(probs[a]) * adv.detach()).backward()
        adv.pow(2).backward()
        a_opt.step(); c_opt.step()
        rh.append(r)
        if ep % eval_every == 0:
            sr = success_rate(actor, env)
            curve.append({"ep": ep, "success": sr, "avg_reward": float(np.mean(rh[-window:]))})
            if sr >= crit:
                sustained += 1
                if hit_ep is None and sustained >= 2:
                    hit_ep = ep
            else:
                sustained = 0
    final_sr = success_rate(actor, env)
    return {"hit_ep": hit_ep, "final_success": final_sr, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=150000)
    ap.add_argument("--emb_steps", type=int, default=120000)
    ap.add_argument("--out", type=str, default="figures/paper/multidim_multitarget.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time(); results = []
    for d in args.dims:
        env0 = MultiDimTargetReach(d, args.k)
        N = env0.n_actions
        g, f, acc = train_sl_embedding(MultiDimTargetReach(d, args.k), seed=0, steps=args.emb_steps)
        print(f"[d={d} N={N}] SL decode acc={acc:.3f}", flush=True)
        for seed in args.seeds:
            r_emb = run("embedding", MultiDimTargetReach(d, args.k), seed, args.episodes, decoder=f)
            r_std = run("standard", MultiDimTargetReach(d, args.k), seed, args.episodes)
            for kind, res in [("embedding", r_emb), ("standard", r_std)]:
                results.append({"d": d, "N": N, "k": args.k, "seed": seed, "agent": kind,
                                "emb_acc": acc, "hit_ep": res["hit_ep"],
                                "final_success": res["final_success"], "curve": res["curve"]})
                print(f"[d={d} N={N} s{seed} {kind:<9}] hit_ep={str(res['hit_ep']):<8} "
                      f"final_success={res['final_success']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rows to {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
