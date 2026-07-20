"""
Track B: where the action-embedding advantage actually lives -- action DIMENSIONALITY.

The centre-out ring is intrinsically 1-D: the rewarding set is a constant fraction of
the circle regardless of the number of actions N, so a plain bandit finds it in an
N-independent number of episodes (see scripts/baseline_pilot.py). No method can show an
N-scaling learning-speed advantage there.

The advantage appears when the action space is intrinsically d-dimensional: with a
k-per-dimension grid there are N = k**d actions and the rewarding region is a
*vanishing* fraction ~ (w)**d of the space. Unstructured RL then needs ~ (1/w)**d
episodes (exponential in d); an agent that explores a fixed d-D embedding learned by
supervised learning stays polynomial. This is the minimal, shallow-linear toy version of
R2's "hundreds of joints and muscles" concern.

We sweep d = 1,2,3 (fixed k) and compare episodes-to-criterion for:
  - standard   : plain actor-critic over N = k**d discrete actions
  - embedding  : SL-pretrained d-D action embedding + AC in d-D
  - random_emb : same architecture, embedding left at random init (control)
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------------------
# Environment: d-dimensional grid reach
# --------------------------------------------------------------------------------------
class MultiDimReach:
    """Reach in [0,1]^d. Action = one cell of a k-per-dim displacement grid (N = k**d).

    Actions are structured: adjacent grid cells produce adjacent endpoints, so nearby
    actions have similar consequences -- the property that makes a low-D embedding useful.
    """

    def __init__(self, d, k, order=3, reward_radius=None, reward_for_hit=1.0,
                 penalty_for_miss=-0.1, step=0.35):
        self.d = d
        self.k = k
        self.n_actions = k ** d
        self.order = order
        self.reward_value = reward_for_hit
        self.penalty_for_miss = penalty_for_miss
        self.start = np.full(d, 0.5)

        # per-dim displacement levels in [-step, +step]
        self.levels = np.linspace(-step, step, k)
        # enumerate all grid actions as displacement vectors
        self.action_vecs = np.array(list(itertools.product(self.levels, repeat=d)))  # (N, d)

        # reward tolerance: default one grid spacing per dimension (so ~1 correct level/dim)
        spacing = (2 * step) / (k - 1) if k > 1 else 2 * step
        self.reward_radius = reward_radius if reward_radius is not None else 0.5 * spacing

        # target = a fixed action's endpoint (so a unique optimum exists on the grid)
        self.target_action = self.n_actions // 2 + 1  # arbitrary interior cell
        self.target = self.start + self.action_vecs[self.target_action]

        self.n_features = d * 2 * order
        self.current = self.start.copy()

    def reset(self):
        self.current = self.start.copy()
        return self.current

    def get_features(self, xy):
        feats = []
        for i in range(self.d):
            for f in range(1, self.order + 1):
                phase = np.pi * f * xy[i]
                feats.append(np.cos(phase))
                feats.append(np.sin(phase))
        return torch.tensor(feats, dtype=torch.float32)

    def endpoint(self, action_idx):
        return self.start + self.action_vecs[action_idx]

    def act(self, action_idx):
        nxt = self.endpoint(action_idx)
        d = np.linalg.norm(nxt - self.target)
        reward = self.reward_value if d <= self.reward_radius else self.penalty_for_miss
        self.current = nxt
        return nxt, reward, True

    def optimal_reward(self):
        return self.reward_value

    def rewarding_fraction(self):
        n_good = sum(np.linalg.norm(self.endpoint(a) - self.target) <= self.reward_radius
                     for a in range(self.n_actions))
        return n_good / self.n_actions


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------
class Encoder(nn.Module):
    """g: (phi(s), phi(s')) -> d-D embedding."""

    def __init__(self, state_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(2 * state_dim, emb_dim)
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / math.sqrt(2 * state_dim))
        nn.init.zeros_(self.linear.bias)

    def forward(self, s, ns):
        return torch.tanh(self.linear(torch.cat([s, ns], dim=-1)))


class Decoder(nn.Module):
    """f: d-D embedding -> action logits."""

    def __init__(self, emb_dim, n_actions):
        super().__init__()
        self.linear = nn.Linear(emb_dim, n_actions)

    def forward(self, e):
        return self.linear(e)


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.linear = nn.Linear(state_dim, n_actions)
        nn.init.normal_(self.linear.weight, 0.0, 1e-2)
        nn.init.zeros_(self.linear.bias)

    def forward(self, s):
        return self.linear(s)


class EmbActor(nn.Module):
    def __init__(self, state_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(state_dim, emb_dim)
        nn.init.normal_(self.linear.weight, 0.0, 1e-2)
        nn.init.zeros_(self.linear.bias)

    def forward(self, s):
        return torch.tanh(self.linear(s))


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 1)

    def forward(self, s):
        return self.fc(s)


# --------------------------------------------------------------------------------------
# SL embedding pretraining (mirrors scripts/embedding_learning.py)
# --------------------------------------------------------------------------------------
def train_embedding(env, seed, steps, lr=0.01, temp=0.2):
    torch.manual_seed(seed)
    g = Encoder(env.n_features, env.d)
    f = Decoder(env.d, env.n_actions)
    opt = optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr,
                      weight_decay=1e-4, betas=(0.95, 0.999))
    loss_fn = nn.NLLLoss()
    rng = np.random.RandomState(42)
    env.reset()
    cfeat = env.get_features(env.start)
    correct = []
    for it in range(steps):
        opt.zero_grad()
        idx = rng.randint(env.n_actions)
        nxt = env.endpoint(idx)
        nfeat = env.get_features(nxt)
        emb = g(cfeat, nfeat)
        pred = f(emb)
        loss = loss_fn(torch.log_softmax(pred / temp, dim=0).unsqueeze(0), torch.tensor([idx]))
        loss.backward()
        opt.step()
        correct.append(int(torch.argmax(pred).item() == idx))
        if len(correct) > 3000:
            correct.pop(0)
    return g, f, float(np.mean(correct))


# --------------------------------------------------------------------------------------
# RL training
# --------------------------------------------------------------------------------------
def reward_decay(reward, r_min, r_max, v_max, v_min):
    reward = max(min(reward, r_max), r_min)
    ratio = (reward - r_min) / (r_max - r_min)
    return v_max - ratio * (v_max - v_min)


def run_standard(env, seed, max_episodes, actor_lr, gamma=0.99, window=200):
    torch.manual_seed(seed); np.random.seed(seed)
    actor = DiscreteActor(env.n_features, env.n_actions)
    critic = Critic(env.n_features)
    a_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    c_opt = optim.Adam(critic.parameters(), lr=actor_lr * 5)
    opt = env.optimal_reward()
    thresh = 0.9 * opt
    rewards, curve, hit = [], [], None
    for ep in range(max_episodes):
        env.reset()
        s = env.get_features(env.start)
        avg = np.mean(rewards[-window:]) if len(rewards) >= window else -0.1
        temp = 0.3 + 2.7 * math.exp(-0.1 * max(0.0, avg) * 10)  # anneal exploration by reward
        a_opt.zero_grad(); c_opt.zero_grad()
        logits = actor(s)
        probs = F.softmax(logits / temp, dim=-1)
        probs = probs + 0.008; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        nxt, r, done = env.act(a)
        value = critic(s)
        adv = torch.tensor(r) - value
        (-torch.log(probs[a]) * adv.detach()).backward()
        (adv.pow(2)).backward()
        a_opt.step(); c_opt.step()
        rewards.append(r)
        roll = np.mean(rewards[-window:])
        curve.append(roll)
        if hit is None and len(rewards) >= window and roll >= thresh:
            hit = ep
    return {"hit_ep": hit, "final_reward": float(np.mean(rewards[-window:])), "opt": float(opt)}


def run_embedding(env, g, f, seed, max_episodes, actor_lr, std_max=0.8, std_min=0.2,
                  inv_temp=0.8, window=200):
    torch.manual_seed(seed); np.random.seed(seed)
    actor = EmbActor(env.n_features, env.d)
    critic = Critic(env.n_features)
    a_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    c_opt = optim.Adam(critic.parameters(), lr=actor_lr * 5)
    opt = env.optimal_reward()
    thresh = 0.9 * opt
    rewards, curve, hit = [], [], None
    for ep in range(max_episodes):
        env.reset()
        s = env.get_features(env.start)
        avg = np.mean(rewards[-window:]) if len(rewards) >= window else -0.1
        std = reward_decay(avg, -0.1, 0.4, std_max, std_min)
        a_opt.zero_grad(); c_opt.zero_grad()
        mean = actor(s)
        emb = torch.tanh(torch.normal(mean, torch.tensor(std)))
        logits = f(emb)
        probs = F.softmax(logits / inv_temp, dim=-1)
        probs = probs + 0.008; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        nxt, r, done = env.act(a)
        value = critic(s)
        adv = torch.tensor(r) - value
        # policy gradient on the tanh-squashed Gaussian mean
        logstd = torch.log(torch.tensor(std))
        pre = 0.5 * (torch.log1p(emb.clamp(-0.999, 0.999)) - torch.log1p(-emb.clamp(-0.999, 0.999)))
        var = torch.exp(logstd) ** 2
        logp = (-0.5 * ((pre - mean) ** 2 / var + 2 * logstd + math.log(2 * math.pi))).sum()
        logp = logp - (2 * (math.log(2) - pre - F.softplus(-2 * pre))).sum()
        (-logp * adv.detach()).backward()
        (adv.pow(2)).backward()
        a_opt.step(); c_opt.step()
        rewards.append(r)
        roll = np.mean(rewards[-window:])
        curve.append(roll)
        if hit is None and len(rewards) >= window and roll >= thresh:
            hit = ep
    return {"hit_ep": hit, "final_reward": float(np.mean(rewards[-window:])), "opt": float(opt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_episodes", type=int, default=40000)
    ap.add_argument("--emb_steps", type=int, default=80000)
    ap.add_argument("--std_actor_lr", type=float, default=5e-3)
    ap.add_argument("--emb_actor_lr", type=float, default=5e-4)
    ap.add_argument("--out", type=str, default="figures/paper/multidim_scaling.json")
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for d in args.dims:
        env0 = MultiDimReach(d, args.k)
        N = env0.n_actions
        frac = env0.rewarding_fraction()
        g, f, emb_acc = train_embedding(MultiDimReach(d, args.k), seed=0, steps=args.emb_steps)
        torch.manual_seed(777)
        er = MultiDimReach(d, args.k)
        gr = Encoder(er.n_features, d); fr = Decoder(d, N)
        print(f"[d={d}] N={N} rewarding_fraction={frac:.4g} emb_SL_acc={emb_acc:.3f}", flush=True)
        for seed in args.seeds:
            env = MultiDimReach(d, args.k)
            r_emb = run_embedding(env, g, f, seed, args.max_episodes, args.emb_actor_lr)
            env2 = MultiDimReach(d, args.k)
            r_rnd = run_embedding(env2, gr, fr, seed, args.max_episodes, args.emb_actor_lr)
            env3 = MultiDimReach(d, args.k)
            r_std = run_standard(env3, seed, args.max_episodes, args.std_actor_lr)
            for name, res in [("embedding", r_emb), ("random_emb", r_rnd), ("standard", r_std)]:
                row = {"d": d, "N": N, "k": args.k, "seed": seed, "agent": name,
                       "rewarding_fraction": frac, "emb_acc": emb_acc, **res}
                results.append(row)
                print(f"[d={d}|N={N}|seed={seed}|{name:<11}] hit_ep={str(res['hit_ep']):<7} "
                      f"final={res['final_reward']:.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rows to {out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
