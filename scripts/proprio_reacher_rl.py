"""
Full RL on the proprioceptive two-joint reacher, using the clean action-torus embedding.

SL (inverse model, regression): g(joint_feat, next_joint_feat) -> emb ; f: emb -> (cos/sin
per joint) continuous command. Gives a clean joint torus (recover-joint R^2 ~ 1.0).

RL: reach a sampled fingertip target. Embedding agent: actor(target_fp) -> emb ; decode emb
-> nearest joint config -> FK -> reward if fingertip within radius. Standard agent:
actor(target_fp) -> logits over k^2 configs. Compare success + episodes-to-criterion as k
(hence #actions) grows. Motor equivalence is handled at the task level (many configs reach
a target); the embedding stays a clean torus.
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

from scripts.two_joint_arm import TwoJointArm
from scripts.proprio_reacher import joint_feats, mlp


def train_embedding(env, seed, emb_dim=4, steps=200000, lr=1e-3, hidden=128, emb_state="joint"):
    """emb_state: 'joint' (proprioceptive action torus) or 'fingertip' (task/effect space).
    Regression target stays the joint command (inverse model) either way."""
    torch.manual_seed(seed); rng = np.random.RandomState(42)
    feat = (lambda c: joint_feats(env, c)) if emb_state == "joint" else (lambda c: env.features(env.fk(c)))
    g = mlp(2 * 12, emb_dim, hidden); f = nn.Linear(emb_dim, 4)
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=lr, weight_decay=1e-4)
    home = feat(env.home)
    for it in range(steps):
        opt.zero_grad()
        a = rng.randint(env.n_actions); t = env.levels[env.configs[a]]
        target = torch.tensor([math.cos(t[0]), math.sin(t[0]), math.cos(t[1]), math.sin(t[1])], dtype=torch.float32)
        emb = torch.tanh(g(torch.cat([home, feat(env.configs[a])], -1)))
        (((f(emb) - target) ** 2).mean()).backward(); opt.step()
    with torch.no_grad():
        embs = np.array([torch.tanh(g(torch.cat([home, feat(env.configs[c])], -1))).numpy()
                         for c in range(env.n_actions)])
    return g, f, embs


def reward_decay(r, rmin, rmax, vmax, vmin):
    r = max(min(r, rmax), rmin); return vmax - (r - rmin) / (rmax - rmin) * (vmax - vmin)


class Critic(nn.Module):
    def __init__(self, sd):
        super().__init__(); self.fc = nn.Linear(sd, 1)

    def forward(self, s):
        return self.fc(s)


def success_rate(pick, env):
    hits = 0
    for idx in range(len(env.eval_targets)):
        tgt = env.sample_target(idx)                 # fixed continuous reachable target
        a = pick(env.target_features())
        hits += env.hit(a, tgt)
    return hits / len(env.eval_targets)


def run(kind, env, seed, episodes, embs=None, emb_dim=4, hidden=128, actor_lr=1e-3,
        crit=0.8, eval_every=5000, temp_max=1.5, temp_min=0.3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sd = env.n_features
    critic = Critic(sd)
    if kind == "embedding":
        emb_t = torch.tensor(embs, dtype=torch.float32)   # (N, emb_dim) config embeddings on the torus
        actor = mlp(sd, emb_dim, hidden)                  # target_fp -> desired torus point
        params = actor.parameters()

        def pick(s, explore=None):
            e = torch.tanh(actor(s))
            if explore is not None:
                e = torch.tanh(e + explore)
            # nearest config on the torus (decode); redundant configs are equivalent
            d = ((emb_t - e) ** 2).sum(-1)
            logits = -d
            return logits
    else:
        actor = mlp(sd, env.n_actions, hidden)
        params = actor.parameters()

        def pick(s, explore=None):
            return actor(s)

    a_opt = torch.optim.Adam(params, lr=actor_lr); c_opt = torch.optim.Adam(critic.parameters(), lr=actor_lr * 5)
    curve = []; hit_ep = None; sustained = 0
    for ep in range(episodes):
        env.sample_target(); s = env.target_features()
        temp = temp_min + (temp_max - temp_min) * (1 - ep / episodes)   # gentle episode-annealed exploration
        a_opt.zero_grad(); c_opt.zero_grad()
        logits = pick(s)
        probs = F.softmax(logits / temp, -1); probs = probs + 0.008; probs = probs / probs.sum()
        a = torch.multinomial(probs, 1).item()
        dist = float(np.linalg.norm(env.fingertips[a] - env.target))     # dense, distance-shaped reward
        r = 1.0 if dist <= env.reward_radius else -0.2 * dist
        v = critic(s); adv = torch.tensor(float(r)) - v
        (-torch.log(probs[a]) * adv.detach()).backward(); adv.pow(2).backward()
        a_opt.step(); c_opt.step()
        if ep % eval_every == 0:
            sr = success_rate(lambda ss: torch.argmax(pick(ss)).item(), env)
            curve.append({"ep": ep, "success": sr})
            if sr >= crit:
                sustained += 1
                if hit_ep is None and sustained >= 2:
                    hit_ep = ep
            else:
                sustained = 0
    return {"hit_ep": hit_ep, "final_success": success_rate(lambda ss: torch.argmax(pick(ss)).item(), env),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--emb_dim", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--episodes", type=int, default=200000)
    ap.add_argument("--emb_steps", type=int, default=200000)
    ap.add_argument("--reward_radius", type=float, default=0.25)
    # fingertip (consequence) embedding: motor-equivalent configs collapse -> reaching is learnable/scalable.
    # (joint-state gives the proprioceptive TORUS, which scatters IK solutions and breaks the reaching decode.)
    ap.add_argument("--emb_state", choices=["joint", "fingertip"], default="fingertip")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--actor_lr", type=float, default=1e-3)
    # which agents to run (lets us pick the best lr PER agent in one figure: embedding@3e-4, standard@1e-4)
    ap.add_argument("--agents", nargs="+", choices=["embedding", "standard"], default=["embedding", "standard"])
    ap.add_argument("--out", type=str, default="figures/paper/proprio_reacher_rl_fixed.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time(); results = []
    for k in args.ks:
        env0 = TwoJointArm(k, reward_radius=args.reward_radius); N = env0.n_actions
        for seed in args.seeds:
            res_by_kind = {}
            if "embedding" in args.agents:
                # per-seed embedding (standing fix): retrain per seed so one bad embedding-init can't dent a k
                g, f, embs = train_embedding(TwoJointArm(k, reward_radius=args.reward_radius), seed,
                                             emb_dim=args.emb_dim, steps=args.emb_steps, emb_state=args.emb_state)
                res_by_kind["embedding"] = run("embedding", TwoJointArm(k, reward_radius=args.reward_radius), seed,
                                               args.episodes, embs=embs, emb_dim=args.emb_dim,
                                               eval_every=args.eval_every, actor_lr=args.actor_lr)
            if "standard" in args.agents:
                res_by_kind["standard"] = run("standard", TwoJointArm(k, reward_radius=args.reward_radius), seed,
                                              args.episodes, eval_every=args.eval_every, actor_lr=args.actor_lr)
            for kind, res in res_by_kind.items():
                results.append({"k": k, "N": N, "seed": seed, "agent": kind, "hit_ep": res["hit_ep"],
                                "final_success": res["final_success"], "curve": res["curve"]})
                print(f"[k={k} N={N} s{seed} {kind:<9}] hit_ep={str(res['hit_ep']):<8} success={res['final_success']:.2f}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
