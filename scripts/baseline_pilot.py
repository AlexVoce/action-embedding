"""
A1 baseline pilot.

Question: on the centre-out ring task, does the RL sample-complexity of a plain
policy-gradient baseline actually grow with the number of actions N? And does the
action-embedding agent beat it?

Hypothesis (to be tested): NO. The ring is intrinsically 1-D, so the rewarding set
is a constant fraction of the circle regardless of N -> both agents converge in an
N-independent number of episodes -> parity. This motivates moving to a multi-D
action task where the advantage should appear.

Compares three agents under both binary and graded reward, across N:
  - standard   : plain actor-critic over N discrete actions (ACLearningAgent)
  - embedding  : SL-pretrained action embedding + AC in 2-D (ACLearningAgentWithEmbedding)
  - random_emb : same architecture but g,f left at random init (control)

Metric: episodes to reach a rolling-average-reward threshold; also final reward.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from core.continuous_env import ReachTask
from core.agent import (
    ACLearningAgent,
    ACLearningAgentWithEmbedding,
    ActionEmbeddingPredictor,
    ActionMapping,
)

BASE_CONFIG = {
    # NOTE: paper uses actor_lr=1e-5 over 200k episodes. For the pilot we use matched,
    # higher lrs so both agents converge in ~25k episodes (fair: same schedule family).
    "actor_lr": 5e-4,
    "critic_lr": 2.5e-3,
    "fg_lr": 0.0,
    "w_decay_fg": 0,
    "embedding_dim": 2,
    "num_actions": 24,
    "grid_size": (20, 20),
    "reach_length": 1,
    "reach_angle": np.radians(135),
    "max_steps": 1,
    "max_episodes": 20000,
    "max_reward_policy_annealing": 0.4,
    "policy_std_max": 0.8,
    "max_reward_target_annealing": 0.5,
    "reward_radius_max": 0.5,
    "reward_radius_min": 0.2,
    "reward_for_hit": 1.0,
    "penalty_for_miss": -0.1,
    "log_to_wandb": False,
    "fourier_order": 3,
    "inv_temp": 0.8,
    "policy_noise": 0.008,
    "discount_factor": 0.99,
    "use_random_policy": False,
    "log_interval": 10000,
    "seed": 0,
    "policy_std": 0.2,
}


class PilotReachTask(ReachTask):
    """ReachTask with an optional graded (dense) reward."""

    def __init__(self, config, reward_mode="binary", graded_sigma=0.5):
        super().__init__(config)
        self.reward_mode = reward_mode
        self.graded_sigma = graded_sigma

    def get_reward(self, next_xy):
        if self.reward_mode == "binary":
            return super().get_reward(next_xy)
        # graded: smooth bump peaking (=reward_for_hit) at the target
        nx, ny = next_xy
        tx, ty = self.target_xy
        d = math.hypot(nx - tx, ny - ty)
        return float(self.reward_value * math.exp(-(d * d) / (2 * self.graded_sigma ** 2)))

    def in_terminal_state(self):
        # single-step task: always terminal after one action
        return True


def reward_decay(reward, r_min, r_max, v_max, v_min):
    reward = max(min(reward, r_max), r_min)
    ratio = (reward - r_min) / (r_max - r_min)
    return v_max - ratio * (v_max - v_min)


def optimal_reward(env):
    """Best achievable single-step reward (target sits on an action angle)."""
    best = -1e9
    cx, cy = env.start_xy
    for a in env.actions:
        nx, ny = env.move_in_direction(cx, cy, a)
        env_next = (nx, ny)
        best = max(best, env.get_reward(env_next))
    return best


def train_embedding(config, n_actions, seed, steps, lr=0.01, temp=0.2):
    """Compact reproduction of scripts/embedding_learning.py (SL of action from transition)."""
    cfg = {**config, "num_actions": n_actions}
    env = PilotReachTask(cfg, reward_mode="binary")
    torch.manual_seed(seed)
    np.random.seed(seed)
    sd = env.n_features
    g = ActionEmbeddingPredictor(sd, 2, lecun_init=True, lecun_scale=1.0)
    f = ActionMapping(2, n_actions)
    optg = torch.optim.AdamW(g.parameters(), lr=lr, weight_decay=1e-4, betas=(0.95, 0.999))
    optf = torch.optim.AdamW(f.parameters(), lr=lr, weight_decay=1e-4, betas=(0.95, 0.999))
    loss_fn = torch.nn.NLLLoss()
    rng = np.random.RandomState(42)

    env.reset()
    centre = np.array(env.current_xy, dtype=float)
    cfeat = env.get_features(tuple(centre))
    final_acc = 0.0
    correct_hist = []
    for it in range(steps):
        optf.zero_grad()
        optg.zero_grad()
        idx = rng.randint(n_actions)
        a = env.actions[idx]
        env.current_xy = tuple(centre)
        nxy, _, _ = env.act(a)
        nfeat = env.get_features(nxy)
        emb = g(cfeat, nfeat)
        pred = f(emb)
        loss = loss_fn(torch.log_softmax(pred / temp, dim=0).unsqueeze(0), torch.tensor([idx]))
        loss.backward()
        optf.step()
        optg.step()
        correct_hist.append(int(torch.argmax(pred).item() == idx))
        if len(correct_hist) > 2000:
            correct_hist.pop(0)
    final_acc = float(np.mean(correct_hist))
    return g, f, final_acc


def make_embedding_agent(env, config, g, f):
    agent = ACLearningAgentWithEmbedding(
        env, config, fg_load_path=None, full_model_load_path=None,
        f_plastic=False, g_plastic=False,
    )
    agent.g = g
    agent.f = f
    return agent


def run_rl_embedding(env, config, agent, max_episodes, window=100):
    opt = optimal_reward(env)
    thresh = 0.9 * opt
    reward_hist = []
    curve = []
    hit_ep = None
    for ep in range(max_episodes):
        env.reset()
        feats = env.get_features(env.current_xy)
        avg_reward = np.mean(reward_hist[-window:]) if len(reward_hist) >= window else -0.1
        std = reward_decay(avg_reward, -0.1, config["max_reward_policy_annealing"],
                           config["policy_std_max"], config["policy_std"])
        agent.set_policy_std(std)
        agent.actor_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, noise=True)
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, emb, nfeats, reward, done)
        reward_hist.append(reward)
        roll = np.mean(reward_hist[-window:])
        curve.append(roll)
        if hit_ep is None and len(reward_hist) >= window and roll >= thresh:
            hit_ep = ep
    return {
        "hit_ep": hit_ep,
        "final_reward": float(np.mean(reward_hist[-window:])),
        "opt": float(opt),
        "curve": [float(c) for c in curve[::50]],
    }


def run_rl_standard(env, config, max_episodes, actor_lr, window=100):
    opt = optimal_reward(env)
    thresh = 0.9 * opt
    cfg = {**config, "actor_lr": actor_lr, "critic_lr": actor_lr * 5}
    agent = ACLearningAgent(env, cfg)
    reward_hist = []
    curve = []
    hit_ep = None
    for ep in range(max_episodes):
        env.reset()
        feats = env.get_features(env.current_xy)
        avg_reward = np.mean(reward_hist[-window:]) if len(reward_hist) >= window else -0.1
        # reward-based temperature annealing (fair exploration schedule)
        agent.update_temperature(max(0.0, avg_reward) * 10, min_temp=0.3, max_temp=3.0)
        a_idx, log_prob = agent.select_action(feats, noise=True)
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, log_prob, nfeats, reward, done)
        reward_hist.append(reward)
        roll = np.mean(reward_hist[-window:])
        curve.append(roll)
        if hit_ep is None and len(reward_hist) >= window and roll >= thresh:
            hit_ep = ep
    return {
        "hit_ep": hit_ep,
        "final_reward": float(np.mean(reward_hist[-window:])),
        "opt": float(opt),
        "curve": [float(c) for c in curve[::50]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_actions", type=int, nargs="+", default=[24, 96, 384])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reward_modes", type=str, nargs="+", default=["binary", "graded"])
    ap.add_argument("--max_episodes", type=int, default=20000)
    ap.add_argument("--emb_steps", type=int, default=40000)
    ap.add_argument("--std_actor_lrs", type=float, nargs="+", default=[1e-3])
    ap.add_argument("--out", type=str, default="figures/paper/baseline_pilot_results.json")
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for N in args.n_actions:
        # train embeddings once per N (RL seed varies below); report SL accuracy
        g, f, emb_acc = train_embedding(BASE_CONFIG, N, seed=0, steps=args.emb_steps)
        gr, fr, _ = None, None, None
        # random-embedding control uses fresh random init
        torch.manual_seed(999)
        sd = PilotReachTask({**BASE_CONFIG, "num_actions": N}).n_features
        gr = ActionEmbeddingPredictor(sd, 2, lecun_init=True, lecun_scale=1.0)
        fr = ActionMapping(2, N)
        print(f"[N={N}] embedding SL accuracy={emb_acc:.3f}", flush=True)

        for reward_mode in args.reward_modes:
            for seed in args.seeds:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                cfg = {**BASE_CONFIG, "num_actions": N}

                # embedding agent
                env = PilotReachTask(cfg, reward_mode=reward_mode)
                ea = make_embedding_agent(env, cfg, g, f)
                r_emb = run_rl_embedding(env, cfg, ea, args.max_episodes)

                # random-embedding control
                env2 = PilotReachTask(cfg, reward_mode=reward_mode)
                ra = make_embedding_agent(env2, cfg, gr, fr)
                r_rnd = run_rl_embedding(env2, cfg, ra, args.max_episodes)

                # standard baseline (best over lr grid)
                best_std = None
                for lr in args.std_actor_lrs:
                    env3 = PilotReachTask(cfg, reward_mode=reward_mode)
                    r = run_rl_standard(env3, cfg, args.max_episodes, actor_lr=lr)
                    if best_std is None or (r["final_reward"] > best_std["final_reward"]):
                        best_std = r
                        best_std["actor_lr"] = lr

                for agent_name, res in [("embedding", r_emb), ("random_emb", r_rnd), ("standard", best_std)]:
                    row = {"N": N, "reward_mode": reward_mode, "seed": seed,
                           "agent": agent_name, "emb_acc": emb_acc, **res}
                    results.append(row)
                    print(f"[N={N}|{reward_mode}|seed={seed}|{agent_name}] "
                          f"hit_ep={res['hit_ep']} final={res['final_reward']:.3f} opt={res['opt']:.3f}",
                          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rows to {out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
