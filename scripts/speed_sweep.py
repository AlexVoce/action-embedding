"""
Track B' (faithful learning-speed test): policy-representation efficiency.

Same centre-out ring task, binary reward, matched exploration curriculum. We compare,
as a function of the number of actions N:
  - embedding : ACLearningAgentWithEmbedding (2-D actor + frozen SL-pretrained f,g)
  - standard  : ACLearningAgent (N-way softmax actor over discrete actions)

Hypothesis: the embedding agent's episodes-to-criterion is ~flat in N (it always
optimises a 2-D policy), while the standard agent's grows with N (it must concentrate
policy mass and suppress N-1 competing logits). This operationalises Fig A.1's parameter
count as a *learning-speed* result under faithful binary reward -- no reward shaping.

Uses the paper's real agent classes and training curriculum (reward-based std / target-
radius annealing), NOT a rewrite, so the embedding side actually converges.
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
from core.agent import ACLearningAgentWithEmbedding, ACLearningAgent
from definitions import paper_model_path

BASE_CONFIG = {
    # paper-faithful lr (config.py): validated to converge the embedding policy.
    "actor_lr": 1e-5, "critic_lr": 1e-4, "fg_lr": 0.0, "w_decay_fg": 0,
    "embedding_dim": 2, "num_actions": 24, "grid_size": (20, 20), "reach_length": 1,
    "reach_angle": np.radians(135), "max_steps": 1, "max_episodes": 150000,
    "max_reward_policy_annealing": 0.4, "policy_std_max": 0.8,
    "max_reward_target_annealing": 0.5, "reward_radius_max": 0.5, "reward_radius_min": 0.2,
    "reward_for_hit": 1.0, "penalty_for_miss": -0.1, "log_to_wandb": False,
    "fourier_order": 3, "inv_temp": 0.8, "policy_noise": 0.008, "discount_factor": 0.99,
    "use_random_policy": False, "seed": 0, "policy_std": 0.2,
}


def reward_based_decay(reward, r_min, r_max, v_max, v_min):
    reward = max(min(reward, r_max), r_min)
    ratio = (reward - r_min) / (r_max - r_min)
    return v_max - ratio * (v_max - v_min)


def angular_error_deg(env, action_angle):
    """Circular distance (deg) between the taken action angle and the target reach angle."""
    diff = np.arctan2(np.sin(action_angle - env.reach_angle), np.cos(action_angle - env.reach_angle))
    return abs(np.degrees(diff))


def fg_path_for(N):
    return Path(paper_model_path) / f"action_embedding_model_seed_0_weight_decay_fg_0.0001_n_action_{N}_fourier_basis.pth"


def train_run(agent_kind, N, seed, cfg, max_episodes, crit_deg=10.0, crit_window=1000,
              crit_sustain=3000, log_every=500):
    """Train one agent; return learning curve + first episode criterion is sustained."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = ReachTask({**cfg, "num_actions": N})

    if agent_kind == "embedding":
        agent = ACLearningAgentWithEmbedding(env, {**cfg, "num_actions": N},
                                             fg_load_path=str(fg_path_for(N)),
                                             f_plastic=False, g_plastic=False)
    else:
        agent = ACLearningAgent(env, {**cfg, "num_actions": N})

    reward_hist, err_hist = [], []
    curve = []
    hit_ep = None
    sustained = 0
    window = 100
    for ep in range(max_episodes):
        env.reset()
        feats = env.get_features(env.current_xy)
        avg_reward = np.mean(reward_hist[-window:]) if len(reward_hist) >= window else -0.1

        # shared exploration + target-radius curriculum
        new_radius = reward_based_decay(avg_reward, -0.1, cfg["max_reward_target_annealing"],
                                        cfg["reward_radius_max"], cfg["reward_radius_min"])
        env.set_target_radius(new_radius)

        if agent_kind == "embedding":
            std = reward_based_decay(avg_reward, -0.1, cfg["max_reward_policy_annealing"],
                                     cfg["policy_std_max"], cfg["policy_std"])
            agent.set_policy_std(std)
            agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
            a_idx, emb, mean_emb, logstd = agent.select_action(feats, noise=True)
            action = env.actions[a_idx]
            nxt, reward, done = env.act(action)
            nfeats = env.get_features(nxt)
            agent.update(feats, a_idx, emb, nfeats, reward, done)
        else:
            # match exploration schedule: anneal softmax temperature by reward
            agent.update_temperature(max(0.0, avg_reward) * 10, min_temp=0.3, max_temp=3.0)
            agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
            a_idx, log_prob = agent.select_action(feats, noise=True)
            action = env.actions[a_idx]
            nxt, reward, done = env.act(action)
            nfeats = env.get_features(nxt)
            agent.update(feats, a_idx, log_prob, nfeats, reward, done)

        reward_hist.append(reward)
        err_hist.append(angular_error_deg(env, action))

        if len(err_hist) >= crit_window:
            mean_err = np.mean(err_hist[-crit_window:])
            if mean_err < crit_deg:
                sustained += 1
            else:
                sustained = 0
            if hit_ep is None and sustained >= crit_sustain:
                hit_ep = ep
        if ep % log_every == 0:
            curve.append({"ep": ep, "reward": float(np.mean(reward_hist[-window:])),
                          "err_deg": float(np.mean(err_hist[-window:]))})
        # early stop once criterion firmly reached (saves compute; hit_ep already recorded)
        if hit_ep is not None and ep > hit_ep + 2000:
            break

    return {"hit_ep": hit_ep,
            "final_err_deg": float(np.mean(err_hist[-window:])),
            "final_reward": float(np.mean(reward_hist[-window:])),
            "n_episodes_run": len(reward_hist),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_actions", type=int, nargs="+", default=[24, 96, 384])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--agents", type=str, nargs="+", default=["embedding", "standard"])
    ap.add_argument("--max_episodes", type=int, default=150000)
    ap.add_argument("--out", type=str, default="figures/paper/speed_sweep.json")
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for N in args.n_actions:
        if "embedding" in args.agents and not fg_path_for(N).exists():
            print(f"[skip N={N}] missing embedding file {fg_path_for(N).name}", flush=True)
            continue
        for seed in args.seeds:
            for kind in args.agents:
                res = train_run(kind, N, seed, BASE_CONFIG, args.max_episodes)
                row = {"N": N, "seed": seed, "agent": kind, **{k: v for k, v in res.items() if k != "curve"}}
                results.append({**row, "curve": res["curve"]})
                print(f"[N={N:<4} s{seed} {kind:<10}] hit_ep={str(res['hit_ep']):<7} "
                      f"final_err={res['final_err_deg']:.1f} n_run={res['n_episodes_run']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rows to {out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
