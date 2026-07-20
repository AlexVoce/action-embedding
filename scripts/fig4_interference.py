"""
Fig 4 (dual-adaptation interference), SL-ring vs random embedding.

Two targets separated by `sep` degrees are adapted to OPPOSITE visuomotor rotations,
interleaved, sharing one decoder f (SL decoder-adaptation). We then measure each target's
adaptation amount with the Fig-3I quantity (calculate_generalization), and report the mean
of the two. Interference => adaptation amount drops when the targets are close (their f
updates overlap and the opposite rotations conflict). We compare the SL-learned ring vs a
random 2-D embedding to test whether the interference profile depends on the learned
structure (R2 #6 / Woolley 2007).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from adaptation.adaptation_generalization_test import calculate_generalization
from scripts.track_a_control import BASE, base_model_path


def adapt_amount_at_target(agent, base_agent, cfg, target_deg, rotation_deg):
    acfg = {**cfg, "reach_angle": np.radians(target_deg), "rotation_angle": rotation_deg,
            "seed": cfg["seed"]}
    agent.env.reach_angle = np.radians(target_deg)
    agent.env.set_visuomotor_rotation(np.radians(rotation_deg).item())
    df = calculate_generalization(agent, base_agent, acfg)
    d = df.set_index("angle from target")["adaptation amount"]
    near = [a for a in [-15, 0, 15] if a in d.index]
    return float(d.loc[near].mean()) if near else float(d.loc[0]) if 0 in d.index else np.nan


def run(condition, seed, sep, cfg, adapt_episodes=100000, fg_lr=1e-4, phase_prob=0.5):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = base_model_path(condition, seed, 135, cfg["num_actions"])
    t1, t2 = 135.0, (135.0 + sep) % 360
    r1, r2 = -30, 30
    acfg = {**cfg, "seed": seed, "actor_lr": 0.0, "critic_lr": 0.0, "fg_lr": fg_lr, "inv_temp": 2.5,
            "import_policy_mean": False, "policy_mean_to_import": None, "num_actions": cfg["num_actions"]}
    env1 = ReachTask({**acfg, "reach_angle": np.radians(t1)}, adaptation_rotation=np.radians(r1).item())
    agent = ACLearningAgentWithEmbedding(env1, {**acfg, "reach_angle": np.radians(t1)},
                                         full_model_load_path=str(base), f_plastic=True, g_plastic=False,
                                         actor_plastic=False, critic_plastic=False)
    base_agent = copy.deepcopy(agent)

    # embedding (policy mean) for target 2, from the neutral encoder
    embs = agent.get_action_embeddings_via_g()
    actions_deg = np.round(np.degrees(agent.env.actions))
    t2_idx = int(np.argmin(np.abs(((actions_deg - t2 + 180) % 360) - 180)))
    t2_mean = torch.tensor(embs[t2_idx])

    env2 = ReachTask({**acfg, "reach_angle": np.radians(t2)}, adaptation_rotation=np.radians(r2).item())

    for ep in range(adapt_episodes):
        phase1 = np.random.rand() < phase_prob
        env = env1 if phase1 else env2
        agent.env = env
        env.reset()
        feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad(); agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(
            feats, random_policy=False,
            import_policy_mean=(not phase1), policy_mean=(None if phase1 else t2_mean))
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, emb, nfeats, reward, done)

    a1 = adapt_amount_at_target(agent, base_agent, {**acfg}, t1, r1)
    a2 = adapt_amount_at_target(agent, base_agent, {**acfg}, t2, r2)
    return {"condition": condition, "seed": seed, "sep": sep, "adapt_t1": a1, "adapt_t2": a2,
            "adapt_mean": float(np.nanmean([a1, a2]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["sl", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seps", type=int, nargs="+", default=[30, 60, 90, 135, 180])
    ap.add_argument("--adapt_episodes", type=int, default=100000)
    ap.add_argument("--out", type=str, default="figures/paper/fig4_interference.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = []
    for condition in args.conditions:
        for sep in args.seps:
            for seed in args.seeds:
                r = run(condition, seed, sep, {**BASE, "seed": seed}, adapt_episodes=args.adapt_episodes)
                results.append(r)
                print(f"[fig4 {condition} sep={sep} s{seed}] adapt_mean={r['adapt_mean']:.1f} "
                      f"(t1={r['adapt_t1']:.1f} t2={r['adapt_t2']:.1f})", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
