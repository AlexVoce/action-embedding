"""
Adaptation-mechanism comparison: WHY multiple systems.

Same visuomotor rotation applied to the SL-embedding base policy, adapted two ways:
  - sl : fast, error-driven — decoder f adapts via the self-supervised NLL update
         (cortico-cerebellar mechanism; actor/critic frozen).
  - rl : slow, reward-driven — the actor/critic adapt by policy gradient on reward
         (basal-ganglia mechanism; decoder f frozen).

Measures how fast each mechanism re-reaches the target under rotation (angular error of the
achieved endpoint vs target). Prediction (the paper's thesis): SL adaptation is much faster
than RL adaptation -> a rationale for a fast SL adaptation system alongside slow RL.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from scripts.track_a_control import BASE, base_model_path
from definitions import revision_fig_dir


def achieved_err_deg(action_rad, rotation_deg, target_deg):
    """Angular error (deg) of the achieved endpoint (action + rotation) vs the target."""
    true_ang = action_rad + math.radians(rotation_deg)
    d = math.atan2(math.sin(true_ang - math.radians(target_deg)),
                   math.cos(true_ang - math.radians(target_deg)))
    return abs(math.degrees(d))


def adapt(mode, seed, target_deg, rotation_deg, cfg, adapt_episodes, crit_deg=15.0,
          window=500, policy_std=0.25):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = base_model_path("sl", seed, target_deg, cfg["num_actions"])
    if not base.exists():
        raise FileNotFoundError(f"missing SL base model {base}")

    acfg = {**cfg, "reach_angle": np.radians(target_deg), "seed": seed, "inv_temp": 2.5,
            "rotation_angle": rotation_deg}
    if mode == "sl":
        acfg = {**acfg, "actor_lr": 0.0, "critic_lr": 0.0, "fg_lr": 1e-4}
        f_pl, a_pl, c_pl = True, False, False
    else:  # rl
        acfg = {**acfg, "actor_lr": 1e-4, "critic_lr": 5e-4, "fg_lr": 0.0}
        f_pl, a_pl, c_pl = False, True, True

    env = ReachTask(acfg, adaptation_rotation=np.radians(rotation_deg).item())
    agent = ACLearningAgentWithEmbedding(env, acfg, full_model_load_path=str(base),
                                         f_plastic=f_pl, g_plastic=False,
                                         actor_plastic=a_pl, critic_plastic=c_pl)
    agent.set_policy_std(policy_std)

    # target action's g-embedding (g is frozen -> fixed); f-decode at this point is the Fig-3 quantity
    tgt_idx = int(round(target_deg / 360.0 * env.n_actions)) % env.n_actions
    with torch.no_grad():
        e_T = torch.tensor(agent.get_action_embeddings_via_g()[tgt_idx], dtype=torch.float32)
    f0 = agent.f.linear.weight.detach().clone()

    errs, curve, recover_ep = [], [], None
    for ep in range(adapt_episodes):
        env.reset()
        feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        if f_pl:
            agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False,
                                                           import_policy_mean=False, policy_mean=None)
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, emb, nfeats, reward, done)
        # Fig 3E quantity: achieved error of the TAKEN action (its distribution swaps as it re-learns)
        errs.append(achieved_err_deg(action, rotation_deg, target_deg))
        if len(errs) >= window:
            m = np.mean(errs[-window:])
            if recover_ep is None and m < crit_deg:
                recover_ep = ep
        if ep % 2000 == 0:
            curve.append({"ep": ep,
                          "taken_err": float(np.mean(errs[-window:])) if len(errs) >= window else 90.0})
    return {"mode": mode, "seed": seed, "recover_ep": recover_ep,
            "final_err": float(np.mean(errs[-window:])), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target_deg", type=int, default=135)
    ap.add_argument("--rotation_deg", type=int, default=-30)
    ap.add_argument("--adapt_episodes", type=int, default=60000)
    ap.add_argument("--out", type=str, default="figures/paper/adapt_mechanism.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = []
    for mode in ["sl", "rl"]:
        for seed in args.seeds:
            r = adapt(mode, seed, args.target_deg, args.rotation_deg, {**BASE}, args.adapt_episodes)
            results.append(r)
            print(f"[adapt {mode} s{seed}] recover_ep={r['recover_ep']} final_err={r['final_err']:.1f}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    # summary
    for mode in ["sl", "rl"]:
        rs = [x["recover_ep"] for x in results if x["mode"] == mode and x["recover_ep"] is not None]
        print(f"  {mode}: recovered {len(rs)}/{len(args.seeds)} seeds; median recover_ep="
              f"{int(np.median(rs)) if rs else 'n/a'}", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
