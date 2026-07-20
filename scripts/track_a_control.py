"""
Track A: is the SL-LEARNED embedding structure necessary for the phenomena?

Compares, using the paper's real agent + generalization machinery:
  - condition=sl      : SL-pretrained ring embedding (the paper's model) -> reference
  - condition=random  : random (untrained) frozen g,f of identical architecture ->
                        tests whether ANY 2-D bottleneck yields the local-generalization
                        phenomenon, or whether the SL-learned ring is required (R1's
                        "is the structure necessary" / R2 #6 "learned vs imposed a priori").

Two stages:
  --stage base   : train actor/critic on one target (frozen g,f), save a full model.
  --stage adapt  : load the base model, apply a visuomotor rotation, adapt f (self-
                   supervised, actor/critic frozen), then compute the Fig-3 generalization
                   curve (calculate_generalization) vs the pre-adaptation agent.

Naming is condition-tagged so nothing collides with the paper's own files.
"""
import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch

from core.continuous_env import ReachTask
from core.agent import ACLearningAgentWithEmbedding
from adaptation.adaptation_generalization_test import calculate_generalization
from definitions import paper_model_path, paper_fig_dir

# base-training curriculum (mirrors core/config.py + scripts/policy_learning.py)
BASE = {
    "actor_lr": 1e-5, "critic_lr": 1e-4, "fg_lr": 0.0, "w_decay_fg": 0,
    "embedding_dim": 2, "num_actions": 24, "grid_size": (20, 20), "reach_length": 1,
    "reach_angle": np.radians(135), "max_steps": 1, "max_episodes": 200000,
    "max_reward_policy_annealing": 0.4, "policy_std_max": 0.8,
    "max_reward_target_annealing": 0.5, "reward_radius_max": 0.5, "reward_radius_min": 0.2,
    "reward_for_hit": 1.0, "penalty_for_miss": -0.1, "log_to_wandb": False,
    "fourier_order": 3, "inv_temp": 0.8, "policy_noise": 0.008, "discount_factor": 0.99,
    "use_random_policy": False, "seed": 0, "policy_std": 0.2,
}


def sl_fg_path(N):
    return Path(paper_model_path) / f"action_embedding_model_seed_0_weight_decay_fg_0.0001_n_action_{N}_fourier_basis.pth"


def base_model_path(condition, seed, target_deg, N):
    return Path(paper_model_path) / f"trackA_base_{condition}_seed{seed}_target{target_deg}_nact{N}.pth"


def reward_based_decay(reward, r_min, r_max, v_max, v_min):
    reward = max(min(reward, r_max), r_min)
    ratio = (reward - r_min) / (r_max - r_min)
    return v_max - ratio * (v_max - v_min)


def build_base_agent(condition, cfg, env):
    """SL: load pretrained ring embedding. random: fresh random frozen g,f."""
    fg = str(sl_fg_path(cfg["num_actions"])) if condition == "sl" else None
    agent = ACLearningAgentWithEmbedding(env, cfg, fg_load_path=fg,
                                         f_plastic=False, g_plastic=False)
    return agent


def train_base(condition, seed, target_deg, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = {**cfg, "reach_angle": np.radians(target_deg), "seed": seed}
    env = ReachTask(cfg)
    agent = build_base_agent(condition, cfg, env)

    reward_history = []
    window = 100
    for ep in range(cfg["max_episodes"]):
        env.reset()
        feats = env.get_features(env.current_xy)
        avg = np.mean(reward_history[-window:]) if len(reward_history) >= window else -0.1
        agent.set_policy_std(reward_based_decay(avg, -0.1, cfg["max_reward_policy_annealing"],
                                                cfg["policy_std_max"], cfg["policy_std"]))
        env.set_target_radius(reward_based_decay(avg, -0.1, cfg["max_reward_target_annealing"],
                                                 cfg["reward_radius_max"], cfg["reward_radius_min"]))
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, noise=True)
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, emb, nfeats, reward, done)
        reward_history.append(reward)
        if ep % 20000 == 0:
            print(f"[base {condition} s{seed}] ep {ep} avg_reward={np.mean(reward_history[-window:]):.3f}", flush=True)

    out = base_model_path(condition, seed, target_deg, cfg["num_actions"])
    torch.save({
        "g_state_dict": agent.g.state_dict(), "f_state_dict": agent.f.state_dict(),
        "actor_state_dict": agent.actor.state_dict(), "critic_state_dict": agent.critic.state_dict(),
        "params": {"state_dim": env.n_features, "embedding_dim": 2, "n_actions": env.n_actions},
    }, out)
    print(f"[base {condition} s{seed}] saved {out.name} final_avg_reward={np.mean(reward_history[-window:]):.3f}", flush=True)


def _greedy_achieved_err(agent, env, target_deg, rotation_deg):
    """Greedy policy action (actor-mean -> f -> argmax) -> achieved endpoint error vs target (deg)."""
    import math
    feats = env.get_features(env._center if hasattr(env, "_center") else
                             (env.env_shape[0] / 2, env.env_shape[1] / 2))
    with torch.no_grad():
        a_rad = env.actions[torch.argmax(agent.f(torch.tanh(agent.actor(feats)))).item()]
    true = a_rad + math.radians(rotation_deg)
    d = math.atan2(math.sin(true - math.radians(target_deg)), math.cos(true - math.radians(target_deg)))
    return abs(math.degrees(d))


def train_adapt(condition, seed, target_deg, rotation_deg, cfg, adapt_episodes=100000,
                monitor_every=0, fg_lr=1e-4, adapt_mode="sl", use_gemb_policy=False,
                align_actor_to_gemb=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.plotting import find_angle_difference
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = base_model_path(condition, seed, target_deg, cfg["num_actions"])
    if not base.exists():
        raise FileNotFoundError(f"missing base model {base} -- run --stage base first")

    # adaptation config. adapt_mode: 'sl' = error-driven decoder f update (actor/critic frozen);
    #                                'rl' = reward-driven actor/critic update (decoder f frozen).
    if adapt_mode == "sl":
        lrs = {"actor_lr": 0.0, "critic_lr": 0.0, "fg_lr": fg_lr}
        f_pl, a_pl, c_pl = True, False, False
    else:
        lrs = {"actor_lr": 1e-4, "critic_lr": 5e-4, "fg_lr": 0.0}
        f_pl, a_pl, c_pl = False, True, True
    acfg = {**cfg, "reach_angle": np.radians(target_deg), "seed": seed, **lrs, "inv_temp": 2.5,
            "rotation_angle": rotation_deg, "angle_diff_criterion": 10.0,
            "import_policy_mean": False, "policy_mean_to_import": None,
            "save_figs_locally": False, "fig_dir": str(paper_fig_dir)}
    env = ReachTask(acfg, adaptation_rotation=np.radians(rotation_deg).item())
    agent = ACLearningAgentWithEmbedding(env, acfg, full_model_load_path=str(base),
                                         f_plastic=f_pl, g_plastic=False,
                                         actor_plastic=a_pl, critic_plastic=c_pl)
    base_agent = copy.deepcopy(agent)
    mon_dir = Path(paper_fig_dir).parent / "revision_figures" / f"sl_adapt_monitor_fglr{fg_lr}"
    if monitor_every:
        mon_dir.mkdir(parents=True, exist_ok=True)
    mon_curve = []
    relearn_curve = []  # Fig 3E: taken-action angular error per episode
    greedy_curve = []   # noise-free behavioural error: policy's preferred action decoded through f
    # optionally drive the policy from the target's g-embedding (the action-representation
    # space where f is adapted), instead of the actor's mean, so decoder rotation -> behaviour.
    gemb_t = None
    gemb_pre = None
    if use_gemb_policy or align_actor_to_gemb:
        gembs = agent.get_action_embeddings_via_g()
        t_idx = int(np.argmin(np.abs(((np.round(np.degrees(env.actions)) - target_deg + 180) % 360) - 180)))
        gemb_t = torch.tensor(gembs[t_idx], dtype=torch.float32)
        # pre-tanh version: select_action applies tanh, so importing atanh(g_emb) lands the policy
        # exactly ON the g-embedding (radius ~1, on the ring) where the decoder adaptation is strong,
        # rather than the tanh-squashed point inside the ring.
        gemb_pre = torch.atanh(torch.clamp(gemb_t, -0.999, 0.999))
        if align_actor_to_gemb:
            # the policy for a single target SHOULD point at that target's action representation
            # (its g-embedding); undertrained RL leaves the actor mean off it. Set the (constant)
            # actor output to the g-embedding so SL (f-plastic) and RL (actor-plastic) both adapt
            # from an aligned policy -> a fair mechanism comparison.
            with torch.no_grad():
                gc = torch.clamp(gemb_t, -0.999, 0.999)
                agent.actor.mean_head.weight.zero_()
                agent.actor.mean_head.bias.copy_(torch.atanh(gc))

    for ep in range(adapt_episodes):
        if monitor_every and ep % monitor_every == 0:
            gerr = _greedy_achieved_err(agent, env, target_deg, rotation_deg)
            mon_curve.append({"ep": ep, "greedy_err": gerr})
            try:
                agent.plot_f_output()
                plt.title(f"ep {ep}  greedy_err={gerr:.0f}deg  (fg_lr={fg_lr})")
                plt.savefig(mon_dir / f"heatmap_seed{seed}_ep{ep:06d}.png", dpi=110, bbox_inches="tight")
                plt.close("all")
            except Exception as e:
                print("heatmap fail:", e, flush=True)
            print(f"[monitor s{seed} fg_lr={fg_lr}] ep {ep} greedy_err={gerr:.1f}", flush=True)
        env.reset()
        feats = env.get_features(env.current_xy)
        agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
        if f_pl:
            agent.f_g_optimizer.zero_grad()
        a_idx, emb, mean_emb, logstd = agent.select_action(feats, random_policy=False,
                                                           import_policy_mean=use_gemb_policy, policy_mean=gemb_pre)
        action = env.actions[a_idx]
        nxt, reward, done = env.act(action)
        nfeats = env.get_features(nxt)
        agent.update(feats, a_idx, emb, nfeats, reward, done)
        # Fig 3E quantity: angular error of the taken action (paper's find_angle_difference)
        relearn_curve.append(abs(find_angle_difference(env, action)))
        # noise-free behavioural error: decode the policy's MEAN embedding through the current f
        # (removes the exploration noise floor so SL-vs-RL adaptation speed is directly comparable)
        with torch.no_grad():
            pm = gemb_pre if use_gemb_policy else mean_emb
            g_act = int(torch.argmax(agent.f(torch.tanh(pm))).item())
        greedy_curve.append(abs(find_angle_difference(env, env.actions[g_act])))

    df = calculate_generalization(agent, base_agent, acfg)
    tag = f"trackA_{condition}_seed{seed}_target{target_deg}_rot{rotation_deg}"
    out = Path(paper_fig_dir) / f"generalization_{tag}.csv"
    df.to_csv(out)
    # summarise local vs global generalization
    d = df[["rotation generalization", "angle from target"]].set_index("angle from target")
    loc = d[(d.index >= -45) & (d.index <= 45)]["rotation generalization"].mean()
    glob = d[(d.index < -45) | (d.index > 45)]["rotation generalization"].mean()
    print(f"[adapt {condition} s{seed}] saved {out.name}  local_gen={loc:.1f}  global_gen={glob:.1f}  "
          f"locality(local-global)={loc-glob:.1f}", flush=True)
    # save Fig 3E re-learning curve (rolling mean of taken-action error) for the mechanism comparison
    import json as _json
    w = 500
    roll = [float(np.mean(relearn_curve[max(0, i - w):i + 1])) for i in range(0, len(relearn_curve), w)]
    roll_g = [float(np.mean(greedy_curve[max(0, i - w):i + 1])) for i in range(0, len(greedy_curve), w)]
    rl_out = Path(paper_fig_dir) / f"relearn_{condition}_{adapt_mode}_seed{seed}_rot{rotation_deg}.json"
    rl_out.write_text(_json.dumps({"condition": condition, "adapt_mode": adapt_mode, "seed": seed,
                                   "rotation_deg": rotation_deg, "roll_err": roll, "roll_greedy": roll_g,
                                   "step": w}))
    print(f"[adapt {condition} {adapt_mode} s{seed}] relearn: start~{roll[1] if len(roll)>1 else 90:.0f} "
          f"end~{roll[-1]:.0f} deg", flush=True)
    return agent, base_agent, acfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "adapt"], required=True)
    ap.add_argument("--condition", choices=["sl", "random"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_deg", type=int, default=135)
    ap.add_argument("--rotation_deg", type=int, default=-30)
    ap.add_argument("--max_episodes", type=int, default=200000)
    ap.add_argument("--adapt_episodes", type=int, default=100000)
    args = ap.parse_args()
    cfg = {**BASE, "max_episodes": args.max_episodes}
    if args.stage == "base":
        train_base(args.condition, args.seed, args.target_deg, cfg)
    else:
        train_adapt(args.condition, args.seed, args.target_deg, args.rotation_deg, cfg,
                    adapt_episodes=args.adapt_episodes)


if __name__ == "__main__":
    main()
