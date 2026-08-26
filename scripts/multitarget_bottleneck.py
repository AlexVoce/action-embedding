"""
Track A, RL-only bottleneck (R1's literal ask), fair multi-target version.

The paper trains one policy per target (target implicit in the weights), so a single-
target RL bottleneck has no reason to organise the whole action space. Here we make the
target READABLE from the input using the SAME Fourier basis: the actor sees phi(target_xy)
-- the Fourier code of where to reach -- and must produce the right action for ANY target.
A linear readout of this Fourier code CAN produce the circular action mapping (the basis
linearises the periodic structure), whereas a raw scalar angle cannot.

Agents (all target-conditioned on phi(target)):
  - bottleneck : phi(target) -> Linear(->2) -> tanh -> Linear(2->N) -> logits   [RL only]
  - standard   : phi(target) -> Linear(->N) -> logits                          [RL only]
  - sl         : phi(target) -> Linear(->2) -> tanh ; frozen pretrained f decodes 2->N

--stage base  : train across all targets; report whether the 2-D bottleneck forms a ring.
(--stage adapt to follow: rotation + two adaptation variants -> generalization.)
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.continuous_env import ReachTask
from core.agent import ActionEmbeddingPredictor, ActionMapping
from definitions import paper_model_path

BASE = {
    "grid_size": (20, 20), "reach_length": 1, "reach_angle": np.radians(135),
    "fourier_order": 3, "num_actions": 24, "max_steps": 1,
    "reward_radius_max": 0.5, "reward_radius_min": 0.2, "reward_for_hit": 1.0,
    "penalty_for_miss": -0.1, "discount_factor": 0.99, "inv_temp": 0.8,
    "policy_noise": 0.008, "policy_std": 0.2, "policy_std_max": 0.8,
    "max_reward_policy_annealing": 0.4, "max_reward_target_annealing": 0.5,
}


class MultiTargetReach(ReachTask):
    """Centre-out reach where a fresh target direction is sampled each episode and the
    agent observes the Fourier code of the target location."""

    def __init__(self, cfg, adaptation_rotation=0.0):
        super().__init__(cfg, adaptation_rotation=adaptation_rotation)
        self._center = (self.env_shape[0] / 2, self.env_shape[1] / 2)
        self.reach_length = cfg["reach_length"]
        self.target_idx = 0
        # CONSTANT-DIFFICULTY mode: continuous target angles (not the action grid) with a fixed
        # angular success tolerance. For N>=9 every continuous target has an action within tol,
        # so ceiling=1.0 at every N (no spacing-vs-TOL artifact e.g. the old N=16 dip) and the
        # axis measured is pure action SELECTION -> baselines can tie SL at low N.
        self.continuous = bool(cfg.get("continuous_targets", False))
        self.angular_tol = np.radians(cfg.get("angular_tol_deg", 20.0))
        self.target_angle = float(self.actions[0])
        if self.continuous:
            rng = np.random.RandomState(12345)
            evs, tol = [], np.radians(20.0)      # fixed 20deg eval set (reachable-only)
            while len(evs) < cfg.get("n_eval", 500):
                th = rng.uniform(0, 2 * np.pi)
                if np.min(np.abs(np.angle(np.exp(1j * (self.actions - th))))) <= tol:
                    evs.append(th)
            self.eval_target_angles = np.array(evs)
        else:
            self.eval_target_angles = self.actions

    def _rand_reachable_angle(self):
        while True:
            th = np.random.uniform(0, 2 * np.pi)
            if np.min(np.abs(np.angle(np.exp(1j * (self.actions - th))))) <= self.angular_tol:
                return th

    def set_angular_tol(self, tol_rad):
        self.angular_tol = float(tol_rad)

    def sample_target(self, idx=None):
        if self.continuous:
            theta = self._rand_reachable_angle() if idx is None else float(self.eval_target_angles[idx])
        else:
            self.target_idx = np.random.randint(self.n_actions) if idx is None else idx
            theta = self.actions[self.target_idx]
        self.target_angle = float(theta)
        self.reach_angle = theta
        self.target_xy = (self._center[0] + self.reach_length * np.cos(theta),
                          self._center[1] + self.reach_length * np.sin(theta))
        self.current_xy = self._center
        return self.target_idx

    def get_reward(self, next_xy):
        if not self.continuous:
            return super().get_reward(next_xy)
        ang = math.atan2(next_xy[1] - self._center[1], next_xy[0] - self._center[0])
        err = abs(np.angle(np.exp(1j * (ang - self.target_angle))))
        return self.reward_value if err <= self.angular_tol else self.penalty_for_miss

    def reset(self):
        self.current_xy = self._center

    def target_features(self):
        return self.get_features(self.target_xy)


class BottleneckActor(nn.Module):
    def __init__(self, state_dim, emb_dim, n_actions):
        super().__init__()
        self.enc = nn.Linear(state_dim, emb_dim)
        self.dec = nn.Linear(emb_dim, n_actions)
        nn.init.normal_(self.enc.weight, 0.0, 1e-2); nn.init.zeros_(self.enc.bias)

    def bottleneck(self, s):
        return torch.tanh(self.enc(s))

    def forward(self, s):
        return self.dec(self.bottleneck(s))


class StandardActor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.lin = nn.Linear(state_dim, n_actions)
        nn.init.normal_(self.lin.weight, 0.0, 1e-2); nn.init.zeros_(self.lin.bias)

    def forward(self, s):
        return self.lin(s)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 1)

    def forward(self, s):
        return self.fc(s)


def reward_decay(reward, r_min, r_max, v_max, v_min):
    reward = max(min(reward, r_max), r_min)
    return v_max - (reward - r_min) / (r_max - r_min) * (v_max - v_min)


def angular_err(env, action_angle, target_angle):
    d = np.arctan2(np.sin(action_angle - target_angle), np.cos(action_angle - target_angle))
    return abs(np.degrees(d))


def ring_score(codes_2d, target_angles):
    """How well the 2-D codes lie on a ring ordered by target angle: circular correlation
    between the code's polar angle and the target angle. 1 = perfect ring."""
    ang = np.arctan2(codes_2d[:, 1], codes_2d[:, 0])
    a = np.unwrap(ang) - np.mean(np.unwrap(ang))
    t = np.unwrap(target_angles) - np.mean(np.unwrap(target_angles))
    denom = np.sqrt((a**2).sum() * (t**2).sum())
    return float((a * t).sum() / denom) if denom > 0 else 0.0


def eval_greedy(actor, env, agent_kind, tol_deg=20.0):
    """Mean greedy angular error (deg) + within-tol success rate over the eval targets; + 2-D codes."""
    errs, codes, tangs, hits = [], [], [], 0
    with torch.no_grad():
        for idx in range(len(env.eval_target_angles)):
            env.sample_target(idx); s = env.target_features()
            a_idx = torch.argmax(actor(s)).item()
            e = angular_err(env, env.actions[a_idx], env.target_angle)
            errs.append(e); hits += int(e <= tol_deg)
            tangs.append(env.target_angle)
            if agent_kind in ("bottleneck", "sl"):
                codes.append(actor.bottleneck(s).numpy())
    return (float(np.mean(errs)), (np.array(codes) if codes else None),
            np.array(tangs), hits / len(env.eval_target_angles))


def train_base(agent_kind, seed, cfg, episodes, actor_lr=1e-4, critic_lr=5e-4,
               crit_deg=20.0, eval_every=5000, tag=""):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = MultiTargetReach(cfg)
    sd = env.n_features
    critic = Critic(sd)
    if agent_kind == "bottleneck":
        actor = BottleneckActor(sd, 2, env.n_actions)
    elif agent_kind == "standard":
        actor = StandardActor(sd, env.n_actions)
    elif agent_kind == "sl":
        actor = BottleneckActor(sd, 2, env.n_actions)  # dec replaced by frozen f below
        ck = torch.load(Path(paper_model_path) /
                        f"action_embedding_model_seed_0_weight_decay_fg_0.0001_n_action_{env.n_actions}_fourier_basis.pth")
        f = ActionMapping(2, env.n_actions); f.load_state_dict(ck["f_state_dict"])
        actor.dec = f
        for p in actor.dec.parameters():
            p.requires_grad_(False)
    a_opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    reward_hist = []; window = 200
    curve = []; hit_ep = None; sustained = 0
    for ep in range(episodes):
        env.sample_target()
        s = env.target_features()
        avg = np.mean(reward_hist[-window:]) if len(reward_hist) >= window else -0.1
        if getattr(env, "continuous", False):
            # proven two-joint recipe: gentle episode-annealed temp + dense reward, fixed difficulty
            std_expl = 0.3 + 1.2 * (1 - ep / episodes)          # temp 1.5 -> 0.3 over training
            env.set_angular_tol(np.radians(20.0))
        else:
            std_expl = reward_decay(avg, -0.1, cfg["max_reward_policy_annealing"], 3.0, 0.5)  # softmax temp
            env.set_target_radius(reward_decay(avg, -0.1, cfg["max_reward_target_annealing"],
                                               cfg["reward_radius_max"], cfg["reward_radius_min"]))
        a_opt.zero_grad(); c_opt.zero_grad()
        logits = actor(s)
        probs = F.softmax(logits / std_expl, dim=-1)
        probs = probs + cfg["policy_noise"]; probs = probs / probs.sum()
        a_idx = torch.multinomial(probs, 1).item()
        action = env.actions[a_idx]
        if getattr(env, "continuous", False):
            err = abs(np.angle(np.exp(1j * (action - env.target_angle))))  # dense distance-shaped reward
            reward = 1.0 if err <= env.angular_tol else -0.2 * (err / np.pi)
        else:
            nxt, reward, done = env.act(action)
        value = critic(s)
        adv = torch.tensor(float(reward)) - value
        (-torch.log(probs[a_idx]) * adv.detach()).backward()
        adv.pow(2).backward()
        a_opt.step(); c_opt.step()
        reward_hist.append(reward)
        if ep % eval_every == 0:
            gerr, _, _, succ = eval_greedy(actor, env, agent_kind)
            curve.append({"ep": ep, "greedy_err": gerr, "success": succ,
                          "avg_reward": float(np.mean(reward_hist[-window:]))})
            if gerr < crit_deg:
                sustained += 1
                if hit_ep is None and sustained >= 2:  # sustained across 2 evals
                    hit_ep = ep
            else:
                sustained = 0
            if ep % 25000 == 0:
                print(f"[{agent_kind} s{seed} N={env.n_actions}] ep {ep} "
                      f"greedy_err={gerr:.1f} success={succ:.3f} avg_reward={np.mean(reward_hist[-window:]):.3f}", flush=True)

    mean_err, codes, tangs, final_success = eval_greedy(actor, env, agent_kind)
    rscore = ring_score(codes, tangs) if codes is not None else float("nan")
    print(f"[{agent_kind} s{seed} N={env.n_actions}] DONE mean_greedy_err={mean_err:.2f}deg  "
          f"success={final_success:.3f}  hit_ep={hit_ep}  ring_score={rscore:.3f}", flush=True)

    out = Path(paper_model_path) / f"multitarget_{agent_kind}{tag}_seed{seed}_nact{env.n_actions}.pth"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "kind": agent_kind, "n_actions": env.n_actions, "state_dim": sd,
                "mean_greedy_err": mean_err, "final_success": final_success, "ring_score": rscore,
                "hit_ep": hit_ep, "curve": curve, "seed": seed, "continuous": getattr(env, "continuous", False),
                "actor_lr": actor_lr, "critic_lr": critic_lr, "episodes": episodes, "crit_deg": crit_deg}, out)
    print(f"[{agent_kind} s{seed}] saved {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base"], default="base")
    ap.add_argument("--agent", choices=["bottleneck", "standard", "sl"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_actions", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=300000)
    ap.add_argument("--crit_deg", type=float, default=20.0)
    ap.add_argument("--actor_lr", type=float, default=1e-4)
    ap.add_argument("--continuous_targets", action="store_true")  # constant-difficulty continuous targets
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--tag", type=str, default="")  # appended to saved filename
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = {**BASE, "num_actions": args.num_actions,
           "continuous_targets": args.continuous_targets, "n_eval": args.n_eval}
    train_base(args.agent, args.seed, cfg, args.episodes, actor_lr=args.actor_lr,
               critic_lr=args.actor_lr * 5, crit_deg=args.crit_deg, eval_every=args.eval_every, tag=args.tag)


if __name__ == "__main__":
    main()
