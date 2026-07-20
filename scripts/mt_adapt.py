"""
Track A / A1b: does adaptation generalize LOCALLY, and does it depend on whether the
2-D decoder was SL-learned (ring) or RL-learned (bottleneck)?

Fair comparison: both agents are multi-target BottleneckActors (phi(target)->enc->2D->dec->N),
differing ONLY in the decoder's origin:
  - sl         : dec = SL-pretrained ring (frozen during base training)
  - bottleneck : dec = RL-learned during base training
  - random     : dec = random fixed (control)

Adaptation (enc frozen, dec adapts) at target T=135deg under visuomotor rotation rho, via:
  - rl : reward-driven policy gradient on dec              (the 'fully RL' adaptation)
  - sl : error-driven inverse update  dec(enc(phi(achieved_location))) -> action_taken
         (the cerebellar/self-supervised mechanism the paper's model uses)

Generalization: after adapting at T, for every test target theta measure the greedy action
shift vs the pre-adaptation policy, normalise by rotation and by the trained-target
adaptation, and report locality = mean(gen | |theta-T|<=45) - mean(gen | >45).
"""
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.multitarget_bottleneck import MultiTargetReach, BottleneckActor, BASE
from core.agent import ActionMapping
from definitions import paper_model_path


def load_agent(agent_kind, seed, N):
    ck = torch.load(Path(paper_model_path) / f"multitarget_{agent_kind}_seed{seed}_nact{N}.pth",
                    map_location="cpu")
    env = MultiTargetReach({**BASE, "num_actions": N})
    actor = BottleneckActor(env.n_features, 2, N)
    if agent_kind == "sl":
        f = ActionMapping(2, N)
        actor.dec = f  # match architecture used at train time
    actor.load_state_dict(ck["actor"])
    return actor, env


def greedy_action_angles(actor, env):
    angs = []
    with torch.no_grad():
        for idx in range(env.n_actions):
            env.sample_target(idx)
            a = torch.argmax(actor(env.target_features())).item()
            angs.append(env.actions[a])
    return np.array(angs)


def circ_shift(a1, a0):
    return np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))


def adapt(actor, env, target_deg, rotation_deg, variant, episodes=40000,
          dec_lr=1e-3, temp=1.0, std_noise=0.008):
    env.adaptation_rotation = np.radians(rotation_deg).item()
    tgt_idx = int(round(target_deg / 360 * env.n_actions)) % env.n_actions
    # freeze enc, adapt dec only
    for p in actor.enc.parameters():
        p.requires_grad_(False)
    for p in actor.dec.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(actor.dec.parameters(), lr=dec_lr)
    nll = torch.nn.NLLLoss()
    for ep in range(episodes):
        env.sample_target(tgt_idx)
        env.current_xy = env._center
        s = env.target_features()
        opt.zero_grad()
        z = actor.bottleneck(s)
        logits = actor.dec(z)
        probs = F.softmax(logits / temp, -1); probs = probs + std_noise; probs = probs / probs.sum()
        a_idx = torch.multinomial(probs, 1).item()
        action = env.actions[a_idx]
        # movement lands with rotation applied
        nxt = env.get_next_xy(action)
        hit = np.linalg.norm(np.array(nxt) - np.array(env.target_xy)) <= env.target_radius
        reward = 1.0 if hit else -0.1
        if variant == "rl":
            v = 0.0  # single-step, no critic during adaptation (advantage = reward - baseline)
            adv = torch.tensor(reward - 0.0)
            (-torch.log(probs[a_idx]) * adv.detach()).backward()
            opt.step()
        else:  # sl: inverse update — code of the ACHIEVED location should map to the taken action
            ach_feat = env.get_features(nxt)
            z_ach = actor.bottleneck(ach_feat)
            pred = actor.dec(z_ach)
            loss = nll(torch.log_softmax(pred / temp, 0).unsqueeze(0), torch.tensor([a_idx]))
            loss.backward()
            opt.step()


def generalization(base_actor, adapted_actor, env, target_deg, rotation_deg):
    a0 = greedy_action_angles(base_actor, env)
    a1 = greedy_action_angles(adapted_actor, env)
    rho = np.radians(rotation_deg)
    shift = circ_shift(a1, a0)                      # radians, per target
    adapt_amt = np.degrees(shift) / (-rotation_deg) * 100.0   # 100% = fully compensates rho
    target_angles_deg = np.degrees(env.actions)
    ang_from_T = np.array([round(np.degrees(circ_shift(np.radians(target_deg), np.radians(t))))
                           for t in target_angles_deg])
    T_idx = int(np.argmin(np.abs(ang_from_T)))
    denom = adapt_amt[T_idx] if abs(adapt_amt[T_idx]) > 1e-6 else np.nan
    rot_gen = adapt_amt / denom * 100.0
    loc = np.nanmean(rot_gen[np.abs(ang_from_T) <= 45])
    glob = np.nanmean(rot_gen[np.abs(ang_from_T) > 45])
    return {"ang_from_T": ang_from_T.tolist(), "rot_gen": rot_gen.tolist(),
            "adapt_amt": adapt_amt.tolist(), "local": float(loc), "global": float(glob),
            "locality": float(loc - glob), "adapt_at_T": float(adapt_amt[T_idx])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["sl", "bottleneck"])
    ap.add_argument("--variants", nargs="+", default=["rl", "sl"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--target_deg", type=int, default=135)
    ap.add_argument("--rotation_deg", type=int, default=-30)
    ap.add_argument("--episodes", type=int, default=40000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for agent_kind in args.agents:
        for seed in args.seeds:
            base_actor, env = load_agent(agent_kind, seed, args.N)
            for variant in args.variants:
                actor = copy.deepcopy(base_actor)
                adapt(actor, env, args.target_deg, args.rotation_deg, variant, episodes=args.episodes)
                env.adaptation_rotation = 0.0  # measure greedy policy without rotation confound
                res = generalization(base_actor, actor, env, args.target_deg, args.rotation_deg)
                print(f"[{agent_kind:<10} adapt={variant:<2} s{seed}] "
                      f"adapt_at_T={res['adapt_at_T']:6.1f}%  local={res['local']:6.1f}  "
                      f"global={res['global']:6.1f}  locality={res['locality']:6.1f}", flush=True)


if __name__ == "__main__":
    main()
