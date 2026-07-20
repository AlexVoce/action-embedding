"""R1 part-b: does the fully-RL bottleneck show the LOCAL adaptation-generalization profile?

Protocol mirrors the single-target visuomotor-rotation paradigm (Krakauer; our Fig 3I): adapt on
ONE probe target under a rotation, then measure the after-effect (change in greedy commanded angle)
at EVERY target as a function of angular distance from the probe. A locally-generalizing system
shifts nearby targets and leaves far ones untouched.

  * bottleneck: adapt by RL on the probe only (reward-driven). The shared 2-D bottleneck is the only
                route by which the correction can spill to other targets.

usage: bottleneck_generalization.py <bottleneck|sl> <seed> [N=24] [rotation_deg=-30] [adapt_eps=30000]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from core.agent import ActionMapping
from definitions import paper_model_path, revision_fig_dir
from scripts.multitarget_bottleneck import (BASE, MultiTargetReach, BottleneckActor, StandardActor,
                                            Critic, reward_decay)

AGENT = sys.argv[1]
SEED = int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ROT_DEG = float(sys.argv[4]) if len(sys.argv) > 4 else -30.0
ADAPT_EPS = int(sys.argv[5]) if len(sys.argv) > 5 else 30000
ROT = np.radians(ROT_DEG)
mp = Path(paper_model_path)


def wrap_deg(d):
    return np.degrees(np.arctan2(np.sin(d), np.cos(d)))


def commanded(actor, env):
    """Greedy commanded angle (rad) for every target."""
    out = []
    with torch.no_grad():
        for idx in range(env.n_actions):
            env.sample_target(idx)
            out.append(env.actions[torch.argmax(actor(env.target_features())).item()])
    return np.array(out)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = {**BASE, "num_actions": N}
    env = MultiTargetReach(cfg, adaptation_rotation=0.0)
    sd = env.n_features
    ck = torch.load(mp / f"multitarget_{AGENT}_seed{SEED}_nact{N}.pth", map_location="cpu")
    if AGENT == "standard":
        actor = StandardActor(sd, env.n_actions)
    else:
        actor = BottleneckActor(sd, 2, env.n_actions)
        if AGENT == "sl":
            actor.dec = ActionMapping(2, env.n_actions)
    actor.load_state_dict(ck["actor"])
    critic = Critic(sd); critic.load_state_dict(ck["critic"])

    probe = int(np.argmin(np.abs(wrap_deg(env.actions - np.radians(135)))))
    probe_deg = float(np.degrees(env.actions[probe]))
    base_cmd = commanded(actor, env)

    lr = ck.get("actor_lr", 1e-3)
    a_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=lr * 5)
    env.set_visuomotor_rotation(ROT)
    reward_hist = []; window = 200
    CRIT_DEG = 10.0                       # early-stop when the probe is compensated (as in Fig 3I)
    sustained = 0; stopped_ep = ADAPT_EPS
    for ep in range(ADAPT_EPS):
        env.sample_target(probe)                       # adapt on the probe target only
        s = env.target_features()
        avg = np.mean(reward_hist[-window:]) if len(reward_hist) >= window else -0.1
        temp = reward_decay(avg, -0.1, BASE["max_reward_policy_annealing"], 3.0, 0.5)
        env.set_target_radius(reward_decay(avg, -0.1, BASE["max_reward_target_annealing"],
                                           BASE["reward_radius_max"], BASE["reward_radius_min"]))
        a_opt.zero_grad(); c_opt.zero_grad()
        probs = F.softmax(actor(s) / temp, dim=-1) + BASE["policy_noise"]
        probs = probs / probs.sum()
        a_idx = torch.multinomial(probs, 1).item()
        _, reward, _ = env.act(env.actions[a_idx])
        value = critic(s)
        adv = torch.tensor(float(reward)) - value
        (-torch.log(probs[a_idx]) * adv.detach()).backward()
        adv.pow(2).backward()
        a_opt.step(); c_opt.step()
        reward_hist.append(reward)
        if ep % 500 == 0:                              # early-stopping check on the probe's greedy error
            env.sample_target(probe)
            with torch.no_grad():
                pa = env.actions[torch.argmax(actor(env.target_features())).item()]
            perr = abs(wrap_deg(pa + ROT - env.actions[probe]))
            sustained = sustained + 1 if perr < CRIT_DEG else 0
            if ep % 10000 == 0:
                print(f"[{AGENT}-gen s{SEED} N{N}] ep {ep} probe_err={perr:.1f} "
                      f"avg_reward={np.mean(reward_hist[-window:]):.3f}", flush=True)
            if sustained >= 2:
                stopped_ep = ep
                print(f"[{AGENT}-gen s{SEED} N{N}] early stop at ep {ep} (probe_err={perr:.1f})", flush=True)
                break

    env.set_visuomotor_rotation(0.0)
    post_cmd = commanded(actor, env)
    dist = wrap_deg(env.actions - np.radians(135))          # angle from probe (deg)
    shift = wrap_deg(post_cmd - base_cmd)                    # after-effect (deg)
    ang_err = wrap_deg(post_cmd + ROT - env.actions)        # angular error given the rotation (deg)
    order = np.argsort(dist)
    out = {"agent": AGENT, "seed": SEED, "N": N, "rotation_deg": ROT_DEG, "probe_deg": probe_deg,
           "stopped_ep": int(stopped_ep),
           "dist_from_probe": dist[order].tolist(), "commanded_shift": shift[order].tolist(),
           "angular_error": ang_err[order].tolist()}
    fp = Path(revision_fig_dir) / f"gen_{AGENT}_s{SEED}_N{N}.json"
    fp.write_text(json.dumps(out))
    at_probe = shift[probe]
    print(f"[{AGENT}-gen s{SEED}] probe shift={at_probe:.1f} (ideal {-ROT_DEG:.0f}); "
          f"mean|far|shift={np.mean(np.abs(shift[np.abs(dist) > 60])):.1f} -> {fp.name}", flush=True)


if __name__ == "__main__":
    main()
