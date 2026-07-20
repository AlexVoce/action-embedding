"""R1 part-b: adaptation of the fully-RL bottleneck vs the SL embedding under a visuomotor
rotation, in the SAME multi-target reach setting.

Both agents are first trained multi-target (they both recover the action ring). We then impose a
global visuomotor rotation (commanded angle a -> achieved angle a+rot) and let each system ADAPT:

  * sl        : freeze the RL-trained encoder, make the SL decoder f plastic, and recalibrate it
                REWARD-FREE from the achieved transitions -- g(centre, achieved) -> action_taken
                (the cortico-cerebellar inverse-model signal). No reward is used.
  * bottleneck: the fully-RL system has no separable decoder, so it can only adapt by RL --
                continue actor-critic training under the rotation (reward-driven).

We log the recovery curve (mean ACHIEVED angular error over all targets vs adaptation episode)
and the final per-target achieved error (generalization profile). The prediction: SL recovers
fast and reward-free; the fully-RL bottleneck recovers slowly (or not, in a comparable budget)
because it must wait for reward.

usage: adapt_multitarget.py <sl|bottleneck> <seed> [N=24] [rotation_deg=-30] [adapt_eps=60000]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from core.agent import ActionEmbeddingPredictor, ActionMapping
from definitions import paper_model_path, revision_fig_dir
from scripts.multitarget_bottleneck import (
    BASE, MultiTargetReach, BottleneckActor, Critic, reward_decay)

AGENT = sys.argv[1]
SEED = int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ROT_DEG = float(sys.argv[4]) if len(sys.argv) > 4 else -30.0
ADAPT_EPS = int(sys.argv[5]) if len(sys.argv) > 5 else 60000
ROT = np.radians(ROT_DEG)
EVAL_EVERY = 1000
mp = Path(paper_model_path)


def ang_err_deg(a, b):
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return abs(np.degrees(d))


def eval_achieved(actor, env):
    """Mean ACHIEVED angular error (deg) over all targets, plus per-target array.
    Achieved angle = greedy commanded action + rotation."""
    errs = []
    with torch.no_grad():
        for idx in range(env.n_actions):
            env.sample_target(idx)
            s = env.target_features()
            a_idx = torch.argmax(actor(s)).item()
            achieved = env.actions[a_idx] + ROT
            errs.append(ang_err_deg(achieved, env.actions[idx]))
    return float(np.mean(errs)), np.array(errs)


def load_actor(env, sd):
    fp = mp / f"multitarget_{AGENT}_seed{SEED}_nact{N}.pth"
    ck = torch.load(fp, map_location="cpu")
    actor = BottleneckActor(sd, 2, env.n_actions)
    if AGENT == "sl":
        f = ActionMapping(2, env.n_actions)
        actor.dec = f
    actor.load_state_dict(ck["actor"])
    return actor, ck


def adapt_sl(env, actor):
    """Reward-free decoder recalibration: g(centre, achieved) -> action_taken."""
    emb = torch.load(mp / f"action_embedding_model_seed_0_weight_decay_fg_0.0001_"
                          f"n_action_{env.n_actions}_fourier_basis.pth", map_location="cpu")
    g = ActionEmbeddingPredictor(env.n_features, 2)
    g.load_state_dict(emb["g_state_dict"])
    for p in g.parameters():
        p.requires_grad_(False)
    for p in actor.enc.parameters():           # encoder frozen
        p.requires_grad_(False)
    for p in actor.dec.parameters():           # decoder plastic
        p.requires_grad_(True)
    opt = torch.optim.Adam(actor.dec.parameters(), lr=2e-4)
    centre_feats = env.get_features(env._center)
    temp = 0.25
    curve = []
    for ep in range(ADAPT_EPS):
        env.sample_target()
        s = env.target_features()
        with torch.no_grad():
            z = actor.bottleneck(s)
        probs = F.softmax(actor.dec(z) / temp, dim=-1) + BASE["policy_noise"]
        probs = probs / probs.sum()
        a_idx = torch.multinomial(probs, 1).item()
        next_xy, _, _ = env.act(env.actions[a_idx])          # achieved = commanded + rot
        g_emb = g(centre_feats, env.get_features(next_xy))
        loss = F.cross_entropy(actor.dec(g_emb).unsqueeze(0), torch.tensor([a_idx]))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % EVAL_EVERY == 0:
            m, _ = eval_achieved(actor, env)
            curve.append({"ep": ep, "achieved_err": m})
            if ep % 10000 == 0:
                print(f"[sl s{SEED} N{N}] ep {ep} achieved_err={m:.1f}", flush=True)
    return curve


def adapt_rl(env, actor, ck):
    """Fully-RL adaptation: continue actor-critic under the rotation (reward-driven)."""
    critic = Critic(env.n_features); critic.load_state_dict(ck["critic"])
    for p in actor.parameters():
        p.requires_grad_(True)
    lr = ck.get("actor_lr", 1e-3)
    a_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=lr * 5)
    reward_hist = []; window = 200; curve = []
    for ep in range(ADAPT_EPS):
        env.sample_target()
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
        if ep % EVAL_EVERY == 0:
            m, _ = eval_achieved(actor, env)
            curve.append({"ep": ep, "achieved_err": m})
            if ep % 10000 == 0:
                print(f"[bottleneck s{SEED} N{N}] ep {ep} achieved_err={m:.1f} "
                      f"avg_reward={np.mean(reward_hist[-window:]):.3f}", flush=True)
    return curve


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = {**BASE, "num_actions": N}
    env = MultiTargetReach(cfg, adaptation_rotation=ROT)
    sd = env.n_features
    actor, ck = load_actor(env, sd)
    pre_m, _ = eval_achieved(actor, env)
    print(f"[{AGENT} s{SEED} N{N}] pre-adapt achieved_err={pre_m:.1f} (rotation {ROT_DEG:g} deg)", flush=True)
    curve = adapt_sl(env, actor) if AGENT == "sl" else adapt_rl(env, actor, ck)
    post_m, post_per = eval_achieved(actor, env)
    tangs = [float(np.degrees(env.actions[i])) for i in range(env.n_actions)]
    out = {"agent": AGENT, "seed": SEED, "N": N, "rotation_deg": ROT_DEG,
           "adapt_eps": ADAPT_EPS, "pre_err": pre_m, "post_err": post_m,
           "curve": curve, "final_per_target": post_per.tolist(), "target_deg": tangs}
    fp = Path(revision_fig_dir) / f"adapt_mt_{AGENT}_s{SEED}_N{N}.json"
    fp.write_text(json.dumps(out))
    print(f"[{AGENT} s{SEED} N{N}] DONE pre={pre_m:.1f} post={post_m:.1f} -> {fp.name}", flush=True)


if __name__ == "__main__":
    main()
