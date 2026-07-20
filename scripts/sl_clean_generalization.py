"""Clean SL-system counterpart to bottleneck_generalization.py, SAME probe-adaptation protocol.

The SL policy IS the learned embedding: for target theta the movement code is g(centre, target_theta)
and the commanded action is argmax f(g(centre, target_theta)). No RL encoder (removing the
RL-encoder confound of the multi-target `sl` agent), and we use a good per-seed embedding.

Adaptation is REWARD-FREE decoder recalibration on the PROBE target only: the agent moves, the
achieved (rotated) transition gives a self-supervised label g(centre, achieved) -> action_taken,
and f is updated to predict it. We then measure the after-effect (change in greedy commanded angle)
at EVERY target vs angular distance from the probe -- the same generalization read-out as the
RL-bottleneck script. Prediction: LOCAL (shift concentrated near the probe), matching Fig 3I.

usage: sl_clean_generalization.py <emb_seed> [N=24] [rotation_deg=-30] [adapt_eps=4000]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from core.agent import ActionEmbeddingPredictor, ActionMapping
from definitions import paper_model_path, revision_fig_dir
from scripts.multitarget_bottleneck import BASE, MultiTargetReach

SEED = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
ROT_DEG = float(sys.argv[3]) if len(sys.argv) > 3 else -30.0
ADAPT_EPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
ROT = np.radians(ROT_DEG)
mp = Path(paper_model_path)


def wrap_deg(d):
    return np.degrees(np.arctan2(np.sin(d), np.cos(d)))


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = {**BASE, "num_actions": N}
    env = MultiTargetReach(cfg, adaptation_rotation=0.0)
    emb = torch.load(mp / f"action_embedding_model_seed_{SEED}_weight_decay_fg_0.0001_"
                          f"n_action_{N}_fourier_basis.pth", map_location="cpu")
    g = ActionEmbeddingPredictor(env.n_features, 2); g.load_state_dict(emb["g_state_dict"])
    f = ActionMapping(2, env.n_actions); f.load_state_dict(emb["f_state_dict"])
    for p in g.parameters():
        p.requires_grad_(False)
    centre = env.get_features(env._center)

    def code(idx):
        env.sample_target(idx)
        return g(centre, env.target_features())

    def commanded_all():
        out = []
        with torch.no_grad():
            for idx in range(env.n_actions):
                out.append(env.actions[torch.argmax(f(code(idx))).item()])
        return np.array(out)

    base_cmd = commanded_all()
    probe = int(np.argmin(np.abs(wrap_deg(env.actions - np.radians(135)))))
    probe_deg = float(np.degrees(env.actions[probe]))

    opt = torch.optim.Adam(f.parameters(), lr=5e-4)
    env.set_visuomotor_rotation(ROT)
    temp = TEMP
    for ep in range(ADAPT_EPS):
        with torch.no_grad():
            z = code(probe)                                    # adapt on probe only
        probs = F.softmax(f(z) / temp, dim=-1) + BASE["policy_noise"]
        probs = probs / probs.sum()
        a_idx = torch.multinomial(probs, 1).item()
        env.sample_target(probe)
        next_xy, _, _ = env.act(env.actions[a_idx])            # achieved = commanded + rot
        g_emb = g(centre, env.get_features(next_xy))
        loss = F.cross_entropy(f(g_emb).unsqueeze(0), torch.tensor([a_idx]))
        opt.zero_grad(); loss.backward(); opt.step()

    env.set_visuomotor_rotation(0.0)
    post_cmd = commanded_all()
    dist = wrap_deg(env.actions - np.radians(135))
    shift = wrap_deg(post_cmd - base_cmd)
    ang_err = wrap_deg(post_cmd + ROT - env.actions)        # angular error given the rotation (deg)
    order = np.argsort(dist)
    out = {"agent": "sl_clean", "seed": SEED, "N": N, "rotation_deg": ROT_DEG, "probe_deg": probe_deg,
           "dist_from_probe": dist[order].tolist(), "commanded_shift": shift[order].tolist(),
           "angular_error": ang_err[order].tolist()}
    fp = Path(revision_fig_dir) / f"gen_sl_clean_s{SEED}_N{N}.json"
    fp.write_text(json.dumps(out))
    print(f"[sl_clean s{SEED}] probe shift={shift[probe]:.1f} (ideal {-ROT_DEG:.0f}); "
          f"mean|far|shift={np.mean(np.abs(shift[np.abs(dist) > 60])):.1f} -> {fp.name}", flush=True)


if __name__ == "__main__":
    main()
