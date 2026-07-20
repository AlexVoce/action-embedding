"""Sweep the decoder-adaptation learning rate to find the value where the SL adaptation rotates
the decode to the CORRECT compensation (greedy error -> 0) and stabilises, rather than
over-rotating (the ~45 deg overshoot). roll_greedy: 30=unadapted(135), 0=compensated(165),
45=overshoot(210)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.track_a_control import train_adapt, BASE

for fg_lr in [1e-4, 3e-5, 1e-5, 3e-6]:
    train_adapt("sl", 0, 135, -30, {**BASE, "seed": 0}, adapt_episodes=40000,
                adapt_mode="sl", use_gemb_policy=False, align_actor_to_gemb=False, fg_lr=fg_lr)
    import json
    d = json.load(open("figures/paper/relearn_sl_sl_seed0_rot-30.json"))
    g = [round(x, 1) for x in d["roll_greedy"]]
    print("fg_lr=%.0e  greedy-decode-err trajectory: %s" % (fg_lr, g[::3]), flush=True)
    print("           min=%.1f (best compensation)  final=%.1f" % (min(g), g[-1]), flush=True)
