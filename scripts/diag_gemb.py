import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from core.continuous_env import ReachTask
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config as acfg

cfg = dict(acfg)
cfg.update(seed=0, reach_angle=float(np.radians(135)), rotation_angle=-30, log_to_wandb=False, actor_tanh=False)
envR = ReachTask(cfg, adaptation_rotation=float(np.radians(-30)))
env0 = ReachTask(cfg)
agent = load_trained_full_model_basetask(cfg, envR, 135, 0)

def gemb_of(env, tdeg):
    ti = int(np.argmin(np.abs(((np.round(np.degrees(env.actions)) - tdeg + 180) % 360) - 180)))
    env.reset()
    s = env.get_features(env.current_xy)
    nxt, _, _ = env.act(env.actions[ti])
    with torch.no_grad():
        return agent.g(s, env.get_features(nxt))

for nm, env in [("norot", env0), ("rot", envR)]:
    ge = gemb_of(env, 135)
    dec = float(np.degrees(envR.actions[int(torch.argmax(agent.f(ge)))]))
    print("%s: gemb(action135) radius=%.2f  base-f decodes to %.0f deg" % (nm, float(torch.norm(ge)), dec))
print("env0 rotation:", getattr(env0, "adaptation_rotation", "n/a"), " envR rotation:", getattr(envR, "adaptation_rotation", "n/a"))
# also: base actor direction and its decode
feats = envR.get_features(envR.current_xy)
with torch.no_grad():
    ao = agent.actor(feats)
    print("base actor raw output:", ao.numpy().round(3), " decodes to %.0f" % float(np.degrees(envR.actions[int(torch.argmax(agent.f(torch.tanh(ao))))])))
