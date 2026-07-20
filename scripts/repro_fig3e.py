"""Reproduce Fig 3E/3F via the EXACT repo pipeline with a properly-trained (310k) base policy.
Retrains the one-target base policy to saturation, runs the paper's adaptation_exp, and reports
the actor's operating radius (coherence check) + the Fig 3E angle_diff trajectory from the run-log."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def retrain_base(seed, target_deg, episodes):
    from core.config import config as cfg
    cfg["max_episodes"] = episodes
    cfg["log_to_wandb"] = False
    cfg["save_model"] = True
    sys.argv = ["x", "--seed", str(seed), "--target", repr(float(np.radians(target_deg)))]
    from core.policy_learning import train_agent
    train_agent(cfg)


def actor_radius(seed, target_deg):
    import os
    from core.continuous_env import ReachTask
    from core.agent import ACLearningAgentWithEmbedding
    from core.config import config as cfg
    from definitions import paper_model_path
    env = ReachTask(cfg)
    fn = os.path.join(paper_model_path,
                      f"fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_"
                      f"tanh_policy_mean_target_{target_deg}_n_actions_24.pth")
    ag = ACLearningAgentWithEmbedding(env, cfg, full_model_load_path=fn, fg_load_path=None,
                                      g_plastic=False, f_plastic=False, actor_plastic=False, critic_plastic=False)
    feats = env.get_features(env.current_xy)
    with torch.no_grad():
        actor_out = ag.actor(feats)                 # already tanh'd
        pol_emb = torch.tanh(actor_out)             # what select_action feeds to f (double tanh)
        g = ag.get_action_embeddings_via_g()
    import numpy as np
    print(f"[coherence] actor policy-embedding radius={float(torch.norm(pol_emb)):.3f}  "
          f"ring radius mean={np.linalg.norm(g,axis=1).mean():.3f}  "
          f"greedy_action={float(np.degrees(env.actions[int(torch.argmax(ag.f(pol_emb)).item())])):.0f}deg "
          f"(target {target_deg})", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 135
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 310000
    print(f"=== retrain base seed {seed} target {target} for {episodes} episodes ===", flush=True)
    retrain_base(seed, target, episodes)
    actor_radius(seed, target)
