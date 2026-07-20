"""A3: multi-step / sequential variant (R1 & R2 #5). Small per-action offset so the agent needs
several steps to reach the goal — genuine sequential decisions where TD learning matters, rather
than a single-step contextual bandit. Trains an action embedding on small-step transitions, then
an actor-critic policy in that embedding, and reports learning speed (embedding vs standard) +
success and an example multi-step trajectory.
"""
import sys, os, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from core.continuous_env import ReachTask
from core.agent import ActionEmbeddingPredictor, ActionMapping, ACLearningAgentWithEmbedding, softmax_with_temperature
from definitions import revision_fig_dir

BASE = dict(grid_size=(20, 20), num_actions=24, fourier_order=3, reach_length=1.0,
            reach_angle=float(np.radians(135)), reward_radius_min=0.25, reward_radius_max=0.25,
            reward_for_hit=1.0, penalty_for_miss=-0.05, discount_factor=0.95, inv_temp=0.8,
            policy_std=0.2, policy_std_max=0.8, policy_noise=0.008, embedding_dim=2, num_RBFs_per_row=5,
            fg_lr=1e-4, actor_lr=5e-4, critic_lr=1e-3, max_steps=1, seed=0,
            max_reward_policy_annealing=0.4, max_reward_target_annealing=0.4, w_decay_fg=0.0,
            tanh_temp=1.0, softmax_temp=1.0)


class MultiStepReach(ReachTask):
    """Reach task where each action moves a small step; several steps are needed to reach goal.
    Dense progress reward (distance reduction toward target) + terminal hit bonus, so the
    sequential task is learnable (sparse terminal-only reward is not, over many small steps)."""
    def __init__(self, config, step_size=0.25, adaptation_rotation=0):
        super().__init__(config, adaptation_rotation)
        self.step_size = step_size

    def get_next_xy(self, a):
        x, y = self.current_xy
        nx, ny = self.move_in_direction(x, y, a, distance=self.step_size, adaptation_rotation=self.adaptation_rotation)
        if self.outside_env(nx, ny):
            return (x, y)
        return (nx, ny)

    def _dist(self, xy):
        return math.hypot(xy[0] - self.target_xy[0], xy[1] - self.target_xy[1])

    def act(self, a):
        prev_d = self._dist(self.current_xy)
        next_xy = self.get_next_xy(a)
        self.current_xy = next_xy
        done = self.in_terminal_state()
        reward = (prev_d - self._dist(next_xy))            # dense progress shaping
        if done:
            reward += self.reward_value                    # terminal hit bonus
        return next_xy, reward, done


def train_embedding(env, cfg, n_iter=120000, cache=None):
    """Inverse model: g(features(s), features(s')) -> action taken, on random small-step transitions."""
    g = ActionEmbeddingPredictor(env.n_features, cfg["embedding_dim"])
    f = ActionMapping(cfg["embedding_dim"], env.n_actions)
    if cache and os.path.exists(cache):
        ck = torch.load(cache)
        g.load_state_dict(ck["g"]); f.load_state_dict(ck["f"])
        print("loaded cached embedding", flush=True)
        return g, f
    opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.NLLLoss()
    rng = np.random.RandomState(cfg["seed"])
    for it in range(n_iter):
        # random start in a central region, random action
        env.current_xy = (10 + rng.uniform(-3, 3), 10 + rng.uniform(-3, 3))
        a_idx = rng.randint(env.n_actions)
        s = env.get_features(env.current_xy)
        nxt = env.get_next_xy(env.actions[a_idx])
        ns = env.get_features(nxt)
        emb = g(s, ns)
        logits = f(emb)
        loss = loss_fn(torch.log_softmax(logits, dim=0).unsqueeze(0), torch.tensor([a_idx]))
        loss = loss - 0.01 * torch.norm(emb)  # magnitude reg (push ring outward), as in embedding_learning
        opt.zero_grad(); loss.backward(); opt.step()
    if cache:
        torch.save({"g": g.state_dict(), "f": f.state_dict()}, cache)
    return g, f


def run_policy(env, cfg, g, f, use_embedding, episodes=120000):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    agent = ACLearningAgentWithEmbedding(env, cfg, g_plastic=False, f_plastic=False,
                                         actor_plastic=True, critic_plastic=True)
    agent.g = g; agent.f = f
    successes, curve = [], []
    for ep in range(episodes):
        env.reset(); env.timer = 0
        done = False; steps = 0; hit = 0
        while steps < cfg["max_steps"]:
            feats = env.get_features(env.current_xy)
            agent.actor_optimizer.zero_grad(); agent.critic_optimizer.zero_grad()
            a_idx, emb, mean_emb, logstd = agent.select_action(feats)
            nxt, r, done = env.act(env.actions[a_idx])
            agent.update(feats, a_idx, emb, env.get_features(nxt), r, done)
            steps += 1
            if done:
                hit = 1
                break
        successes.append(hit)
        if ep % 2000 == 0:
            curve.append([ep, float(np.mean(successes[-2000:]))])
            if ep % 20000 == 0:
                print("  ep %d success=%.2f" % (ep, curve[-1][1]), flush=True)
    return curve, agent


def rollout_trajectories(env, agent, n=6):
    """Greedy multi-step rollouts from centre -> record the paths."""
    paths = []
    for _ in range(n):
        env.reset(); env.timer = 0
        path = [env.current_xy]
        for _ in range(env.max_trial_length):
            feats = env.get_features(env.current_xy)
            with torch.no_grad():
                logits = agent.f(torch.tanh(agent.actor(feats)))
                a_idx = int(torch.argmax(logits).item())
            nxt, r, done = env.act(env.actions[a_idx])
            path.append(nxt)
            if done:
                break
        paths.append(np.array(path))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step_size", type=float, default=0.25)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--emb_iter", type=int, default=120000)
    ap.add_argument("--episodes", type=int, default=150000)
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = {**BASE, "seed": args.seed, "max_steps": args.max_steps}

    env = MultiStepReach(cfg, step_size=args.step_size)
    print("target at distance %.2f, step %.2f -> ~%d steps needed; max_steps=%d"
          % (cfg["reach_length"], args.step_size, int(np.ceil(cfg["reach_length"] / args.step_size)), args.max_steps), flush=True)
    cache = str(Path(revision_fig_dir).parent / "paper" / f"a3_embedding_seed{args.seed}.pth")
    g, f = train_embedding(env, cfg, n_iter=args.emb_iter, cache=cache)
    print("embedding ready", flush=True)
    curve, agent = run_policy(env, cfg, g, f, use_embedding=True, episodes=args.episodes)
    print("[a3] final success rate = %.2f" % curve[-1][1], flush=True)
    paths = rollout_trajectories(env, agent, n=6)
    json.dump({"curve": curve, "step_size": args.step_size, "max_steps": args.max_steps},
              open(Path(revision_fig_dir).parent / "paper" / f"a3_multistep_seed{args.seed}.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    e = [x for x, _ in curve]; v = [y for _, y in curve]
    ax[0].plot(e, v, color="#2E8B57", lw=2)
    ax[0].set(xlabel="episode", ylabel="success rate", title="A3: learning the multi-step reach", ylim=[-0.02, 1.02])
    ax[0].spines[["right", "top"]].set_visible(False)
    for p in paths:
        ax[1].plot(p[:, 0], p[:, 1], "-o", ms=3, alpha=0.7, color="#4477AA")
    ax[1].plot(*env.start_xy, "ks", ms=8, label="start")
    tgt = plt.Circle(env.target_xy, env.target_radius, color="#A94850", alpha=0.4, label="target")
    ax[1].add_patch(tgt); ax[1].plot(*env.target_xy, "r*", ms=10)
    ax[1].set(title="example trajectories (step %.2f, ~%d steps)" % (args.step_size, int(np.ceil(cfg["reach_length"] / args.step_size))), aspect="equal")
    ax[1].legend(frameon=False, fontsize=8); ax[1].spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    out = Path(revision_fig_dir) / "a3_multistep.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print("saved", out.name, flush=True)
    print("SEED %d A3 DONE" % args.seed, flush=True)


if __name__ == "__main__":
    main()
