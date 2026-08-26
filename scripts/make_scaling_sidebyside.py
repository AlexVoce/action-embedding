"""Side-by-side scaling figure (data panels only; schematics to be drawn separately).
Left  = single-joint reach: episodes-to-criterion vs N  (standard / RL-bottleneck / SL embedding).
Right = two-joint reacher : final success rate vs N=k^2 (standard / embedding), FIXED (fingertip) data.
"""
import glob, re, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from definitions import paper_fig_dir, revision_fig_dir

COL = {"standard": "#A94850", "bottleneck": "#3B6FB0", "sl": "#2E8B57", "embedding": "#2E8B57"}
matplotlib.rc("font", size=7)
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42


def agg(d):  # (agent,N)->[vals]  ->  per agent: sorted (N, mean, sem)
    out = defaultdict(list)
    for (a, N), v in d.items():
        v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if v:
            out[a].append((N, float(np.mean(v)), float(np.std(v) / max(1, np.sqrt(len(v))))))
    return {a: sorted(p) for a, p in out.items()}


# --- single-joint: final SUCCESS RATE vs N ---
# prefer the constant-difficulty continuous-target sweep at each agent's best lr; fall back to the
# old grid-aligned-target models.
sj_file = (Path(paper_fig_dir) / "single_joint_success_cont.json")
if not sj_file.exists():
    sj_file = Path(paper_fig_dir) / "single_joint_success.json"
sj = defaultdict(list)
for r in json.loads(sj_file.read_text()):
    sj[(r["agent"], r["N"])].append(r["success"])
sj = agg(sj)

# --- two-joint: final SUCCESS RATE vs N=k^2 ---
# prefer the final per-agent-best-lr run (embedding@3e-4, standard@1e-4, 4 seeds, per-seed embeddings);
# fall back to the older sweeps.
tj = defaultdict(list)
files = (glob.glob(str(Path(paper_fig_dir) / "prrfp_final_*_k*.json"))
         or glob.glob(str(Path(paper_fig_dir) / "prrfp3_k*.json"))
         or glob.glob(str(Path(paper_fig_dir) / "prrfp2_k*.json")))
for fp in files:
    for r in json.load(open(fp)):
        tj[(r["agent"], r["N"])].append(r["final_success"])
tj = agg(tj)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for agent, lab in [("standard", "Standard RL (full-rank)"), ("bottleneck", "Fully-RL 2-D bottleneck"),
                   ("sl", "SL embedding (ours)")]:
    if agent not in sj:
        continue
    Ns, ms, ss = (np.array(z) for z in zip(*sj[agent]))
    ax[0].plot(Ns, ms, marker="o", color=COL[agent], label=lab, lw=2)
    ax[0].fill_between(Ns, ms - ss, ms + ss, color=COL[agent], alpha=0.2, lw=0)
ax[0].set(xscale="log", xlabel="number of actions  N", ylabel="final success rate",
          title="A.  Single-joint reach", ylim=[0, 1.03])

for agent, lab in [("standard", "Standard RL"), ("embedding", "Embedding (ours)")]:
    if agent not in tj:
        continue
    Ns, ms, ss = (np.array(z) for z in zip(*tj[agent]))
    ax[1].plot(Ns, ms, marker="o", color=COL[agent], label=lab, lw=2)
    ax[1].fill_between(Ns, ms - ss, ms + ss, color=COL[agent], alpha=0.2, lw=0)
ax[1].set(xscale="log", xlabel="number of actions  N = k²", ylabel="final success rate",
          title="B.  Two-joint reacher", ylim=[0, 1.03])

for a in ax:
    a.legend(frameon=False, fontsize=8)
    a.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "scaling_single_vs_two_joint.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".svg"))
print("saved", out.name)
print("single-joint Ns:", {a: [p[0] for p in sj[a]] for a in sj})
print("two-joint pts:", {a: [(p[0], round(p[1], 2)) for p in tj[a]] for a in tj})
