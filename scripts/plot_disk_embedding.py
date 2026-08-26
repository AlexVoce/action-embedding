"""Visualise the two two-joint-arm embeddings: fingertip (consequence) = workspace disk,
joint (proprioceptive) = torus. PCA to 2-D, coloured by workspace polar angle."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from scripts.two_joint_arm import TwoJointArm
from scripts.proprio_reacher_rl import train_embedding
from core.plotting import set_plotting_defaults
from definitions import revision_fig_dir

k = int(sys.argv[1]) if len(sys.argv) > 1 else 24
env = TwoJointArm(k)
fps = env.fingertips                                   # (N,2) fingertip positions
ang = np.arctan2(fps[:, 1], fps[:, 0])                 # workspace polar angle (colour)
set_plotting_defaults()

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
sc = ax[0].scatter(fps[:, 0], fps[:, 1], c=ang, cmap="twilight", s=10)
ax[0].set(title="fingertip positions\n(the reachable WORKSPACE)", xlabel="x", ylabel="y")
ax[0].set_aspect("equal")

for i, (est, lbl) in enumerate([("fingertip", "fingertip-state embedding\n= WORKSPACE (disk); motor-equiv. collapse"),
                                ("joint", "joint-state embedding\n= TORUS (config space)")]):
    _, _, embs = train_embedding(TwoJointArm(k), 0, emb_dim=4, steps=120000, emb_state=est)
    uniq = len(set(map(tuple, np.round(embs, 2))))
    proj = PCA(n_components=2).fit_transform(embs)
    ax[i + 1].scatter(proj[:, 0], proj[:, 1], c=ang, cmap="twilight", s=10)
    ax[i + 1].set(title=f"{lbl}\n({uniq}/{env.n_actions} distinct pts)", xlabel="PC1", ylabel="PC2")
    print(f"{est}: {uniq}/{env.n_actions} distinct embedding points", flush=True)

fig.suptitle("Two-joint arm (k=%d): consequence embedding folds the torus onto the workspace disk "
             "(coloured by workspace angle)" % k, fontsize=10)
for a in ax:
    a.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
out = Path(revision_fig_dir) / "disk_vs_torus_embedding.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out.name, flush=True)
