"""Analyse the single-joint continuous-target lr grid. For each agent: success by (N, lr),
best lr (avg over N). Writes single_joint_success_cont.json = per-seed success at each agent's
best lr, for the scaling figure. Run on azure (needs torch to read the .pth checkpoints)."""
import glob, re, json, os
from collections import defaultdict
import numpy as np, torch
from definitions import paper_model_path, paper_fig_dir

per = defaultdict(list)  # (agent,N,lr) -> [(seed,succ)]
for fp in glob.glob(os.path.join(paper_model_path, "multitarget_*_cont_lr*_seed*_nact*.pth")):
    m = re.search(r"multitarget_(standard|bottleneck|sl)_cont_lr([0-9.e+-]+)_seed(\d+)_nact(\d+)\.pth",
                  os.path.basename(fp))
    if not m:
        continue
    ag, lr, s, N = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
    ck = torch.load(fp, map_location="cpu")
    fs = ck.get("final_success")
    if fs is not None:
        per[(ag, N, lr)].append((s, float(fs)))

mean = {k: float(np.mean([x for _, x in v])) for k, v in per.items()}
Ns = sorted(set(N for _, N, _ in mean)); lrs = sorted(set(lr for _, _, lr in mean))
best_lr = {}
for ag in ["standard", "bottleneck", "sl"]:
    print(f"\n=== {ag}: final success by (N rows, lr cols) ===")
    print("    N\\lr " + "".join(f"{lr:>9.0e}" for lr in lrs))
    for N in Ns:
        row = [mean.get((ag, N, lr), float("nan")) for lr in lrs]
        print(f"{N:6d}  " + "".join(f"{v:9.2f}" if not np.isnan(v) else "     --  " for v in row))
    avg = {lr: np.nanmean([mean.get((ag, N, lr), np.nan) for N in Ns]) for lr in lrs}
    best_lr[ag] = max(avg, key=lambda x: (avg[x] if not np.isnan(avg[x]) else -1))
    print(f"  --> best lr for {ag}: {best_lr[ag]:.0e} (avg {avg[best_lr[ag]]:.2f})")

# strongest-baseline presentation: best lr per (agent, N) -- each point optimally tuned, so any
# residual gap is real, not a fixed-lr handicap. (Baseline best lr is genuinely N-dependent.)
rows = []
chosen = {}
for ag in ["standard", "bottleneck", "sl"]:
    for N in Ns:
        cand = [(lr, mean[(ag, N, lr)]) for lr in lrs if (ag, N, lr) in mean]
        if not cand:
            continue
        lr_best = max(cand, key=lambda x: x[1])[0]
        chosen[(ag, N)] = f"{lr_best:.0e}"
        for s, succ in per[(ag, N, lr_best)]:
            rows.append({"agent": ag, "N": N, "seed": s, "success": succ, "lr": lr_best})
out = os.path.join(paper_fig_dir, "single_joint_success_cont.json")
json.dump(rows, open(out, "w"))
print(f"\nwrote {len(rows)} rows (best lr per (agent,N)) -> single_joint_success_cont.json")
for ag in ["standard", "bottleneck", "sl"]:
    print(f"  {ag:10s} lr/N: " + " ".join(f"{N}:{chosen.get((ag, N), '--')}" for N in Ns))
