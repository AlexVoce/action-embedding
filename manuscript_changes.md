# Manuscript change checklist (PCOMPBIOL-D-26-00710 revision)

Line-anchored edits for **`manuscipt/greenstreet_geerts_gallego_clopath_2025_ploscb_revisions/main.tex`**
(556 lines; bib = `refs.bib`, `\addbibresource` at line 58). Response letter is otherwise done; every sim is complete.
Line numbers are as of this reading — they'll shift as you insert text, so work bottom-to-top or re-grep.
Tags: reviewer point + plan code. Owner **Lead** unless noted. `[ ]` = to do, `[x]` = already done.

---

## Already resolved (verify only)
- [x] **C2 / A6 — Park citation** (R1 Fig 2 ref 63 / R2 #7): you already fixed it. Confirmed: every Park cite in `main.tex` (incl. both in the Fig 2D caption, line 195) is `park_conjoint_2025` (Junchol Park, Neuron 2025 — the BG paper). The wrong one, `park_competition_2025` (Core Francisco Park, *"Competition Dynamics… In-Context Learning"*, arXiv:2412.01003), is **not cited anywhere**. → Just (a) recompile and eyeball the Fig 2 reference, and (b) optionally delete the now-unused `park_competition_2025` entry from `refs.bib` (biblatex won't print it if uncited, so harmless to leave). Response letter's R2 #7/C2 section already states it's corrected.

---

## Abstract (lines 107–115)
- [ ] **Scaling = demonstrated** (R2 #4, B5). The abstract currently doesn't claim the scaling result. Add a clause (end of line 113, after the generalisation/adaptation sentence) stating you now show it, e.g.:
  > "We further demonstrate that learning in this structured embedding space accelerates reinforcement learning as the action space grows, both for large discrete action sets and a two-joint arm."
  Keep it "demonstrated," not "predicted" — only the extrapolation to very high-DOF biological control stays a prediction (put that caveat in the Discussion, not here).

## Introduction (lines 124–158)
- [ ] **Foreground fast-SL / slow-RL in the Intro** (R1.1, B1). At **line 156** (*"reinforcement learning, which is typically slow \cite{Botvinick2019,wang_prefrontal_2018}"*): **remove the `wang_prefrontal_2018` cite here** (keep `Botvinick2019`) and instead engage Wang/meta-RL in a dedicated Discussion paragraph (below). Keep/foreground the slow-synaptic-RL vs fast-SL framing in the intro, noting SL can learn from every transition without reward (reward-free adaptation — see Discussion).
- [ ] **Name "motor equivalence"** (R1, C1). **Line 156** already describes *"multiple joint configurations that can achieve the same end-effector position or movement outcome"* — insert the term: "…the same end-effector outcome (**motor equivalence**)."
- [ ] **Scaling claim → cite the new demonstrated figure** (R2 #4, B5). **Line 171** ends *"…particularly as the size of the action space increases \cite{chandak_learning_2019} (Supplemental Figure \ref{fig:scaling})."* Reword to point at the new **SI** scaling figure as the demonstration, keeping the parameter-count supplement (`fig:scaling`) as the mechanistic explanation: "…increases, which we demonstrate directly within our model (Supplementary Figure \ref{fig:scaling_demo}; parameter-count intuition in Supplementary Figure \ref{fig:scaling})."

## Results
- [ ] **Demonstrated-scaling paragraph (figure in SI)** (R1 part-a, R2 #4). **After line 205** (which currently just asserts *"learning with such a representation speeds up learning when there are many possible actions \cite{chandak_learning_2019}"* with no own demonstration), add 2–3 sentences pointing to the new **SI** figure `fig:scaling_demo` (`speed_vs_nactions.pdf` + two-joint reacher `proprio_reacher_scaling_prrfp.pdf`): SL stays ~N-independent and low-error while standard RL's episodes-to-criterion explode with N; the reacher shows the same crossover. **Honesty note:** in the single-step, single-target task there is no speed benefit (manifold ≈1-D) — the advantage is a property of large, structured action spaces.
- [ ] **Park comparison is qualitative, in MAIN text** (R2 #1, C7). **Line 201**: *"placed either close together (small $\Delta = 30\degree$) as in Park et al."* → "(small $\Delta = 30\degree$ — widened from the $10\degree$ used by Park et al. \cite{park_conjoint_2025} for ease of visualisation; the comparison is qualitative)". (Currently this caveat lives only in Methods, line 460.)
- [ ] **Justify the large ($165\degree$) offset** (R1, Fig 2C). Also near **line 201** (or Fig 2 caption line 194), add one sentence on why the large-offset condition is shown even though Park deliberately avoided large offsets (e.g. to probe the model's prediction across the *full* similarity range, where selection vs specification diverge most).
- [ ] **"Structure is learned, not imposed" — make explicit + cite controls** (R2 #6, A7). **Line 201** already says *"unlike the encoding models used by Park et al. which assume a fixed representational structure, our model learns this structure through interaction with the environment."* Strengthen and cite the new **one-hot** and **random-embedding** control figures (appendix items below) as direct evidence.

## Adaptation section
- [ ] **Parkinson's nuance** (R2 #3 minor, B10). **Line 227**: *"…not disrupted by basal ganglia dysfunction according to studies in Parkinsonian patients \cite{bedard2011basal,leow2012impaired,marinelli2009learning}."* Add a qualifier: initial adaptation is intact, but PD studies report impaired **savings / consolidation / retention** (refs 70–72). One clause + citations.

## Interference / Woolley 2007 (lines 238–258)
- [ ] **Fix "indistinguishable"** (B7). **Line 252**: *"targets that either lay on top of each other in the visual space (indistinguishable)"* — in Woolley the two targets were **cued by red/blue background colour**, i.e. distinguishable by colour though co-located. Reword accordingly.
- [ ] **Woolley figure fidelity** (B7). **Fig 4 caption (lines 242–247)** / panel refs: note that in Fig 4A,B **black = initial trial, grey = final trial**, and state which phase (PRE / TRAINING / POST) each panel shows and its correspondence to the model.

## Discussion (lines 260–322) — needs author-voice prose
- [ ] **Separate the two levels of explanation** (R2 #1, B2). Add a short paragraph (near the Discussion opening, **lines 261–265**) stating explicitly which conclusions follow from the **computational** framework (SL/RL) vs which depend on the assumed **biological** mapping (cortex/cerebellum/BG).
- [ ] **Clarify cerebellum `g` vs `f` (no reword needed, just APPEND a clause)** (R2.2, B3). R2.2 read line 298 as "cerebellum⇒g" contradicting the adaptation section (lines 227/230: cerebellum⇒f). But line 298 is already non-committal ("could", "either view", "open question") and is about the cortex-vs-cerebellum division of the SL system — under "cerebellum alone" the cerebellum does the WHOLE SL role (g AND f), so no real contradiction. Fix = keep line 298 as-is and **append** a clarifying clause (don't commit to any regional split). Append after "…direct projections to the striatum.":
  > Note that in our framework the supervised system encompasses *both* the learned embedding and its fast recalibration; a cerebellar contribution to rapid adaptation (Figures \ref{fig:adaptation}–\ref{fig:interference}) is therefore part of this same supervised role, not a separate one, and which sub-computations are carried out by cortex versus cerebellum remains open.
  Response-letter 2.2 updated to match (sentence was ambiguous; we clarify rather than "fix a contradiction").
- [ ] **NEW Discussion subsubsection: meta-reinforcement learning** (R1.1, B1). Add ~line 288 (in "Relationship to previous RL models"). Two-paragraph draft is in the chat / response-letter R1.1 — key points: Wang/Botvinick's fast learner is a *reward-meta-learned RL* system in PFC recurrent activity capturing *task* structure; ours is a *supervised* system capturing *action* structure in cortico-cerebellar circuits → complementary, different substrates; our embedding-pretraining is an amortised "learning to learn" but via supervision not meta-RL. Cites: `wang_prefrontal_2018`, `Botvinick2019` (both already in bib).
- [ ] **Hebbian-slowness caveat + resolution** (R1.1, ties to B3). In the cortico-cerebellar SL subsection (**lines 289–298**). 3-sentence draft in chat. Acknowledge cortex classically learns slowly via Hebbian plasticity; resolve by attributing the speed-critical error-driven part to the cerebellum (fast supervised learner) and noting cortex may support faster error-based learning than the strict Hebbian view. **Cites already in bib:** `marr1969theory`, `albus_theory_1971`, `ito1970inhibitory`, `li2020cortico`, `pemberton_cerebellar-driven_2024`, `Lillicrap2016a`. **Cites to ADD to refs.bib** (bibtex in chat): `hebb1949`, `feldman2009` (or `buonomano_merzenich1998`), `raymond_medina2018`, `lillicrap2020` (optional — can reuse `Lillicrap2016a`).
- [ ] **Cerebellum–BG disynaptic pathways** (R2 #3, B4). Add a paragraph in the *"other models of corticocerebellar-BG interactions"* subsection (**lines 301–308**): dentate→striatum and STN→cerebellar cortex; note the cortex-independent cerebellar projection to striatum is relevant to the Fig 2 specification-like striatal activity. Add citations.
- [ ] **Reward-free adaptation argument** (B1 backbone, from R1 part-b(3)). **Lines 286–287** already note SL adaptation needs no slow reward-based updates. Expand into the full argument: the prediction objective is available on *every* transition, rewarded or not, so SL can (and should) adapt from sensory error without waiting for reward — a fully-RL system cannot. Port the wording from the response letter's R1 part-b(3).
- [ ] **Inverse-model distinction** (R1 minor, R2 #4, B8). **Lines 294–296** already touch this; expand: a classical inverse model is goal/reward-conditioned and used as a *controller*, whereas here the module inverts the transition model to learn a *representation*.
- [ ] **Departure from forward/inverse-model view** (R1 minor, B9). **Lines 294–295**: expand the contrast with the prevailing forward/inverse view (refs 100, [35]).
- [ ] **Contextual-bandit acknowledgment + multi-step** (R1, R2 #5, B6). In *Experimental predictions and future work* (**~line 317**), acknowledge the main tasks are single-step (≈ contextual bandit) with limited TD role, and point to the new **multi-step** appendix figure showing the framework extends to sequential tasks.

## Methods
- [ ] **C4 — std-update equation typo** (R1). **Line 365** reads `\text{std} \leftarrow \text{std} - \frac{...}{...}(\text{std}_{max} - \text{std}_{min})`. It should mirror the radius update (line 374): change the leading `\text{std}` to `\text{std}_{max}`, i.e.
  `\text{std} \leftarrow \text{std}_{max} - \frac{ r - r_{min}}{r_{max} - r_{min}} (\text{std}_{max} - \text{std}_{min})`.
- [ ] **C5 — embedding learning rate α_e** (R1) — ⚠️ **VERIFY VALUE FIRST.** Two problems: (i) **Table, line 401** — the *Embedding Learning* column shows `---`; it needs the real value. (ii) **Text, line 448** — states the embedding was pre-trained *"using a learning rate of $1 \times 10^{-4}$"*, but `scripts/run_seeds_embedding.sh` calls `embedding_learning.py` with **no `--learning_rate`**, so it used the **default 0.01**. So 1e-4 and 0.01 conflict. **Check which lr your published embedding models were actually trained with**, then make line 401 and line 448 agree on it. (Saved `.pth` files don't store the lr, so this needs your memory / a re-run record. My best read from the code is **0.01**.)

## New figures to add — ALL to Supporting Information (per decision: keep Figs 1–4 as the main story)
Export PDFs into the manuscript `figs/` dir first (current revision figures are `figures/revision_figures/*.png`).
Add as new `\begin{figure}` blocks in the SI section (after **line 554**), each with a caption + SI legend after the reference list (Journal req. 6), and referenced from the relevant Results/Discussion sentence:
- [ ] `fig:scaling_demo` — `speed_vs_nactions.pdf` + `proprio_reacher_scaling_prrfp.pdf` (R1 part-a efficacy, R2 #4 scaling) — referenced from line 171 & the new post-205 paragraph
- [ ] one-hot grid-world ring `onehot_gridworld_linear.pdf` (R1 state-rep, R2 #6)
- [ ] two-joint torus/workspace `proprio_vs_fingertip_arm.pdf` (motor equivalence, R1.m1 / R2)
- [ ] random-embedding control `random_control_generalization.pdf` (R2 #6)
- [ ] generalisation-scale sensitivity `a4_sensitivity.pdf` (R2 #8)
- [ ] multi-step reach `a3_multistep.pdf` (R1 single-step / R2 #5)
- [ ] **fully-RL baseline, learning** `bottleneck_vs_sl_learning.pdf` (R1.4b) — recovers ring, no learning benefit.
- [ ] **adaptation transfer, SL vs standard full-RL** `adaptation_generalization_bars.pdf` (R1.4b adaptation + R1.2) — transfer to untrained targets (30° baseline − |error|; +gen/−interference), near vs far bars; SL near +20°/far +8° (local generalisation), standard RL near +1°/far −17° (no transfer, interference). Transfer-vs-baseline metric so standard's failure is unambiguous. (Supersedes profile versions `adaptation_generalization_sl_vs_rl` / `bottleneck_vs_sl_generalization`.)
- Note: existing SI figures are currently *inside* the manuscript (lines 502–554) — Journal req. 6 also requires moving those out into Supporting Information with legends after the references.

## Figures needing regeneration (modeler, not a .tex text edit)
- [ ] **C3 — Fig 3I y-axis** (R1): regenerate Fig 3I with a non-inverted / relabelled axis ("negative angular error" or flip sign), ideally plotting Krakauer 2000's transformed angular error; clarify the quantity in the **caption (line 218)**.
- [ ] **C6 — define M2 / ACA** (R2 #2 minor): the Fig 2D caption (**line 195**) names "M1, and M2" while the underlying Park data panels are STR/MOp/**ACA** — reconcile the labels and spell out M2 (secondary motor cortex) / ACA (anterior cingulate area) at first use.

## Front matter / submission (writing, not body)
- [ ] **Author Summary** (D3): a draft **already exists commented-out at lines 117–120** (~110 words). Uncomment, expand to 150–200 words, place between Abstract and Introduction.
- Competing interests (line 483–484) and Code/Data availability (line 486–487) exist but may need the full PLOS wording (D7–D9) — corresponding author.
