
# Revision Triage — PCOMPBIOL-D-26-00710

_"Why motor learning involves multiple systems: an algorithmic perspective" — PLOS Computational Biology_

**Decision:** Major revision · **Deadline:** 5 Sep 2026 (extension available on request)

## Bottom line

Both reviewers explicitly state the core idea is sound and novel and that their concerns do not undermine it. The bulk of the requests are "clarify / acknowledge / add a supplementary control," not "the result is wrong." Realistic estimate: **3–6 weeks of active work**, dominated by a handful of new baseline/control simulations; the rest is writing and mechanical journal formatting. The two months to the deadline are comfortable.

The concerns converge, so this is ~6–7 discrete experiments/analyses (not 15): a **baseline comparison** and an **alternative state representation** are the ones both reviewers care most about.

## At a glance

| Bucket                           | Count | Effort                                  |
| -------------------------------- | :---: | --------------------------------------- |
| A. New simulations & analyses    |   7   | High — the real work (A1, A2 are large) |
| B. Framing, writing & discussion |  10   | Low each, but many — ~2–3 days total    |
| C. Minor text / figure fixes     |   7   | Trivial — under a day                   |
| D. Journal requirements / admin  |  10   | Mechanical — ~half a day + letter       |

**Effort key:** S = under half a day · M = half to two days · L = multi-day (new sims + write-up).

## A. New simulations & analyses — the heavy lifting

| #   | Item / what to do                                                                                                                                                                                                                                                                                               | Effort | Owner   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: | ------- |
| A1  | **Baseline comparison** — show the embedding actually accelerates learning vs plain policy gradient, AND add an RL-only low-rank / bottleneck baseline to test whether it reproduces the same adaptation & interference profiles. Both reviewers hit this; the efficacy claim is currently asserted, not shown. |   L    | Modeler |
| A2  | **Alternative state representation** — repeat key results with a one-hot / grid-world state (or otherwise) so the Fourier basis is not pre-encoding x/y. Address R1's PCA-of-state-differences finding that the state rep already contains the action structure. Sharpest technical concern.                    |   L    | Modeler |
| A3  | **Multi-step / sequential task** — add a variant with a smaller per-action offset requiring several steps to reach the goal, so it is not purely single-step. Supplement is fine.                                                                                                                               |  M–L   | Modeler |
| A4  | **Hyperparameter sensitivity** — show how the generalization scale (Fig 3I) and dual-adaptation interference scale (Fig 4G) depend on tuned parameters; clarify what is a prediction vs a fit.                                                                                                                  |   M    | Modeler |
| A5  | **Park 2025 offset** — either directly simulate Park's small-offset (10°) condition, OR explicitly justify including the large-offset condition Park deliberately avoided. (Sim = M; text-only justification = S.)                                                                                              | M / S  | Modeler |
| A6  | **Fig 2 selection baseline** — justify the "selection encoding model" (currently replotted from an in-context-learning paper) or replace it with a baseline that reflects selection-based accounts of BG function.                                                                                              |   M    | Modeler |
| A7  | **Specification-activity control** — disentangle what follows from RL within the embedding vs from the embedding structure itself; make the stronger claim (structure is learned through interaction, not imposed a priori) explicit. Links to A2.                                                              |   M    | Modeler |

## B. Framing, writing & discussion

|#|Item / what to do|Effort|Owner|
|---|---|:-:|---|
|B1|Foreground the fast-SL / slow-RL framing in the Introduction (currently only in Discussion). Clarify the Wang 2018 meta-RL distinction and the tension that cortex is usually thought to learn slowly via Hebbian updates.|M|Lead + Clopath|
|B2|Separate the two levels of explanation: SL/RL (computational) vs cortex/cerebellum/BG (biological). State which conclusions follow from the framework and which depend on the assumed mapping. (R2 #1)|M|Lead + Clopath|
|B3|Reconcile the cerebellum's role: encoder _g_ (embedding) in the Discussion vs decoder _f_ (fast adaptation) in Figs 3–4. Define cortex's role explicitly. (R2 #2)|M|Lead + Clopath|
|B4|Add discussion of cerebellum–BG disynaptic pathways (dentate→striatum, STN→cerebellar cortex), esp. the cortex-independent cerebellar projection to striatum relevant to Fig 2. Add citations. (R2 #3)|S–M|Lead|
|B5|Soften scaling claims in the abstract & introduction; frame high-dimensional relevance as a prediction, matching the Discussion. (R1 baselines, R2 #4)|S|Lead|
|B6|Acknowledge the single-step task is closer to a contextual bandit; note the limited role of TD learning and how the framework could extend to multi-step. Pairs with A3. (R1, R2 #5)|S|Lead|
|B7|**Woolley 2007 fixes:** correct "indistinguishable" (targets cued by red/blue background colour); note black = initial / grey = final trial in Fig 4A,B; state which phase (PRE/TRAINING/POST) each panel shows and its correspondence to the model.|M|Lead|
|B8|Inverse-model discussion: note a classical inverse model is goal/reward-conditioned and used as a controller, whereas here the module inverts the transition model to learn a representation. Develop the significance of that different role. (R1 minor, R2 #4)|S–M|Lead|
|B9|Expand discussion of the departure from the prevailing forward/inverse-model view (refs 100, [35]), since the work departs from it. (R1 minor)|S|Lead|
|B10|Add nuance on Parkinson's: refs 70–72 support intact initial adaptation but report impairments in savings/consolidation/retention. (R2 #3 minor)|S|Lead|

## C. Minor text & figure fixes

|#|Item / what to do|Effort|Owner|
|---|---|:-:|---|
|C1|Use the term "motor equivalence" for multiple joint configs achieving the same outcome. (R1)|S|Lead|
|C2|Fig 2: reference 63 points to the wrong paper — fix. (R1)|S|Lead|
|C3|Fig 3I: inverted y-axis is confusing vs 3E — relabel "negative angular error" / flip sign; ideally plot Krakauer 2000's transformed angular error and clarify what that quantity is. (R1)|S|Lead|
|C4|std update equation should read std ← std_max − … (mirror the radius update). (R1)|S|Modeler|
|C5|Report the embedding learning rate α_e in the "Embedding Learning" column. (R1)|S|Modeler|
|C6|Define M2 / ACA (Fig 2). (R2 #2 minor)|S|Lead|
|C7|State in the MAIN text (not just Methods) that the Park comparison is qualitative, and that the 30° small-separation is widened from Park's 10° for visualisation. (R2 #1 minor)|S|Lead|

## D. Journal requirements & submission admin

|#|Item / what to do|Effort|Owner|
|---|---|:-:|---|
|D1|Complete CRediT contributions for Greenstreet, Geerts, Gallego & Clopath in the submission form.|S|All authors|
|D2|Upload manuscript source file (.docx/.rtf/.tex). If .tex, upload as "LaTeX Source File" and keep the PDF as "Manuscript".|S|Corresp.|
|D3|Write a 150–200 word Author Summary for a general audience, placed between the Abstract and Introduction.|S–M|Lead|
|D4|Export main figures as separate .tif/.eps files; run them through PLOS's NAAS tool.|S|Modeler|
|D5|Provide figure files for Fig 1 A–K, 2 A–D, 3 A–I, 4 A–G (referenced on pp. 3, 8, 11, 14).|S|Modeler|
|D6|Move supplementary figures out of the manuscript into "Supporting Information"; add each SI legend after the reference list.|S|Modeler|
|D7|Rewrite Financial Disclosure in full sentences: grant numbers + author initials per source, funder role, and any author salaries.|S|Corresp.|
|D8|Provide a Competing Interests statement (incl. co-authors); e.g. "The authors have declared that no competing interests exist."|S|Corresp.|
|D9|Confirm the Data Availability statement meets PLOS's full-availability policy (both reviewers answered "Yes" — likely fine; just verify code/data links).|S|Corresp.|
|D10|Assemble the three required files: Response to Reviewers letter, Revised Manuscript with Track Changes, and a clean unmarked Manuscript.|M|Lead + Clopath|

## Suggested order of attack

1. **Start the simulations now (A1, A2 first)** — they gate the response letter and take the longest.
2. **In parallel, knock out C and D** (trivial, and D can be delegated to co-authors / corresponding author).
3. **Write B once the sims land**, so the framing matches the new results.
4. **Assemble D10 last** (letter + tracked-changes + clean manuscript).

_"Owner" is a suggestion — adjust to your team. "Lead" = whoever drives the writing; "Modeler" = whoever runs the code; "Corresp." = Clopath for submission-form items._