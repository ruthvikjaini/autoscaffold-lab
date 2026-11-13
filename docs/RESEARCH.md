# RESEARCH.md — autoscaffold-lab (Day 20, 2026-11-13)

## 1. Trending-first check (2026-11-13, ~00:00 ET)

Sources checked: gitdiscover.org (daily picks Nov 12), findarepo.com
papers-with-code + AI-agents (updated Nov 12), topopensource.ai (updated
Nov 12 21:30 UTC), github.locker deep-learning.

- Skill/prompt packs dominate again: janestreet/agent_skills (+2,094 stars/24h,
  9 days old), mattpocock/skills (~322K), anthropics/skills (362 -> 988/day).
- Agent frameworks/platforms: morluto/rea ~454K (+3,354/24h), deepseek-ai/
  deepseek-harness ~279K, stablyai/orca ~123K, meta/astryx ~86K, Ebony-Vinyl/
  dsh-our-free-model 71K (+9.1K/7d), NousResearch/hermes-agent 263K.
- Papers-with-code risers: google-research/timesfm 35K (+221/7d, needs the
  pretrained checkpoint), LearningCircuit/local-deep-research 9.6K (needs a
  real LLM), chenxiachan/textfold 1.0K (a folding UI, no benchmarkable
  training weakness).
- New daily picks: multica-ai/multicaGo 55.8K (agent workspace), ruvnet/
  RuViewRust 104.1K (WiFi perception), Tencent/LoopForgeGo (agent workspace).
- **Verdict: no trending repo picked.** Same as Days 6-19: the trending
  code repos either need a multi-GB/TB checkpoint, a real LLM, NIM/API keys,
  or a GPU fleet, or are skill packs / platforms with no measurable
  one-night CPU weakness. Paper fallback per the standing rule.

## 2. Paper chosen

Zhang, Hu, Peng, Li, Bhatt, Zeng & Hong (Univ. of Minnesota + JPMorgan AI
Research), "AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning
of Language Models", arXiv:2611.14515, submitted 10 Nov 2026.
Verified on arXiv abstract + full HTML 2026-11-13. No paper code read or
used; implementation from the paper's equations and Algorithms 1-2.

### 2.1 What the paper does

RLVR collapses each rollout to one scalar reward, wasting the information in
the trajectory. AutoScaffold puts a frozen teacher LLM in the loop: every
K = 10 training steps the teacher inspects the student's training experience
through query tools (Q_stat class statistics, Q_traj individual trajectories,
Q_group all G attempts on one task, plus a memory of its last three
decisions), diagnoses bottlenecks, and writes scaffolds in two forms:
(i) class-level strategies shared across a task class, injected into a
fraction q[C] <= q_max = 0.5 of rollout groups, and (ii) instance-level hints
for individual hard tasks (in math: a solution prefix revealing the first
r_x fraction of the reference solution). Proposed class-level scaffolds are
validated on held-out tasks V_C (success with none / current / proposed,
matched seeds) before deployment; the teacher withdraws support once
unscaffolded success exceeds 50%. Scaffolds are removed at evaluation.
Teacher: GPT-5.5, at most B = 15 queries per update, a few dollars per run.

### 2.2 Paper results (kept separate from ours)

- ALFWorld (Qwen2.5-1.5B, GiGPO): 93.9% vs 86.7% GiGPO (+7.2). Largest
  gains on the weakest classes: Look +19.7, Cool +16.2, Pick2 +15.3.
- Search QA (Qwen2.5-3B, GiGPO): 46.1% EM vs 42.1% (+4.0).
- Math (Qwen3-4B, GRPO): 70.1% vs 68.7% (+1.4), on par with SAGE (70.0%);
  (Nemotron-1.5B, DAPO): 62.4% vs 60.4% (+2.0), beating QuestA (61.1%)
  in ~1/3 the steps (350 vs 1,100).
- KernelBench (Qwen3-8B, rubric GRPO): correctness 59% -> 63% (L1),
  50% -> 54% (L2); Fast1 20% -> 25% (L2).
- Ablations (ALFWorld): no validation 93.9% -> 79.8% (below the 86.7%
  baseline); fixed Skill0 skill bank as scaffold 78.6% (15.3 below
  AutoScaffold, 8.1 below baseline); fixed failure sample instead of
  interactive queries -8.6; q_max = 1 (scaffold every group) 86.0%
  (slightly below GiGPO baseline).

### 2.3 Theory (Appendix A) we test directly

- Prop. 1: mixed-outcome group probability u(p) = 1 - (1-p)^G - p^G,
  ~Gp for small p. Tasks with Gp << 1 rarely contribute signal.
- Prop. 2: expected within-group reward variance (G-1)/G * p(1-p),
  peaks at p = 1/2.
- Props. 3-4: class scaffolds move p toward 1/2 (max signal);
  instance scaffolds bring p = 0 tasks into training at all.
- Too-strong scaffolds (p -> 1) also kill the signal; hence q_max < 1
  (privileged-information dropout) and withdrawal above 50%.

### 2.4 Other candidates considered / held

- Salloum & Ali, "Tree-Ising: Repairing Language Models with Bit-Level
  Ising Edits" (arXiv:2611.14210, 10 Nov 2026; HF daily papers #1 on
  2026-11-11, 86 upvotes): weight-space repair as per-row QUBO over BF16
  weights, solved by simulated annealing / simulated bifurcation; 54.6%
  MATH-500 vs 51.7% agent edit, 7.9% base; GSM8K 21.6% -> 69.4% on
  Qwen3.5-2B. Strong, futuristic (quantum-inspired solver, QAOA on IBM
  Heron) — HELD as the leading candidate for a future night. Not picked
  tonight because the "loop direction / concluding direction" repair
  objective maps less cleanly onto a tiny arithmetic transformer than
  AutoScaffold's prompt-scaffold mechanism, which reuses the lab's
  proven GRPO harness.
- "Margin-Guided Token Credit for RL with Verifiable Rewards" (MCPO,
  arXiv:2611.14015, 10 Nov 2026): reweights advantage by token-level
  margin/hesitation; +6.55 macro avg@32 over GRPO. HELD (same RLVR
  cluster; token-credit angle is a clean future night).
- Liu et al., "No Free Lunch in RLVR" (arXiv:2610.39295): held a tenth
  night. RLVR cluster; AutoScaffold is fresher (10 Nov) and its
  mechanism (training-time scaffolds) is a new axis vs Days 1/7/10/13/
  15/17/19, which were all GRPO-variant mechanics.
- Reward Budgeting (arXiv:2611.05942), KAPPO (arXiv:2611.03146): held.
- Interaction-as-a-Judge (arXiv:2610.17895): needs a computer-use agent
  + real apps; not CPU-reproducible in one night.

## 3. Design: autoscaffold-lab

### 3.1 Testbed

Five task classes of multi-step arithmetic, response format
`<expr> = <s1> = <s2> = <ans> EOS` (two-step) or three-step for `mix3`,
verifiable binary reward (1 iff final answer correct and format valid),
plus a shaped segment reward used only for diagnostics:

| class | form | strategy the class needs | designed failure mode |
|---|---|---|---|
| add | a+b+c | left-to-right | carry slip in s1 |
| sub | a-b-c | left-to-right, never regroup | grouping a-(b-c) |
| prec | a+b*c / a*b+c | multiply before adding | left-to-right evaluation |
| mod | a*b mod m | multiply, then remainder | modding the wrong operand / skipping mod |
| mix3 | a op b op c op d | do one op at a time, write each result | skipping an intermediate step |

Operands 0-9 (a,b,c,d), m in {3..9}. SFT warm-start on all classes but
with limited steps, so hard classes (prec/mod/mix3) sit at low-but-nonzero
success — the regime where Props. 1/3/4 say scaffolds add signal.
Calibration target: warm-start scaffold-free pass@1 ~0.55 add, ~0.45 sub,
~0.25 prec, ~0.20 mod, ~0.05 mix3. If a class is at 0.00 after SFT,
instance scaffolds are the only way it enters training (Prop. 4) —
that is itself a measured outcome, not a bug.

### 3.2 Student

1-layer causal transformer, d=64, 4 heads, vocab ~40, NumPy, manual
backprop (same harness lineage as Days 10/15/17/19). GRPO: G=8,
clip 0.2, k1 KL beta 0.02, lr 3e-4, 120 steps, batch 12 prompts/step,
3 seeds. Scaffolds are plain text prepended to the prompt; scaffold
tokens receive no gradient (paper Eq. 3). Eval is always scaffold-free
(greedy + sampled pass@1 on 240 held-out problems, 48/class).

### 3.3 Teacher (the honest deviation)

The paper's teacher is GPT-5.5. No LLM is available in this lab, so the
teacher is a deterministic diagnostician that implements the paper's
decision protocol exactly, with a fixed strategy library instead of
free-text generation. Stated plainly in the README; the protocol, not
the prose, is what is reproduced:

- Evidence queries: Q_stat (per-class success with/without scaffold,
  counts of all-fail / all-succeed groups), Q_traj (failed trajectories
  by class), Q_group (all G attempts on one task), memory (last 3
  decisions). Query budget B = 15 per update, same as the paper.
- Diagnosis: each failed trajectory is classified into a fixed failure
  taxonomy by the verifier (wrong-intermediate, wrong-final,
  precedence-error, grouping-error, mod-error, skipped-step,
  format-error). The dominant failure per class is the diagnosis.
- Proposal: dominant failure -> strategy text from a fixed library
  (e.g. precedence-error -> "Multiply before adding: find the
  multiplication first, write its result, then add."). Instance-level:
  for tasks all-fail in the last block, a solution prefix revealing the
  first r_x fraction of the reference derivation (r starts 0.5, +0.25
  if still all-fail, -0.25 if solved, in [0, 1]).
- Validation: held-out V_C (16 tasks/class, excluded from training).
  Current student evaluated with none / current / proposed scaffold,
  n=2 rollouts/task, matched seeds. Adopt proposed iff
  v_new > v_cur; keep current iff v_cur >= v_new; remove the scaffold
  iff v_none >= max(v_cur, v_new) and v_none > 0.5 (withdrawal).
- Injection: q[C] starts 0.5, capped at q_max = 0.5. If v_none > 0.5,
  q[C] halves each update (withdrawal); if v_none < 0.2 and the
  scaffold helps (v_cur > v_none), q[C] returns to 0.5.
- Memory: last 3 (block, class, diagnosis, decision, v_none before/
  after). If the last edit for a class coincided with a v_none drop,
  the teacher reverts to the previous scaffold at the next update.

### 3.4 Arms (main grid, 3 seeds each, 120 steps)

1. `grpo` — plain GRPO, no scaffold (paper's RL baseline).
2. `autoscaffold` — full protocol above (K=10, q_max=0.5, validation).
3. `fixed` — oracle hand-written class strategies (the library's best
   text for each class), q=0.5 fixed, never updated, no instance hints.
   Tests adaptivity (paper: fixed Skill0 bank -15.3).
4. `novalid` — AutoScaffold without held-out validation (teacher's
   proposals deployed as-is). Tests validation (paper: -14.1).
5. `qmax1` — AutoScaffold with q_max = 1.0 (scaffold in every group).
   Tests scaffold dependence (paper: below baseline).

### 3.5 Extensions

- E1 scaffold dependence: eval accuracy with scaffold vs without, at
  steps 0/40/80/120, per arm (does the AutoScaffold student become
  less dependent as q falls?).
- E2 Props. 1/3 check: measured mixed-group fraction u_hat(p) per class
  per block vs the theoretical u(p) curve; correlation between block
  mixed-group fraction and next-block accuracy gain.
- E3 teacher ablation: fixed sample of failures (SkillRL-style) instead
  of interactive queries — teacher sees only 4 failed trajectories per
  weak class, no Q_stat/Q_group/memory (seed 0-1).
- E4 transfer: train on add/sub/prec/mod only; scaffold-free eval on
  mix3 (unseen class, no scaffold ever written for it) vs the grpo arm.
- E5 instance-only: class scaffolds disabled, instance prefixes only
  (seed 0-1) — isolates the Prop. 4 contribution.

### 3.6 What would falsify the paper here

- autoscaffold <= grpo on scaffold-free eval (null headline, as in
  Days 10/15/19 — reported plainly if so).
- Validation rejecting every proposal (teacher never helps) or adopting
  harmful ones (v_new < v_none on V_C after deployment).
- qmax1 >= autoscaffold (no dependence penalty) — would contradict
  the paper's ablation.
- Mixed-group fraction not tracking u(p), or not predicting gains.

## 4. Bugs / design fixes during the night

- (tests) Branding test flagged itself: `tests/test_branding.py` is
  excluded from its own scan (same pattern as previous nights).
- (tests) `test_tasks.test_precedence_detection`: the left-to-right
  alt segments for `2+3*4` are `[5, 20]`, not `[20, 20]` — the test
  was wrong, the classifier was right.
- (tests) KL-sign regression test, three iterations: (1) a 5-step
  gradient drift at lr 1e-2 scrambles the policy (it starts emitting
  scaffold word tokens) and makes every GRPO batch degenerate, so the
  no-signal skip fires and nothing moves; (2) a gentle parameter-noise
  drift keeps mixed groups, but 10 GRPO updates with beta=0.5 still
  raise the KL (0.258 -> 0.382) — the reward term legitimately moves
  the policy away from the reference, so "KL must decrease after a
  drift" is the wrong property to assert; (3) final form, mirroring
  the Day 15 test: from the reference, beta=+0.5 keeps the symmetric
  KL small while beta=-0.5 (the flipped sign) drives it far away —
  the sign is discriminated by construction.
- (tests) The KL and rewarded-logprob tests need a warm-start that
  actually yields mixed groups: 60 SFT steps on 4 problems just
  memorises them (all rewards 0 on fresh problems). Add-only SFT
  (300 steps, 200 problems, d=32) reaches mean reward 0.27 with 7/8
  mixed groups and is what both tests use.
- (calibration) SFT 1200 steps (loss 0.75): held-out greedy 0.333,
  sampled 0.156, per-class add 0.28 / sub 0.19 / prec 0.08 / mod 0.22 /
  mix3 0.00 — undertrained, mix3 has no signal at all. SFT 2500
  steps (loss 0.31, ~5.5 min): greedy 0.579, sampled 0.382,
  mixed_frac 0.671, per-class add 0.57 / sub 0.59 / prec 0.23 /
  mod 0.37 / mix3 0.16. The grid uses 2500 steps.
- (teacher) Memory is per-class last-3 decisions, not a global
  last-3 (a global cap evicts other classes' edits after one update).
- (teacher) `instance-only` mode needs `enable_class=False`: without
  it the teacher's update would also deploy class strategies.
- (extensions) E2 correlation first returned None: the code paired
  consecutive blocks (10 steps apart) with the eval curve (sampled
  every 20 steps), so no block pair matched two curve points. Fixed
  to pair consecutive curve intervals with the mean mixed fraction of
  the blocks inside each interval; corr = 0.358 over 45 intervals.
- (teacher, known limitation) The memory-revert rule fires when
  v_none drops > 0.05 after an adopt/remove, but V_C is only 12
  tasks/class x 2 rollouts, so v_none has sampling noise ~0.09 and
  the threshold is inside the noise. Spurious reverts are possible;
  they are self-correcting (the teacher can re-adopt next block) and
  are recorded in the decisions log, but a 0.15 threshold would be
  cleaner. Left as-is for this night; flagged for a future run.



## 5. Results (3-seed means, 120 GRPO steps, 240 held-out problems)

Warm-start (2500 SFT steps): greedy 0.533, sampled 0.319.

| arm | greedy | sampled |
|---|---|---|
| grpo | 0.599 | 0.488 |
| autoscaffold | 0.594 | 0.493 |
| fixed | 0.583 | 0.471 |
| novalid | 0.581 | 0.489 |
| qmax1 | 0.592 | 0.491 |

Per-class sampled: grpo add 0.815 / sub 0.742 / prec 0.377 / mod 0.373 /
mix3 0.134; autoscaffold 0.819 / 0.760 / 0.392 / 0.366 / 0.128; fixed
0.769 / 0.694 / 0.385 / 0.369 / 0.136; novalid 0.806 / 0.733 / 0.411 /
0.383 / 0.110; qmax1 0.807 / 0.753 / 0.396 / 0.362 / 0.135.
Learning curves track: step 20 grpo 0.556 vs autoscaffold 0.562; step
60 0.589 vs 0.590; step 120 0.599 vs 0.594.

HEADLINE IS A NULL: AutoScaffold is indistinguishable from plain GRPO.
The paper's gains (+7.2 ALFWorld, +4.0 QA) and all three ablations
(fixed −15.3, no-validation −14.1, q_max=1 below baseline) do NOT
reproduce here. What reproduces is the first half of the mechanism:
while shown, scaffolds lift V_C success by ~+0.05 on average — too
small to change the learned policy in 120 steps.

Teacher behaviour (autoscaffold arm): 9 adopts, 34 validation
rejections, 3 removals, 0 memory reverts across 3 seeds. The teacher
proposes often (diagnosis fires) but validation rarely confirms — the
adopt bar (v_new > v_cur on 12 tasks x 2 rollouts) is strict relative
to the noise. The rejected scaffolds were harmless, not harmful
(novalid = autoscaffold). Instance prefixes: ~280 of 600 training tasks
carry one at the end of training.

Extensions (results/extensions.json):
- E1: scaffolded eval hurts — step 120 grpo plain 0.490 vs scaffolded
  0.395; autoscaffold 0.493 vs 0.388. Gap concentrated in add (−0.24)
  and prec (−0.21).
- E2: measured mixed fraction 0.539 vs u(class-mean p) 0.841
  (MAE 0.303); per class add 0.76/0.91, sub 0.71/0.92, prec 0.44/0.88,
  mod 0.56/0.97, mix3 0.23/0.53. Prop. 1 holds per task, not per class
  mean (within-class heterogeneity). Corr(mixed frac, next gain) =
  0.358 over 45 intervals — weak positive, the predicted direction.
- E3 fixed-sample teacher: 0.617/0.509, 0.600/0.492 (mean 0.609/0.501),
  nominally best but within seed noise of AutoScaffold. Interactive
  queries add nothing measurable.
- E4 transfer: mix3 held-out greedy 0.188/0.125 for both grpo and
  autoscaffold — identical, no transfer.
- E5 instance-only: 0.621/0.519, 0.579/0.485 (mean 0.600/0.502) —
  within noise of everything else.
- E6 strategy-vs-cue (step-120 checkpoints, means over both arms,
  3 seeds): add plain 0.815 / own 0.561 / wrong 0.007; prec plain
  0.375 / own 0.163 / wrong 0.069. The model treats the scaffold as a
  class cue: a mismatched strategy collapses accuracy to ~0. The E2
  correlation fix (curve steps are every 20, blocks every 10 — the
  first version paired block k with block k+1 and found no matching
  curve points, returning corr None) is documented in §4.


## 6. Files and runtime

- Grid: 15 arms x 120 steps, ~35 min total (each arm ~90-120 s);
  SFT warm-starts 2500 steps ~5.5 min each, one per seed, shared
  across arms. Extensions: E1 ~7 min (24 checkpoint evals), E3/E4/E5
  8 training runs ~2 min each, E6 ~1 min.
- Checkpoints in checkpoints/ are NOT shipped (gitignored); results/
  JSON files are shipped and cross-footed by scripts/crossfoot.py.
