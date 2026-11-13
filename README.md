# autoscaffold-lab

A CPU-only, NumPy-only reproduction and extension of **AutoScaffold**
(Zhang et al., arXiv:2611.14515, 10 Nov 2026): a frozen teacher turns a
student's RL training experience into adaptive scaffolds — class-level
strategies and instance-level solution prefixes — injected into training
prompts, validated on held-out tasks, and withdrawn as the student
improves. Scaffolds are removed at evaluation, so the measured gain is
the student's own capability.

## What the paper claims

RL with verifiable rewards (RLVR) collapses each rollout to one scalar,
wasting the information in the trajectory. AutoScaffold puts a teacher in
the loop: every K = 10 steps it inspects the student's experience through
query tools (class statistics, individual trajectories, rollout groups,
memory of past decisions), diagnoses bottlenecks, and writes two kinds of
scaffold — strategies shared across a task class, injected into a
fraction q[C] ≤ q_max = 0.5 of rollout groups, and hints for individual
hard tasks (in math: a solution prefix revealing the first r_x of the
reference solution). Proposed class scaffolds are validated on held-out
tasks V_C (success with none / current / proposed) before deployment, and
support is withdrawn once unscaffolded success exceeds 50%. The theory
(Appendix A) says group-relative methods learn only from mixed-outcome
groups: u(p) = 1 − (1−p)^G − p^G, so scaffolds help by moving a class's
success rate toward 1/2, and instance scaffolds bring p = 0 tasks into
training at all. Reported gains (scaffold-free eval): +7.2 ALFWorld,
+4.0 search QA, +1.4–2.0 math, +3–5 KernelBench. Ablations: no validation
−14.1 points; fixed skill bank −15.3; q_max = 1 (scaffold every group)
below the RL baseline.

## What this repo does

The same protocol on a tiny CPU testbed: a 1-layer d=64 causal
transformer (NumPy, manual backprop) doing multi-step arithmetic with a
verifiable binary reward, trained by GRPO (G=8, clip 0.2, k1 KL β=0.02,
lr 3e-4) from an SFT warm-start. Five task classes, each with its own
designed failure mode:

| class | form | failure mode the strategy addresses |
|---|---|---|
| add | a+b+c | carry slip in the first step |
| sub | a−b−c | regrouping as a−(b−c) |
| prec | a+b\*c / a\*b+c | left-to-right evaluation (precedence) |
| mod | a\*b mod m | forgetting / misplacing the mod |
| mix3 | a op b op c op d | skipping a written intermediate step |

Response format: `<s1> = <s2> = <ans> EOS` (two segments, three for
mix3). The teacher is a deterministic diagnostician implementing the
paper's decision protocol exactly (evidence queries under the same
B = 15 budget, failure taxonomy, held-out validation, q adaptation,
memory of the last three decisions per class, instance prefixes).
**Deviation, stated plainly:** the paper's teacher is GPT-5.5 writing
free-text scaffolds; no LLM is available in this lab, so the teacher
selects from a fixed strategy library and writes solution prefixes from
verified reference derivations. The protocol — diagnose, propose,
validate, deploy, remember — is what is reproduced, not the prose.

## Results (3-seed means, 120 GRPO steps, 240 held-out problems)

The warm-start greedy 0.533 (3-seed mean).

| arm | greedy | sampled |
|---|---|---|
| grpo | 0.599 | 0.488 |
| autoscaffold | 0.594 | 0.493 |
| fixed | 0.583 | 0.471 |
| novalid | 0.581 | 0.489 |
| qmax1 | 0.592 | 0.491 |

Final sampled pass@1 per class (3-seed means):

| arm | class | sampled |
|---|---|---|
| grpo | add | 0.815 |
| grpo | sub | 0.742 |
| grpo | prec | 0.377 |
| grpo | mod | 0.373 |
| grpo | mix3 | 0.134 |
| autoscaffold | add | 0.819 |
| autoscaffold | sub | 0.760 |
| autoscaffold | prec | 0.392 |
| autoscaffold | mod | 0.366 |
| autoscaffold | mix3 | 0.128 |
| fixed | add | 0.769 |
| fixed | sub | 0.694 |
| fixed | prec | 0.385 |
| fixed | mod | 0.369 |
| fixed | mix3 | 0.136 |
| novalid | add | 0.806 |
| novalid | sub | 0.733 |
| novalid | prec | 0.411 |
| novalid | mod | 0.383 |
| novalid | mix3 | 0.110 |
| qmax1 | add | 0.807 |
| qmax1 | sub | 0.753 |
| qmax1 | prec | 0.396 |
| qmax1 | mod | 0.362 |
| qmax1 | mix3 | 0.135 |

Teacher decisions (autoscaffold arm): seed 0 adopts 3, seed 1 adopts 4,
seed 2 adopts 2. Across the three seeds the teacher proposed class
scaffolds 43 times and validation rejected 34 of them; it removed a
scaffold 3 times and never reverted from memory.

**Headline: null.** AutoScaffold (0.594 greedy / 0.493 sampled) is
statistically indistinguishable from plain GRPO (0.599 / 0.488); the
learning curves track each other at every checkpoint (step 20: 0.562 vs
0.556; step 60: 0.590 vs 0.589; step 120: 0.594 vs 0.599). GRPO alone
lifts the warm-start from 0.533 / 0.319 to ~0.60 / 0.49; the scaffolds
add nothing measurable on scaffold-free eval. The paper's +7.2
ALFWorld / +4.0 QA gains do **not** reproduce on this testbed. The
ablations do not reproduce either: the fixed oracle scaffold is
slightly *worse* than baseline (0.583 vs 0.599 — the paper's direction
but a far smaller gap than its −15.3), no-validation is the same as
AutoScaffold (0.581 vs 0.594, not the paper's −14.1), and q_max = 1
shows no dependence penalty (0.592 vs 0.599, not below baseline) —
because the teacher withdraws q as classes improve. What does
reproduce is the mechanism's first half: while a scaffold is shown, the
student's success on V_C is higher by ~0.05 on average
(v_cur − v_none over deployed scaffolds). That lift is real but too
small to change what the policy learns in 120 steps.

## Extensions

- **E1 — scaffold dependence:** at step 120, evaluating with the
  oracle class strategy *hurts*: E1 step-120 grpo plain sampled 0.490, scaffolded 0.395; E1 step-120 autoscaffold plain sampled 0.493, scaffolded 0.388. The gap is concentrated in add (−0.24) and prec
  (−0.21); sub/mod/mix3 are ~neutral. The student does not benefit
  from the strategy text at eval — see E6.
- **E2 — Props. 1/3 check:** E2 MAE 0.303, corr 0.358. The measured
  mixed-group fraction (0.539) sits 0.30 below the theoretical u(p)
  computed from the class-mean success rate (0.841), in every class
  (add 0.76 vs 0.91, prec 0.44 vs 0.88, mix3 0.23 vs 0.53). Prop. 1
  holds per task, not per class mean: success probability varies
  within a class, so most groups are degenerate even when the class
  mean sits near 1/2. The correlation between a block's mixed fraction
  and the next interval's gain is weakly positive (r = 0.358, 45
  intervals), the direction Prop. 1 predicts, but the gains are small
  and noisy.
- **E3 — fixed failure sample:** the teacher sees only 4 failed
  trajectories per weak class instead of interactive queries
  (SkillRL-style). | e3-fixed-sample/seed0 | 0.617 | 0.509 |
  | e3-fixed-sample/seed1 | 0.600 | 0.492 | — mean 0.609 / 0.501,
  nominally the best arm but within seed noise of AutoScaffold
  (0.594 / 0.493). Interactive queries add no measurable value here;
  the paper's −8.6 ablation does not reproduce.
- **E4 — transfer:** train on add/sub/prec/mod only; scaffold-free
  eval on the unseen mix3 class.
  | e4-transfer-grpo/seed0 mix3 | 0.188 |
  | e4-transfer-grpo/seed1 mix3 | 0.125 |
  | e4-transfer-autoscaffold/seed0 mix3 | 0.188 |
  | e4-transfer-autoscaffold/seed1 mix3 | 0.125 | — identical.
  Class strategies do not transfer to an unseen class, as expected:
  any benefit is class-local.
- **E5 — instance-only:** class scaffolds disabled, solution prefixes
  only. | e5-instance-only/seed0 | 0.621 | 0.519 |
  | e5-instance-only/seed1 | 0.579 | 0.485 | — mean 0.600 / 0.502,
  again within noise of AutoScaffold and plain GRPO. Instance prefixes
  alone carry no measurable edge either, despite ~280 of 600 training
  tasks carrying a prefix by the end of training.
- **E6 — strategy or class cue?** Step-120 checkpoints on add/prec
  with no scaffold, the class's own strategy, and a mismatched
  strategy (means over both arms, 3 seeds): E6 add plain 0.815, own 0.561, wrong 0.007; E6 prec plain 0.375, own 0.163, wrong 0.069.
  The model treats the scaffold as a class cue, not as advice: its own
  strategy hurts, and a mismatched strategy collapses accuracy to
  ~0.00–0.07 (the model emits the wrong response format for the cued
  class). A tiny model with absolute positions also pays a positional
  cost for the prepended tokens. This is a testbed limitation, not a
  property of the paper's method — but it is the mechanism that makes
  E1's scaffolded eval worse, and it bounds what a rule-based teacher
  can show on this scale.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest tests -q        # 25 tests
python scripts/demo.py           # ~1 min: warm-start + one teacher update
python scripts/run_grid.py       # 5 arms x 3 seeds x 120 steps (~35 min)
python scripts/run_extensions.py # E1-E5 (~20 min)
python scripts/run_e6.py         # E6: strategy-vs-cue diagnostic (~1 min)
python scripts/crossfoot.py --extensions  # README numbers vs results/*.json
```

`scripts/calibrate_sft.py` trains the warm-start (2500 SFT steps,
~5.5 min) and reports per-class accuracy. `results/grid.json` and
`results/extensions.json` hold every measured number; `scripts/crossfoot.py`
recomputes every README figure from them.

## Method notes

- SFT warm-start: 2500 steps, batch 32, lr 3e-3, label smoothing 0.05,
  on 2000 problems; prompts mix plain (60%), class-strategy (30%), and
  instance-prefix (10%) forms so the student learns to use scaffolds
  without depending on them.
- GRPO: batch 12 problems × G=8, temperature 1.0, one epoch per batch
  (ratio 1, clip vacuous), KL inside the advantage. Batches with no
  mixed-outcome group are skipped entirely — with all rewards equal,
  Adam would take a full-size step on the tiny KL gradient and drift
  the policy for no signal.
- Teacher: K=10, q_max=0.5, B=15, V_C = 12 tasks/class. Adopts a
  proposed scaffold only if v_new > v_cur on V_C; withdraws a scaffold
  once v_none > 0.5; halves q[C] on withdrawal, restores it while
  v_none < 0.2 and the scaffold helps. Instance prefixes grow while a
  task stays all-fail and shrink once it is solved; the final answer is
  never revealed.
- All numbers in this README are simulation numbers from this testbed,
  cross-footed against `results/*.json`. Paper numbers are quoted
  separately above and are not ours.

## Limitations

- Tiny model, synthetic arithmetic: the mechanism (scaffolds move p
  toward 1/2, validation filters harmful scaffolds, q_max < 1 prevents
  dependence) is what is tested, not the paper's absolute gains.
- The teacher is rule-based, not an LLM: it cannot invent a strategy
  outside its library, so the "automatic discovery" part of the paper
  is only partially reproduced.
- The student treats scaffolds as class cues, not advice (E6): a
  mismatched strategy collapses accuracy, and even the matching
  strategy hurts at eval. Prepended scaffolds also shift absolute
  positions, which a tiny model is sensitive to. A separator token or
  a scaffold placed after the expression would separate the content
  effect from the positional one.
- 3 seeds, 120 steps: small-sample; seed variance is reported, not hidden.
- The fixed arm uses the same strategy library as the teacher, so it
  tests adaptivity and validation, not the quality of hand-written
  scaffolds vs discovered ones.

## Sources

- Zhang, Hu, Peng, Li, Bhatt, Zeng & Hong, "AutoScaffold: AI-Guided
  Scaffolding for Reinforcement Learning of Language Models",
  arXiv:2611.14515 (10 Nov 2026). https://arxiv.org/abs/2611.14515
- Trending check (12 Nov 2026): GitHub trending / gitdiscover /
  findarepo / topopensource — skill packs and agent platforms dominate;
  no trending repo with a measurable one-night CPU weakness (see
  docs/RESEARCH.md §1).
- Held for a future night: Salloum & Ali, "Tree-Ising: Repairing
  Language Models with Bit-Level Ising Edits", arXiv:2611.14210;
  "Margin-Guided Token Credit for RL with Verifiable Rewards" (MCPO),
  arXiv:2611.14015.
