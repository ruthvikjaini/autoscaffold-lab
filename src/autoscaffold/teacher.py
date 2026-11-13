"""The deterministic teacher: AutoScaffold's decision protocol (paper
Algorithms 1-2, Sec. 3.2) with a rule-based diagnostician in place of
the paper's GPT-5.5 teacher.

The protocol is reproduced, not the prose: evidence is gathered through
the paper's query tools (Q_stat class statistics, Q_traj individual
trajectories, Q_group all G attempts on one task, plus a memory of the
last three decisions) under the same query budget B = 15; the teacher
diagnoses each class's dominant failure mode from the fixed taxonomy,
proposes class-level strategies from a fixed library, validates each
proposed class scaffold on held-out tasks V_C (success with none /
current / proposed, matched seeds) before deploying it, adapts the
injection probability q[C] (withdrawal once unscaffolded success
exceeds 50%), and writes instance-level solution prefixes for
persistently unsolved tasks. The deviation — a deterministic
diagnostician instead of a frozen LLM — is stated in the README.
"""

import numpy as np
from collections import Counter

from .tasks import prompt_tokens
from .vocab import CLASSES, STRATEGIES, STRATEGY_TARGETS


def task_key(problem):
    return (problem["class"], tuple(problem["expr"]))


class Teacher:
    def __init__(self, valid_problems, q_max=0.5, query_budget=15,
                 validate=True, interactive=True, instance=True,
                 enable_class=True):
        """valid_problems: held-out V_C tasks (list, excluded from training).

        q_max: cap on class-scaffold injection probability (paper 0.5).
        validate: run held-out validation before deploying a proposed
            class scaffold (False = the paper's no-validation ablation).
        interactive: gather evidence through the query tools (False =
            fixed failure sample only, SkillRL-style ablation).
        instance: maintain instance-level solution prefixes.
        enable_class: deploy class-level strategies (False = the
            instance-only ablation: prefixes only).
        """
        self.valid = valid_problems
        self.q_max = q_max
        self.query_budget = query_budget
        self.validate = validate
        self.interactive = interactive
        self.instance = instance
        self.strategies = {c: None for c in CLASSES}  # deployed text
        self.q = {c: q_max for c in CLASSES}
        self.prefix_frac = {}  # task_key -> revealed fraction
        self.enable_class = enable_class
        self.memory = {c: [] for c in CLASSES}  # class -> last 3 decisions
        self.decisions = []  # full decision log for the paper trail

    # ------------------------------------------------------------ prompts
    def build_prompt(self, problem, rng):
        """Build one training prompt: Bernoulli(q[C]) class scaffold +
        the task's instance prefix (if any). Returns (prompt, info)."""
        cls = problem["class"]
        z = rng.random() < self.q[cls]
        strategy = self.strategies[cls] if z else None
        frac = self.prefix_frac.get(task_key(problem), 0.0)
        prompt = prompt_tokens(problem, strategy=strategy, prefix_frac=frac)
        return prompt, {"scaffolded": bool(strategy), "prefix_frac": frac}

    # ------------------------------------------------------------ evidence
    def q_stat(self, block_log):
        """Per-class success with/without scaffold, all-fail / all-succeed
        group counts. Returns (stats, queries_used)."""
        stats = {}
        for cls in CLASSES:
            groups = [g for g in block_log if g["class"] == cls]
            if not groups:
                stats[cls] = None
                continue
            sc = [g for g in groups if g["scaffolded"]]
            un = [g for g in groups if not g["scaffolded"]]
            stats[cls] = {
                "n": len(groups),
                "success": float(np.mean([g["mean_reward"] for g in groups])),
                "success_scaffolded": (float(np.mean([g["mean_reward"] for g in sc]))
                                       if sc else None),
                "success_plain": (float(np.mean([g["mean_reward"] for g in un]))
                                 if un else None),
                "n_all_fail": sum(1 for g in groups if g["mean_reward"] == 0.0),
                "n_all_succeed": sum(1 for g in groups if g["mean_reward"] == 1.0),
                "n_mixed": sum(1 for g in groups
                               if 0.0 < g["mean_reward"] < 1.0),
            }
        return stats, 1

    def q_traj(self, block_log, cls, outcome="fail", k=4, rng=None):
        """Up to k individual trajectories from class `cls` with the
        given outcome. Returns (trajectories, queries_used)."""
        rng = rng or np.random.default_rng(0)
        cands = []
        for g in block_log:
            if g["class"] != cls:
                continue
            for tr in g["trajs"]:
                if (outcome == "fail" and tr["reward"] == 0.0) or \
                   (outcome == "succeed" and tr["reward"] == 1.0):
                    cands.append(tr)
        if not cands:
            return [], 1
        idx = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
        return [cands[i] for i in idx], 1

    def q_group(self, block_log, cls, rng=None):
        """All G attempts on one all-fail task of class `cls`."""
        rng = rng or np.random.default_rng(0)
        cands = [g for g in block_log
                 if g["class"] == cls and g["mean_reward"] == 0.0]
        if not cands:
            return None, 1
        return cands[int(rng.integers(0, len(cands)))], 1

    # ------------------------------------------------------------ diagnose
    @staticmethod
    def diagnose(trajs):
        """Dominant failure mode among failed trajectories, or None."""
        modes = [t["mode"] for t in trajs if t["reward"] == 0.0]
        if not modes:
            return None, Counter()
        counts = Counter(modes)
        # Fixed priority order breaks ties deterministically.
        priority = ["precedence", "grouping", "mod", "skipped-step",
                    "wrong-intermediate", "wrong-final", "format"]
        best = max(counts.items(),
                   key=lambda kv: (kv[1], -priority.index(kv[0])))
        return best[0], counts

    # ------------------------------------------------------------ validate
    def _eval_on_valid(self, model, cls, strategy, n=2, seed=0, greedy=False):
        """Success rate of `model` on V_C with the given class scaffold
        (None = no scaffold). Sampled rollouts, matched seeds."""
        from .evaluate import evaluate_group
        problems = [p for p in self.valid if p["class"] == cls]
        rng = np.random.default_rng(seed)
        return evaluate_group(model, problems, group=n, rng=rng,
                              strategy=strategy, temperature=1.0,
                              greedy=greedy)

    # ------------------------------------------------------------ update
    def update(self, model, block_log, block, rng):
        """One teacher update at the end of a block (paper Algorithm 2).

        Returns a dict describing the decisions taken (also appended to
        self.decisions).
        """
        queries = 0
        record = {"block": block, "queries": 0, "classes": {}}

        # --- gathering evidence (initial: V_C success none/current) ---
        evidence = {}
        for cls in CLASSES:
            v_none = self._eval_on_valid(model, cls, None, seed=1000 + block)
            v_cur = self._eval_on_valid(
                model, cls, self.strategies[cls], seed=1000 + block) \
                if self.strategies[cls] else None
            evidence[cls] = {"v_none": v_none, "v_cur": v_cur}
        queries += 1  # Q_stat counts as one query round
        stats, _ = self.q_stat(block_log)

        # --- memory: revert the last edit if it coincided with a drop ---
        for cls in CLASSES:
            entries = self.memory[cls]
            if not entries:
                continue
            entry = entries[-1]
            if entry["decision"] in ("adopt", "remove") and \
                    evidence[cls]["v_none"] < entry["v_none"] - 0.05:
                self.strategies[cls] = entry.get("previous_strategy")
                record["classes"].setdefault(
                    cls, {})["memory_revert"] = True

        # --- per-class diagnosis, proposal, validation, deployment ---
        for cls in CLASSES:
            st = stats[cls]
            ev = evidence[cls]
            crec = {"v_none": ev["v_none"], "v_cur": ev["v_cur"]}
            record["classes"][cls] = crec
            if st is None:
                continue
            weak = ev["v_none"] < 0.5
            if self.interactive:
                trajs, _ = self.q_traj(block_log, cls, "fail", k=4, rng=rng)
                queries += 1
                grp, _ = self.q_group(block_log, cls, rng=rng)
                queries += 1
                if grp is not None:
                    trajs = trajs + grp["trajs"]
                if queries > self.query_budget:
                    trajs = trajs[:4]
            else:
                # Fixed failure sample (SkillRL-style): 4 failed
                # trajectories, no statistics, no groups, no memory.
                trajs, _ = self.q_traj(block_log, cls, "fail", k=4, rng=rng)
                queries += 1
            diagnosis, counts = self.diagnose(trajs)
            crec["diagnosis"] = diagnosis
            crec["failure_counts"] = dict(counts)
            crec["n_block_groups"] = st["n"]
            crec["block_success"] = st["success"]

            # Propose: deploy the class strategy only when the dominant
            # failure is one the strategy addresses and the class is
            # weak; withdraw once the student succeeds without help.
            proposal = None
            if self.enable_class and weak \
                    and diagnosis in STRATEGY_TARGETS[cls] \
                    and self.strategies[cls] is None:
                proposal = STRATEGIES[cls]
            if proposal is not None:
                crec["proposal"] = "adopt"
                if self.validate:
                    v_new = self._eval_on_valid(model, cls, proposal,
                                                seed=1000 + block)
                    crec["v_new"] = v_new
                    if v_new > (ev["v_cur"] if ev["v_cur"] is not None
                                else ev["v_none"]):
                        self._deploy(cls, proposal, ev["v_none"], "adopt",
                                     record, crec)
                    else:
                        crec["decision"] = "rejected-by-validation"
                else:
                    self._deploy(cls, proposal, ev["v_none"], "adopt",
                                 record, crec)
            elif self.strategies[cls] is not None and ev["v_none"] > 0.5:
                # Withdrawal: the student no longer needs the scaffold.
                crec["proposal"] = "remove"
                if self.validate:
                    if ev["v_none"] >= (ev["v_cur"] or 0.0):
                        self._deploy(cls, None, ev["v_none"], "remove",
                                     record, crec)
                    else:
                        crec["decision"] = "kept-despite-withdrawal-signal"
                else:
                    self._deploy(cls, None, ev["v_none"], "remove",
                                 record, crec)
            else:
                crec["decision"] = "unchanged"

            # --- injection probability: withdraw as the student improves
            if ev["v_none"] > 0.5:
                self.q[cls] = max(0.0, self.q[cls] * 0.5)
            elif ev["v_none"] < 0.2 and ev["v_cur"] is not None \
                    and ev["v_cur"] > ev["v_none"]:
                self.q[cls] = self.q_max
            crec["q"] = self.q[cls]

        # --- instance-level prefixes for persistently unsolved tasks ---
        if self.instance:
            n_prefix = self._update_instance_prefixes(block_log)
            record["n_instance_prefixes"] = n_prefix
        record["queries"] = queries
        self.decisions.append(record)
        return record

    def _deploy(self, cls, strategy, v_none, decision, record, crec):
        prev = self.strategies[cls]
        self.strategies[cls] = strategy
        crec["decision"] = decision
        self.memory[cls].append(
            {"block": record["block"], "class": cls,
             "diagnosis": crec.get("diagnosis"),
             "decision": decision, "v_none": v_none,
             "previous_strategy": prev})
        self.memory[cls] = self.memory[cls][-3:]

    def _update_instance_prefixes(self, block_log):
        """Adjust per-task solution prefixes from the block log.

        Tasks all-fail in unscaffolded groups get (or keep) a prefix;
        the revealed fraction grows while the task stays all-fail and
        shrinks once the task is solved. Returns the number of tasks
        currently carrying a prefix.
        """
        by_task = {}
        for g in block_log:
            if g["scaffolded"]:
                continue
            key = task_key(g["problem"])
            by_task.setdefault(key, []).append(g["mean_reward"])
        for key, rewards in by_task.items():
            all_fail = all(r == 0.0 for r in rewards)
            cls = key[0]
            n_seg = 3 if cls == "mix3" else 2
            step = 1.0 / n_seg
            cur = self.prefix_frac.get(key, 0.0)
            if all_fail:
                if cur == 0.0:
                    self.prefix_frac[key] = step
                elif cur < (n_seg - 1) / n_seg - 1e-9:
                    self.prefix_frac[key] = min(cur + step,
                                                (n_seg - 1) / n_seg)
            else:
                if cur > 0.0:
                    self.prefix_frac[key] = max(0.0, cur - step)
                    if self.prefix_frac[key] == 0.0:
                        del self.prefix_frac[key]
        return len(self.prefix_frac)
