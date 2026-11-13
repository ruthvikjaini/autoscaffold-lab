"""Demo: train a small warm-start, show one teacher update, and print
sample scaffolded vs plain rollouts. Runs in ~1 minute.
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from autoscaffold.evaluate import evaluate
from autoscaffold.model import Transformer
from autoscaffold.sft import train_sft
from autoscaffold.tasks import prompt_tokens, sample_problems
from autoscaffold.teacher import Teacher
from autoscaffold.vocab import STRATEGIES, VOCAB_SIZE, decode


def main():
    rng = np.random.default_rng(0)
    train = sample_problems(200, seed=100)
    heldout = sample_problems(10, seed=500)
    model = Transformer(VOCAB_SIZE, max_len=64, d=64, seed=0)
    train_sft(model, train, steps=150, batch=32, lr=3e-3, rng=rng,
              log_every=0)
    print("warm-start (150 SFT steps) held-out greedy:",
          evaluate(model, heldout, group=1, rng=rng, greedy=True)["greedy"])

    # one teacher update on a fake all-fail prec block
    from autoscaffold.grpo import rollout_batch
    valid = sample_problems(2, seed=400)
    teacher = Teacher(valid)
    problems = [p for p in sample_problems(12, seed=300)
                if p["class"] == "prec"][:4]
    prompts = [prompt_tokens(p) for p in problems]
    trajs = rollout_batch(model, problems, prompts, group=8, rng=rng)
    block_log = []
    for gi, p in enumerate(problems):
        g = [t for t in trajs if t["group"] == gi]
        block_log.append({"problem": p, "class": "prec",
                          "mean_reward": float(np.mean([t["reward"] for t in g])),
                          "scaffolded": False, "trajs": g})
    rec = teacher.update(model, block_log, block=1, rng=rng)
    print("teacher decision for prec:",
          rec["classes"]["prec"].get("decision"),
          "| diagnosis:", rec["classes"]["prec"].get("diagnosis"))

    # show a scaffolded vs plain rollout on one prec problem
    p = problems[0]
    print("problem:", decode(prompt_tokens(p)))
    print("strategy:", decode(STRATEGIES["prec"]))
    for label, prompt in (("plain", prompt_tokens(p)),
                          ("scaffolded", prompt_tokens(
                              p, strategy=STRATEGIES["prec"]))):
        gens = model.generate([prompt], max_new=24, greedy=True, rng=rng)
        print(f"  {label}: {decode(gens[0])}  (answer {p['answer']})")


if __name__ == "__main__":
    main()
