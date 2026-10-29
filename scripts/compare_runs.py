"""Summarize FlashREINFORCE runs from their train.log (Molt's per-step status lines).

python scripts/compare_runs.py outputs/qwen-math-1.5b outputs/qwen-math-1.5b-turn_linear [--upto 1200]

Steps beyond --upto are ignored so runs at different progress compare on the same budget.
A restarted log keeps the last record of each step.
"""

import argparse
import ast
import re
import statistics
from pathlib import Path

STEP = re.compile(r"Global step (\d+): (\{.*\})")
EVAL = re.compile(r"Eval at step (\d+): (\{.*\})")


def load(run: Path):
    steps, evals = {}, {}
    for line in (run / "train.log").open(errors="ignore"):
        for pattern, table in ((STEP, steps), (EVAL, evals)):
            m = pattern.search(line)
            if m:
                table[int(m.group(1))] = ast.literal_eval(m.group(2))
    return steps, evals


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def summarize(run: Path, upto: int):
    steps, evals = load(run)
    steps = {k: v for k, v in sorted(steps.items()) if k <= upto}
    evals = {k: v for k, v in sorted(evals.items()) if k <= upto}
    if not steps:
        return {"run": run.name, "steps": 0}
    rows = list(steps.values())
    grad = [r["actor_grad_norm"] for r in rows]
    out = {
        "run": run.name,
        "steps": max(steps),
        "reward(last100)": mean([r["reward"] for r in rows[-100:]]),
        "grad_norm mean": mean(grad),
        "grad_norm std": statistics.pstdev(grad),
        "grad_norm |diff| mean": mean([abs(a - b) for a, b in zip(grad[1:], grad)]),
        # Median: the step after an eval also absorbs the eval's wall clock.
        "step_time s (median)": statistics.median([r["timing/step_total"] for r in rows]),
        "make_experience s": mean([r["timing/make_experience"] for r in rows]),
        "response_len": mean([r["response_length"] for r in rows]),
    }
    for key in ("baseline/mse", "baseline/pred_std", "baseline/adv_pos_frac", "baseline/adv_neg_frac"):
        if key in rows[-1]:
            out[key] = mean([r[key] for r in rows if key in r])
    if evals:
        last = max(evals)
        for key, value in evals[last].items():
            if key.endswith("_pass1"):
                out[f"{key}@{last}"] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--upto", type=int, default=10**9, help="last global step to include")
    args = parser.parse_args()
    summaries = [summarize(run, args.upto) for run in args.runs]
    keys = list(dict.fromkeys(k for s in summaries for k in s))
    width = max(map(len, keys))
    for key in keys:
        cells = [s.get(key, "-") for s in summaries]
        print(f"{key:<{width}}  " + "  ".join(f"{c:>14.4f}" if isinstance(c, float) else f"{c!s:>14}" for c in cells))


if __name__ == "__main__":
    main()
