"""Summarize FlashREINFORCE runs from their train.log (Molt's per-step status lines).

python scripts/compare_runs.py outputs/qwen-math-1.5b outputs/qwen-math-1.5b-turn_linear [--upto 1200]

Steps beyond --upto are ignored so runs at different progress compare on the same budget: a
fixed number of rollouts (--upto), of cumulative generated tokens (--budget_tokens, millions) or
of training wall-clock (--budget_hours). A restarted log keeps the last record of each step.
"""

import argparse
import ast
import re
import statistics
from pathlib import Path

STEP = re.compile(r"Global step (\d+): (\{.*\})")
EVAL = re.compile(r"Eval at step (\d+): (\{.*\})")
LENGTH_KEYS = (
    "length/success_mean",
    "length/fail_mean",
    "length/success_truncated_rate",
    "length/fail_truncated_rate",
    "lw/w_mean",
    "lw/w_p10",
    "lw/w_p90",
    "lw/at_min_frac",
    "lw/at_max_frac",
    "lw/long_fail_share_unweighted",
    "lw/long_fail_share_weighted",
)


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


def budget_step(steps, budget_tokens, budget_hours):
    """The last step whose cumulative generated tokens (M) and wall-clock (h) are within budget."""
    tokens = hours = 0.0
    last = 0
    for step, r in sorted(steps.items()):
        tokens += r["rollout/response_length_mean"] * r["rollout/num_samples"] / 1e6
        hours += r["timing/step_total"] / 3600
        if tokens > budget_tokens or hours > budget_hours:
            break
        last = step
    return last


def summarize(run: Path, upto: int, budget_tokens: float, budget_hours: float):
    steps, evals = load(run)
    upto = min(upto, budget_step(steps, budget_tokens, budget_hours))
    steps = {k: v for k, v in sorted(steps.items()) if k <= upto}
    evals = {k: v for k, v in sorted(evals.items()) if k <= upto}
    if not steps:
        return {"run": run.name, "steps": 0}
    rows = list(steps.values())
    grad = [r["actor_grad_norm"] for r in rows]
    out = {
        "run": run.name,
        "steps": max(steps),
        "rollouts (cum)": sum(r["rollout/num_rollouts"] for r in rows),
        "gen tokens M (cum)": sum(r["rollout/response_length_mean"] * r["rollout/num_samples"] for r in rows) / 1e6,
        "wall-clock h (cum)": sum(r["timing/step_total"] for r in rows) / 3600,
        "reward(last100)": mean([r["reward"] for r in rows[-100:]]),
        "grad_norm mean": mean(grad),
        "grad_norm std": statistics.pstdev(grad),
        "grad_norm |diff| mean": mean([abs(a - b) for a, b in zip(grad[1:], grad)]),
        # Median: the step after an eval also absorbs the eval's wall clock.
        "step_time s (median)": statistics.median([r["timing/step_total"] for r in rows]),
        "make_experience s": mean([r["timing/make_experience"] for r in rows]),
        "response_len": mean([r["response_length"] for r in rows]),
        "truncated": mean([r["truncated"] for r in rows]),
    }
    for key in ("baseline/mse", "baseline/pred_std", "baseline/adv_pos_frac", "baseline/adv_neg_frac", *LENGTH_KEYS):
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
    parser.add_argument("--budget_tokens", type=float, default=float("inf"), help="cumulative generated tokens (M)")
    parser.add_argument("--budget_hours", type=float, default=float("inf"), help="cumulative step wall-clock (h)")
    args = parser.parse_args()
    summaries = [summarize(run, args.upto, args.budget_tokens, args.budget_hours) for run in args.runs]
    keys = list(dict.fromkeys(k for s in summaries for k in s))
    width = max(map(len, keys))
    for key in keys:
        cells = [s.get(key, "-") for s in summaries]
        print(f"{key:<{width}}  " + "  ".join(f"{c:>14.4f}" if isinstance(c, float) else f"{c!s:>14}" for c in cells))


if __name__ == "__main__":
    main()
