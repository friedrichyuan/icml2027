"""Compare budget-aware task-sampling runs at equal resource budgets.

Reads <run>/task_sampler/{eval,trajectories,refresh}.jsonl and prints, per run:
  1. every eval point with the budget actually spent (policy tokens, wall time net of eval,
     environment interactions, new rollouts) and pass@1 per benchmark;
  2. mean pass@1 interpolated at shared budget points (tokens / wall / rollouts) and the
     normalized area under the pass@1-vs-tokens curve;
  3. eval pass@1 per task bucket at the last eval;
  4. training-side per-bucket share / success / cost / truncation and the final p / p0.

  python scripts/summarize_task_sampler.py outputs/qwen-math-1.5b-TS_*_s42 [--json out.json]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(path):
    d = os.path.join(path, "task_sampler")
    evals = {}
    for row in read_jsonl(os.path.join(d, "eval.jsonl")):
        evals[row["step"]] = row  # a resumed run may repeat an eval; keep the latest
    evals = sorted(evals.values(), key=lambda r: r["budget/policy_tokens"])
    for row in evals:
        bench = [
            v for k, v in row.items() if re.fullmatch(r"eval_(?!bucket_).+_pass1", k)
        ]
        row["mean_pass1"] = float(np.mean(bench)) if bench else float("nan")
        row["net_wall_hours"] = (
            row["budget/wall_seconds"] - row["budget/eval_seconds"]
        ) / 3600
    return {
        "name": os.path.basename(path.rstrip("/")),
        "evals": evals,
        "trajectories": read_jsonl(os.path.join(d, "trajectories.jsonl")),
        "refresh": read_jsonl(os.path.join(d, "refresh.jsonl")),
    }


def table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    def line(row):
        return "  ".join(str(x).rjust(w) for x, w in zip(row, widths))

    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
    print()


def interp(run, axis, x):
    xs = np.array([r[axis] for r in run["evals"]], dtype=float)
    ys = np.array([r["mean_pass1"] for r in run["evals"]], dtype=float)
    if len(xs) == 0 or x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--json", help="also write the aligned comparison here")
    args = p.parse_args()
    runs = [
        load_run(r)
        for pattern in args.runs
        for r in sorted(glob.glob(pattern)) or [pattern]
    ]
    runs = [r for r in runs if r["evals"]]
    if not runs:
        raise SystemExit("no task_sampler/eval.jsonl found")

    for run in runs:
        print(f"=== {run['name']}: eval points ===")
        bench = sorted(
            {
                k
                for r in run["evals"]
                for k in r
                if re.fullmatch(r"eval_(?!bucket_).+_pass1", k)
            }
        )
        table(
            ["step", "tokens(M)", "wall-eval(h)", "env_steps", "rollouts", "mean"]
            + [b[5:-6] for b in bench],
            [
                [
                    r["step"],
                    f"{r['budget/policy_tokens'] / 1e6:.1f}",
                    f"{r['net_wall_hours']:.2f}",
                    int(r["budget/env_interactions"]),
                    int(r["budget/rollouts"]),
                    f"{r['mean_pass1']:.4f}",
                ]
                + [f"{r.get(b, float('nan')):.4f}" for b in bench]
                for r in run["evals"]
            ],
        )

    aligned = {}
    for axis, label, scale in (
        ("budget/policy_tokens", "policy tokens (M)", 1e6),
        ("net_wall_hours", "wall time net of eval (h)", 1),
        ("budget/rollouts", "new rollouts (k)", 1e3),
    ):
        # Shared range: up to the smallest final budget, at each run's eval budgets.
        top = min(r["evals"][-1][axis] for r in runs)
        points = sorted(
            {r_[axis] for r in runs for r_ in r["evals"] if 0 < r_[axis] <= top}
        )
        if not points:
            continue
        print(f"=== mean pass@1 at equal {label} ===")
        rows = []
        for run in runs:
            vals = [interp(run, axis, x) for x in points]
            aligned.setdefault(run["name"], {})[axis] = dict(zip(points, vals))
            rows.append(
                [run["name"]] + ["-" if v is None else f"{v:.4f}" for v in vals]
            )
        table(["run"] + [f"{x / scale:.2f}" for x in points], rows)

    print("=== normalized AUC of mean pass@1 over policy tokens (shared range) ===")
    lo = max(r["evals"][0]["budget/policy_tokens"] for r in runs)
    hi = min(r["evals"][-1]["budget/policy_tokens"] for r in runs)
    rows = []
    for run in runs:
        if hi > lo:
            grid = np.linspace(lo, hi, 200)
            auc = float(np.mean([interp(run, "budget/policy_tokens", x) for x in grid]))
        else:
            auc = float("nan")
        aligned.setdefault(run["name"], {})["auc_tokens"] = auc
        rows.append([run["name"], f"{lo / 1e6:.1f}-{hi / 1e6:.1f}M", f"{auc:.4f}"])
    table(["run", "range", "AUC"], rows)

    last_keys = {k for run in runs for k in run["evals"][-1]}
    buckets = sorted(
        k[12:-6]
        for k in last_keys
        if k.startswith("eval_bucket_") and k.endswith("_pass1")
    )
    if buckets:
        print("=== eval pass@1 per task bucket at the last eval ===")
        table(
            ["run", "tokens(M)"] + buckets,
            [
                [run["name"], f"{run['evals'][-1]['budget/policy_tokens'] / 1e6:.1f}"]
                + [
                    f"{run['evals'][-1].get(f'eval_bucket_{b}_pass1', float('nan')):.4f}"
                    for b in buckets
                ]
                for run in runs
            ],
        )

    for run in runs:
        trajs = [t for t in run["trajectories"] if "policy_token_count" in t]
        if not trajs:
            continue
        stats = defaultdict(
            lambda: {"n": 0, "success": 0, "tokens": 0, "truncated": 0, "infra": 0}
        )
        for t in run["trajectories"]:
            s = stats[t["bucket_id"]]
            if "policy_token_count" not in t:
                s["infra"] += 1
                continue
            s["n"] += 1
            s["success"] += t["success"]
            s["tokens"] += t["policy_token_count"]
            s["truncated"] += t["termination_reason"] == "truncated"
        final = {
            b["bucket_id"]: b
            for b in (run["refresh"][-1]["buckets"] if run["refresh"] else [])
        }
        total = sum(s["n"] for s in stats.values())
        print(f"=== {run['name']}: training rollouts per bucket ===")
        table(
            [
                "bucket",
                "share",
                "p0",
                "final p/p0",
                "success",
                "tokens/traj",
                "truncated",
                "infra",
            ],
            [
                [
                    b,
                    f"{s['n'] / total:.3f}",
                    f"{final[b]['p0']:.3f}" if b in final else "-",
                    f"{final[b]['probability_ratio']:.2f}" if b in final else "-",
                    f"{s['success'] / max(s['n'], 1):.3f}",
                    f"{s['tokens'] / max(s['n'], 1):.0f}",
                    f"{s['truncated'] / max(s['n'], 1):.3f}",
                    s["infra"],
                ]
                for b, s in sorted(stats.items())
            ],
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"aligned": aligned, "evals": {r["name"]: r["evals"] for r in runs}},
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
