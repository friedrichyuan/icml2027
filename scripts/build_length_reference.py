#!/usr/bin/env python
"""Freeze a reference length sample for --algo.length_weight from a plain-Flash run's
--logger.rollout_lengths_file dump, and print the calibrated T0 / c for each alpha.

  python scripts/build_length_reference.py outputs/qwen-math-1.5b-batch_mean/rollout_lengths.jsonl \
      --steps 1:200 --out refs/flash_len_s1-200.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs-molt"))

import torch  # noqa: E402
from molt.trainer.algorithm.length_weight import LengthWeight  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="rollout_lengths.jsonl written by --logger.rollout_lengths_file")
    parser.add_argument("--steps", default=None, help="inclusive step range LO:HI (default: every step in the dump)")
    parser.add_argument("--out", required=True, help="reference JSON for --algo.length_weight.reference_file")
    parser.add_argument("--alphas", default="0.75,1.0,1.25")
    parser.add_argument("--raw_weight_min", type=float, default=0.5)
    parser.add_argument("--raw_weight_max", type=float, default=2.0)
    args = parser.parse_args()

    lo, hi = (int(x) for x in args.steps.split(":")) if args.steps else (0, float("inf"))
    lengths, rewards, truncated, steps = [], [], [], []
    with open(args.dump) as f:
        for line in f:
            record = json.loads(line)
            if lo <= record["step"] <= hi:
                steps.append(record["step"])
                lengths += record["lengths"]
                rewards += record["rewards"]
                truncated += record["truncated"]
    if not lengths:
        sys.exit(f"no samples in {args.dump} for steps {args.steps}")
    if len(set(steps)) != len(steps):
        sys.exit(f"{args.dump} has repeated steps (a resumed run?); pick a --steps range without repeats")

    frozen = LengthWeight.calibrate(lengths, 1.0, args.raw_weight_min, args.raw_weight_max)
    reference = {
        "id": frozen.reference_stats_id,
        "source": os.path.abspath(args.dump),
        "steps": [min(steps), max(steps)],
        "num_steps": len(steps),
        "num_samples": len(lengths),
        "success_rate": sum(r > 0 for r in rewards) / len(rewards),
        "truncated_rate": sum(truncated) / len(truncated),
        "lengths": lengths,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reference, f)

    print(
        f"{args.out}: id={reference['id']} steps {reference['steps'][0]}-{reference['steps'][1]} "
        f"n={len(lengths)} success={reference['success_rate']:.3f} truncated={reference['truncated_rate']:.3f}"
    )
    print(f"T0 (median) = {frozen.reference_length}")
    for alpha in (float(a) for a in args.alphas.split(",")):
        lw = LengthWeight.calibrate(lengths, alpha, args.raw_weight_min, args.raw_weight_max)
        raw = lw.raw(torch.tensor(lengths))
        print(
            f"alpha={alpha}: c={lw.calibration_scale:.6f}  at_min={(raw <= lw.raw_weight_min).float().mean():.3f}"
            f"  at_max={(raw >= lw.raw_weight_max).float().mean():.3f}"
        )


if __name__ == "__main__":
    main()
