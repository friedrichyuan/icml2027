"""Add a fixed ``task_bucket`` column to a prompt dataset (budget-aware task sampling).

The rule is decided before training and never looks at rewards: the first topic family whose
keywords match the problem text wins, in the order below. Buckets holding less than
``--min_share`` of the TRAIN split are merged into ``other``; the eval split reuses the train
bucket set, so both splits share one vocabulary.

  python scripts/assign_task_buckets.py --src data/qwen-math --dst data/qwen-math-bucketed
"""

import argparse
import ast
import json
import re
from collections import Counter

import datasets

TOPICS = [
    (
        "geometry",
        (
            r"triangle|circle|polygon|(?<!perfect )square(?! root)|rectangle|angle|radius|diameter|"
            r"perimeter|area|tangent|parallelogram|trapezoid|hexagon|pentagon|cube|sphere|cylinder|cone|"
            r"prism|pyramid|tetrahedron|coordinate plane|\\triangle|\\angle|\\overline|"
            r"perpendicular|inscribed|circumscribed|chord|vertex|vertices"
        ),
    ),
    (
        "counting_probability",
        (
            r"probability|how many ways|number of ways|arrangements?|permutations?|"
            r"combinations?|choose|committee|dice|die |coins?|cards?|randomly|"
            r"expected value|subsets?|how many (?:distinct |different )?"
            r"(?:ordered |positive )?(?:sequences|strings|paths|arrangements)"
        ),
    ),
    (
        "number_theory",
        (
            r"divisible|divisor|remainder|prime|modulo|\\pmod|\\mod|gcd|lcm|"
            r"greatest common|least common|integer solutions|digits?|base \d|"
            r"factorial|perfect square|congruen"
        ),
    ),
    (
        "sequences_series",
        (
            r"sequence|series|a_\{?n|a_\{?1|recursive|recurrence|arithmetic progression|"
            r"geometric progression|\\sum|\\prod|terms?\b"
        ),
    ),
    (
        "functions_inequalities",
        (
            r"f\(x\)|function|polynomial|roots?|inequalit|maximum|minimum|"
            r"real numbers? x|log|\\log|\\sin|\\cos|\\tan|complex number|"
            r"\\sqrt"
        ),
    ),
]


def problem_text(prompt) -> str:
    if isinstance(prompt, str):
        try:
            prompt = ast.literal_eval(prompt)
        except (ValueError, SyntaxError):
            return prompt
    if isinstance(prompt, list):
        return " ".join(
            m.get("content", "") for m in prompt if m.get("role", "user") == "user"
        )
    return str(prompt)


def topic_of(text: str) -> str:
    low = text.lower()
    for name, pattern in TOPICS:
        if re.search(pattern, low):
            return name
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="dataset dir with train/ and eval/ (load_from_disk)",
    )
    p.add_argument("--dst", required=True)
    p.add_argument("--input_key", default="prompt")
    p.add_argument("--bucket_key", default="task_bucket")
    p.add_argument("--min_share", type=float, default=0.05)
    args = p.parse_args()

    train = datasets.load_from_disk(f"{args.src}/train")
    raw = [topic_of(problem_text(x)) for x in train[args.input_key]]
    counts = Counter(raw)
    kept = {
        b for b, n in counts.items() if n >= args.min_share * len(raw) and b != "other"
    }
    rule = {name: (name if name in kept else "other") for name, _ in TOPICS} | {
        "other": "other"
    }

    for split in ("train", "eval"):
        ds = (
            train
            if split == "train"
            else datasets.load_from_disk(f"{args.src}/{split}")
        )
        buckets = [rule[topic_of(problem_text(x))] for x in ds[args.input_key]]
        if args.bucket_key in ds.column_names:
            ds = ds.remove_columns(args.bucket_key)
        ds = ds.add_column(args.bucket_key, buckets)
        ds.save_to_disk(f"{args.dst}/{split}")
        by_source = (
            Counter(zip(ds["datasource"], buckets))
            if "datasource" in ds.column_names
            else {}
        )
        print(
            f"[{split}] {len(ds)} rows:",
            json.dumps(dict(Counter(buckets).most_common())),
        )
        for (src, b), n in sorted(by_source.items()):
            print(f"    {src:>16s} {b:<24s} {n}")
    with open(f"{args.dst}/bucket_rule.json", "w") as f:
        json.dump(
            {
                "topics": TOPICS,
                "merge": rule,
                "train_counts": counts,
                "min_share": args.min_share,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
