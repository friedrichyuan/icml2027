"""Build the Qwen2.5-7B-Instruct multi-turn Python-tool datasets from data/qwen-math.

Train: the same 7,500 DAPO-Math-17k prompts. Eval: AMC23, Minerva, AIME25 (the paper's tool
setting). Each problem is unwrapped from the DAPO "Answer:" instruction, re-wrapped with a
tool-use instruction asking for a \\boxed{} answer, and gets a `tools` column with the
python_executor schema (train with --data.tools_key tools).

    python scripts/prepare_python_tool.py [--src data/qwen-math] [--out data/qwen-tool]
"""

import argparse
import importlib.util
from pathlib import Path

from datasets import Dataset, load_from_disk

ROOT = Path(__file__).resolve().parent.parent
EVAL_SOURCES = ("AMC23", "Minerva", "AIME25")
DAPO_HEAD = "where $Answer is the answer to the problem.\n\n"
DAPO_TAIL = '\n\nRemember to put your answer on its own line after "Answer:".'
TOOL_INSTRUCTION = (
    "Solve the following math problem. You may call the python_executor tool to run Python "
    "code (sympy and numpy are available) for calculations or to check your reasoning; print() "
    "the values you need. When you are confident, stop calling the tool and put your final "
    "answer within \\boxed{{}}.\n\n{problem}"
)


def _tool_schema() -> dict:
    path = ROOT / "labs-molt" / "examples" / "python" / "tools" / "python_executor.py"
    spec = importlib.util.spec_from_file_location("python_executor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TOOL.schema


def unwrap_dapo(content: str) -> str:
    start = content.index(DAPO_HEAD) + len(DAPO_HEAD)
    end = content.rindex(DAPO_TAIL)
    return content[start:end].strip()


def convert(ds, tools: list, sources=None) -> Dataset:
    rows = []
    for row in ds:
        if sources and row["datasource"] not in sources:
            continue
        problem = unwrap_dapo(row["prompt"][0]["content"])
        rows.append(
            {
                "datasource": row["datasource"],
                "prompt": [{"role": "user", "content": TOOL_INSTRUCTION.format(problem=problem)}],
                "reward_model": row["reward_model"],
                "tools": tools,
            }
        )
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "qwen-math")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "qwen-tool")
    args = parser.parse_args()

    tools = [_tool_schema()]
    train = convert(load_from_disk(str(args.src / "train")), tools)
    eval_ds = convert(load_from_disk(str(args.src / "eval")), tools, EVAL_SOURCES)
    args.out.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(args.out / "train"))
    eval_ds.save_to_disk(str(args.out / "eval"))
    counts = {s: eval_ds["datasource"].count(s) for s in EVAL_SOURCES}
    print(f"wrote {len(train)} train + {len(eval_ds)} eval rows {counts} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
