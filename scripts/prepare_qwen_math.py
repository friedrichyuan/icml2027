"""Build the Qwen2.5-Math-1.5B FlashREINFORCE datasets in Molt's on-disk format.

Train: first 7,500 unique prompts from BytedTsinghua-SIA/DAPO-Math-17k.
Eval: MATH-500, AMC23, Minerva, AIME25, OlympiadBench, wrapped in the same
user instruction DAPO uses so the math grader sees an Answer: line.
"""

import ast
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import Dataset, load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "qwen-math"
DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\nRemember to put your answer on its own "
    'line after "Answer:".'
)


def _user_prompt(text: str) -> list:
    return [{"role": "user", "content": text}]


def _row(datasource: str, prompt_text: str, answer: str) -> dict:
    answer = str(answer).strip()
    return {
        "datasource": datasource,
        "prompt": _user_prompt(prompt_text),
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "response": [{"role": "assistant", "content": "\\boxed{" + answer + "}"}],
    }


def _unwrap_answer(value) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value]
        return parts[0] if len(parts) == 1 else ", ".join(parts)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, list):
            return _unwrap_answer(parsed)
    return text


def prepare_train(max_train: int = 7500) -> Dataset:
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    seen = set()
    keep = []
    for idx, row in enumerate(train_ds):
        try:
            key = row["prompt"][0]["content"]
        except (KeyError, IndexError, TypeError):
            key = str(row.get("prompt"))
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
        if len(keep) >= max_train:
            break
    print(f"train: scanned to row {idx}, kept {len(keep)} unique prompts", flush=True)
    selected = train_ds.select(keep)

    def format_row(example):
        answer = example["reward_model"].get("ground_truth", "")
        return {
            "datasource": example.get("data_source") or "dapo_math_17k",
            "prompt": example["prompt"],
            "reward_model": {"ground_truth": str(answer), "style": example["reward_model"].get("style", "rule")},
            "response": [{"role": "assistant", "content": "\\boxed{" + str(answer) + "}"}],
        }

    return selected.map(format_row, remove_columns=selected.column_names)


def prepare_eval() -> Dataset:
    specs = [
        ("HuggingFaceH4/MATH-500", "test", "MATH-500", "problem", "answer"),
        ("math-ai/amc23", "test", "AMC23", "question", "answer"),
        ("math-ai/minervamath", "test", "Minerva", "question", "answer"),
        ("math-ai/aime25", "test", "AIME25", "problem", "answer"),
        ("math-ai/olympiadbench", "test", "OlympiadBench", "question", "final_answer"),
    ]
    rows = []
    for repo, split, name, question_key, answer_key in specs:
        ds = load_dataset(repo, split=split)
        n = 0
        for example in ds:
            question = example[question_key]
            if not isinstance(question, str) or not question.strip():
                continue
            answer = _unwrap_answer(example[answer_key])
            if not answer:
                continue
            rows.append(_row(name, DAPO_INSTRUCTION.format(problem=question.strip()), answer))
            n += 1
        print(f"eval {name}: {n}", flush=True)
    return Dataset.from_list(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = prepare_train()
    eval_ds = prepare_eval()
    train.save_to_disk(OUT / "train")
    eval_ds.save_to_disk(OUT / "eval")
    print(f"wrote {len(train)} train + {len(eval_ds)} eval rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
