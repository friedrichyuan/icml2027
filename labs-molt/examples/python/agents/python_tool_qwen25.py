"""Multi-turn math env for Qwen2.5-Instruct with a Python tool (Hermes JSON tool calls).

The prompt is rendered with ``--data.tools_key tools`` so the Qwen2.5 chat template adds the
``# Tools`` system block. Each turn the model may emit one or more

    <tool_call>
    {"name": "python_executor", "arguments": {"code": "..."}}
    </tool_call>

blocks; the env runs them in the sandboxed ``python_executor`` and feeds the outputs back as
``<tool_response>`` blocks in a user turn, exactly as the Qwen2.5 template renders ``tool``
messages. The episode ends when the model commits a ``\\boxed{}`` answer outside its tool calls
or stops calling tools; only that committed answer is graded. At ``MAX_AGENT_TURNS`` an episode
still calling tools is truncated with reward 0.

Environment variables:
  MAX_AGENT_TURNS         (default 10)  — model turns per episode.
  PYTHON_EXECUTOR_TIMEOUT (default 10)  — wall-clock per tool call (s).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
from pathlib import Path

import torch

from molt.agents import Env, Result, StepEnvRunner

logger = logging.getLogger(__name__)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GRADER = _load_module("math_grader", _PROJECT_DIR / "utils" / "math_grader.py")
_PYTHON_EXECUTOR = _load_module("python_executor", _PROJECT_DIR / "tools" / "python_executor.py").TOOL
_TOOLS = {_PYTHON_EXECUTOR.schema["function"]["name"]: _PYTHON_EXECUTOR}

_MAX_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "10"))
# A call cut off by <|im_end|> or the token cap still counts, so the model gets an error back
# instead of having an unfinished call graded as its final answer.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|<\|im_end\|>|$)", re.DOTALL)
_CALL_FORMAT = 'A call must be JSON, e.g. {"name": "python_executor", "arguments": {"code": "print(1 + 1)"}}.'


def parse_tool_calls(text: str) -> list[dict | str]:
    """Return one entry per ``<tool_call>`` block: ``{"name", "arguments"}`` or an error string."""
    calls = []
    for body in _TOOL_CALL_RE.findall(text):
        try:
            call = json.loads(body.strip(), strict=False)
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args, strict=False)
            if not isinstance(call.get("name"), str) or not isinstance(args, dict):
                raise ValueError("expected {\"name\": str, \"arguments\": object}")
            calls.append({"name": call["name"], "arguments": args})
        except (ValueError, AttributeError) as exc:
            calls.append(f"Error: could not parse tool call ({exc}). {_CALL_FORMAT}")
    return calls


def committed_answer(text: str) -> str:
    """The last ``\\boxed{}``/``\\fbox{}`` outside tool calls (code may print a boxed string)."""
    prose = _TOOL_CALL_RE.sub("", text)
    boxed = _GRADER._last_braced_command(prose, r"\boxed") or _GRADER._last_braced_command(prose, r"\fbox")
    return boxed.strip() if boxed else ""


def grade(answer: str, label) -> float:
    if not answer or not label:
        return 0.0
    try:
        return float(_GRADER.score_response(f"\\boxed{{{answer}}}", "", label).get("reward", 0.0))
    except Exception as exc:
        logger.debug("grader failed: %s", exc)
        return 0.0


def tool_observation(close: str, outputs: list[str]) -> str:
    responses = "".join(f"\n<tool_response>\n{out}\n</tool_response>" for out in outputs)
    return f"{close}\n<|im_start|>user{responses}<|im_end|>\n<|im_start|>assistant\n"


def _run_call(call: dict | str) -> str:
    if isinstance(call, str):
        return call
    tool = _TOOLS.get(call["name"])
    if tool is None:
        return f"Error: tool `{call['name']}` is not available. Available: {list(_TOOLS)}."
    output = tool.execute(call["arguments"])
    # The sandbox is a script, not a REPL: a bare trailing expression prints nothing.
    return "(no output; use print() to show values)" if output == "(no output)" else output


class PythonToolEnv(Env):
    def __init__(self):
        self.turn = 0
        self.tool_calls = 0

    async def reset(self, state):
        self.turn = 0
        self.tool_calls = 0
        return {"observation": state["observation"]}

    async def step(self, state) -> Result:
        action = state["action_text"]
        self.turn += 1
        # vLLM keeps a matched stop token in token_ids; close the turn only if it is missing.
        close = "" if action.endswith("<|im_end|>") else "<|im_end|>"

        calls = parse_tool_calls(action)
        answer = committed_answer(action)
        if answer or not calls:
            reward = torch.tensor(grade(answer, state.get("label")), dtype=torch.float32)
            return Result(reward=reward, terminated=True, info=self._info(reward, answer))

        self.tool_calls += len(calls)
        zero = torch.tensor(0.0)
        if self.turn >= _MAX_TURNS:
            return Result(reward=zero, terminated=False, truncated=True, info=self._info(zero, ""))

        outputs = [await asyncio.to_thread(_run_call, call) for call in calls]
        return Result(
            reward=zero, observation=tool_observation(close, outputs), terminated=False, info=self._info(zero, "")
        )

    def _info(self, reward, answer: str) -> dict:
        return {
            "tool_calls": torch.tensor(float(self.tool_calls)),
            "turns": torch.tensor(float(self.turn)),
            "committed_answer": torch.tensor(1.0 if answer else 0.0),
            "correct": reward,
        }


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(PythonToolEnv)
