import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_AGENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "agents" / "python_tool_qwen25.py"


def _load_agent():
    spec = importlib.util.spec_from_file_location("python_tool_qwen25_agent", _AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agent = _load_agent()
LABEL = {"ground_truth": "34", "style": "rule"}


def _call(code: str) -> str:
    return '<tool_call>\n{"name": "python_executor", "arguments": {"code": "' + code + '"}}\n</tool_call>'


def _step(env, action):
    return asyncio.run(env.step({"action_text": action, "label": LABEL, "observation_text": ""}))


def test_parse_tool_calls_handles_multiple_raw_newlines_and_string_arguments():
    text = (
        _call("print(1)")
        + '\n<tool_call>\n{"name": "python_executor", "arguments": {"code": "x = 2\nprint(x)"}}\n</tool_call>'
        + '\n<tool_call>{"name": "python_executor", "arguments": "{\\"code\\": \\"print(3)\\"}"}</tool_call>'
    )
    calls = agent.parse_tool_calls(text)
    assert [c["arguments"]["code"] for c in calls] == ["print(1)", "x = 2\nprint(x)", "print(3)"]


def test_parse_tool_calls_reports_malformed_and_unclosed_calls():
    calls = agent.parse_tool_calls("<tool_call>\n{not json}\n</tool_call> <tool_call>\n{\"name\": \"python_executor\"")
    assert len(calls) == 2 and all(isinstance(c, str) and c.startswith("Error") for c in calls)
    assert agent.parse_tool_calls("plain reasoning, no tools") == []


def test_committed_answer_ignores_boxed_inside_tool_code():
    assert agent.committed_answer(_call("print('\\\\boxed{7}')")) == ""
    assert agent.committed_answer("so the answer is \\boxed{\\frac{1}{2}}<|im_end|>") == "\\frac{1}{2}"


def test_tool_turn_then_committed_answer(monkeypatch):
    monkeypatch.setattr(agent, "_TOOLS", {"python_executor": SimpleNamespace(execute=lambda args: "34")})
    env = agent.PythonToolEnv()
    asyncio.run(env.reset({"observation": "p", "label": LABEL}))

    first = _step(env, "let me compute\n" + _call("print(34)") + "<|im_end|>")
    assert not first.terminated and not first.truncated and float(first.reward) == 0.0
    assert first.observation == "\n<|im_start|>user\n<tool_response>\n34\n</tool_response><|im_end|>\n<|im_start|>assistant\n"

    final = _step(env, "The answer is \\boxed{34}.<|im_end|>")
    assert final.terminated and float(final.reward) == 1.0
    assert final.observation == ""
    assert float(final.info["tool_calls"]) == 1.0 and float(final.info["turns"]) == 2.0


def test_no_tool_call_without_answer_ends_with_zero_reward():
    env = agent.PythonToolEnv()
    result = _step(env, "The answer is 34.<|im_end|>")
    assert result.terminated and float(result.reward) == 0.0
    assert float(result.info["committed_answer"]) == 0.0


def test_turn_cap_truncates_without_running_the_tool(monkeypatch):
    def fail(args):
        raise AssertionError("tool should not run on the last turn")

    monkeypatch.setattr(agent, "_MAX_TURNS", 1)
    monkeypatch.setattr(agent, "_TOOLS", {"python_executor": SimpleNamespace(execute=fail)})
    result = _step(agent.PythonToolEnv(), _call("print(1)"))
    assert result.truncated and not result.terminated and float(result.reward) == 0.0


def test_silent_code_gets_a_print_hint():
    result = _step(agent.PythonToolEnv(), _call("x = 1 + 1\\nx"))
    assert "(no output; use print() to show values)" in result.observation


def test_silent_code_gets_a_print_hint():
    result = _step(agent.PythonToolEnv(), _call("x = 1 + 1\\nx"))
    assert "(no output; use print() to show values)" in result.observation


def test_unknown_tool_and_parse_errors_are_fed_back():
    env = agent.PythonToolEnv()
    action = '<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call><tool_call>oops</tool_call>'
    result = _step(env, action)
    assert "tool `bash` is not available" in result.observation
    assert "could not parse tool call" in result.observation
    assert result.observation.startswith("<|im_end|>\n<|im_start|>user")
