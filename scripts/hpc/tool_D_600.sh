#!/bin/bash
# Qwen2.5-7B-Instruct multi-turn Python tool, replay arm D (K=2 + o_proj Adam reset), seed 42,
# stops at step 600. Needs data/qwen-tool (python scripts/prepare_python_tool.py).
# Re-running resumes from outputs/qwen7b-tool-replay_D_s42/state.
#   bash scripts/hpc/tool_D_600.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK=tool ARM=D SEED=42 STOP_STEP=600 exec bash "$ROOT/scripts/run_replay_hpc.sh" "$@"
