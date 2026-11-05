#!/bin/bash
# Qwen2.5-7B-Instruct multi-turn Python tool, original FlashREINFORCE, seed 42, stops at step 600.
# Needs data/qwen-tool (python scripts/prepare_python_tool.py).
# Re-running resumes from outputs/Qwen-7B-Tool/flash_s42/state.
#   bash scripts/hpc/tool_flash_600.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASK=tool ARM=flash SEED=42 STOP_STEP=600 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" "$@"
