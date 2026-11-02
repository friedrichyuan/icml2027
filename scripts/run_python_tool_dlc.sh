#!/bin/bash
# FlashREINFORCE Qwen2.5-7B-Instruct multi-turn Python tool (paper setting
# qwen2p5_7b_instruct_python_tool_10turn) on a DLC/HPC node that already runs the molt image.
# Reuses run_qwen_math_dlc.sh and overrides the recipe flags that differ (last flag wins).
#
#   python scripts/prepare_python_tool.py            # once: data/qwen-math -> data/qwen-tool
#   bash scripts/run_python_tool_dlc.sh [extra train flags]
#   TASK=tool ARM=A SEED=42 STOP_STEP=600 bash scripts/run_replay_hpc.sh   # with auto-resume
#
# MODEL defaults to Qwen2.5-7B-Instruct under $ROOT, $ROOT/data or $ROOT/data/models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${MODEL:-}" ]; then
  for base in "$ROOT" "$ROOT/data" "$ROOT/data/models" "$ROOT/models"; do
    if [ -e "$base/Qwen2.5-7B-Instruct/config.json" ]; then
      MODEL="$base/Qwen2.5-7B-Instruct"
      break
    fi
  done
fi
: "${MODEL:?set MODEL to the Qwen2.5-7B-Instruct directory (looked under $ROOT and $ROOT/data)}"
export MODEL
export TRAIN="${TRAIN:-$ROOT/data/qwen-tool/train}"
export EVAL="${EVAL:-$ROOT/data/qwen-tool/eval}"
for path in "$TRAIN" "$EVAL"; do
  [ -e "$path" ] || { echo "missing $path; run: python scripts/prepare_python_tool.py" >&2; exit 1; }
done
export OUT="${OUT:-$ROOT/outputs/qwen7b-tool}"
export MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-10}"

# 7B checkpoints are ~15 GB (HF) and ~90 GB (DCP state with Adam): keep only the latest of each
# and skip the extra full checkpoint on every new best eval.
exec bash "$ROOT/scripts/run_qwen_math_dlc.sh" \
  --data.tools_key tools \
  --data.max_len 8192 \
  --rollout.max_new_tokens 6144 \
  --train.num_episodes 68 \
  --train.async_queue_size 1 \
  --algo.advantage.is_correction_threshold 1e-3 \
  --train.agent_path "$ROOT/labs-molt/examples/python/agents/python_tool_qwen25.py" \
  --eval.n_samples_per_prompt 4 \
  --eval.eval_at_start \
  --ckpt.save_steps 50 \
  --ckpt.max_num 1 \
  --ckpt.dcp_max_num 1 \
  --ckpt.best_metric_key none \
  "$@"
