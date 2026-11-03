#!/bin/bash
# Qwen2.5-Math-1.5B, ablation F (no replay, K=1 + o_proj Adam reset), seed 42, stops at step 400.
# Re-running resumes from outputs/qwen-math-1.5b-replay_F_s42/state.
#   bash scripts/hpc/replay_F_400.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM=F SEED=42 STOP_STEP=400 exec bash "$ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
