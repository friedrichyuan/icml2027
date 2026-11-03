#!/bin/bash
# Qwen2.5-Math-1.5B, replay arm D (K=2 + o_proj Adam reset), seed 42, full 34-episode recipe.
# Re-running resumes from outputs/qwen-math-1.5b-replay_D_s42/state.
#   bash scripts/hpc/replay_D_full.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM=D SEED=42 STOP_STEP=0 exec bash "$ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
