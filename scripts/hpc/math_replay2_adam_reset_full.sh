#!/bin/bash
# Qwen2.5-Math-1.5B, K=2 replay + o_proj Adam reset, seed 42, full 34-episode recipe.
# Re-running resumes from outputs/Qwen-Math-1.5B/replay2_adam_reset_s42/state.
#   bash scripts/hpc/math_replay2_adam_reset_full.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=replay2_adam_reset SEED=42 STOP_STEP=0 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
