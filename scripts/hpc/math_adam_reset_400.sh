#!/bin/bash
# Qwen2.5-Math-1.5B, no replay (K=1) + o_proj Adam reset, seed 42, stops at step 400.
# Re-running resumes from outputs/Qwen-Math-1.5B/adam_reset_s42/state.
#   bash scripts/hpc/math_adam_reset_400.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=adam_reset SEED=42 STOP_STEP=400 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
