#!/bin/bash
# Qwen2.5-Math-1.5B, K=2 replay + Adam reset on all params, seed 42, stops at step 400.
# Re-running resumes from outputs/Qwen-Math-1.5B/replay2_adam_reset_all_s42/state.
#   bash scripts/hpc/math_replay2_adam_reset_all_400.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=replay2_adam_reset_all SEED=42 STOP_STEP=400 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
