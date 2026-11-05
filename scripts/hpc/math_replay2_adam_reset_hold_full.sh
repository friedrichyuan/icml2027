#!/bin/bash
# Qwen2.5-Math-1.5B, replay2_adam_reset, but skip the replay Adam step when that batch's
# sequence-gate rejection exceeds 0.3. Seed 42, full recipe.
# Re-running resumes from outputs/Qwen-Math-1.5B/replay2_adam_reset_hold_s42/state.
#   bash scripts/hpc/math_replay2_adam_reset_hold_full.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=replay2_adam_reset_hold SEED=42 STOP_STEP=0 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
