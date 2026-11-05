#!/bin/bash
# Qwen2.5-Math-1.5B, K=2 replay whose replay updates clip each token's ratio to the rollout policy
# (--train.replay.ppo_clip), stops at step 400. Does a bounded replay keep replay2's early gain?
# Compare with flash, flash_lr2e-6 and replay2_adam_reset of the same seed at steps 100-400.
# Re-running resumes from outputs/Qwen-Math-1.5B/replay2_ppo_clip_s<seed>.
#   bash scripts/hpc/math_replay2_ppo_clip_400.sh
#   SEED=1 bash scripts/hpc/math_replay2_ppo_clip_400.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=replay2_ppo_clip SEED="${SEED:-42}" STOP_STEP=400 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
