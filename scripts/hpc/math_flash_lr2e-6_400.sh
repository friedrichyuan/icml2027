#!/bin/bash
# Qwen2.5-Math-1.5B, original FlashREINFORCE with 2x the Adam LR (2e-6), stops at step 400.
# Control for replay2: K=2 takes twice the optimizer steps per batch; if this arm matches its early
# gain, replay adds nothing beyond more update magnitude. Compare with flash and replay2_adam_reset
# of the same seed at steps 100-400. Re-running resumes from outputs/Qwen-Math-1.5B/flash_lr2e-6_s<seed>.
#   bash scripts/hpc/math_flash_lr2e-6_400.sh
#   SEED=1 bash scripts/hpc/math_flash_lr2e-6_400.sh
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=flash_lr2e-6 SEED="${SEED:-42}" STOP_STEP=400 exec bash "$CODE_ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
