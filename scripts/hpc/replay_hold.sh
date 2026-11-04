#!/bin/bash
# hold: replay D, but skip the replay Adam step when that batch's sequence-gate rejection exceeds
# 0.3. Seed 42, full recipe. Re-running resumes from outputs/...-replay_hold_s42/state.
#   bash scripts/hpc/replay_hold.sh
#   TASK=tool bash scripts/hpc/replay_hold.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM=hold SEED=42 STOP_STEP=0 exec bash "$ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
