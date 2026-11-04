#!/bin/bash
# Budget-aware task sampling, mode=signal_only, on original Flash (seed 42). Ends itself after the eval at
# 60M generated policy tokens (evals every 15M); re-run the same command to resume.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM=TS_signal_only SEED=42 STOP_STEP=0 exec bash "$ROOT/scripts/run_replay_hpc.sh" \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
