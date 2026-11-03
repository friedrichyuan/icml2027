#!/bin/bash
# Run replay / reset arms one after another on one HPC node via run_replay_hpc.sh.
# Each item is ARM:SEED[:STOP_STEP] (STOP_STEP defaults to 400). An arm that already has its eval
# at STOP_STEP is skipped and a failed arm does not stop the queue, so re-submitting the same
# command resumes where it left off.
#
#   bash scripts/run_replay_confirm_hpc.sh A:42:800 F:42 D_all:42
#   bash scripts/run_replay_confirm_hpc.sh          # the full confirmation set below
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Multi-seed A/B/D (A to 800 for the wall-clock-matched comparison), the K=1 + optim_only
# control F, and the reset-scope ablations.
default=(
  A:42:800 A:1:800 A:2:800
  B:42 B:1 B:2
  D:42 D:1 D:2
  F:42 F:1 F:2
  D_all:42 D_down:42
)
items=("$@")
[ ${#items[@]} -gt 0 ] || items=("${default[@]}")

# 14 arms: keep only the latest HF snapshot and resumable state per arm, no best-eval copies.
ckpt=(--ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none)

failed=()
for item in "${items[@]}"; do
  IFS=: read -r arm seed stop <<< "$item"
  echo "$(date '+%F %T') === $arm seed $seed stop ${stop:-400}"
  if ! ARM="$arm" SEED="$seed" STOP_STEP="${stop:-400}" bash "$ROOT/scripts/run_replay_hpc.sh" "${ckpt[@]}"; then
    failed+=("$item")
  fi
done
echo "$(date '+%F %T') queue done; failed: ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
