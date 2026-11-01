#!/bin/bash
# One replay / reset arm on an HPC node that already runs the molt image (no docker wrapper).
# Stops once the eval at STOP_STEP lands (STOP_STEP=0 runs the full 34-episode recipe). The LR
# schedule is the full recipe's, so a 400-step arm matches the local runs. A crash or a hang
# (train.log silent for HANG_MIN, e.g. a lost eval request) resumes from $OUT/state, and so does
# re-submitting the same command after a preemption.
#
#   ARM=D SEED=1 bash scripts/run_replay_hpc.sh
#   ARM=B SEED=2 STOP_STEP=400 bash scripts/run_replay_hpc.sh [extra train flags]
#
# Arms: A original Flash | B K=2 | C K=2+shrink | D K=2+optim_only | E K=1+shrink
#       B4 K=4 | D4 K=4+optim_only
# GPU / model overrides (CUDA_VISIBLE_DEVICES, VLLM_NUM_ENGINES, ACTOR_GPUS, MODEL) pass through
# to run_qwen_math_dlc.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:?set ARM to one of A B C D E B4 D4}"
SEED="${SEED:-42}"
STOP_STEP="${STOP_STEP:-400}"
HANG_MIN="${HANG_MIN:-90}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"

case "$ARM" in
  A) flags="" ;;
  B) flags="--train.replay.update_ratio 2" ;;
  C) flags="--train.replay.update_ratio 2 --train.reset.mode shrink --train.reset.eps 0.01" ;;
  D) flags="--train.replay.update_ratio 2 --train.reset.mode optim_only" ;;
  E) flags="--train.reset.mode shrink --train.reset.eps 0.01" ;;
  B4) flags="--train.replay.update_ratio 4" ;;
  D4) flags="--train.replay.update_ratio 4 --train.reset.mode optim_only" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
export OUT="${OUT:-$ROOT/outputs/qwen-math-1.5b-replay_${ARM}_s${SEED}}"
mkdir -p "$OUT"
log="$OUT/train.log"

reached() { [ "$STOP_STEP" -gt 0 ] && grep -q "Eval at step $STOP_STEP:" "$log" 2>/dev/null; }

stop_run() {
  kill -- "-$1" 2>/dev/null
  sleep 30
  kill -9 -- "-$1" 2>/dev/null
  ray stop --force >/dev/null 2>&1
}

for ((attempt = 0; attempt <= MAX_RESTARTS; attempt++)); do
  if reached; then
    echo "$(date '+%F %T') $ARM seed $SEED already has its eval at step $STOP_STEP"
    exit 0
  fi
  echo "$(date '+%F %T') arm=$ARM seed=$SEED attempt=$attempt flags: $flags $*" | tee -a "$log"
  # Own process group so the whole Ray driver tree can be stopped at STOP_STEP.
  # shellcheck disable=SC2086
  setsid bash "$ROOT/scripts/run_qwen_math_dlc.sh" $flags --train.seed "$SEED" --ckpt.load_enable "$@" &
  pid=$!
  while sleep 60; do
    if reached; then
      echo "$(date '+%F %T') $ARM seed $SEED reached eval at step $STOP_STEP" | tee -a "$log"
      stop_run "$pid"
      exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      rc=$?
      if [ "$rc" -eq 0 ]; then
        echo "$(date '+%F %T') $ARM seed $SEED finished" | tee -a "$log"
        exit 0
      fi
      echo "$(date '+%F %T') exited with $rc; resuming from checkpoint" | tee -a "$log"
      break
    fi
    if [ -n "$(find "$log" -mmin +"$HANG_MIN" 2>/dev/null)" ]; then
      echo "$(date '+%F %T') train.log silent for ${HANG_MIN}m; restarting from checkpoint" | tee -a "$log"
      stop_run "$pid"
      break
    fi
  done
  ray stop --force >/dev/null 2>&1
done
echo "$(date '+%F %T') $ARM seed $SEED gave up after $MAX_RESTARTS restarts" | tee -a "$log"
exit 1
