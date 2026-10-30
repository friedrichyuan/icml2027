#!/bin/bash
# Run the replay / reset arms one after another on the docker recipe, each until its eval at
# STOP_STEP lands. A run whose train.log is silent for HANG_MIN minutes (e.g. an eval request
# that never returns) or that dies early is resumed from its latest checkpoint.
#   STOP_STEP=400 setsid nohup scripts/run_replay_queue.sh B C D E > outputs/replay_queue.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOP_STEP="${STOP_STEP:-400}"
HANG_MIN="${HANG_MIN:-90}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"

flags_for() {
  case "$1" in
    B) echo "--train.replay.update_ratio 2" ;;
    C) echo "--train.replay.update_ratio 2 --train.reset.mode shrink --train.reset.eps 0.01" ;;
    D) echo "--train.replay.update_ratio 2 --train.reset.mode optim_only" ;;
    E) echo "--train.reset.mode shrink --train.reset.eps 0.01" ;;
    B4) echo "--train.replay.update_ratio 4" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

for arm in "$@"; do
  flags="$(flags_for "$arm")" || exit 1
  out="$ROOT/outputs/qwen-math-1.5b-replay_$arm"
  name="frl-replay-$arm"
  restarts=0
  resume=""
  while true; do
    echo "$(date '+%F %T') start $arm (restart $restarts) $flags $resume"
    docker rm -f "$name" >/dev/null 2>&1
    # shellcheck disable=SC2086
    OUT="$out" NAME="$name" "$ROOT/scripts/run_qwen_math_docker.sh" $flags $resume || exit 1
    resume="--ckpt.load_enable"
    while true; do
      sleep 60
      if grep -q "Eval at step $STOP_STEP:" "$out/train.log" 2>/dev/null; then
        echo "$(date '+%F %T') $arm reached eval at step $STOP_STEP"
        docker stop -t 60 "$name" >/dev/null
        break 2
      fi
      if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then
        echo "$(date '+%F %T') $arm container exited early"
        break
      fi
      if [ -n "$(find "$out/train.log" -mmin +"$HANG_MIN" 2>/dev/null)" ]; then
        echo "$(date '+%F %T') $arm train.log silent for ${HANG_MIN}m; restarting from checkpoint"
        break
      fi
    done
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      echo "$(date '+%F %T') $arm gave up after $MAX_RESTARTS restarts"
      docker rm -f "$name" >/dev/null 2>&1
      break
    fi
  done
done
echo "$(date '+%F %T') queue done"
