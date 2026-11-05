#!/bin/bash
# One replay / reset arm on an HPC node that already runs the molt image (no docker wrapper).
# Stops once the eval at STOP_STEP lands (STOP_STEP=0 runs the full 34-episode recipe). The LR
# schedule is the full recipe's, so a 400-step arm matches the local runs. A crash or a hang
# (train.log silent for HANG_MIN, e.g. a lost eval request) resumes from $OUT/state, and so does
# re-submitting the same command after a preemption.
#
#   ARM=replay2_adam_reset SEED=1 bash scripts/run_replay_hpc.sh
#   ARM=replay2 SEED=2 STOP_STEP=400 bash scripts/run_replay_hpc.sh [extra train flags]
#   TASK=tool ARM=replay2_adam_reset SEED=42 STOP_STEP=600 bash scripts/run_replay_hpc.sh
#
# Tasks (output dir): math Qwen2.5-Math-1.5B, run_qwen_math_dlc.sh (outputs/Qwen-Math-1.5B, default)
#                     tool Qwen2.5-7B-Instruct multi-turn Python tool, run_python_tool_dlc.sh
#                          (outputs/Qwen-7B-Tool)
# Arms land in $DATA_ROOT/outputs/<task dir>/<arm>_s<seed>. K = actor updates per new batch.
#   flash                        original FlashREINFORCE
#   flash_lr2e-6                 flash with 2x the Adam LR (does replay gain more than extra steps?)
#   replay2 / replay4            K=2 / K=4
#   replay2_shrink               K=2 + shrink o_proj toward its RL-start weights
#   replay2_adam_reset           K=2 + reset the o_proj Adam moments
#   replay2_adam_reset_all       same reset on all params
#   replay2_adam_reset_down_proj same reset on MLP down_proj
#   replay4_adam_reset           K=4 + o_proj Adam reset
#   replay2_adam_reset_hold      replay2_adam_reset, plus skip the replay Adam step when that
#                                batch's sequence-gate rejection exceeds 0.3 (LR schedule advances)
#   shrink / adam_reset          K=1 with the o_proj shrink / Adam reset
# GPU / model overrides (CUDA_VISIBLE_DEVICES, VLLM_NUM_ENGINES, ACTOR_GPUS, MODEL) pass through
# to the task launcher, and so do CODE_ROOT (checkout to train) and DATA_ROOT (weights, datasets,
# outputs), see run_qwen_math_dlc.sh. Assumes the job owns its node: before every launch it kills
# leftover Ray / vLLM processes and waits up to GPU_WAIT_MIN minutes for the GPUs to empty.
set -uo pipefail
export CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-$CODE_ROOT}"
case "${TASK:-math}" in
  math) launcher="$CODE_ROOT/scripts/run_qwen_math_dlc.sh" task_dir=Qwen-Math-1.5B ;;
  tool) launcher="$CODE_ROOT/scripts/run_python_tool_dlc.sh" task_dir=Qwen-7B-Tool ;;
  *) echo "unknown TASK=$TASK (math or tool)" >&2; exit 1 ;;
esac
ARM="${ARM:?set ARM to one of the arms listed at the top of this script}"
SEED="${SEED:-42}"
STOP_STEP="${STOP_STEP:-400}"
HANG_MIN="${HANG_MIN:-90}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-10}"

case "$ARM" in
  flash) flags=() ;;
  flash_lr2e-6) flags=(--actor.adam.lr 2e-6) ;;
  replay2) flags=(--train.replay.update_ratio 2) ;;
  replay2_shrink) flags=(--train.replay.update_ratio 2 --train.reset.mode shrink --train.reset.eps 0.01) ;;
  replay2_adam_reset) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only) ;;
  shrink) flags=(--train.reset.mode shrink --train.reset.eps 0.01) ;;
  adam_reset) flags=(--train.reset.mode optim_only) ;;
  replay2_adam_reset_all) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only --train.reset.params '*') ;;
  replay2_adam_reset_down_proj) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only --train.reset.params '*.down_proj.weight') ;;
  replay4) flags=(--train.replay.update_ratio 4) ;;
  replay4_adam_reset) flags=(--train.replay.update_ratio 4 --train.reset.mode optim_only) ;;
  replay2_adam_reset_hold) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only --train.replay.max_filter_ratio 0.3) ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
# free_gpus kills every Ray / vLLM process on the node, so a second launcher here would kill the
# first one's training. Refuse to start while another holds the node lock.
lock=/tmp/molt_replay_hpc.lock
exec 9> "$lock"
if ! flock -n 9; then
  echo "another run_replay_hpc.sh is running on this node ($(cat "$lock.owner" 2>/dev/null)); not starting $ARM seed $SEED" >&2
  exit 1
fi
echo "$ARM seed $SEED pid $$ since $(date '+%F %T')" > "$lock.owner"

export OUT="${OUT:-$DATA_ROOT/outputs/$task_dir/${ARM}_s${SEED}}"
mkdir -p "$OUT"
log="$OUT/train.log"

reached() { [ "$STOP_STEP" -gt 0 ] && grep -q "Eval at step $STOP_STEP:" "$log" 2>/dev/null; }

# Bash already puts a background job in its own process group (pgid == pid), so
# kill -- -pid tears down the launcher and its children. Ray / vLLM EngineCore
# workers often escape that group — always pkill them here too, otherwise a
# successful "reached STOP_STEP" exit leaves GPUs busy on DLC.
stop_run() {
  kill -- "-$1" 2>/dev/null || kill "$1" 2>/dev/null
  sleep 30
  kill -9 -- "-$1" 2>/dev/null || kill -9 "$1" 2>/dev/null
  ray stop --force >/dev/null 2>&1
  pkill -9 -f 'molt.cli.train_rl_ray|EngineCore|RolloutRayActor|ray::' 2>/dev/null
  sleep 5
}

# vLLM EngineCore processes are spawned by Ray workers and can outlive `ray stop`, keeping
# ~0.85 of each rollout GPU; a retry then fails at vLLM startup. Kill leftovers and wait until
# every visible GPU is (nearly) empty before launching.
free_gpus() {
  ray stop --force >/dev/null 2>&1
  pkill -9 -f 'molt.cli.train_rl_ray|EngineCore|RolloutRayActor|ray::' 2>/dev/null
  command -v nvidia-smi >/dev/null || return 0
  local gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" used=unknown
  for ((i = 0; i < GPU_WAIT_MIN * 6; i++)); do
    used="$(nvidia-smi -i "$gpus" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
    [ "${used:-0}" -lt 2048 ] && return 0
    sleep 10
  done
  echo "$(date '+%F %T') GPUs $gpus still busy after ${GPU_WAIT_MIN}m (max used ${used} MiB):" | tee -a "$log"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv | tee -a "$log"
  return 1
}

for ((attempt = 0; attempt <= MAX_RESTARTS; attempt++)); do
  if reached; then
    echo "$(date '+%F %T') $ARM seed $SEED already has its eval at step $STOP_STEP"
    exit 0
  fi
  free_gpus || exit 1
  echo "$(date '+%F %T') arm=$ARM seed=$SEED attempt=$attempt flags: ${flags[*]} $*" | tee -a "$log"
  # Do NOT wrap with bare `setsid`: when this script's background job is already a
  # process-group leader (typical under DLC), setsid forks and the parent exits at
  # once, so $! dies within a second. The watchdog then treats training as finished
  # or crashed and relaunches — that was creating a fresh tb/rl_* every few minutes.
  # 9>&-: leftover vLLM must not inherit the node lock and block the next run.
  bash "$launcher" "${flags[@]}" --train.seed "$SEED" --ckpt.load_enable "$@" 9>&- &
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
      # A true full-recipe finish is exit 0 with STOP_STEP=0. Exit 0 before STOP_STEP
      # is the old setsid false-finish; treat it as a crash and resume.
      if [ "$rc" -eq 0 ] && { [ "$STOP_STEP" -eq 0 ] || reached; }; then
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
