#!/bin/bash
# One replay / reset arm on an HPC node that already runs the molt image (no docker wrapper).
# Stops once the eval at STOP_STEP lands (STOP_STEP=0 runs the full 34-episode recipe). The LR
# schedule is the full recipe's, so a 400-step arm matches the local runs. A crash or a hang
# (train.log silent for HANG_MIN, e.g. a lost eval request) resumes from $OUT/state, and so does
# re-submitting the same command after a preemption.
#
#   ARM=D SEED=1 bash scripts/run_replay_hpc.sh
#   ARM=B SEED=2 STOP_STEP=400 bash scripts/run_replay_hpc.sh [extra train flags]
#   TASK=tool ARM=D SEED=42 STOP_STEP=600 bash scripts/run_replay_hpc.sh
#
# Tasks: math Qwen2.5-Math-1.5B (run_qwen_math_dlc.sh, default)
#        tool Qwen2.5-7B-Instruct multi-turn Python tool (run_python_tool_dlc.sh)
# Arms: A original Flash | B K=2 | C K=2+shrink | D K=2+optim_only | E K=1+shrink
#       F K=1+optim_only | D_all / D_down D with the Adam reset on all params / MLP down_proj
#       B4 K=4 | D4 K=4+optim_only
#       TS_original / TS_signal_only / TS_cost_only / TS_combined: budget-aware task sampling on
#       original Flash; needs TRAIN/EVAL with a task_bucket column (scripts/hpc/sampler_*.sh), and
#       ends itself after the eval at TS_MAX_TOKENS policy tokens (evals every TS_EVAL_TOKENS)
# GPU / model overrides (CUDA_VISIBLE_DEVICES, VLLM_NUM_ENGINES, ACTOR_GPUS, MODEL) pass through
# to the task launcher. Assumes the job owns its node: before every launch it kills leftover
# Ray / vLLM processes and waits up to GPU_WAIT_MIN minutes for the GPUs to empty.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${TASK:-math}" in
  math) launcher="$ROOT/scripts/run_qwen_math_dlc.sh" run_prefix=qwen-math-1.5b ;;
  tool) launcher="$ROOT/scripts/run_python_tool_dlc.sh" run_prefix=qwen7b-tool ;;
  *) echo "unknown TASK=$TASK (math or tool)" >&2; exit 1 ;;
esac
ARM="${ARM:?set ARM to one of A B C D E F D_all D_down B4 D4 TS_original TS_signal_only TS_cost_only TS_combined}"
SEED="${SEED:-42}"
STOP_STEP="${STOP_STEP:-400}"
HANG_MIN="${HANG_MIN:-90}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-10}"

case "$ARM" in
  A) flags=() ;;
  B) flags=(--train.replay.update_ratio 2) ;;
  C) flags=(--train.replay.update_ratio 2 --train.reset.mode shrink --train.reset.eps 0.01) ;;
  D) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only) ;;
  E) flags=(--train.reset.mode shrink --train.reset.eps 0.01) ;;
  F) flags=(--train.reset.mode optim_only) ;;
  D_all) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only --train.reset.params '*') ;;
  D_down) flags=(--train.replay.update_ratio 2 --train.reset.mode optim_only --train.reset.params '*.down_proj.weight') ;;
  B4) flags=(--train.replay.update_ratio 4) ;;
  D4) flags=(--train.replay.update_ratio 4 --train.reset.mode optim_only) ;;
  TS_original | TS_signal_only | TS_cost_only | TS_combined)
    flags=(--task_sampler.mode "${ARM#TS_}" --eval.steps 1000000
      --task_sampler.eval_every_tokens "${TS_EVAL_TOKENS:-15000000}"
      --task_sampler.max_policy_tokens "${TS_MAX_TOKENS:-60000000}") ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac
case "$ARM" in TS_*) run_name="$ARM" ;; *) run_name="replay_$ARM" ;; esac
# free_gpus kills every Ray / vLLM process on the node, so a second launcher here would kill the
# first one's training. Refuse to start while another holds the node lock.
lock=/tmp/molt_replay_hpc.lock
exec 9> "$lock"
if ! flock -n 9; then
  echo "another run_replay_hpc.sh is running on this node ($(cat "$lock.owner" 2>/dev/null)); not starting $ARM seed $SEED" >&2
  exit 1
fi
echo "$ARM seed $SEED pid $$ since $(date '+%F %T')" > "$lock.owner"

if [[ "$ARM" == TS_* && -z "${TRAIN:-}" ]]; then
  [ "${TASK:-math}" = math ] || { echo "TS_* arms need TRAIN/EVAL with a task_bucket column" >&2; exit 1; }
  src="$ROOT/data/qwen-math"; [ -e "$src/train" ] || src="$ROOT/qwen-math"
  dst="$ROOT/data/qwen-math-bucketed"
  [ -e "$dst/eval" ] || python "$ROOT/scripts/assign_task_buckets.py" --src "$src" --dst "$dst" || exit 1
  export TRAIN="$dst/train" EVAL="$dst/eval"
fi
export OUT="${OUT:-$ROOT/outputs/${run_prefix}-${run_name}_s${SEED}}"
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
