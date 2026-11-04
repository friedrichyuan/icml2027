#!/bin/bash
# Deploy and train on a fresh multi-GPU machine: FlashREINFORCE, Qwen2.5-Math-1.5B, DAPO math.
#
#   git clone -b task-sampler <repo-url> icml2027 && cd icml2027
#   bash scripts/quickstart.sh                         # original Flash (full 34-episode recipe)
#   SAMPLER=combined bash scripts/quickstart.sh        # budget-aware task sampling, 60M-token run
#   GPUS=0,1,2,3 OUT=outputs/try bash scripts/quickstart.sh --train.num_episodes 1
#
# SAMPLER: original | signal_only | cost_only | combined. Extra arguments go to train_rl_ray.
# Runs in the molt image when docker is available (IMAGE), else in the current Python env
# (NATIVE=1 forces that; it needs the image's packages: torch, vllm, ray, ...). The model and the
# datasets are fetched once into data/ (HF_ENDPOINT defaults to hf-mirror.com); re-running the
# same command resumes from $OUT/state. Half the GPUs serve vLLM, the other half train.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
IMAGE="${IMAGE:-hijkzzz/molt:latest}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
n="$(awk -F, '{print NF}' <<< "$GPUS")"
[ "$n" -ge 2 ] || { echo "need at least 2 GPUs (got GPUS=$GPUS)" >&2; exit 1; }
SAMPLER="${SAMPLER:-}"
export OUT="${OUT:-$ROOT/outputs/quickstart-${SAMPLER:-flash}}" MODEL="${MODEL:-}" TRAIN="${TRAIN:-}" EVAL="${EVAL:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" VLLM_NUM_ENGINES=$((n / 2)) ACTOR_GPUS=$((n - n / 2))

run() {
  if [ "${NATIVE:-0}" = 1 ] || ! command -v docker > /dev/null; then
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$ROOT/labs-molt" "$@"
  else
    docker run --rm --gpus "\"device=$GPUS\"" --ipc=host --shm-size=64g \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
      -v "$ROOT:$ROOT" -v "$HOME/.cache:/root/.cache" -w "$ROOT" \
      -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n - 1)))" -e PYTHONPATH="$ROOT/labs-molt" \
      -e OUT -e MODEL -e TRAIN -e EVAL -e HF_ENDPOINT -e VLLM_NUM_ENGINES -e ACTOR_GPUS \
      --entrypoint "$1" "$IMAGE" "${@:2}"
  fi
}

if [ -z "$MODEL" ] && [ ! -e data/models/Qwen2.5-Math-1.5B/config.json ]; then
  run python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Math-1.5B', local_dir='data/models/Qwen2.5-Math-1.5B')"
fi
[ -e data/qwen-math/eval ] || run python scripts/prepare_qwen_math.py

flags=()
if [ -n "$SAMPLER" ]; then
  [ -e data/qwen-math-bucketed/eval ] ||
    run python scripts/assign_task_buckets.py --src data/qwen-math --dst data/qwen-math-bucketed
  export TRAIN="$ROOT/data/qwen-math-bucketed/train" EVAL="$ROOT/data/qwen-math-bucketed/eval"
  flags=(--task_sampler.mode "$SAMPLER" --eval.steps 1000000
    --task_sampler.eval_every_tokens 15000000 --task_sampler.max_policy_tokens 60000000)
fi
run bash scripts/run_qwen_math_dlc.sh ${flags[@]+"${flags[@]}"} --ckpt.load_enable \
  --ckpt.max_num 1 --ckpt.dcp_max_num 1 --ckpt.best_metric_key none "$@"
