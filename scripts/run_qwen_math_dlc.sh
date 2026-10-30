#!/bin/bash
# FlashREINFORCE Qwen2.5-Math-1.5B on a DLC node that already runs the molt image.
# No docker wrapper: this script is the container entrypoint.
#
#   ROOT=/mnt/data/yuanmingqi/code/icml2027 bash scripts/run_qwen_math_dlc.sh
#   OUT=outputs/qwen-math-1.5b-lw_a0.75 bash scripts/run_qwen_math_dlc.sh --algo.length_weight.enabled ...
#
# Defaults: 8 visible GPUs (0-7), 4 vLLM + 4 actor, model/data under $ROOT/data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/labs-molt"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_AGENT_TURNS=1
export WANDB_MODE=disabled

# DLC nodes expose their GPUs as 0..N-1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# Prefer the ctx8192 wrapper; fall back to a plain Qwen2.5-Math-1.5B tree under data/.
if [ -z "${MODEL:-}" ]; then
  for candidate in \
    "$ROOT/data/models/Qwen2.5-Math-1.5B-ctx8192" \
    "$ROOT/data/models/Qwen2.5-Math-1.5B" \
    "$ROOT/data/Qwen2.5-Math-1.5B-ctx8192" \
    "$ROOT/data/Qwen2.5-Math-1.5B"
  do
    if [ -e "$candidate/config.json" ] || [ -e "$candidate/model.safetensors" ] || [ -e "$candidate/model.safetensors.index.json" ]; then
      MODEL="$candidate"
      break
    fi
  done
fi
: "${MODEL:?set MODEL to the Qwen2.5-Math-1.5B directory under $ROOT/data}"
TRAIN="${TRAIN:-$ROOT/data/qwen-math/train}"
EVAL="${EVAL:-$ROOT/data/qwen-math/eval}"
OUT="${OUT:-$ROOT/outputs/qwen-math-1.5b}"
mkdir -p "$OUT"

for path in "$MODEL" "$TRAIN" "$EVAL"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
# Soft links into /data/models break on DLC if that host path is absent.
if [ -L "$MODEL/model.safetensors" ] && [ ! -e "$MODEL/model.safetensors" ]; then
  echo "broken weight symlink: $MODEL/model.safetensors -> $(readlink "$MODEL/model.safetensors")" >&2
  echo "put a real Qwen2.5-Math-1.5B tree under $ROOT/data (or set MODEL=...)" >&2
  exit 1
fi

echo "ROOT=$ROOT MODEL=$MODEL OUT=$OUT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$OUT/train.log"

if ray status >/dev/null 2>&1; then
  echo "A Ray cluster is already running; refusing to join it." >&2
  ray status >&2 || true
  exit 1
fi
# Raise open-file limits when the image allows it (Ray + vLLM open many fds).
ulimit -n 1048576 2>/dev/null || true

ray start --head --num-gpus="$((VLLM_NUM_ENGINES + ACTOR_GPUS))" --disable-usage-stats
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$ROOT/labs-molt"
python -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL" \
  --data.prompt_dataset "$TRAIN" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.apply_chat_template \
  --data.max_samples 7500 \
  --data.max_len 6144 \
  --rollout.batch_size 128 \
  --rollout.vllm_generate_batch_size 128 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens 4096 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size 128 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 34 \
  --train.async_queue_size 8 \
  --train.force_on_policy \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines "$VLLM_NUM_ENGINES" \
  --vllm.tensor_parallel_size 1 \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation sdpa \
  --actor.gradient_checkpoint full \
  --actor.optim adam \
  --actor.adam.lr 1e-6 \
  --actor.adam.weight_decay 0.1 \
  --actor.lr_scheduler cosine_with_min_lr \
  --actor.lr_warmup_ratio 0.03 \
  --actor.min_lr_ratio 0.1 \
  --actor.max_norm 1.0 \
  --actor.loss_agg_mode seq-mean-token-mean \
  --actor.entropy_coef 0 \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_mode mask \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold 3e-3 \
  --algo.kl.init_coef 0 \
  --train.agent_path "$ROOT/labs-molt/examples/python/agents/math.py" \
  --eval.dataset "$EVAL" \
  --eval.steps 100 \
  --eval.batch_size 8 \
  --eval.n_samples_per_prompt 16 \
  --eval.temperature 0.7 \
  --eval.top_p 0.7 \
  --ckpt.output_dir "$OUT/hf" \
  --ckpt.path "$OUT/state" \
  --ckpt.save_steps 50 \
  --ckpt.save_hf \
  --ckpt.max_num 40 \
  --ckpt.dcp_max_num 2 \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "$OUT/tb" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"
