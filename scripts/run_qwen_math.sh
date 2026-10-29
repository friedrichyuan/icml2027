#!/bin/bash
# FlashREINFORCE Qwen2.5-Math-1.5B on 7.5k deduplicated DAPO prompts.
# Paper budget is 256k rollouts / step ~2000: 34 passes of 7500 prompts.
# Eight 4090s: vLLM on the first four visible devices, actor on the last four.
# Attention is SDPA without sequence packing because Transformer Engine / flash-attn
# are not built for this CUDA 13 + Ada stack.
set -euo pipefail

export PATH="/home/ps/miniconda3/envs/flashreinforce/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_AGENT_TURNS=1
export WANDB_MODE=disabled
# vLLM is placed first, so 2-5 sample and 6-9 train. The actor cards share one PCIe switch.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7,8,9}"
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/data/models/Qwen2.5-Math-1.5B-ctx8192}"
TRAIN="${TRAIN:-$ROOT/data/qwen-math/train}"
EVAL="${EVAL:-$ROOT/data/qwen-math/eval}"
OUT="${OUT:-$ROOT/outputs/qwen-math-1.5b}"
mkdir -p "$OUT"

# Shared machine: wait until two 4090s have enough free memory. vLLM is placed
# on the first visible device, the actor on the second.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  while true; do
    if picked=$(python - << 'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
rows = []
for line in out.strip().splitlines():
    idx_s, total_s, used_s = [part.strip() for part in line.split(",")]
    idx, total, used = int(idx_s), int(total_s), int(used_s)
    rows.append((total - used, idx, total, used))
rows.sort(reverse=True)
# vLLM is placed on the first visible GPU. Give the freer card to the actor.
actor, vllm = rows[0], rows[1]
# Other tenants releasing memory mid-init trips vLLM's profiler, so both cards
# must be idle, not merely "enough free bytes".
if actor[3] > 1024 or vllm[3] > 1024 or actor[0] < 36 * 1024 or vllm[0] < 36 * 1024:
    summary = ", ".join(f"{idx}:{used}MiB" for _, idx, _, used in sorted(rows, key=lambda r: r[1]))
    print(f"waiting for two 4090s with enough free memory ({summary})", file=sys.stderr)
    sys.exit(1)
util = min(0.85, max(0.35, (vllm[0] - 1536) / vllm[2]))
print(f"{vllm[1]},{actor[1]}")
print(f"{util:.2f}")
PY
    ); then
      break
    fi
    sleep 20
  done
  export CUDA_VISIBLE_DEVICES="$(printf '%s\n' "$picked" | sed -n '1p')"
  export VLLM_GPU_MEMORY_UTILIZATION="$(printf '%s\n' "$picked" | sed -n '2p')"
fi
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} vLLM util=${VLLM_GPU_MEMORY_UTILIZATION}" | tee -a "$OUT/train.log"

if ray status >/dev/null 2>&1; then
  echo "A Ray cluster is already running; refusing to join it." >&2
  ray status >&2 || true
  exit 1
fi

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
