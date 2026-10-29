#!/bin/bash
# Launch run_qwen_math.sh in a detached hijkzzz/molt:latest container on host GPUs 2-9.
# Extra arguments are appended to the Molt command (later flags override the recipe), e.g.
#   OUT=outputs/qwen-math-1.5b-turn_linear scripts/run_qwen_math_docker.sh --algo.advantage.flash_baseline turn_linear
# Follow with: docker logs -f <name>   (name defaults to frl-<basename of OUT>)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${OUT:-$ROOT/outputs/qwen-math-1.5b-docker}")"
NAME="${NAME:-frl-$(basename "$OUT")}"
HOST_GPUS="${HOST_GPUS:-2,3,4,5,6,7,8,9}"
NUM_GPUS="$(awk -F, '{print NF}' <<< "$HOST_GPUS")"

# The repo is mounted at its host path, so OUT and the dataset/model paths mean the same inside;
# data/models/* symlinks into /data/models. Inside the container the cards are renumbered 0..N-1.
docker run -d --name "$NAME" --gpus "\"device=$HOST_GPUS\"" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 --shm-size=64g \
  -v "$ROOT:$ROOT" -v /data/models:/data/models:ro -v "$HOME/.cache:/root/.cache" \
  -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))" \
  -e PYTHONPATH="$ROOT/labs-molt" -e OUT="$OUT" \
  --entrypoint bash hijkzzz/molt:latest "$ROOT/scripts/run_qwen_math.sh" "$@"
echo "started $NAME -> $OUT"
