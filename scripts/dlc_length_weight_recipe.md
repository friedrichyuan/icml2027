# DLC: fixed length-weight ablation (FlashREINFORCE, Qwen2.5-Math-1.5B)

Code root on the node: `/mnt/data/yuanmingqi/code/icml2027`.  
The job already runs inside the molt image — use the `*_dlc.sh` scripts, not the docker wrappers.

Weights must resolve under `$ROOT/data` (set `MODEL=...` if the directory name differs).  
Prompt/eval datasets: `$ROOT/data/qwen-math/{train,eval}`.

Each full arm is ~33 h on 8 GPUs (4 vLLM + 4 actor).

## 1. Dump lengths from a plain Flash run (reference sample)

```bash
cd /mnt/data/yuanmingqi/code/icml2027
mkdir -p outputs/flash_ref refs
OUT=outputs/flash_ref bash scripts/run_qwen_math_dlc.sh \
  --logger.rollout_lengths_file /mnt/data/yuanmingqi/code/icml2027/outputs/flash_ref/rollout_lengths.jsonl
```

Stop after a few hundred steps once the dump is thick enough (e.g. steps 1–200), then freeze the reference:

```bash
python scripts/build_length_reference.py \
  outputs/flash_ref/rollout_lengths.jsonl --steps 1:200 --out refs/flash_len.json
```

The script prints `T0` and per-α `c`. Reuse the same `refs/flash_len.json` for every α arm.

## 2. Run the three α arms (one job each)

```bash
cd /mnt/data/yuanmingqi/code/icml2027
REF=refs/flash_len.json

ALPHA=1.0  bash scripts/run_length_weight_dlc.sh   # control: w=1
ALPHA=0.75 bash scripts/run_length_weight_dlc.sh   # up-weight long
ALPHA=1.25 bash scripts/run_length_weight_dlc.sh   # down-weight long
```

Outputs land in `outputs/qwen-math-1.5b-lw_a{1.0,0.75,1.25}/`.  
Startup log should contain a line like:
`[LengthWeight] alpha=... T0=... c=... reference=...`

## 3. Optional: turn_linear × length_weight (round 2)

Append the baseline switch; length-weight is orthogonal:

```bash
ALPHA=0.75 REF=refs/flash_len.json bash scripts/run_length_weight_dlc.sh \
  --algo.advantage.flash_baseline turn_linear
```

## 4. Compare

```bash
python scripts/compare_runs.py \
  outputs/qwen-math-1.5b-lw_a1.0 \
  outputs/qwen-math-1.5b-lw_a0.75 \
  outputs/qwen-math-1.5b-lw_a1.25 \
  --budget_tokens 2000
```

## Sanity checks before a long job

```bash
# weights / data present
ls data/models/*/config.json data/qwen-math/train data/qwen-math/eval

# 5-step smoke (override episodes via train flags if you add a short recipe later)
OUT=outputs/smoke bash scripts/run_qwen_math_dlc.sh --train.num_episodes 1 --data.max_samples 256
```

`CUDA_VISIBLE_DEVICES`, `VLLM_NUM_ENGINES`, `ACTOR_GPUS`, `MODEL`, `OUT` are all overridable env vars.
