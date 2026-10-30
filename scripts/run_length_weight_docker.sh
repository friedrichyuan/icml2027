#!/bin/bash
# One --algo.length_weight arm of the 1.5B math recipe in docker (run_qwen_math_docker.sh).
#   ALPHA=0.75 REF=refs/flash_len.json scripts/run_length_weight_docker.sh [extra train flags]
# ALPHA=1.0 is the control: the weight is exactly 1, the loss is bit-identical to plain Flash,
# and it logs the same length/* and lw/* diagnostics as the other arms.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ALPHA:?set ALPHA, e.g. 0.75 / 1.0 / 1.25}"
: "${REF:?set REF to a reference JSON from scripts/build_length_reference.py}"
REF="$(realpath "$REF")"
[[ "$REF" == "$ROOT"/* ]] || { echo "REF must live under $ROOT (the only host path mounted in the container)" >&2; exit 1; }
export OUT="${OUT:-$ROOT/outputs/qwen-math-1.5b-lw_a${ALPHA}}"
export NAME="${NAME:-frl-qwen-math-1.5b-lw_a${ALPHA}}"
exec "$ROOT/scripts/run_qwen_math_docker.sh" \
  --algo.length_weight.enabled \
  --algo.length_weight.alpha "$ALPHA" \
  --algo.length_weight.reference_file "$REF" \
  --algo.length_weight.raw_weight_min "${RAW_MIN:-0.5}" \
  --algo.length_weight.raw_weight_max "${RAW_MAX:-2.0}" \
  "$@"
