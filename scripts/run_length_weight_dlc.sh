#!/bin/bash
# One --algo.length_weight arm on DLC (run_qwen_math_dlc.sh).
#
#   ALPHA=0.75 REF=refs/flash_len.json bash scripts/run_length_weight_dlc.sh
# ALPHA=1.0 is the control: w_i = 1, loss bit-identical to plain Flash, same lw/* logs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ALPHA:?set ALPHA, e.g. 0.75 / 1.0 / 1.25}"
: "${REF:?set REF to a reference JSON from scripts/build_length_reference.py}"
REF="$(realpath "$REF")"
[ -f "$REF" ] || { echo "missing reference file: $REF" >&2; exit 1; }
export OUT="${OUT:-$ROOT/outputs/qwen-math-1.5b-lw_a${ALPHA}}"
exec "$ROOT/scripts/run_qwen_math_dlc.sh" \
  --algo.length_weight.enabled \
  --algo.length_weight.alpha "$ALPHA" \
  --algo.length_weight.reference_file "$REF" \
  --algo.length_weight.raw_weight_min "${RAW_MIN:-0.5}" \
  --algo.length_weight.raw_weight_max "${RAW_MAX:-2.0}" \
  "$@"
