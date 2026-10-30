#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY=iac
MODEL=Qwen/Qwen3-8B
CLASSIFIER="${ROOT}/checkpoints/intent_e5_large_instruct_30cls.pt"
STATS=""
CLOCK=wall
CAPACITY=""
MEM_FRACTION=0.85
PORT=30000
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) POLICY="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --stats) STATS="$2"; shift 2 ;;
        --clock) CLOCK="$2"; shift 2 ;;
        --capacity) CAPACITY="$2"; shift 2 ;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

source "${ROOT}/scripts/env.sh"

ARGS=(--model-path "${MODEL}" --port "${PORT}" --mem-fraction-static "${MEM_FRACTION}"
      --radix-eviction-policy "${POLICY}" --enable-cache-report)
[[ -n "${CAPACITY}" ]] && ARGS+=(--max-total-tokens "${CAPACITY}")
if [[ "${POLICY}" == iac ]]; then
    ARGS+=(--iac-clock "${CLOCK}" --iac-classifier-path "${CLASSIFIER}")
    [[ -n "${STATS}" ]] && ARGS+=(--iac-stats-path "${STATS}")
fi

exec python -m sglang.launch_server "${ARGS[@]}" "${EXTRA[@]}"
