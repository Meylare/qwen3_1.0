#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HOME_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv5"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/qwen35_9b_base_pairwise_ab}"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/data/raw/train_fixed.jsonl}"
MODEL_PATH="${MODEL_PATH:-${HOME_ROOT}/models/Qwen3.5-9B-Base}"
SMOKE_PAIRS="${SMOKE_PAIRS:-10}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
MAX_STEPS="${MAX_STEPS:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
CACHE_PROCESSED_SAMPLES="${CACHE_PROCESSED_SAMPLES:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
DISABLE_UNSLOTH_COMPILE="${DISABLE_UNSLOTH_COMPILE:-1}"
TRAIN_ARGS_EXTRA="${TRAIN_ARGS_EXTRA:-}"
STAGE_TIMINGS_JSON="${OUTPUT_DIR}/stage_timings.json"
STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

current_time_sec() {
  python3 - <<'PY'
import time
print(f"{time.perf_counter():.6f}")
PY
}

elapsed_sec() {
  python3 - <<'PY' "$1" "$2"
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{end - start:.4f}")
PY
}

write_stage_timings_json() {
  python3 - <<'PY' "$STAGE_TIMINGS_JSON" "$STARTED_AT_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PREPARE_STAGE_SEC:-}" "${TRAIN_STAGE_SEC:-}" "${INFER_STAGE_SEC:-}" "${TOTAL_STAGE_SEC:-}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "started_at_utc": sys.argv[2],
    "finished_at_utc": sys.argv[3],
    "prepare_smoke_dataset_sec": float(sys.argv[4]) if sys.argv[4] else None,
    "train_pairwise_ab_logits_sec": float(sys.argv[5]) if sys.argv[5] else None,
    "infer_pair_checkpoint_sec": float(sys.argv[6]) if sys.argv[6] else None,
    "total_runtime_sec": float(sys.argv[7]) if sys.argv[7] else None,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: ${VENV_DIR} not found. Run setup_env.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
mkdir -p "${OUTPUT_DIR}"
TOTAL_START_SEC="$(current_time_sec)"

echo ">>> Preparing smoke dataset"
PREPARE_STAGE_START_SEC="$(current_time_sec)"
python "${SCRIPT_DIR}/prepare_smoke_dataset.py" \
  --input "${TRAIN_JSONL}" \
  --output_dir "${DATA_DIR}" \
  --smoke_pairs "${SMOKE_PAIRS}" \
  --max_video_duration 45 \
  --video_fps 2 \
  --whisper_model "${WHISPER_MODEL}"
PREPARE_STAGE_END_SEC="$(current_time_sec)"
PREPARE_STAGE_SEC="$(elapsed_sec "${PREPARE_STAGE_START_SEC}" "${PREPARE_STAGE_END_SEC}")"
printf '>>> Stage timing: prepare_smoke_dataset_sec=%s\n' "${PREPARE_STAGE_SEC}"

echo ">>> Running pairwise A/B logits training"
TRAIN_CMD=(
  python -u "${SCRIPT_DIR}/train_pairwise_ab_logits.py"
  --model_name "${MODEL_PATH}"
  --train_dataset "${DATA_DIR}/train.jsonl"
  --eval_dataset "${DATA_DIR}/eval.jsonl"
  --output_dir "${OUTPUT_DIR}"
  --preprocess_metrics "${DATA_DIR}/preprocess_metrics.json"
  --max_steps "${MAX_STEPS}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --lora_dropout "${LORA_DROPOUT}"
)

if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  TRAIN_CMD+=(--load_in_4bit)
fi

if [[ "${CACHE_PROCESSED_SAMPLES}" == "0" ]]; then
  TRAIN_CMD+=(--no_cache_processed_samples)
fi

if [[ "${DISABLE_UNSLOTH_COMPILE}" == "0" ]]; then
  TRAIN_CMD+=(--enable_unsloth_compile)
fi

if [[ -n "${TRAIN_ARGS_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${TRAIN_ARGS_EXTRA} )
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

printf '>>> Training config: max_steps=%s max_seq_length=%s load_in_4bit=%s cache_processed_samples=%s dataloader_num_workers=%s dataloader_prefetch_factor=%s lora_dropout=%s disable_unsloth_compile=%s\n' \
  "${MAX_STEPS}" "${MAX_SEQ_LENGTH}" "${LOAD_IN_4BIT}" "${CACHE_PROCESSED_SAMPLES}" "${DATALOADER_NUM_WORKERS}" "${DATALOADER_PREFETCH_FACTOR}" "${LORA_DROPOUT}" "${DISABLE_UNSLOTH_COMPILE}"
TRAIN_STAGE_START_SEC="$(current_time_sec)"
"${TRAIN_CMD[@]}"
TRAIN_STAGE_END_SEC="$(current_time_sec)"
TRAIN_STAGE_SEC="$(elapsed_sec "${TRAIN_STAGE_START_SEC}" "${TRAIN_STAGE_END_SEC}")"
printf '>>> Stage timing: train_pairwise_ab_logits_sec=%s\n' "${TRAIN_STAGE_SEC}"

INFER_DATASET="${DATA_DIR}/eval.jsonl"
if [[ ! -s "${INFER_DATASET}" ]]; then
  INFER_DATASET="${DATA_DIR}/train.jsonl"
fi

echo ">>> Running post-train A/B logits inference on first eval sample"
INFER_STAGE_START_SEC="$(current_time_sec)"
python "${SCRIPT_DIR}/infer_pair_checkpoint.py" \
  --base_model "${MODEL_PATH}" \
  --adapter_dir "${OUTPUT_DIR}/final_lora" \
  --dataset_jsonl "${INFER_DATASET}" \
  --sample_index 0 \
  --response_mode answer_only \
  --output_json "${OUTPUT_DIR}/infer_eval_sample_answer_only.json"
INFER_STAGE_END_SEC="$(current_time_sec)"
INFER_STAGE_SEC="$(elapsed_sec "${INFER_STAGE_START_SEC}" "${INFER_STAGE_END_SEC}")"
printf '>>> Stage timing: infer_pair_checkpoint_sec=%s\n' "${INFER_STAGE_SEC}"

TOTAL_END_SEC="$(current_time_sec)"
TOTAL_STAGE_SEC="$(elapsed_sec "${TOTAL_START_SEC}" "${TOTAL_END_SEC}")"
printf '>>> Stage timing: total_runtime_sec=%s\n' "${TOTAL_STAGE_SEC}"
write_stage_timings_json

echo ">>> Done"
