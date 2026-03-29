#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/output/qwen35_9b_base_gspo_smoke"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/../../train.jsonl}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../../models/Qwen3.5-9B-Base}"
SMOKE_PAIRS="${SMOKE_PAIRS:-10}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: ${VENV_DIR} not found. Run setup_env.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

echo ">>> Preparing smoke dataset"
python "${SCRIPT_DIR}/prepare_smoke_dataset.py" \
  --input "${TRAIN_JSONL}" \
  --output_dir "${DATA_DIR}" \
  --smoke_pairs "${SMOKE_PAIRS}" \
  --max_video_duration 45 \
  --video_fps 2 \
  --whisper_model "${WHISPER_MODEL}"

echo ">>> Running GSPO smoke training"
python "${SCRIPT_DIR}/train_gspo_qwen35_smoke.py" \
  --model_name "${MODEL_PATH}" \
  --train_dataset "${DATA_DIR}/train.jsonl" \
  --eval_dataset "${DATA_DIR}/eval.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --preprocess_metrics "${DATA_DIR}/preprocess_metrics.json"

INFER_DATASET="${DATA_DIR}/eval.jsonl"
if [[ ! -s "${INFER_DATASET}" ]]; then
  INFER_DATASET="${DATA_DIR}/train.jsonl"
fi

echo ">>> Running post-train inference on first eval sample"
python "${SCRIPT_DIR}/infer_pair_checkpoint.py" \
  --base_model "${MODEL_PATH}" \
  --adapter_dir "${OUTPUT_DIR}/final_lora" \
  --dataset_jsonl "${INFER_DATASET}" \
  --sample_index 0 \
  --output_json "${OUTPUT_DIR}/infer_eval_sample.json"

echo ">>> Done"
