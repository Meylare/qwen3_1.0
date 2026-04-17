#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_PREFIX="${RUN_PREFIX:-qwen35_9b_base_pairwise_ab}"
SMOKE_PAIRS="${SMOKE_PAIRS:-10}"
MAX_STEPS="${MAX_STEPS:-32}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${RUN_PREFIX}_${SMOKE_PAIRS}pairs_${MAX_STEPS}steps_${RUN_TIMESTAMP}}"
LOG_FILE="${OUTPUT_DIR}/train.log"
PID_FILE="${OUTPUT_DIR}/train.pid"
COMMAND_FILE="${OUTPUT_DIR}/launch_command.sh"
META_FILE="${OUTPUT_DIR}/launch_meta.txt"

LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
CACHE_PROCESSED_SAMPLES="${CACHE_PROCESSED_SAMPLES:-1}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
DISABLE_UNSLOTH_COMPILE="${DISABLE_UNSLOTH_COMPILE:-0}"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/data/raw/train_fixed.jsonl}"
MODEL_PATH="${MODEL_PATH:-}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
TRAIN_ARGS_EXTRA="${TRAIN_ARGS_EXTRA:-}"

mkdir -p "${OUTPUT_DIR}"

{
  echo "#!/bin/bash"
  echo "set -euo pipefail"
  printf 'export OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
  printf 'export SMOKE_PAIRS=%q\n' "${SMOKE_PAIRS}"
  printf 'export MAX_STEPS=%q\n' "${MAX_STEPS}"
  printf 'export LOAD_IN_4BIT=%q\n' "${LOAD_IN_4BIT}"
  printf 'export MAX_SEQ_LENGTH=%q\n' "${MAX_SEQ_LENGTH}"
  printf 'export LOGGING_STEPS=%q\n' "${LOGGING_STEPS}"
  printf 'export SAVE_STEPS=%q\n' "${SAVE_STEPS}"
  printf 'export DATALOADER_NUM_WORKERS=%q\n' "${DATALOADER_NUM_WORKERS}"
  printf 'export CACHE_PROCESSED_SAMPLES=%q\n' "${CACHE_PROCESSED_SAMPLES}"
  printf 'export LORA_DROPOUT=%q\n' "${LORA_DROPOUT}"
  printf 'export DISABLE_UNSLOTH_COMPILE=%q\n' "${DISABLE_UNSLOTH_COMPILE}"
  printf 'export TRAIN_JSONL=%q\n' "${TRAIN_JSONL}"
  printf 'export WHISPER_MODEL=%q\n' "${WHISPER_MODEL}"
  printf 'export TRAIN_ARGS_EXTRA=%q\n' "${TRAIN_ARGS_EXTRA}"
  if [[ -n "${MODEL_PATH}" ]]; then
    printf 'export MODEL_PATH=%q\n' "${MODEL_PATH}"
  fi
  printf 'cd %q\n' "${SCRIPT_DIR}"
  printf 'exec bash %q\n' "${SCRIPT_DIR}/run_pairwise_ab_logits.sh"
} > "${COMMAND_FILE}"
chmod +x "${COMMAND_FILE}"

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'output_dir=%s\n' "${OUTPUT_DIR}"
  printf 'log_file=%s\n' "${LOG_FILE}"
  printf 'command_file=%s\n' "${COMMAND_FILE}"
} > "${META_FILE}"

nohup "${COMMAND_FILE}" > "${LOG_FILE}" 2>&1 &
PID="$!"
printf '%s\n' "${PID}" > "${PID_FILE}"

printf 'pid=%s\n' "${PID}"
printf 'output_dir=%s\n' "${OUTPUT_DIR}"
printf 'log_file=%s\n' "${LOG_FILE}"
printf 'command_file=%s\n' "${COMMAND_FILE}"
