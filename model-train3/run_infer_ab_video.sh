#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv4"
MODEL_DIR="${MODEL_DIR:-/home/ubuntu/models/Qwen3.5-27B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/home/ubuntu/model-train/data}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-32}"
LOG_FILE="${SCRIPT_DIR}/output/vllm_server.log"
VRAM_CSV="${SCRIPT_DIR}/output/vram_metrics.csv"
RUN_METRICS_JSON="${SCRIPT_DIR}/output/run_metrics.json"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
MEDIA_IO_KWARGS="$(printf '{"video":{"num_frames":%s}}' "${VIDEO_NUM_FRAMES}")"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: ${VENV_DIR} not found. Run setup_env.sh first."
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: model directory not found: ${MODEL_DIR}"
  echo "Download model first with:"
  echo "  source ${VENV_DIR}/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3.5-27B', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False, resume_download=True)\""
  exit 1
fi

source "${VENV_DIR}/bin/activate"

cleanup() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" || true
    wait "${GPU_MONITOR_PID}" || true
  fi

  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo ">>> Stopping vLLM server (PID ${VLLM_PID})"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

START_TS="$(date +%s.%N)"

echo "timestamp,memory_used_mib,memory_total_mib,util_gpu_pct,util_mem_pct" > "${VRAM_CSV}"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits \
      | head -n 1 >> "${VRAM_CSV}"
    sleep 1
  done
) &
GPU_MONITOR_PID=$!

echo ">>> Starting vLLM server on port ${VLLM_PORT}"
: > "${LOG_FILE}"
vllm serve "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser qwen3 \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --media-io-kwargs "${MEDIA_IO_KWARGS}" \
  > "${LOG_FILE}" 2>&1 &
VLLM_PID=$!

echo ">>> Waiting for server readiness"
READY=0
MAX_TRIES=$((READY_TIMEOUT_SEC / 2))
for _ in $(seq 1 "${MAX_TRIES}"); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "ERROR: vLLM server did not become ready in time"
  echo "See logs: ${LOG_FILE}"
  exit 1
fi

READY_TS="$(date +%s.%N)"
SERVER_STARTUP_S="$(awk "BEGIN {print ${READY_TS}-${START_TS}}")"

echo ">>> Running A/B video inference"
INFER_START_TS="$(date +%s.%N)"
OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1" \
OPENAI_API_KEY="EMPTY" \
MODEL_NAME="${MODEL_NAME}" \
OUTPUT_JSON="${SCRIPT_DIR}/output/infer_ab_video_result.json" \
python "${SCRIPT_DIR}/infer_ab_video.py"
INFER_END_TS="$(date +%s.%N)"
TOTAL_END_TS="${INFER_END_TS}"
TOTAL_RUNTIME_S="$(awk "BEGIN {print ${TOTAL_END_TS}-${START_TS}}")"
INFER_WALL_S="$(awk "BEGIN {print ${INFER_END_TS}-${INFER_START_TS}}")"

if [[ -n "${GPU_MONITOR_PID:-}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
  kill "${GPU_MONITOR_PID}" || true
  wait "${GPU_MONITOR_PID}" || true
  unset GPU_MONITOR_PID
fi

read -r PEAK_VRAM_MIB AVG_VRAM_MIB TOTAL_VRAM_MIB <<<"$(awk -F',' '
  NR==1 {next}
  {
    used=$2+0;
    total=$3+0;
    if (used>peak) peak=used;
    sum+=used;
    n+=1;
    total_mem=total;
  }
  END {
    avg=(n>0)?sum/n:0;
    printf "%.0f %.2f %.0f", peak, avg, total_mem;
  }' "${VRAM_CSV}")"

python - <<PY
import json
from pathlib import Path

run_metrics = {
    "model": "${MODEL_NAME}",
    "model_dir": "${MODEL_DIR}",
    "vllm_port": int("${VLLM_PORT}"),
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "video_num_frames": int("${VIDEO_NUM_FRAMES}"),
    "gpu_memory_utilization": float("${GPU_MEMORY_UTILIZATION}"),
    "server_startup_s": round(float("${SERVER_STARTUP_S}"), 3),
    "inference_wall_s": round(float("${INFER_WALL_S}"), 3),
    "total_runtime_s": round(float("${TOTAL_RUNTIME_S}"), 3),
    "peak_vram_mib": int(float("${PEAK_VRAM_MIB}")),
    "avg_vram_mib": round(float("${AVG_VRAM_MIB}"), 2),
    "total_vram_mib": int(float("${TOTAL_VRAM_MIB}")),
}
if run_metrics["total_vram_mib"] > 0:
    run_metrics["peak_vram_pct"] = round(
        run_metrics["peak_vram_mib"] / run_metrics["total_vram_mib"] * 100, 2
    )
else:
    run_metrics["peak_vram_pct"] = None

out_path = Path("${RUN_METRICS_JSON}")
out_path.write_text(json.dumps(run_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved run metrics: {out_path}")
PY

echo ">>> Done"
