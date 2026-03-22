#!/bin/bash
# =============================================================================
# GSPO Training Script for Qwen3-Omni-30B MoE — Thinking, text-only output
# QLoRA (NF4, 4-bit) + vLLM rollout — RTX PRO 6000 Blackwell 96 GB
# =============================================================================
# Requirements:
#   pip install ms-swift[llm] -U
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#   pip install flash-attn --no-build-isolation
#   pip install bitsandbytes>=0.44.0
#   pip install vllm>=0.8.0
#
# VRAM-бюджет (оценка на 1 GPU):
#   Модель 4-bit NF4 (train)   : ~18 GB
#   LoRA-адаптеры bf16         : ~2  GB
#   Активации 65k ctx          : ~20-35 GB
#   Оптимизатор AdamW          : ~8  GB  (только адаптеры, offload на CPU)
#   vLLM rollout (shared GPU)  : ~12 GB  (vllm_gpu_memory_utilization=0.15)
#   Итого                      : ~60-75 GB — в пределах 96 GB
#
# CPU RAM 200 GB: DeepSpeed ZeRO-2 offload оптимизатора
#
# GSPO paper: https://arxiv.org/abs/2506.xxxxx
# ms-swift docs: https://swift.readthedocs.io
# =============================================================================

set -euo pipefail

# Абсолютные пути, чтобы запускать из любой директории
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. PATHS & IDENTIFIERS
# ---------------------------------------------------------------------------
# Thinking text-only вариант Qwen3-Omni-30B:
# принимает видео/аудио на ВХОД, генерирует только текст (без аудио-декодера)
MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"
# MODEL_ID="/path/to/local/Qwen3-Omni-30B-A3B-Thinking"   # или локальный путь

OUTPUT_DIR="${SCRIPT_DIR}/output/qwen3omni_30b_a3b_thinking_gspo"
DATASET_PATH="${SCRIPT_DIR}/data/train.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/val.jsonl"
REWARD_FUNCS_PATH="${SCRIPT_DIR}/reward_functions.py"
DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_zero2.json"

# ---------------------------------------------------------------------------
# 2. DISTRIBUTED TRAINING SETTINGS
# ---------------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MASTER_PORT=${MASTER_PORT:-29500}

# ---------------------------------------------------------------------------
# 3. RUN
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  -m swift.cli.rlhf \
    --rlhf_type gspo \
    --importance_sampling_level sequence \
    \
    --model_id_or_path "${MODEL_ID}" \
    --model_type qwen3-omni \
    --disable_audio_output true \
    \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.15 \
    --vllm_max_model_len 65536 \
    --vllm_enforce_eager false \
    \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    \
    --num_train_epochs 3 \
    --max_steps -1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --gspo_num_generations 6 \
    --gspo_temperature 0.9 \
    --gspo_top_p 0.95 \
    --gspo_max_new_tokens 8192 \
    --gspo_epsilon 0.2 \
    --gspo_beta 0.001 \
    --gspo_norm_rewards true \
    \
    --thinking_mode true \
    --thinking_budget 16384 \
    \
    --reward_funcs_path "${REWARD_FUNCS_PATH}" \
    --reward_funcs virality_accuracy virality_format virality_calibration \
    \
    --sft_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules all-linear \
    \
    --quantization_bit 4 \
    --quant_type nf4 \
    --bnb_4bit_compute_dtype bf16 \
    --bnb_4bit_use_double_quant true \
    \
    --max_length 65536 \
    --truncation_strategy right \
    --gradient_checkpointing true \
    \
    --bf16 true \
    --tf32 true \
    --attn_impl flash_attn \
    \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    \
    --dataloader_num_workers 4 \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    \
    --report_to tensorboard \
    --logging_dir "${OUTPUT_DIR}/tb_logs"
