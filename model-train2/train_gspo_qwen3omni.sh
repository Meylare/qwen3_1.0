#!/bin/bash
# =============================================================================
# GSPO Training Script for Qwen3-Omni-30B MoE — Thinking, text-only output
# QLoRA (NF4, 4-bit) — RTX PRO 6000 Blackwell 96 GB
# =============================================================================
# Requirements:
#   pip install ms-swift[llm] -U
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#   pip install flash-attn --no-build-isolation
#   pip install bitsandbytes>=0.44.0
#   pip install decord "qwen_omni_utils>=0.0.9"
#
# VRAM-бюджет (оценка на 1 GPU):
#   Модель 4-bit NF4 (train)   : ~18 GB
#   LoRA-адаптеры bf16         : ~2  GB
#   Активации 65k ctx          : ~20-35 GB
#   Оптимизатор AdamW          : ~8  GB  (только адаптеры, offload на CPU)
#   Итого                      : ~48-63 GB — в пределах 96 GB
#
# CPU RAM 200 GB: DeepSpeed ZeRO-2 offload оптимизатора
#
# ms-swift docs: https://swift.readthedocs.io
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. PATHS & IDENTIFIERS
# ---------------------------------------------------------------------------
MODEL_ID="/home/ubuntu/models/Qwen3-Omni-30B-A3B-Thinking"
# MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"

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
# num_generations=8:
#   generation_batch_size = per_device_train_batch_size(2) * grad_accum(8) = 16 -> 16 % 8 = 0 OK
#   per_device_eval_batch_size = 8 -> 8 % 8 = 0 OK

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  -m swift.cli.rlhf \
    --rlhf_type grpo \
    --importance_sampling_level sequence \
    \
    --model "${MODEL_ID}" \
    --model_type qwen3_omni_moe \
    \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    \
    --num_train_epochs 3 \
    --max_steps -1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_completion_length 8192 \
    --epsilon 0.2 \
    --beta 0.001 \
    --scale_rewards group \
    \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_enforce_eager \
    --vllm_limit_mm_per_prompt '{"video": 2}' \
    \
    --enable_thinking true \
    \
    --reward_funcs accuracy \
    \
    --freeze_aligner false \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj \
    \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant true \
    \
    --max_length 65536 \
    --truncation_strategy left \
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
