#!/bin/bash
# =============================================================================
# GSPO Smoke-Test Script for Qwen3-Omni-30B MoE
# Fast validation of startup, vLLM rollout, and first optimization steps.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="/home/ubuntu/models/Qwen3-Omni-30B-A3B-Thinking"
OUTPUT_DIR="${SCRIPT_DIR}/output/qwen3omni_30b_a3b_thinking_gspo_test"
DATASET_PATH="${SCRIPT_DIR}/data/train.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/val.jsonl"
DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_zero2.json"

# Smoke-test defaults:
# - single GPU by default for simpler debugging
# - very short rollout and context to reduce KV-cache pressure
# - only a couple of optimizer steps to validate the full path
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29510}"

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
    --num_train_epochs 1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --num_generations 2 \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_completion_length 512 \
    --epsilon 0.2 \
    --beta 0.001 \
    --scale_rewards group \
    \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.44 \
    --vllm_max_model_len 4096 \
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
    --max_length 4096 \
    --truncation_strategy left \
    --gradient_checkpointing true \
    \
    --bf16 true \
    --tf32 true \
    --attn_impl flash_attn \
    \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 1000 \
    \
    --dataloader_num_workers 0 \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    \
    --report_to tensorboard \
    --logging_dir "${OUTPUT_DIR}/tb_logs"
