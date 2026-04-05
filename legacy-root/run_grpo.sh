#!/bin/bash
# run_grpo.sh — запуск GRPO обучения через ms-swift
#
# Подготовка данных (запустить один раз):
#   python dataset_converter.py \
#       --input ./data/train_ready.jsonl \
#       --output ./data/train_swift.jsonl
#
# Debug запуск (20 примеров, проверка что всё работает):
#   bash run_grpo.sh --debug
#
# Полный запуск:
#   bash run_grpo.sh

set -e

MODEL_PATH="/home/ubuntu/models/qwen3-omni-30b-thinking-awq-4bit"
TRAIN_DATA="./data/train_swift.jsonl"
EVAL_DATA="./data/train_swift_eval.jsonl"
OUTPUT_DIR="./checkpoints/virality_grpo_swift"

# Флаги
DEBUG=false
for arg in "$@"; do
    case $arg in
        --debug) DEBUG=true ;;
    esac
done

if [ "$DEBUG" = true ]; then
    echo "=== DEBUG MODE ==="
    MAX_STEPS=20
    SAVE_STEPS=20
    EVAL_STEPS=20
    LOG_STEPS=1
    NUM_GENERATIONS=2
    MAX_COMPLETION_LENGTH=300
    TRAIN_DATA_LIMIT="#20"  # ms-swift синтаксис для лимита примеров
else
    MAX_STEPS=-1  # -1 = весь датасет
    SAVE_STEPS=100
    EVAL_STEPS=100
    LOG_STEPS=5
    NUM_GENERATIONS=2
    MAX_COMPLETION_LENGTH=600
    TRAIN_DATA_LIMIT=""
fi

CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    \
    --model "$MODEL_PATH" \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj \
    \
    --dataset "${TRAIN_DATA}${TRAIN_DATA_LIMIT}" \
    --val_dataset "$EVAL_DATA" \
    \
    --external_plugins ./plugin.py \
    --reward_funcs virality \
    \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_base_url "http://localhost:8091" \
    \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --temperature 0.9 \
    --top_p 0.95 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    \
    --max_length 65000 \
    \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOG_STEPS \
    --output_dir "$OUTPUT_DIR" \
    \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    \
    --log_completions true \
    --report_to wandb \
    \
    --torch_dtype bfloat16 \
    --dataloader_num_workers 0