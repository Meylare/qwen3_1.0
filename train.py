"""
train.py — точка входа для GRPO обучения.

Запуск:
    python train.py

Для дебага на маленьком датасете:
    python train.py --debug

С кастомными путями:
    python train.py --train_data ./data/train.jsonl --output_dir ./runs/exp1
"""
import argparse
import logging
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="./data/train.jsonl")
    p.add_argument("--eval_data", default=None)
    p.add_argument("--model_path", default="./Qwen3-Omni-30B-A3B-Thinking")
    p.add_argument("--output_dir", default="./checkpoints/virality_grpo")
    p.add_argument("--resume", default=None,
                   help="Путь к чекпоинту для продолжения обучения, например ./checkpoints/virality_grpo/checkpoint-latest")
    p.add_argument("--wandb", action="store_true",
                   help="Логировать в Weights & Biases")
    p.add_argument("--run_name", default=None,
                   help="Имя run'а в wandb. По умолчанию генерируется автоматически")
    p.add_argument("--debug", action="store_true",
                   help="Загрузить только 20 примеров, 1 эпоха")
    return p.parse_args()


def load_model_and_processor(
    model_path: str,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules=None,
):
    """
    Загружает Qwen3-Omni в режиме Thinking + Thinker-only (без Talker/речи).

    Thinking модель: <think>...</think> всегда включён безусловно.
    Квантизация: оба (policy + ref) в NF4 4-bit.
    На 96GB: ~15GB policy + ~15GB ref + ~20GB optimizer states + ~30GB KV/activations.
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Qwen3OmniMoeThinkerForConditionalGeneration — text-only вариант:
    #   - generate() возвращает просто тензор (не кортеж), не нужен return_audio=False
    #   - Talker не загружается совсем → экономия ~10GB vs enable_audio_output=False
    #   - Это именно то что содержит Qwen3-Omni-30B-A3B-Thinking чекпоинт
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"Loading processor from {model_path}")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # BitsAndBytes конфиг объявляем один раз — используется для обоих загрузок
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_qlora else None

    common_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # LOCAL_RANK задаётся torchrun при multi-GPU запуске.
        # При одиночном запуске (python train.py) переменная отсутствует → GPU 0.
        # Не используем device_map="auto": при QLoRA + LoRA адаптеры остаются
        # на GPU 0 пока base model распределяется — gradient flow ломается.
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    if bnb_config:
        common_kwargs["quantization_config"] = bnb_config

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )

    logger.info("Loading policy model (Qwen3-Omni Thinker, NF4 4-bit)...")
    model = get_peft_model(
        Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_path, **common_kwargs),
        lora_config,
    )
    # enable_input_require_grads() необходим для gradient checkpointing с LoRA:
    # без него градиент не течёт через frozen base layers в LoRA адаптеры
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    # use_reentrant=False обязателен с PEFT/LoRA на PyTorch >= 2.0
    # use_reentrant=True (default) несовместим с autograd hooks которые добавляет LoRA
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, processor


def main():
    args = parse_args()

    from config import TrainConfig
    from dataset import ViralityDataset
    from omni_trainer import OmniGRPOTrainer, ViralityCollator
    from reward import virality_reward_fn

    cfg = TrainConfig()

    # argparse default перекрывается если явно передан путь;
    # иначе берём из config (позволяет менять путь только в config.py)
    if args.model_path == "./Qwen3-Omni-30B-A3B-Thinking":
        args.model_path = cfg.model.model_path

    # Seed
    torch.manual_seed(cfg.grpo.seed)
    torch.cuda.manual_seed_all(cfg.grpo.seed)
    random.seed(cfg.grpo.seed)
    np.random.seed(cfg.grpo.seed)

    # ── Wandb ────────────────────────────────────────────────
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb не установлен. Запусти: pip install wandb")
        else:
            wandb.init(
                project="virality-grpo",
                name=args.run_name,
                # resume="allow" только при явном --resume.
                # Без этого два запуска с одинаковым именем без --resume
                # тихо продолжат один и тот же run в wandb вместо создания нового.
                resume="allow" if args.resume else None,
                config={
                    # Модель
                    "model_path": args.model_path,
                    "lora_r": cfg.model.lora_r,
                    "lora_alpha": cfg.model.lora_alpha,
                    # GRPO
                    "num_generations": cfg.grpo.num_generations,
                    "max_completion_length": cfg.grpo.max_completion_length,
                    "learning_rate": cfg.grpo.learning_rate,
                    "gradient_accumulation_steps": cfg.grpo.gradient_accumulation_steps,
                    "num_train_epochs": cfg.grpo.num_train_epochs,
                    "warmup_ratio": cfg.grpo.warmup_ratio,
                    # Reward
                    "correct_reward": cfg.reward.correct_reward,
                    "wrong_reward": cfg.reward.wrong_reward,
                    "thinking_max_tokens": cfg.reward.thinking_max_tokens,
                    # Данные
                    "video_fps": cfg.data.video_fps,
                    "max_video_duration": cfg.data.max_video_duration,
                    "train_data": args.train_data,
                    "seed": cfg.grpo.seed,
                },
            )
            logger.info(f"Wandb run: {wandb.run.url}")

    # Override для дебага
    if args.debug:
        logger.info("DEBUG MODE: small dataset, 1 epoch")
        cfg.grpo.num_train_epochs = 1
        cfg.grpo.logging_steps = 1
        cfg.grpo.save_steps = 50
        cfg.grpo.num_generations = 2
        max_samples = 20
    else:
        max_samples = None

    # ── Модель ───────────────────────────────────────────────
    model, processor = load_model_and_processor(
        args.model_path,
        use_qlora=cfg.model.use_qlora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )

    # ── Датасет ──────────────────────────────────────────────
    train_dataset = ViralityDataset(
        data_path=args.train_data,
        video_fps=cfg.data.video_fps,
        max_video_duration=cfg.data.max_video_duration,
        max_samples=max_samples,
    )
    logger.info(f"Train samples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_data:
        eval_dataset = ViralityDataset(
            data_path=args.eval_data,
            video_fps=cfg.data.video_fps,
            max_video_duration=cfg.data.max_video_duration,
            max_samples=50 if args.debug else None,
        )

    # ── Коллатор ─────────────────────────────────────────────
    collator = ViralityCollator(
        processor=processor,
        use_audio_in_video=cfg.data.use_audio_in_video,
        max_length=cfg.grpo.max_context_length,
        device="cuda",
    )

    # Batch size = 1: один видеопэйр за раз.
    # GRPO делает G генераций на каждый пример,
    # поэтому эффективно обрабатываем G пар за шаг.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.grpo.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # CUDA тензоры в коллаторе несовместимы с workers > 0
        pin_memory=False,
    )

    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    # ── Reward функция ───────────────────────────────────────
    # tokenizer нужен для точного подсчёта токенов в <think> (особенно для кириллицы)
    reward_fn = partial(
        virality_reward_fn,
        correct_reward=cfg.reward.correct_reward,
        wrong_reward=cfg.reward.wrong_reward,
        no_answer_penalty=cfg.reward.no_answer_penalty,
        format_bonus=cfg.reward.format_bonus,
        thinking_max_tokens=cfg.reward.thinking_max_tokens,
        length_penalty_max=cfg.reward.length_penalty_max,
        tokenizer=processor.tokenizer,
    )

    # ── Тренер ───────────────────────────────────────────────
    trainer = OmniGRPOTrainer(
        model=model,
        processor=processor,
        reward_fn=reward_fn,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        # GRPO params
        num_generations=cfg.grpo.num_generations,
        max_completion_length=cfg.grpo.max_completion_length,
        # Оптимизация
        learning_rate=cfg.grpo.learning_rate,
        gradient_accumulation_steps=cfg.grpo.gradient_accumulation_steps,
        num_train_epochs=cfg.grpo.num_train_epochs,
        warmup_ratio=cfg.grpo.warmup_ratio,
        # Технические
        use_audio_in_video=cfg.data.use_audio_in_video,
        output_dir=args.output_dir,
        logging_steps=cfg.grpo.logging_steps,
        save_steps=cfg.grpo.save_steps,
        weight_decay=cfg.grpo.weight_decay,
        use_wandb=args.wandb and WANDB_AVAILABLE,
        # temperature должна совпадать с generate() внутри тренера:
        # logprob computation делит logits на это же значение
        temperature=cfg.grpo.temperature,
    )

    # Resume from checkpoint если передан --resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Поехали ──────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()