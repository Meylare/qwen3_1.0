--- START OF FILE text/x-python ---

# src/training/train_stage2.py
"""
Stage 2 Training Script - Viral Prediction with LoRA.
Обучает модель предсказания виральности видео с использованием LoRA адаптеров.
Замораживает всё кроме LoRA параметров и viral_predictor.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from transformers import AutoTokenizer
import pandas as pd

import sys
import os
# Добавляем src директорию в путь
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, src_path)

from models.G2V2_model import G2V2Model
from data.datasets import ViralDataset, viral_collate_fn


def prepare_model_for_stage2(
    model_name: str,
    projector_weights_path: str,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_fake_model: bool = False
) -> G2V2Model:
    """
    Создает и подготавливает модель для Stage 2 обучения.
    Замораживает всё кроме LoRA параметров и viral_predictor.
    """
    if use_fake_model:
        # Создаем tiny модель для теста на CPU
        print("[TEST] Создание фейковой модели для теста...")
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
        config = Qwen2_5_VLConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=512,
        )
        model = G2V2Model(
            model_name=model_name,
            clap_dim=512,
            hidden_dim=128,  # Маленький hidden_dim для теста
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            _test_config=config
        )
    else:
        # Реальная модель
        print("[INFO] Загрузка реальной модели...")
        model = G2V2Model(
            model_name=model_name,
            clap_dim=512,
            hidden_dim=4096,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            projector_weights_path=projector_weights_path
        )

    # 1. Замораживаем ВСЁ по умолчанию
    for param in model.parameters():
        param.requires_grad = False

    # 2. Размораживаем LoRA параметры (если LoRA включен)
    if use_lora and hasattr(model, 'model') and hasattr(model.model, 'get_base_model'):
        try:
            # Получаем LoRA параметры через PEFT
            from peft import get_peft_model_state_dict
            lora_params = []
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    lora_params.append(param)
            print(f"[OK] Разморожено {len(lora_params)} LoRA параметров")
        except Exception as e:
            print(f"[WARN] Ошибка при разморозке LoRA: {e}")

    # 3. Размораживаем viral_predictor
    for param in model.viral_predictor.parameters():
        param.requires_grad = True

    # Проверка обучаемых параметров
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model prepared for Stage 2:")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Total parameters: {total_params:,}")
    print(".1%")

    return model


def create_optimizer_groups(model: G2V2Model, lora_lr: float = 2e-4, predictor_lr: float = 1e-3) -> torch.optim.AdamW:
    """
    Создает оптимизатор AdamW с двумя группами параметров:
    1. LoRA параметры с маленьким LR
    2. Viral predictor с большим LR
    """
    lora_params = []
    predictor_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' in name:
                lora_params.append(param)
            elif 'viral_predictor' in name:
                predictor_params.append(param)

    optimizer_groups = []

    if lora_params:
        optimizer_groups.append({
            'params': lora_params,
            'lr': lora_lr,
            'weight_decay': 0.01,
            'name': 'lora'
        })
        print(f"[INFO] LoRA группа: {len(lora_params)} параметров, LR={lora_lr}")

    if predictor_params:
        optimizer_groups.append({
            'params': predictor_params,
            'lr': predictor_lr,
            'weight_decay': 0.01,
            'name': 'predictor'
        })
        print(f"[INFO] Predictor группа: {len(predictor_params)} параметров, LR={predictor_lr}")

    if not optimizer_groups:
        raise ValueError("Нет обучаемых параметров!")

    return torch.optim.AdamW(optimizer_groups)


def train_stage2(
    model: G2V2Model,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    num_epochs: int = 5,
    save_path: str = "checkpoints/stage2_model",
    device: str = "cuda",
    log_every: int = 10,
    save_every: int = 100
):
    """
    Основной цикл обучения Stage 2.
    """
    model = model.to(device)
    mse_loss = nn.MSELoss()

    print(f"Starting Stage 2 training on {device}")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {train_loader.batch_size}")
    # ... логи ...

    os.makedirs(save_path, exist_ok=True)
    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Перемещаем данные на устройство
            # video_embeds уже [Batch, N, 3584/4096] (готовые)
            video_embeds = batch['video_embeds'].to(device).to(dtype=model.model.dtype)
            
            # Внимание: здесь берем 'video_attention_mask', если он есть в батче (из collate_fn)
            video_mask = batch.get('video_attention_mask')
            if video_mask is not None:
                video_mask = video_mask.to(device)

            # audio_embeds это [Batch, N, 512] (сырые CLAP)
            audio_embeds = batch['audio_embeds'].to(device).to(dtype=model.model.dtype)
            
            audio_mask = batch.get('audio_attention_mask')
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = batch['labels'].to(device).to(dtype=torch.float32)

            # --- ИСПРАВЛЕНИЕ ТУТ ---
            # Убрали ручной вызов get_text_embeddings и get_audio_embeddings.
            # Передаем данные напрямую.
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_features=video_embeds,    # Готовые тензоры из кэша
                audio_features=audio_embeds,    # Сырые векторы CLAP (512) -> Модель сама спроецирует
                video_attention_mask=video_mask, # Передаем маски, если модель обновлена для их приема
                audio_attention_mask=audio_mask,
                task_type="score"
            )  # [batch, 1]

            # MSE Loss
            loss = mse_loss(outputs.squeeze(-1), labels.squeeze(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Статистика
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                current_loss = loss.item()
                print(f"Step {global_step} | Loss: {current_loss:.6f}")

            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, loss.item(), save_path, f"step_{global_step}")

        # Конец эпохи
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} completed | Average loss: {avg_epoch_loss:.6f}")
            
            # Сохранение после каждой эпохи
            save_checkpoint(model, optimizer, global_step, avg_epoch_loss, save_path, f"epoch_{epoch+1}")
            
            # Лучший чекпоинт
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(model, optimizer, global_step, best_loss, save_path, "best")

    # Финальное сохранение
    save_checkpoint(model, optimizer, global_step, best_loss, save_path, "final")
    print(f"Training completed! Best loss: {best_loss:.6f}")


def save_checkpoint(
    model: G2V2Model,
    optimizer: torch.optim.AdamW,
    step: int,
    loss: float,
    save_path: str,
    checkpoint_name: str
):
    """Сохраняет чекпоинт модели и оптимизатора."""
    checkpoint_path = os.path.join(save_path, f"{checkpoint_name}.pt")

    checkpoint = {
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'use_lora': model.use_lora,
            'hidden_dim': model.hidden_dim,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"[SAVE] Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: G2V2Model, optimizer: Optional[torch.optim.AdamW] = None):
    """Загружает чекпоинт."""
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))

    print(f"[LOAD] Checkpoint loaded: {checkpoint_path}")
    print(f"   - Step: {step}")
    print(f"   - Loss: {loss:.6f}")

    return step, loss


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Training: Viral Prediction with LoRA")

    # Данные
    parser.add_argument("--parquet_file", type=str, default=None,
                       help="Путь к parquet файлу с данными для обучения (не нужен при --use_fake_data)")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Корневая директория с тензорами")

    # Модель
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Имя модели для загрузки")
    parser.add_argument("--projector_weights_path", type=str, default="checkpoints/projector.bin",
                       help="Путь к весам проектора из Stage 1")

    # LoRA конфигурация
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Использовать LoRA адаптеры")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")

    # Обучение
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Количество эпох")
    parser.add_argument("--lora_lr", type=float, default=2e-4,
                       help="Learning rate для LoRA параметров")
    parser.add_argument("--predictor_lr", type=float, default=1e-3,
                       help="Learning rate для viral predictor")
    parser.add_argument("--max_len", type=int, default=2048,
                       help="Максимальная длина последовательности")

    # Система
    parser.add_argument("--save_path", type=str, default="checkpoints/stage2",
                       help="Путь для сохранения чекпоинтов")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Устройство для обучения (cuda/cpu)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Путь к чекпоинту для возобновления обучения")
    parser.add_argument("--log_every", type=int, default=10,
                       help="Логировать каждые N шагов")
    parser.add_argument("--save_every", type=int, default=100,
                       help="Сохранять чекпоинт каждые N шагов")

    # Тестовый режим
    parser.add_argument("--use_fake_model", action="store_true",
                       help="Использовать фейковую модель для теста")
    parser.add_argument("--use_fake_data", action="store_true",
                       help="Использовать фейковые данные для теста")

    args = parser.parse_args()

    # Проверка аргументов
    if not args.use_fake_data and args.parquet_file is None:
        parser.error("--parquet_file is required when not using --use_fake_data")

    # Активация окружения conda (если нужно)
    if args.device.startswith("cuda"):
        print("[INFO] Using CUDA. Make sure conda environment 'qwen3-project' is activated")

    # Создаем токенизатор
    print("[INFO] Инициализация токенизатора...")
    if args.use_fake_model:
        # Простой fallback токенизатор для теста
        class SimpleTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1

            def encode(self, text, add_special_tokens=False, **kwargs):
                # Простая токенизация по словам
                words = text.split()
                return [hash(word) % 1000 for word in words]

            def get_vocab(self):
                return {}

        tokenizer = SimpleTokenizer()
        print("[TEST] Created simple tokenizer for testing")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            # Добавляем спец-токены если нужно
            special_tokens = ['<|audio|>', '<|video|>', '<|im_start|>', '<|im_end|>']
            vocab = tokenizer.get_vocab()
            new_tokens = [t for t in special_tokens if t not in vocab]
            if new_tokens:
                tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                print(f"Added special tokens: {new_tokens}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("[TEST] Создание простого токенизатора для теста...")
            class SimpleTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.eos_token_id = 1

                def encode(self, text, add_special_tokens=False, **kwargs):
                    words = text.split()
                    return [hash(word) % 1000 for word in words]

                def get_vocab(self):
                    return {}

            tokenizer = SimpleTokenizer()

    # Создаем датасет
    print("[INFO] Создание датасета...")
    if args.use_fake_data:
        # Создаем фейковый датасет для теста
        class FakeViralDataset(Dataset):
            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "video_embeds": torch.randn(10, 512),  # [seq_v, hidden_v]
                    "audio_embeds": torch.randn(5, 512),   # [seq_a, hidden_a]
                    "input_ids": torch.randint(0, 1000, (50,)),  # [seq_t]
                    "attention_mask": torch.ones(50, dtype=torch.long),  # [seq_t]
                    "label": torch.rand(1) * 10  # [1] - viral index
                }

        dataset = FakeViralDataset(100)
        print("[TEST] Created fake dataset for testing")
    else:
        dataset = ViralDataset(
            parquet_file=args.parquet_file,
            tokenizer=tokenizer,
            max_len=args.max_len,
            data_root=args.data_root
        )

    # Создаем DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # Лучше так, чтобы паддинг точно совпадал с токенизатором модели
        collate_fn=lambda x: viral_collate_fn(x, tokenizer=tokenizer), 
        num_workers=0,
        pin_memory=args.device.startswith("cuda")
    )
    # Создаем и подготавливаем модель
    print("[INFO] Подготовка модели...")
    model = prepare_model_for_stage2(
        model_name=args.model_name,
        projector_weights_path=args.projector_weights_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fake_model=args.use_fake_model
    )

    # Создаем оптимизатор
    print("[INFO] Создание оптимизатора...")
    optimizer = create_optimizer_groups(model, args.lora_lr, args.predictor_lr)

    # Логируем количество обучаемых параметров
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        # Fallback для моделей без PEFT
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

    # Возобновление обучения (если указано)
    start_step = 0
    if args.resume_from:
        print(f"[RESUME] Возобновление обучения из {args.resume_from}")
        start_step, _ = load_checkpoint(args.resume_from, model, optimizer)

    # Запуск обучения
    train_stage2(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        device=args.device,
        log_every=args.log_every,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()