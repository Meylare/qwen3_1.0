# src/training/train_stage1.py
"""
Stage 1 Training Script - Audio Alignment.

Обучает audio_projector генерировать описания аудио.
Замораживает всю модель кроме проектора.
Использует teacher forcing для next token prediction.
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
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

import sys
import os
# Добавляем src директорию в путь
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, src_path)

from models.G2V2_model import G2V2Model


class Stage1Dataset(Dataset):
    """
    Датасет для Stage 1 обучения.
    Загружает закэшированные аудио-векторы и капшны в память.
    """

    def __init__(
        self,
        embeddings_file: str,
        captions_file: str,
        tokenizer: Any,
        max_seq_len: int = 512,
        use_fake_data: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if use_fake_data:
            # Фейковые данные для теста на CPU
            self.embeddings = torch.randn(1000, 512)  # 1000 сэмплов по 512 измерений
            self.captions = [f"Sample audio description {i}" for i in range(1000)]
            print("Using fake data for testing")
        else:
            # Загрузка реальных данных
            if not os.path.exists(embeddings_file):
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            if not os.path.exists(captions_file):
                raise FileNotFoundError(f"Captions file not found: {captions_file}")

            # Загружаем тензор в память
            self.embeddings = torch.load(embeddings_file, map_location='cpu', weights_only=True)
            print(f"Loaded embeddings: {self.embeddings.shape}")

            # Загружаем капшны
            with open(captions_file, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            print(f"Loaded captions: {len(self.captions)}")

            # Проверка соответствия размеров
            if len(self.embeddings) != len(self.captions):
                raise ValueError(f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.captions)} captions")

        print(f"Dataset size: {len(self)} samples")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает:
        - audio_vec: [512] - CLAP вектор
        - input_ids: [seq_len] - токены промпта
        - labels: [seq_len] - labels для обучения (-100 для промпта, токены для ответа)
        """
        audio_vec = self.embeddings[idx]  # [512]
        caption = self.captions[idx]

        # Формируем промпт как в спецификации
        # User: <|audio|> Describe this sound. Assistant: {caption} <|im_end|>
        prompt = f"<|audio|> Describe this sound. Assistant: {caption} <|im_end|>"

        # Токенизируем
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Создаем labels
        labels = [-100] * len(tokens)  # По умолчанию игнорируем все

        # Находим позицию начала ответа (после "Assistant: ")
        assistant_marker = "Assistant: "
        assistant_tokens = self.tokenizer.encode(assistant_marker, add_special_tokens=False)

        # Ищем последовательность токенов assistant_marker
        for i in range(len(tokens) - len(assistant_tokens) + 1):
            if tokens[i:i+len(assistant_tokens)] == assistant_tokens:
                # Нашли начало ответа - начиная с этого места ставим реальные labels
                start_response = i + len(assistant_tokens)
                for j in range(start_response, len(tokens)):
                    labels[j] = tokens[j]
                break

        # Ограничиваем длину
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        return {
            'audio_vec': audio_vec,  # [512]
            'input_ids': torch.tensor(tokens, dtype=torch.long),  # [seq_len]
            'labels': torch.tensor(labels, dtype=torch.long),  # [seq_len]
            'attention_mask': torch.ones(len(tokens), dtype=torch.long)  # [seq_len]
        }


def prepare_model_for_stage1(
    model_name: str,
    clap_dim: int = 512,
    use_lora: bool = False,
    use_fake_model: bool = False
) -> G2V2Model:
    """
    Создает и подготавливает модель для Stage 1 обучения.
    Замораживает всё кроме audio_projector.
    """
    if use_fake_model:
        # Создаем tiny модель для теста на CPU
        print("🧪 Создание фейковой модели для теста...")
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
            clap_dim=clap_dim,
            hidden_dim=128,  # Маленький hidden_dim для теста
            use_lora=False,
            _test_config=config
        )
    else:
        # Реальная модель
        print("🏗️ Загрузка реальной модели...")
        model = G2V2Model(
            model_name=model_name,
            clap_dim=clap_dim,
            use_lora=use_lora
        )

    # 1. Замораживаем ВСЁ
    for param in model.parameters():
        param.requires_grad = False

    # 2. Размораживаем ТОЛЬКО проектор
    for param in model.audio_projector.parameters():
        param.requires_grad = True

    # Проверка
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model prepared for Stage 1:")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable percentage: {trainable_params/total_params:.1%}")

    return model


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Коллат функция для Stage 1 датасета.
    Паддит последовательности до максимальной длины в батче.
    """
    # Определяем максимальную длину в батче
    max_len = max(item['input_ids'].shape[0] for item in batch)

    # Паддим все последовательности
    padded_input_ids = []
    padded_labels = []
    padded_attention_masks = []
    audio_vecs = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]

        # Паддинг справа (как в каузальных моделях)
        pad_len = max_len - seq_len

        if pad_len > 0:
            # input_ids паддим токеном паддинга
            pad_token_id = 0  # Обычно 0 для большинства токенизаторов
            padded_input_ids.append(torch.cat([
                item['input_ids'],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ]))

            # labels паддим -100
            padded_labels.append(torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))

            # attention_mask: 1 для реальных токенов, 0 для паддинга
            padded_attention_masks.append(torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
        else:
            padded_input_ids.append(item['input_ids'])
            padded_labels.append(item['labels'])
            padded_attention_masks.append(item['attention_mask'])

        audio_vecs.append(item['audio_vec'])

    return {
        'audio_vecs': torch.stack(audio_vecs),  # [batch, 512]
        'input_ids': torch.stack(padded_input_ids),  # [batch, max_len]
        'labels': torch.stack(padded_labels),  # [batch, max_len]
        'attention_mask': torch.stack(padded_attention_masks)  # [batch, max_len]
    }


def train_stage1(
    model: G2V2Model,
    train_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    save_path: str = "projector.bin",
    device: str = "cpu",
    log_every: int = 10
):
    """
    Основной цикл обучения Stage 1.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    print(f"Starting Stage 1 training on {device}")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: {train_loader.batch_size}")
    print(f"   - Saving to: {save_path}")

    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            audio_vecs = batch['audio_vecs'].to(device).to(dtype=model.model.dtype)  # [batch, 512]
            input_ids = batch['input_ids'].to(device)    # [batch, seq_len]
            labels = batch['labels'].to(device)          # [batch, seq_len]
            attention_mask = batch['attention_mask'].to(device)  # [batch, seq_len]

            # Получаем текстовые эмбеддинги
            text_embeds = model.get_text_embeddings(input_ids)  # [batch, seq_len, hidden_dim]

            # Проецируем аудио
            audio_embeds = model.get_audio_embeddings(audio_vecs)  # [batch, 1, hidden_dim]

            # Конкатенируем: [audio_embeds, text_embeds]
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)  # [batch, seq_len+1, hidden_dim]

            # Расширяем attention_mask для аудио токена
            audio_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)
            extended_mask = torch.cat([audio_mask, attention_mask], dim=1)  # [batch, seq_len+1]

            # Forward pass через Qwen с inputs_embeds
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                labels=labels,  # Qwen сам посчитает loss
                output_hidden_states=False,
                return_dict=True
            )

            loss = outputs.loss

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
                print(f"Step {global_step} | Loss: {current_loss:.4f}")

                # Сохраняем лучший чекпоинт
                if current_loss < best_loss:
                    best_loss = current_loss
                    save_projector_checkpoint(model.audio_projector, save_path)
                    print(f"Saved best checkpoint: loss={best_loss:.4f}")

        # Конец эпохи
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed | Average loss: {avg_epoch_loss:.4f}")

    # Финальное сохранение
    save_projector_checkpoint(model.audio_projector, save_path)
    print(f"Training completed! Projector saved to {save_path}")
    print(f"   - Final loss: {best_loss:.4f}")


def save_projector_checkpoint(projector: nn.Module, save_path: str):
    """Сохраняет только веса проектора."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(projector.state_dict(), save_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training: Audio Alignment")
    parser.add_argument("--embeddings_file", type=str, default="data/wavcaps_embeddings.pt",
                       help="Путь к файлу с аудио эмбеддингами")
    parser.add_argument("--captions_file", type=str, default="data/wavcaps_captions.json",
                       help="Путь к файлу с описаниями аудио")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Имя модели для загрузки")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Максимальная длина последовательности")
    parser.add_argument("--save_path", type=str, default="checkpoints/projector.bin",
                       help="Путь для сохранения проектора")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство для обучения")
    parser.add_argument("--use_fake_data", action="store_true", help="Использовать фейковые данные для теста")
    parser.add_argument("--use_fake_model", action="store_true", help="Использовать фейковую модель для теста")
    parser.add_argument("--log_every", type=int, default=10, help="Логировать каждые N шагов")

    args = parser.parse_args()

    # Активация окружения conda (если нужно)
    if args.device == "cpu":
        print("Training on CPU. Make sure conda environment 'qwen3-project' is activated")

    # Создаем токенизатор
    print("🔧 Инициализация токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        # Добавляем спец-токены если нужно
        special_tokens = ['<|audio|>', '<|im_end|>']
        vocab = tokenizer.get_vocab()
        new_tokens = [t for t in special_tokens if t not in vocab]
        if new_tokens:
                    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                    print(f"Added special tokens: {new_tokens}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("🧪 Создание простого токенизатора для теста...")
        # Простой fallback токенизатор для теста
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 1000

            def encode(self, text, add_special_tokens=False):
                # Простая токенизация по словам
                words = text.split()
                return [hash(word) % 1000 for word in words]

            def get_vocab(self):
                return {}

        tokenizer = SimpleTokenizer()

    # Создаем датасет
    print("📚 Создание датасета...")
    dataset = Stage1Dataset(
        embeddings_file=args.embeddings_file,
        captions_file=args.captions_file,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        use_fake_data=args.use_fake_data
    )

    # Создаем DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Для CPU
    )

    # Создаем и подготавливаем модель
    print("🤖 Подготовка модели...")
    model = prepare_model_for_stage1(
        model_name=args.model_name,
        clap_dim=512,
        use_lora=False,  # Для Stage 1 LoRA не нужен
        use_fake_model=args.use_fake_model
    )

    # Запуск обучения
    train_stage1(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        device=args.device,
        log_every=args.log_every
    )


if __name__ == "__main__":
    main()