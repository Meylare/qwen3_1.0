# src/training/train_stage3.py
"""
Stage 3 Training Script - Multi-Task Learning.
Обучает модель сразу на трех задачах: Viral Prediction, SFT, DPO.
Использует MultiTaskDataset, MultiTaskCollator и G2V2MultiTaskTrainer.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

import sys
import os
# Добавляем src директорию в путь
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, src_path)

from models.G2V2_model import G2V2Model
from data.datasets import ViralDataset, SFTDataset, DPODataset, viral_collate_fn, _collate_text_sequences


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer=None) -> Dict[str, torch.Tensor]:
    """
    Функция сборки батча для SFTDataset.
    Простой паддинг текстовых последовательностей с labels.
    """
    if not batch:
        return {}

    # Извлечение данных
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Определение pad_token_id
    if tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0  # fallback

    # Паддинг текстовых последовательностей
    text_inputs = _collate_text_sequences(input_ids, attention_masks, pad_token_id)

    # Паддинг labels (используем -100 для паддинга)
    max_len = max(label.size(0) for label in labels)
    padded_labels = []
    for label in labels:
        if label.size(0) < max_len:
            pad_size = max_len - label.size(0)
            pad_tensor = torch.full((pad_size,), -100, dtype=label.dtype)
            padded_label = torch.cat([label, pad_tensor])
        else:
            padded_label = label
        padded_labels.append(padded_label)

    return {
        "input_ids": text_inputs["input_ids"],          # [batch, max_seq]
        "attention_mask": text_inputs["attention_mask"], # [batch, max_seq]
        "labels": torch.stack(padded_labels)             # [batch, max_seq] с -100 для паддинга
    }


def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer=None) -> Dict[str, torch.Tensor]:
    """
    Функция сборки батча для DPODataset.
    Паддинг для prompt, chosen и rejected последовательностей.
    """
    if not batch:
        return {}

    # Извлечение данных
    prompt_input_ids = [item['prompt_input_ids'] for item in batch]
    prompt_attention_masks = [item['prompt_attention_mask'] for item in batch]
    chosen_input_ids = [item['chosen_input_ids'] for item in batch]
    chosen_attention_masks = [item['chosen_attention_mask'] for item in batch]
    rejected_input_ids = [item['rejected_input_ids'] for item in batch]
    rejected_attention_masks = [item['rejected_attention_mask'] for item in batch]

    # Определение pad_token_id
    if tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0  # fallback

    # Паддинг prompt
    prompt_inputs = _collate_text_sequences(prompt_input_ids, prompt_attention_masks, pad_token_id)

    # Паддинг chosen
    chosen_inputs = _collate_text_sequences(chosen_input_ids, chosen_attention_masks, pad_token_id)

    # Паддинг rejected
    rejected_inputs = _collate_text_sequences(rejected_input_ids, rejected_attention_masks, pad_token_id)

    return {
        "prompt_input_ids": prompt_inputs["input_ids"],          # [batch, max_seq]
        "prompt_attention_mask": prompt_inputs["attention_mask"], # [batch, max_seq]
        "chosen_input_ids": chosen_inputs["input_ids"],          # [batch, max_seq]
        "chosen_attention_mask": chosen_inputs["attention_mask"], # [batch, max_seq]
        "rejected_input_ids": rejected_inputs["input_ids"],      # [batch, max_seq]
        "rejected_attention_mask": rejected_inputs["attention_mask"]  # [batch, max_seq]
    }


def multitask_collate_fn(batch: List[Dict[str, Any]], tokenizer=None) -> Dict[str, Any]:
    """
    Универсальная функция сборки батча для MultiTaskDataset.
    Разделяет сэмплы по task_type и применяет соответствующие collate функции.
    """
    if not batch:
        return {}

    # Разделяем сэмплы по типам
    viral_samples = []
    sft_samples = []
    dpo_samples = []

    for sample in batch:
        task_type = sample['task_type']
        if task_type == 'viral':
            viral_samples.append(sample)
        elif task_type == 'sft':
            sft_samples.append(sample)
        elif task_type == 'dpo':
            dpo_samples.append(sample)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # Создаем пустой результат
    result = {}

    # Обрабатываем виральные сэмплы
    if viral_samples:
        viral_batch = viral_collate_fn(viral_samples, tokenizer)
        result.update({
            'viral_' + k: v for k, v in viral_batch.items()
        })

    # Обрабатываем SFT сэмплы
    if sft_samples:
        sft_batch = sft_collate_fn(sft_samples, tokenizer)
        result.update({
            'sft_' + k: v for k, v in sft_batch.items()
        })

    # Обрабатываем DPO сэмплы
    if dpo_samples:
        dpo_batch = dpo_collate_fn(dpo_samples, tokenizer)
        result.update({
            'dpo_' + k: v for k, v in dpo_batch.items()
        })

    # Добавляем информацию о присутствии каждого типа задач
    result['has_viral'] = len(viral_samples) > 0
    result['has_sft'] = len(sft_samples) > 0
    result['has_dpo'] = len(dpo_samples) > 0

    return result


class MultiTaskDataset(Dataset):
    """
    Объединяет три датасета (Viral, SFT, DPO) в один.
    Добавляет поле task_type для каждого сэмпла.
    """

    def __init__(
        self,
        viral_dataset: Optional[ViralDataset] = None,
        sft_dataset: Optional[SFTDataset] = None,
        dpo_dataset: Optional[DPODataset] = None
    ):
        self.viral_dataset = viral_dataset
        self.sft_dataset = sft_dataset
        self.dpo_dataset = dpo_dataset

        # Создаем список всех сэмплов с их типами
        self.samples = []

        # Добавляем виральные сэмплы
        if viral_dataset:
            for i in range(len(viral_dataset)):
                self.samples.append(('viral', i))

        # Добавляем SFT сэмплы
        if sft_dataset:
            for i in range(len(sft_dataset)):
                self.samples.append(('sft', i))

        # Добавляем DPO сэмплы
        if dpo_dataset:
            for i in range(len(dpo_dataset)):
                self.samples.append(('dpo', i))

        print(f"[INFO] MultiTaskDataset создан с {len(self.samples)} сэмплами:")
        if viral_dataset:
            print(f"  - Viral: {len(viral_dataset)} сэмплов")
        if sft_dataset:
            print(f"  - SFT: {len(sft_dataset)} сэмплов")
        if dpo_dataset:
            print(f"  - DPO: {len(dpo_dataset)} сэмплов")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_type, sample_idx = self.samples[idx]

        if task_type == 'viral':
            sample = self.viral_dataset[sample_idx]
        elif task_type == 'sft':
            sample = self.sft_dataset[sample_idx]
        elif task_type == 'dpo':
            sample = self.dpo_dataset[sample_idx]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Добавляем task_type в сэмпл
        sample['task_type'] = task_type
        return sample


class G2V2MultiTaskTrainer(Trainer):
    """
    Кастомный Trainer для многозадачного обучения.
    Переопределяет compute_loss для обработки Viral, SFT и DPO задач.
    """

    def __init__(
        self,
        model: G2V2Model,
        args: TrainingArguments,
        alpha: float = 1.0,  # вес для viral loss
        beta: float = 1.0,   # вес для DPO loss
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model: G2V2Model, inputs: Dict[str, Any], return_outputs: bool = False):
        """
        Вычисляет общий loss для смеси задач.
        """
        total_loss = 0.0
        loss_components = {}

        # 1. VIRAL LOSS - если есть виральные данные
        if inputs.get('has_viral', False):
            viral_loss = self._compute_viral_loss(model, inputs)
            total_loss += self.alpha * viral_loss
            loss_components['viral'] = viral_loss.item()

        # 2. SFT LOSS - если есть SFT данные
        if inputs.get('has_sft', False):
            sft_loss = self._compute_sft_loss(model, inputs)
            total_loss += sft_loss  # SFT loss имеет вес 1.0
            loss_components['sft'] = sft_loss.item()

        # 3. DPO LOSS - если есть DPO данные
        if inputs.get('has_dpo', False):
            dpo_loss = self._compute_dpo_loss(model, inputs)
            total_loss += self.beta * dpo_loss
            loss_components['dpo'] = dpo_loss.item()

        # Логируем компоненты loss
        if hasattr(self.state, 'log_history') and self.state.log_history:
            self.state.log_history[-1].update(loss_components)

        if return_outputs:
            return total_loss, None  # Возвращаем None вместо outputs для простоты
        return total_loss

    def _compute_viral_loss(self, model: G2V2Model, inputs: Dict[str, Any]) -> torch.Tensor:
        """Вычисляет MSE loss для виральной задачи."""
        # Извлекаем виральные данные
        outputs = model(
            input_ids=inputs['viral_input_ids'],
            attention_mask=inputs['viral_attention_mask'],
            video_features=inputs['viral_video_embeds'],
            audio_features=inputs['viral_audio_embeds'],
            video_attention_mask=inputs['viral_video_attention_mask'],
            audio_attention_mask=inputs['viral_audio_attention_mask'],
            task_type="score"
        )  # [batch, 1]

        labels = inputs['viral_labels']  # [batch, 1]

        # MSE Loss
        loss = self.mse_loss(outputs.squeeze(-1), labels.squeeze(-1))
        return loss

    def _compute_sft_loss(self, model: G2V2Model, inputs: Dict[str, Any]) -> torch.Tensor:
        """Вычисляет Cross-Entropy loss для SFT задачи."""
        # Извлекаем SFT данные
        outputs = model(
            input_ids=inputs['sft_input_ids'],
            attention_mask=inputs['sft_attention_mask'],
            task_type="text"
        )  # [batch, vocab_size]

        labels = inputs['sft_labels']  # [batch, seq_len] с -100 для игнорирования

        # Flatten для CrossEntropyLoss
        outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch*seq, vocab]
        labels_flat = labels.view(-1)  # [batch*seq]

        # Cross-Entropy Loss (автоматически игнорирует -100)
        loss = self.ce_loss(outputs_flat, labels_flat)
        return loss

    def _compute_dpo_loss(self, model: G2V2Model, inputs: Dict[str, Any]) -> torch.Tensor:
        """Вычисляет DPO loss для preference optimization."""
        # Извлекаем DPO данные
        prompt_ids = inputs['dpo_prompt_input_ids']
        prompt_mask = inputs['dpo_prompt_attention_mask']
        chosen_ids = inputs['dpo_chosen_input_ids']
        chosen_mask = inputs['dpo_chosen_attention_mask']
        rejected_ids = inputs['dpo_rejected_input_ids']
        rejected_mask = inputs['dpo_rejected_attention_mask']

        batch_size = prompt_ids.size(0)

        # Создаем полные последовательности для chosen и rejected
        # Конкатенируем prompt + response для каждого
        chosen_full_ids = []
        chosen_full_mask = []
        rejected_full_ids = []
        rejected_full_mask = []

        for i in range(batch_size):
            # Chosen: prompt + chosen response
            chosen_seq = torch.cat([prompt_ids[i], chosen_ids[i]], dim=0)
            chosen_seq_mask = torch.cat([prompt_mask[i], chosen_mask[i]], dim=0)
            chosen_full_ids.append(chosen_seq)
            chosen_full_mask.append(chosen_seq_mask)

            # Rejected: prompt + rejected response
            rejected_seq = torch.cat([prompt_ids[i], rejected_ids[i]], dim=0)
            rejected_seq_mask = torch.cat([prompt_mask[i], rejected_mask[i]], dim=0)
            rejected_full_ids.append(rejected_seq)
            rejected_full_mask.append(rejected_seq_mask)

        # Паддинг до одинаковой длины
        max_len = max(seq.size(0) for seq in chosen_full_ids + rejected_full_ids)

        def pad_sequences(sequences, masks, max_len, device):
            padded_seqs = []
            padded_masks = []
            for seq, mask in zip(sequences, masks):
                if seq.size(0) < max_len:
                    pad_len = max_len - seq.size(0)
                    pad_tensor = torch.zeros(pad_len, dtype=seq.dtype, device=device)
                    padded_seq = torch.cat([seq, pad_tensor])
                    padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype, device=device)])
                else:
                    padded_seq = seq[:max_len]
                    padded_mask = mask[:max_len]
                padded_seqs.append(padded_seq)
                padded_masks.append(padded_mask)
            return torch.stack(padded_seqs), torch.stack(padded_masks)

        # Определяем устройство из входных данных
        device = prompt_ids.device

        chosen_padded_ids, chosen_padded_mask = pad_sequences(chosen_full_ids, chosen_full_mask, max_len, device)
        rejected_padded_ids, rejected_padded_mask = pad_sequences(rejected_full_ids, rejected_full_mask, max_len, device)

        # Прогоняем модель для chosen последовательностей
        chosen_outputs = model(
            input_ids=chosen_padded_ids,
            attention_mask=chosen_padded_mask,
            task_type="text"
        )  # [batch, seq, vocab]

        # Прогоняем модель для rejected последовательностей
        rejected_outputs = model(
            input_ids=rejected_padded_ids,
            attention_mask=rejected_padded_mask,
            task_type="text"
        )  # [batch, seq, vocab]

        # Получаем логиты для последнего токена каждой последовательности
        chosen_logits = chosen_outputs[:, -1, :]  # [batch, vocab]
        rejected_logits = rejected_outputs[:, -1, :]  # [batch, vocab]

        # DPO Loss (упрощенная версия)
        # Используем фиксированную температуру и beta
        temperature = 1.0
        beta_dpo = 0.1

        chosen_log_probs = torch.log_softmax(chosen_logits / temperature, dim=-1)
        rejected_log_probs = torch.log_softmax(rejected_logits / temperature, dim=-1)

        # Предполагаем, что target токены - это последний токен response
        # В реальности нужно правильно определить позиции
        chosen_targets = chosen_ids[:, -1]  # Последний токен chosen response
        rejected_targets = rejected_ids[:, -1]  # Последний токен rejected response

        chosen_log_p = chosen_log_probs.gather(-1, chosen_targets.unsqueeze(-1)).squeeze(-1)
        rejected_log_p = rejected_log_probs.gather(-1, rejected_targets.unsqueeze(-1)).squeeze(-1)

        # DPO Loss
        dpo_loss = -torch.log(torch.sigmoid(beta_dpo * (chosen_log_p - rejected_log_p))).mean()

        return dpo_loss


def prepare_model_for_stage3(
    model_name: str,
    stage2_checkpoint: str,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_fake_model: bool = False
) -> G2V2Model:
    """
    Создает и подготавливает модель для Stage 3 обучения.
    Замораживает всё кроме LoRA параметров.
    """
    if use_fake_model:
        # Создаем tiny модель для теста
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
        # Загружаем модель из Stage 2 чекпоинта
        print(f"[INFO] Загрузка модели из Stage 2 чекпоинта: {stage2_checkpoint}")

        # Сначала создаем модель с той же конфигурацией
        model = G2V2Model(
            model_name=model_name,
            clap_dim=512,
            hidden_dim=4096,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Загружаем чекпоинт Stage 2
        checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Stage 2 чекпоинт загружен: {stage2_checkpoint}")

    # Для Stage 3 размораживаем только LoRA параметры (если они есть)
    # Viral predictor остается замороженным, как в Stage 2
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем LoRA параметры
    if use_lora and hasattr(model, 'model') and hasattr(model.model, 'get_base_model'):
        try:
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            print(f"[OK] LoRA параметры разморожены для Stage 3")
        except Exception as e:
            print(f"[WARN] Ошибка при разморозке LoRA: {e}")

    # Проверяем обучаемые параметры
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model prepared for Stage 3:")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Total parameters: {total_params:,}")
    print(".1%")

    return model


def train_stage3(
    model: G2V2Model,
    multitask_dataset: MultiTaskDataset,
    tokenizer: Any,
    alpha: float = 1.0,
    beta: float = 1.0,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    save_path: str = "checkpoints/stage3",
    device: str = "cuda",
    log_every: int = 10,
    save_every: int = 100
):
    """
    Основной цикл обучения Stage 3 - Multi-Task Learning.
    """
    model = model.to(device)

    print(f"Starting Stage 3 training on {device}")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Alpha (viral weight): {alpha}")
    print(f"   - Beta (DPO weight): {beta}")
    print(f"   - Learning rate: {learning_rate}")

    # Создаем DataLoader
    train_loader = DataLoader(
        multitask_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: multitask_collate_fn(x, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=device.startswith("cuda")
    )

    # Настраиваем TrainingArguments
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=log_every,
        save_steps=save_every,
        save_total_limit=3,
        remove_unused_columns=False,  # Важно для custom collate_fn
        dataloader_pin_memory=device.startswith("cuda"),
        dataloader_num_workers=0,
    )

    # Создаем кастомный Trainer
    trainer = G2V2MultiTaskTrainer(
        model=model,
        args=training_args,
        alpha=alpha,
        beta=beta,
        train_dataset=multitask_dataset,  # Для совместимости с HF Trainer
    )

    # Запускаем обучение
    trainer.train()

    # Финальное сохранение
    final_path = os.path.join(save_path, "final_model")
    trainer.save_model(final_path)
    print(f"[SAVE] Final model saved: {final_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Training: Multi-Task Learning")

    # Данные
    parser.add_argument("--viral_parquet", type=str, default=None,
                       help="Путь к parquet файлу с виральными данными")
    parser.add_argument("--sft_parquet", type=str, default=None,
                       help="Путь к parquet файлу с SFT данными")
    parser.add_argument("--dpo_parquet", type=str, default=None,
                       help="Путь к parquet файлу с DPO данными")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Корневая директория с тензорами")

    # Модель
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Имя модели для загрузки")
    parser.add_argument("--stage2_checkpoint", type=str, default="checkpoints/stage2/final.pt",
                       help="Путь к чекпоинту Stage 2")

    # LoRA конфигурация
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Использовать LoRA адаптеры")
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")

    # Multi-task веса
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Вес для viral loss")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Вес для DPO loss")

    # Обучение
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max_len", type=int, default=2048,
                       help="Максимальная длина последовательности")

    # Система
    parser.add_argument("--save_path", type=str, default="checkpoints/stage3",
                       help="Путь для сохранения чекпоинтов")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Устройство для обучения (cuda/cpu)")
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
    if not args.use_fake_data:
        if not args.viral_parquet and not args.sft_parquet and not args.dpo_parquet:
            parser.error("Хотя бы один parquet файл должен быть указан (или используйте --use_fake_data)")

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

    # Создаем датасеты
    print("[INFO] Создание датасетов...")

    viral_dataset = None
    sft_dataset = None
    dpo_dataset = None

    if args.use_fake_data:
        # Создаем фейковые датасеты для теста
        class FakeViralDataset(Dataset):
            def __init__(self, size=50):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "video_embeds": torch.randn(10, 512),
                    "audio_embeds": torch.randn(5, 512),
                    "input_ids": torch.randint(0, 1000, (50,)),
                    "attention_mask": torch.ones(50, dtype=torch.long),
                    "label": torch.rand(1)
                }

        class FakeSFTDataset(Dataset):
            def __init__(self, size=50):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                seq_len = 50
                return {
                    "input_ids": torch.randint(0, 1000, (seq_len,)),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.randint(0, 1000, (seq_len,))
                }

        class FakeDPODataset(Dataset):
            def __init__(self, size=50):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                seq_len = 30
                return {
                    "prompt_input_ids": torch.randint(0, 1000, (seq_len,)),
                    "prompt_attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "chosen_input_ids": torch.randint(0, 1000, (seq_len,)),
                    "chosen_attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "rejected_input_ids": torch.randint(0, 1000, (seq_len,)),
                    "rejected_attention_mask": torch.ones(seq_len, dtype=torch.long)
                }

        viral_dataset = FakeViralDataset(50)
        sft_dataset = FakeSFTDataset(50)
        dpo_dataset = FakeDPODataset(50)
        print("[TEST] Created fake datasets for testing")
    else:
        # Реальные датасеты
        if args.viral_parquet:
            viral_dataset = ViralDataset(
                parquet_file=args.viral_parquet,
                tokenizer=tokenizer,
                max_len=args.max_len,
                data_root=args.data_root
            )

        if args.sft_parquet:
            sft_dataset = SFTDataset(
                parquet_file=args.sft_parquet,
                tokenizer=tokenizer,
                max_len=args.max_len
            )

        if args.dpo_parquet:
            dpo_dataset = DPODataset(
                parquet_file=args.dpo_parquet,
                tokenizer=tokenizer,
                max_len=args.max_len
            )

    # Создаем MultiTaskDataset
    multitask_dataset = MultiTaskDataset(
        viral_dataset=viral_dataset,
        sft_dataset=sft_dataset,
        dpo_dataset=dpo_dataset
    )

    # Создаем и подготавливаем модель
    print("[INFO] Подготовка модели...")
    model = prepare_model_for_stage3(
        model_name=args.model_name,
        stage2_checkpoint=args.stage2_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fake_model=args.use_fake_model
    )

    # Запуск обучения
    train_stage3(
        model=model,
        multitask_dataset=multitask_dataset,
        tokenizer=tokenizer,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        device=args.device,
        log_every=args.log_every,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()