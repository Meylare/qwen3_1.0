# tests/test_datasets.py
"""
Тесты для датасетов G2V2 модели.
Проверяет работу ViralDataset, SFTDataset, DPODataset и collate_fn.
"""

import torch
import pandas as pd
import tempfile
import shutil
import os
from pathlib import Path
import sys
import json
from functools import partial

# Добавляем корневую папку в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.datasets import ViralDataset, SFTDataset, DPODataset, viral_collate_fn


class MockTokenizer:
    """Мок токенизатора для тестирования."""

    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0  # Используем 0 как pad_token_id для совместимости
        self.eos_token_id = 1

    def __call__(self, text, max_length=None, padding=False, truncation=True, return_tensors="pt"):
        """Мок токенизации."""
        if isinstance(text, str):
            # Простая токенизация - каждый символ
            tokens = [ord(c) % self.vocab_size for c in text[:max_length or len(text)]]
            if len(tokens) == 0:
                tokens = [1]  # EOS token
        else:
            # Список текстов
            tokens = []
            for t in text:
                t_tokens = [ord(c) % self.vocab_size for c in t[:max_length or len(t)]]
                if len(t_tokens) == 0:
                    t_tokens = [1]
                tokens.append(t_tokens)

        # Создаем тензоры
        if isinstance(text, str):
            input_ids = torch.tensor([tokens])
            attention_mask = torch.ones_like(input_ids)
        else:
            # Для батча текстов
            max_len = max(len(t) for t in tokens)
            padded_tokens = []
            attention_masks = []

            for t in tokens:
                if len(t) < max_len:
                    pad_size = max_len - len(t)
                    t_padded = t + [0] * pad_size
                    mask = [1] * len(t) + [0] * pad_size
                else:
                    t_padded = t
                    mask = [1] * len(t)

                padded_tokens.append(t_padded)
                attention_masks.append(mask)

            input_ids = torch.tensor(padded_tokens)
            attention_mask = torch.tensor(attention_masks)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Мок chat template."""
        if tokenize:
            # Возвращаем токены напрямую
            text = self._format_messages(messages)
            return self(text, return_tensors="pt")["input_ids"]

        # Возвращаем отформатированный текст
        return self._format_messages(messages)

    def _format_messages(self, messages, add_generation_prompt=False):
        """Форматирование сообщений в текст."""
        result = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        return result


def create_fake_viral_data(temp_dir: str, num_samples: int = 3):
    """Создает фейковые данные для ViralDataset."""

    # Создание parquet файла
    data = []
    for i in range(num_samples):
        account_id = f"account_{i}"
        video_id = f"video_{i}"

        # Создание директории для тензоров
        tensor_dir = Path(temp_dir) / account_id / video_id
        tensor_dir.mkdir(parents=True, exist_ok=True)

        # Создание фейковых тензоров разной длины
        seq_len = 5 + i * 5  # 5, 10, 15 кадров

        # Video embeds: [seq_len, 3584]
        video_tensor = torch.randn(seq_len, 3584)
        torch.save(video_tensor, tensor_dir / "video_embeds.pt")

        # Audio embeds: [seq_len, 512]
        audio_tensor = torch.randn(seq_len, 512)
        torch.save(audio_tensor, tensor_dir / "audio_embeds.pt")

        # Добавление в dataframe
        data.append({
            'account_id': account_id,
            'video_id': video_id,
            'context_str': f"Context for video {i}",
            'transcript_text': f"This is transcript for video {i}",
            'viral_index': 0.1 + i * 0.3  # 0.1, 0.4, 0.7
        })

    df = pd.DataFrame(data)
    parquet_path = Path(temp_dir) / "train_stage2_fake.parquet"
    df.to_parquet(parquet_path)

    return str(parquet_path)


def create_fake_sft_data(temp_dir: str, num_samples: int = 3):
    """Создает фейковые данные для SFTDataset."""

    data = []
    for i in range(num_samples):
        problem_json = json.dumps({
            "problem": f"Technical problem {i}",
            "details": f"Detailed description {i}"
        })

        advice_text = f"This is advice number {i} for solving the problem."

        data.append({
            'problem_json': problem_json,
            'advice_text': advice_text
        })

    df = pd.DataFrame(data)
    parquet_path = Path(temp_dir) / "sft_tech_advice_fake.parquet"
    df.to_parquet(parquet_path)

    return str(parquet_path)


def create_fake_dpo_data(temp_dir: str, num_samples: int = 3):
    """Создает фейковые данные для DPODataset."""

    data = []
    for i in range(num_samples):
        prompt = f"What is the best way to solve problem {i}?"

        chosen = f"The optimal solution {i} involves multiple steps and careful consideration."

        rejected = f"Just do it randomly {i}."

        data.append({
            'prompt_text': prompt,
            'chosen_text': chosen,
            'rejected_text': rejected
        })

    df = pd.DataFrame(data)
    parquet_path = Path(temp_dir) / "dpo_self_play_fake.parquet"
    df.to_parquet(parquet_path)

    return str(parquet_path)


def test_viral_dataset():
    """Тест ViralDataset."""
    print("[VIRAL] Тестирование ViralDataset...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Создание фейковых данных
        parquet_path = create_fake_viral_data(temp_dir, num_samples=3)
        tokenizer = MockTokenizer()

        # Создание датасета
        dataset = ViralDataset(
            parquet_file=parquet_path,
            tokenizer=tokenizer,
            max_len=512,
            data_root=temp_dir
        )

        assert len(dataset) == 3, f"[FAIL] Ожидалось 3 сэмпла, получено {len(dataset)}"

        # Тестирование __getitem__
        sample = dataset[0]
        required_keys = ['video_embeds', 'audio_embeds', 'input_ids', 'attention_mask', 'label']

        for key in required_keys:
            assert key in sample, f"[FAIL] Отсутствует ключ {key}"

        # Проверка размерностей
        assert sample['video_embeds'].shape[1] == 3584, f"[FAIL] Видео эмбеддинги должны иметь размерность 3584, получено {sample['video_embeds'].shape[1]}"
        assert sample['audio_embeds'].shape[1] == 512, f"[FAIL] Аудио эмбеддинги должны иметь размерность 512, получено {sample['audio_embeds'].shape[1]}"
        assert sample['label'].item() >= 0.0 and sample['label'].item() <= 1.0, f"[FAIL] Лейбл должен быть в диапазоне [0,1], получено {sample['label'].item()}"

        # Тестирование второго сэмпла (разная длина)
        sample2 = dataset[1]
        assert sample2['video_embeds'].shape[0] == 10, f"[FAIL] Второй сэмпл должен иметь 10 кадров, получено {sample2['video_embeds'].shape[0]}"

        # Тестирование обработки transcript_text с JSON
        json_transcript = json.dumps([
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.5, "text": "This is a test"}
        ])

        # Создаем новый parquet с JSON transcript
        json_data = [{
            'account_id': 'account_json',
            'video_id': 'video_json',
            'context_str': 'Context text',
            'transcript_text': json_transcript,
            'viral_index': 0.8
        }]
        json_df = pd.DataFrame(json_data)
        json_parquet_path = Path(temp_dir) / "test_json_transcript.parquet"
        json_df.to_parquet(json_parquet_path)

        # Создаем тензоры для JSON теста
        json_tensor_dir = Path(temp_dir) / 'account_json' / 'video_json'
        json_tensor_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(5, 3584), json_tensor_dir / "video_embeds.pt")
        torch.save(torch.randn(5, 512), json_tensor_dir / "audio_embeds.pt")

        json_dataset = ViralDataset(
            parquet_file=str(json_parquet_path),
            tokenizer=tokenizer,
            max_len=512,
            data_root=temp_dir
        )

        json_sample = json_dataset[0]
        # Проверяем что сэмпл создался без ошибок
        assert 'input_ids' in json_sample, "JSON transcript должен быть обработан без ошибок"

        print("[OK] ViralDataset тест пройден")


def test_sft_dataset():
    """Тест SFTDataset."""
    print("[SFT] Тестирование SFTDataset...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Создание фейковых данных
        parquet_path = create_fake_sft_data(temp_dir, num_samples=3)
        tokenizer = MockTokenizer()

        # Создание датасета
        dataset = SFTDataset(
            parquet_file=parquet_path,
            tokenizer=tokenizer,
            max_len=512
        )

        assert len(dataset) == 3, f"[FAIL] Ожидалось 3 сэмпла, получено {len(dataset)}"

        # Тестирование __getitem__
        sample = dataset[0]
        required_keys = ['input_ids', 'attention_mask', 'labels']

        for key in required_keys:
            assert key in sample, f"[FAIL] Отсутствует ключ {key}"

        # Проверка что labels содержат -100
        has_masked_tokens = (sample['labels'] == -100).any().item()
        assert has_masked_tokens, "[FAIL] Labels должны содержать замаскированные токены (-100)"

        print("[OK] SFTDataset тест пройден")


def test_dpo_dataset():
    """Тест DPODataset."""
    print("[DPO] Тестирование DPODataset...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Создание фейковых данных
        parquet_path = create_fake_dpo_data(temp_dir, num_samples=3)
        tokenizer = MockTokenizer()

        # Создание датасета
        dataset = DPODataset(
            parquet_file=parquet_path,
            tokenizer=tokenizer,
            max_len=512
        )

        assert len(dataset) == 3, f"[FAIL] Ожидалось 3 сэмпла, получено {len(dataset)}"

        # Тестирование __getitem__
        sample = dataset[0]
        required_keys = [
            'prompt_input_ids', 'prompt_attention_mask',
            'chosen_input_ids', 'chosen_attention_mask',
            'rejected_input_ids', 'rejected_attention_mask'
        ]

        for key in required_keys:
            assert key in sample, f"[FAIL] Отсутствует ключ {key}"

        # Проверка что все последовательности не пустые
        assert sample['prompt_input_ids'].numel() > 0, "[FAIL] Prompt input_ids не должен быть пустым"
        assert sample['chosen_input_ids'].numel() > 0, "[FAIL] Chosen input_ids не должен быть пустым"
        assert sample['rejected_input_ids'].numel() > 0, "[FAIL] Rejected input_ids не должен быть пустым"

        print("[OK] DPODataset тест пройден")


def test_collate_fn():
    """Тест функции viral_collate_fn."""
    print("[COLLATE] Тестирование viral_collate_fn...")

    # Создание фейковых сэмплов разной длины
    samples = []

    for i in range(3):
        seq_len = 5 + i * 5  # 5, 10, 15

        sample = {
            'video_embeds': torch.randn(seq_len, 3584),    # Разная длина
            'audio_embeds': torch.randn(seq_len, 512),     # Разная длина
            'input_ids': torch.randint(0, 1000, (20 + i * 10,)),  # Разная длина текста
            'attention_mask': torch.ones(20 + i * 10),
            'label': torch.tensor(0.1 + i * 0.3, dtype=torch.float32)
        }
        samples.append(sample)

    # Применение collate_fn
    tokenizer = MockTokenizer()
    batch = viral_collate_fn(samples, tokenizer)

    # Проверка размерностей батча
    batch_size = 3

    # Видео: [batch, max_seq_v=15, hidden_v=3584]
    assert batch['video_embeds'].shape == (batch_size, 15, 3584), f"[FAIL] Видео размерность должна быть {(batch_size, 15, 3584)}, получено {batch['video_embeds'].shape}"
    assert batch['video_attention_mask'].shape == (batch_size, 15), f"[FAIL] Видео маска должна быть {(batch_size, 15)}, получено {batch['video_attention_mask'].shape}"

    # Аудио: [batch, max_seq_a=15, hidden_a=512]
    assert batch['audio_embeds'].shape == (batch_size, 15, 512), f"[FAIL] Аудио размерность должна быть {(batch_size, 15, 512)}, получено {batch['audio_embeds'].shape}"
    assert batch['audio_attention_mask'].shape == (batch_size, 15), f"[FAIL] Аудио маска должна быть {(batch_size, 15)}, получено {batch['audio_attention_mask'].shape}"

    # Текст: [batch, max_seq_t=40]
    assert batch['input_ids'].shape[0] == batch_size, f"[FAIL] Batch size должен быть {batch_size}, получено {batch['input_ids'].shape[0]}"
    assert batch['attention_mask'].shape[0] == batch_size, f"[FAIL] Batch size должен быть {batch_size}, получено {batch['attention_mask'].shape[0]}"

    # Лейблы: [batch, 1]
    assert batch['labels'].shape == (batch_size,), f"[FAIL] Лейблы должны быть {(batch_size,)}, получено {batch['labels'].shape}"

    # Проверка масок (первые элементы должны быть 1, последние 0 для коротких последовательностей)
    # Для видео: первый сэмпл имеет длину 5, так что mask[0, :5] = 1, mask[0, 5:] = 0
    assert batch['video_attention_mask'][0, :5].all(), "[FAIL] Первые 5 элементов видео маски должны быть 1"
    assert not batch['video_attention_mask'][0, 5:].any(), "Остальные элементы видео маски должны быть 0"

    print("[OK] viral_collate_fn тест пройден")


def test_dataloader_integration():
    """Интеграционный тест с DataLoader."""
    print("[DATALOADER] Тестирование интеграции с DataLoader...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Создание фейковых данных
        parquet_path = create_fake_viral_data(temp_dir, num_samples=4)
        tokenizer = MockTokenizer()

        # Создание датасета и DataLoader
        dataset = ViralDataset(
            parquet_file=parquet_path,
            tokenizer=tokenizer,
            max_len=512,
            data_root=temp_dir
        )

        from torch.utils.data import DataLoader

        # Создаем partial функцию с tokenizer
        from functools import partial
        collate_fn_with_tokenizer = partial(viral_collate_fn, tokenizer=tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn_with_tokenizer,
            shuffle=False
        )

        # Получение батча
        batch = next(iter(dataloader))

        # Проверка что батч не пустой и имеет правильные ключи
        expected_keys = [
            'video_embeds', 'video_attention_mask',
            'audio_embeds', 'audio_attention_mask',
            'input_ids', 'attention_mask', 'labels'
        ]

        for key in expected_keys:
            assert key in batch, f"[FAIL] Отсутствует ключ {key} в батче"

        # Проверка batch_size
        assert batch['video_embeds'].shape[0] == 2, f"[FAIL] Batch size должен быть 2, получено {batch['video_embeds'].shape[0]}"
        assert batch['labels'].shape[0] == 2, f"[FAIL] Batch size должен быть 2, получено {batch['labels'].shape[0]}"

        print("[OK] DataLoader интеграция тест пройдена")


def run_all_tests():
    """Запуск всех тестов."""
    print("[TEST] Запуск тестов датасетов G2V2...\n")

    try:
        test_viral_dataset()
        test_sft_dataset()
        test_dpo_dataset()
        test_collate_fn()
        test_dataloader_integration()

        print("\n[SUCCESS] Все тесты пройдены успешно!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Тест провален с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)