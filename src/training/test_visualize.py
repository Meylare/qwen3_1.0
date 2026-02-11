"""
Тестовый скрипт для проверки визуализаций без загрузки модели.
Создает mock данные и тестирует функции визуализации.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.visualize_qwen_features import (
    create_attention_heatmap,
    create_token_importance_plot,
    create_frame_importance_plot,
    create_video_token_heatmap_on_frames,
    map_tokens_to_frames
)
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "visualizations/test_output"


def create_mock_processor():
    """Создает mock processor для тестирования."""
    try:
        # Пытаемся загрузить реальный processor
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True
        )
        return processor
    except Exception as e:
        logger.warning(f"Не удалось загрузить processor: {e}")
        logger.info("Используем mock processor")
        
        # Создаем простой mock
        class MockProcessor:
            class MockTokenizer:
                def convert_ids_to_tokens(self, ids):
                    return [f"token_{i}" for i in ids]
            
            def __init__(self):
                self.tokenizer = self.MockTokenizer()
        
        return MockProcessor()


def create_mock_attention_weights(seq_len=20):
    """Создает mock attention weights."""
    # Создаем реалистичную матрицу attention
    attn = torch.rand(seq_len, seq_len)
    # Делаем диагональ более выраженной
    attn = attn + torch.eye(seq_len) * 2
    # Нормализуем
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def create_mock_input_ids(seq_len=20):
    """Создает mock input_ids."""
    return torch.randint(1000, 2000, (seq_len,))


def create_mock_frames(num_frames=10, height=224, width=224):
    """Создает mock кадры видео."""
    frames = []
    for i in range(num_frames):
        # Создаем разноцветные кадры для визуализации
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Каждый кадр имеет свой цветовой паттерн
        color_intensity = int(255 * (i / num_frames))
        frame[:, :, 0] = color_intensity  # Red channel
        frame[:, :, 1] = 255 - color_intensity  # Green channel
        frame[:, :, 2] = 128  # Blue channel
        frames.append(frame)
    return frames


def test_attention_heatmap():
    """Тестирует создание attention heatmap."""
    logger.info("🧪 Тест: Attention Heatmap")
    
    processor = create_mock_processor()
    attention_weights = create_mock_attention_weights(seq_len=15)
    input_ids = create_mock_input_ids(seq_len=15)
    
    output_path = Path(OUTPUT_DIR) / "test_attention_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_attention_heatmap(
            attention_weights, input_ids, processor,
            str(output_path), max_tokens=15
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_token_importance():
    """Тестирует создание token importance plot."""
    logger.info("🧪 Тест: Token Importance")
    
    processor = create_mock_processor()
    token_importance = torch.rand(20) * 0.5 + 0.1  # Важность от 0.1 до 0.6
    input_ids = create_mock_input_ids(seq_len=20)
    
    output_path = Path(OUTPUT_DIR) / "test_token_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_token_importance_plot(
            token_importance, input_ids, processor,
            str(output_path), top_n=15
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_frame_importance():
    """Тестирует создание frame importance plot."""
    logger.info("🧪 Тест: Frame Importance")
    
    frame_importance = torch.rand(10) * 0.5 + 0.1
    
    output_path = Path(OUTPUT_DIR) / "test_frame_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_frame_importance_plot(frame_importance, str(output_path))
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_video_token_heatmap():
    """Тестирует создание video token heatmap на кадрах."""
    logger.info("🧪 Тест: Video Token Heatmap on Frames")
    
    frames = create_mock_frames(num_frames=5)
    
    # Создаем реалистичную пространственную важность токенов
    # Структура: T=5 кадров, H=14, W=14 токенов на кадр = 980 токенов всего
    T, H_tokens, W_tokens = 5, 14, 14
    num_tokens = T * H_tokens * W_tokens
    
    # Создаем важность токенов с пространственными паттернами
    video_token_importance = torch.zeros(num_tokens)
    for t in range(T):
        for h in range(H_tokens):
            for w in range(W_tokens):
                token_idx = t * (H_tokens * W_tokens) + h * W_tokens + w
                # Создаем паттерн: центр кадра важнее, края менее важны
                center_h, center_w = H_tokens // 2, W_tokens // 2
                dist_from_center = np.sqrt((h - center_h)**2 + (w - center_w)**2)
                max_dist = np.sqrt(center_h**2 + center_w**2)
                importance = 0.3 + 0.7 * (1 - dist_from_center / (max_dist + 1e-8))
                # Добавляем случайность
                importance += torch.rand(1).item() * 0.2 - 0.1
                video_token_importance[token_idx] = max(0.1, min(1.0, importance))
    
    # Создаем video_grid_thw
    video_grid_thw = torch.tensor([[T, H_tokens, W_tokens]])
    
    output_path = Path(OUTPUT_DIR) / "test_video_token_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_video_token_heatmap_on_frames(
            frames, 
            video_token_importance, 
            video_grid_thw,
            str(output_path), 
            alpha=0.4
        )
        logger.info(f"✅ Тест пройден: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def test_map_tokens_to_frames():
    """Тестирует сопоставление токенов с кадрами."""
    logger.info("🧪 Тест: Map Tokens to Frames")
    
    # Создаем mock video_grid_thw: (T=5, H=14, W=14) = 980 токенов на кадр
    video_grid_thw = torch.tensor([[5, 14, 14]])
    num_tokens = 5 * 14 * 14  # 980 токенов всего
    video_token_importance = torch.rand(num_tokens) * 0.5 + 0.1
    
    try:
        frame_importance_list = map_tokens_to_frames(
            video_token_importance, video_grid_thw, num_frames=5
        )
        
        assert len(frame_importance_list) == 5, f"Ожидалось 5 кадров, получено {len(frame_importance_list)}"
        assert all(0 <= imp <= 1 for imp in frame_importance_list), "Важность должна быть в [0, 1]"
        
        logger.info(f"✅ Тест пройден: {frame_importance_list}")
        return True
    except Exception as e:
        logger.error(f"❌ Тест провален: {e}", exc_info=True)
        return False


def run_all_tests():
    """Запускает все тесты."""
    logger.info("🚀 Запуск тестов визуализации...")
    logger.info(f"Результаты будут сохранены в: {OUTPUT_DIR}\n")
    
    results = {}
    
    results['attention_heatmap'] = test_attention_heatmap()
    results['token_importance'] = test_token_importance()
    results['frame_importance'] = test_frame_importance()
    results['video_token_heatmap'] = test_video_token_heatmap()
    results['map_tokens_to_frames'] = test_map_tokens_to_frames()
    
    # Итоги
    logger.info("\n" + "="*50)
    logger.info("📊 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*50)
    logger.info(f"Пройдено: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 Все тесты пройдены успешно!")
    else:
        logger.warning(f"⚠️ Провалено тестов: {total - passed}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
