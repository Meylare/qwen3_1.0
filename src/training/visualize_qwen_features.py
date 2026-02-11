"""
Скрипт для визуализации важности признаков обученной Qwen модели.
Создает тепловые карты attention, важность токенов и кадров видео.
"""

import sys
import os

# Проверяем тестовый режим ДО всех импортов
TEST_MODE = '--test_mode' in sys.argv

# Если тестовый режим, перенаправляем на test_visualize.py
if TEST_MODE and __name__ == "__main__":
    # Добавляем путь к проекту
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Запускаем тестовый скрипт напрямую через subprocess чтобы избежать циклического импорта
    import subprocess
    test_script = os.path.join(os.path.dirname(__file__), 'test_visualize.py')
    result = subprocess.run([sys.executable, test_script], cwd=project_root)
    sys.exit(result.returncode)

# Обычные импорты для реального режима
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, List
import json
from PIL import Image
import platform
import importlib.util

# Добавляем корневую директорию проекта в sys.path для импортов
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Попытка импорта cv2 с fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 не доступен, будет использован numpy fallback для наложения")

# Импорты для реального режима (только при необходимости)
# Эти импорты выполняются лениво при вызове функций, которые их используют
try:
    from transformers import BitsAndBytesConfig, AutoProcessor
    from peft import PeftModel
    _MODEL_IMPORTS_AVAILABLE = True
except ImportError:
    _MODEL_IMPORTS_AVAILABLE = False
    BitsAndBytesConfig = None
    AutoProcessor = None
    PeftModel = None

# Импорты моделей выполняются только при необходимости
ViralPredictorModel = None
ViralVideoDataset = None

def _lazy_import_models():
    """Ленивый импорт моделей только когда они действительно нужны."""
    global ViralPredictorModel, ViralVideoDataset
    if ViralPredictorModel is None:
        viral_model_path = os.path.join(project_root, 'src', 'models', 'viral_model.py')
        spec = importlib.util.spec_from_file_location("viral_model", viral_model_path)
        viral_model_module = importlib.util.module_from_spec(spec)
        sys.modules["viral_model"] = viral_model_module
        spec.loader.exec_module(viral_model_module)
        ViralPredictorModel = viral_model_module.ViralPredictorModel
        
        dataset_path = os.path.join(project_root, 'src', 'data', 'dataset.py')
        spec = importlib.util.spec_from_file_location("dataset", dataset_path)
        dataset_module = importlib.util.module_from_spec(spec)
        sys.modules["dataset"] = dataset_module
        spec.loader.exec_module(dataset_module)
        ViralVideoDataset = dataset_module.ViralVideoDataset


def get_device():
    """
    Определяет доступное устройство с учетом платформы.
    Приоритет: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def is_macos():
    """Проверяет, запущено ли на macOS."""
    return platform.system() == "Darwin"


def _resize_heatmap_numpy(heatmap_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Простое масштабирование тепловой карты через numpy (fallback)."""
    src_h, src_w = heatmap_2d.shape
    
    # Используем простое повторение и обрезку
    if target_h >= src_h and target_w >= src_w:
        # Увеличиваем через повторение
        repeat_h = target_h // src_h
        repeat_w = target_w // src_w
        resized = np.repeat(np.repeat(heatmap_2d, repeat_h, axis=0), repeat_w, axis=1)
        # Обрезаем до нужного размера
        resized = resized[:target_h, :target_w]
    else:
        # Уменьшаем через усреднение
        step_h = src_h / target_h
        step_w = src_w / target_w
        resized = np.zeros((target_h, target_w))
        for i in range(target_h):
            for j in range(target_w):
                start_h = int(i * step_h)
                end_h = int((i + 1) * step_h)
                start_w = int(j * step_w)
                end_w = int((j + 1) * step_w)
                resized[i, j] = heatmap_2d[start_h:end_h, start_w:end_w].mean()
    
    return resized


# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ (не требуют импорта моделей)
# ============================================================================

def create_attention_heatmap(
    attention_weights: torch.Tensor,
    input_ids: torch.Tensor,
    processor,
    output_path: str,
    max_tokens: int = 50
):
    """Создает тепловую карту attention weights."""
    # Ограничиваем размер для читаемости
    seq_len = min(attention_weights.shape[0], max_tokens)
    attn_matrix = attention_weights[:seq_len, :seq_len].cpu().numpy()
    
    # Декодируем токены для подписей
    token_ids = input_ids[:seq_len].cpu().numpy()
    try:
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
        token_labels = [t[:10] + '...' if len(t) > 10 else t for t in tokens]
    except:
        token_labels = [f"Token_{i}" for i in range(seq_len)]
    
    # Создание графика
    plt.figure(figsize=(max(12, seq_len * 0.3), max(10, seq_len * 0.3)))
    sns.heatmap(
        attn_matrix,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.1,
        linecolor='gray',
        square=True
    )
    
    plt.title('Тепловая карта Attention Weights (последний слой)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Key Tokens', fontsize=12)
    plt.ylabel('Query Tokens', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Тепловая карта attention сохранена: {output_path}")
    plt.close()


def create_token_importance_plot(
    token_importance: torch.Tensor,
    input_ids: torch.Tensor,
    processor,
    output_path: str,
    top_n: int = 20
):
    """Создает визуализацию важности токенов."""
    if token_importance is None:
        logger.warning("Token importance не вычислена")
        return
    
    importance_np = token_importance.cpu().numpy()
    token_ids = input_ids.cpu().numpy()
    
    # Сортируем по важности
    top_indices = np.argsort(importance_np)[-top_n:][::-1]
    
    # Декодируем токены
    try:
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids[top_indices])
        token_labels = [t[:15] + '...' if len(t) > 15 else t for t in tokens]
    except:
        token_labels = [f"Token_{i}" for i in top_indices]
    
    top_importance = importance_np[top_indices]
    
    # Создание графика
    plt.figure(figsize=(max(10, top_n * 0.5), 6))
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, top_n))
    bars = plt.barh(range(top_n), top_importance, color=colors, edgecolor='black', linewidth=0.5)
    
    plt.yticks(range(top_n), token_labels, fontsize=9)
    plt.xlabel('Важность токена (Gradient Magnitude)', fontsize=12, fontweight='bold')
    plt.title(f'Топ-{top_n} важных токенов (Gradient-based)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Добавление значений
    for i, val in enumerate(top_importance):
        plt.text(val + max(top_importance) * 0.01, i, f"{val:.4f}", va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ График важности токенов сохранен: {output_path}")
    plt.close()


def create_frame_importance_plot(
    frame_importance: torch.Tensor,
    output_path: str
):
    """Создает визуализацию важности кадров видео."""
    if frame_importance is None:
        logger.warning("Frame importance не вычислена")
        return
    
    importance_np = frame_importance.cpu().numpy()
    num_frames = len(importance_np)
    
    # Создание графика
    plt.figure(figsize=(max(10, num_frames * 0.3), 6))
    plt.plot(range(num_frames), importance_np, marker='o', linewidth=2, markersize=8, color='#FF6B35')
    plt.fill_between(range(num_frames), importance_np, alpha=0.3, color='#FF6B35')
    
    plt.xlabel('Номер кадра', fontsize=12, fontweight='bold')
    plt.ylabel('Важность кадра (Gradient Magnitude)', fontsize=12, fontweight='bold')
    plt.title('Важность кадров видео для предсказания', fontsize=14, fontweight='bold', pad=20)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(range(num_frames))
    
    # Выделяем топ-3 кадра
    top_3_indices = np.argsort(importance_np)[-3:][::-1]
    for idx in top_3_indices:
        plt.scatter([idx], [importance_np[idx]], s=200, color='red', marker='*', zorder=5)
        plt.text(idx, importance_np[idx] + max(importance_np) * 0.05, f'#{idx}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ График важности кадров сохранен: {output_path}")
    plt.close()


def map_tokens_to_frames(
    video_token_importance: torch.Tensor,
    video_grid_thw: Optional[torch.Tensor] = None,
    num_frames: int = 10
) -> List[float]:
    """Сопоставляет важность видео-токенов с кадрами."""
    if video_token_importance is None:
        return None
    
    if video_grid_thw is not None:
        T, H, W = video_grid_thw[0].cpu().numpy().astype(int)
        tokens_per_frame = H * W
        
        frame_importance_list = []
        for t in range(T):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(video_token_importance))
            frame_tokens = video_token_importance[start_idx:end_idx]
            frame_importance_list.append(frame_tokens.mean().item())
        
        return frame_importance_list
    else:
        # Fallback: равномерное распределение
        tokens_per_frame = len(video_token_importance) // num_frames
        frame_importance_list = []
        for t in range(num_frames):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(video_token_importance))
            frame_tokens = video_token_importance[start_idx:end_idx]
            frame_importance_list.append(frame_tokens.mean().item())
        
        return frame_importance_list


def create_video_token_heatmap_on_frames(
    frames: List[np.ndarray],
    video_token_importance: torch.Tensor,
    video_grid_thw: Optional[torch.Tensor] = None,
    output_path: str = None,
    alpha: float = 0.5,
    frame_importance_list: Optional[List[float]] = None  # Для обратной совместимости
):
    """
    Создает пространственную тепловую карту важности видео-токенов, наложенную на реальные кадры.
    
    Args:
        frames: Список реальных кадров видео [num_frames] каждый [H_frame, W_frame, 3]
        video_token_importance: Важность каждого видео-токена [num_video_tokens]
        video_grid_thw: Структура токенов [batch, 3] -> (T, H_tokens, W_tokens)
        output_path: Путь для сохранения
        alpha: Прозрачность тепловой карты
        frame_importance_list: Устаревший параметр (для обратной совместимости)
    """
    # Обратная совместимость: если передан frame_importance_list без video_token_importance
    if video_token_importance is None and frame_importance_list is not None:
        logger.warning("Используется устаревший режим с frame_importance_list")
        return _create_simple_frame_heatmap(frames, frame_importance_list, output_path, alpha)
    
    if video_token_importance is None:
        logger.warning("video_token_importance не предоставлена")
        return
    
    if video_grid_thw is None:
        logger.warning("video_grid_thw не предоставлен, используем равномерное распределение")
        num_frames = len(frames)
        tokens_per_frame = len(video_token_importance) // num_frames
        frame_importance_list = []
        for t in range(num_frames):
            start_idx = t * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(video_token_importance))
            frame_importance_list.append(video_token_importance[start_idx:end_idx].mean().item())
        return _create_simple_frame_heatmap(frames, frame_importance_list, output_path, alpha)
    
    # Извлекаем структуру токенов
    T, H_tokens, W_tokens = video_grid_thw[0].cpu().numpy().astype(int)
    tokens_per_frame = H_tokens * W_tokens
    
    # Проверяем соответствие
    if len(frames) != T:
        logger.warning(f"Несоответствие: {len(frames)} кадров vs {T} временных шагов")
        min_len = min(len(frames), T)
        frames = frames[:min_len]
        T = min_len
    
    num_frames = len(frames)
    
    # Создаем коллаж кадров
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    axes = axes.flatten()
    
    # Нормализуем важность всех токенов для единой цветовой шкалы
    importance_np = video_token_importance.cpu().numpy()
    importance_min = importance_np.min()
    importance_max = importance_np.max()
    importance_range = importance_max - importance_min + 1e-8
    
    for frame_idx in range(num_frames):
        ax = axes[frame_idx]
        
        # Получаем оригинальный кадр
        frame = frames[frame_idx]
        
        # Конвертируем кадр в RGB если нужно
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        frame_rgb = frame[:, :, :3] if frame.shape[2] >= 3 else frame
        frame_h, frame_w = frame_rgb.shape[:2]
        
        # Извлекаем важность токенов для этого кадра
        start_token_idx = frame_idx * tokens_per_frame
        end_token_idx = min(start_token_idx + tokens_per_frame, len(video_token_importance))
        frame_tokens_importance = importance_np[start_token_idx:end_token_idx]
        
        # Создаем пространственную карту важности [H_tokens, W_tokens]
        token_importance_2d = frame_tokens_importance.reshape(H_tokens, W_tokens)
        
        # Масштабируем до размера кадра
        # Приоритет: cv2 > scipy > numpy fallback
        if CV2_AVAILABLE:
            try:
                import cv2
                spatial_heatmap = cv2.resize(token_importance_2d.astype(np.float32), 
                                            (frame_w, frame_h), 
                                            interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.warning(f"Ошибка cv2.resize: {e}, используем numpy fallback")
                spatial_heatmap = _resize_heatmap_numpy(token_importance_2d, frame_h, frame_w)
        else:
            # Пробуем scipy
            try:
                from scipy.ndimage import zoom
                zoom_h = frame_h / H_tokens
                zoom_w = frame_w / W_tokens
                spatial_heatmap = zoom(token_importance_2d, (zoom_h, zoom_w), order=1)
            except ImportError:
                # Fallback: простое масштабирование через numpy
                spatial_heatmap = _resize_heatmap_numpy(token_importance_2d, frame_h, frame_w)
        
        # Нормализуем для цветовой карты
        spatial_heatmap_normalized = (spatial_heatmap - importance_min) / importance_range
        
        # Применяем цветовую карту 'hot' (красный = высоко, черный = низко)
        import matplotlib.cm as cm
        colormap = cm.get_cmap('hot')
        heatmap_rgb = colormap(spatial_heatmap_normalized)[:, :, :3]  # [H, W, 3]
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        
        # Накладываем на кадр
        if CV2_AVAILABLE:
            try:
                overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            except:
                overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        else:
            overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        ax.imshow(overlay)
        
        # Вычисляем среднюю важность для этого кадра
        avg_importance = frame_tokens_importance.mean()
        max_importance = frame_tokens_importance.max()
        
        ax.set_title(f'Frame {frame_idx}\nAvg: {avg_importance:.3f} | Max: {max_importance:.3f}', 
                    fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # Скрываем лишние оси
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Пространственная тепловая карта важности видео-токенов на кадрах', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Тепловая карта на кадрах сохранена: {output_path}")
    plt.close()


def _create_simple_frame_heatmap(
    frames: List[np.ndarray],
    frame_importance_list: List[float],
    output_path: str,
    alpha: float = 0.5
):
    """Устаревшая версия с однородной важностью для всего кадра."""
    num_frames = len(frames)
    importance_array = np.array(frame_importance_list)
    importance_normalized = (importance_array - importance_array.min()) / (importance_array.max() - importance_array.min() + 1e-8)
    
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    axes = axes.flatten()
    
    for idx, (frame, importance) in enumerate(zip(frames, importance_normalized)):
        ax = axes[idx]
        
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        H, W = frame.shape[:2]
        heatmap = np.ones((H, W)) * importance
        
        import matplotlib.cm as cm
        colormap = cm.get_cmap('hot')
        heatmap_rgb = colormap(heatmap)[:, :, :3]
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        
        frame_rgb = frame[:, :, :3] if frame.shape[2] >= 3 else frame
        if CV2_AVAILABLE:
            try:
                overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            except:
                overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        else:
            overlay = (frame_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        ax.imshow(overlay)
        ax.set_title(f'Frame {idx}\nImportance: {frame_importance_list[idx]:.4f}', 
                    fontsize=9, fontweight='bold')
        ax.axis('off')
    
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Тепловая карта важности видео-токенов на кадрах', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Тепловая карта на кадрах сохранена: {output_path}")
    plt.close()


# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С МОДЕЛЬЮ (требуют импорта моделей)
# ============================================================================

# Эти функции будут добавлены позже, когда понадобятся для реального режима
# Пока что тестовый режим работает только с функциями визуализации выше
