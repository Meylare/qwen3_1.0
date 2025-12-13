# src/models/__init__.py
"""
Модуль моделей для предсказания вирусности видео.
Экспортирует все основные компоненты и модели.
"""
from .components import (
    RMSNorm,
    Projector,
    PredictorHead,
    ViralPredictor,  # Алиас для PredictorHead
    MLPAdapter,      # Алиас для Projector
)

from .g2v2_model import (
    G2V2Model,
)

# Алиас для AudioProjector (то же самое, что Projector)
AudioProjector = Projector

__all__ = [
    # Компоненты
    'RMSNorm',
    'Projector',
    'PredictorHead',
    'ViralPredictor',
    'MLPAdapter',
    'AudioProjector',
    # Модели
    'G2V2Model',
]
