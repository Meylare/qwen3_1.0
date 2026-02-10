# Inference module
from .optimizer import ViralOptimizer, OptimizationResult
from .advisor import TimecodeAdvisor, VideoAdvice, SegmentAnalysis, FrameAdvice, ChangeIntensity

__all__ = [
    # Блок C: Оптимизатор
    "ViralOptimizer",
    "OptimizationResult",
    # Блок D: Советник
    "TimecodeAdvisor",
    "VideoAdvice",
    "SegmentAnalysis",
    "FrameAdvice",
    "ChangeIntensity",
]