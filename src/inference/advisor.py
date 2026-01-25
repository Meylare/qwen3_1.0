"""
Блок D: Советник по таймкодам.
Анализирует 3D delta_z и генерирует советы, привязанные к конкретным моментам видео.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum


class ChangeIntensity(Enum):
    """Интенсивность изменений в сегменте."""
    NONE = "none"           # Изменений не требуется
    LOW = "low"             # Небольшие корректировки
    MEDIUM = "medium"       # Заметные изменения
    HIGH = "high"           # Существенные изменения
    CRITICAL = "critical"   # Критические изменения


@dataclass
class SegmentAnalysis:
    """Анализ одного сегмента видео."""
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    delta_norm: float           # Средняя норма дельты в сегменте
    max_delta_norm: float       # Максимальная норма дельты
    intensity: ChangeIntensity  # Классификация интенсивности
    direction_vector: torch.Tensor  # Усредненный вектор направления изменений
    contribution_percent: float  # Вклад сегмента в общую дельту (%)


@dataclass
class FrameAdvice:
    """Совет для конкретного кадра/момента."""
    frame_idx: int
    time_sec: float
    delta_norm: float
    rank: int                   # Ранг по важности (1 = самый важный)
    advice_text: str            # Текстовый совет


@dataclass
class VideoAdvice:
    """Полный набор советов для видео."""
    video_duration_sec: float
    total_frames: int
    overall_score_improvement: float
    segments: List[SegmentAnalysis]
    top_frames: List[FrameAdvice]      # Топ кадров с наибольшими изменениями
    summary: str                        # Краткое резюме
    detailed_advice: List[str]          # Детальные советы по сегментам


class TimecodeAdvisor:
    """
    Блок D: Анализатор delta_z с генерацией советов по таймкодам.
    
    Принимает 3D delta_z из оптимизатора и:
    1. Разбивает видео на сегменты (начало, середина, конец)
    2. Для каждого сегмента считает силу изменений (норму ΔZ)
    3. Генерирует советы, привязанные к сегментам
    """
    
    def __init__(
        self,
        fps: float = 2.0,                    # Кадров в секунду (зависит от процессора)
        num_segments: int = 3,               # Количество сегментов (начало, середина, конец)
        intensity_thresholds: Optional[Dict[str, float]] = None,
        top_k_frames: int = 5                # Сколько топ-кадров выводить
    ):
        """
        Args:
            fps: Частота кадров видео-токенов (обычно 2 FPS для Qwen2.5-Omni)
            num_segments: Количество сегментов для разбиения
            intensity_thresholds: Пороги для классификации интенсивности
            top_k_frames: Количество топ-кадров для детального анализа
        """
        self.fps = fps
        self.num_segments = num_segments
        self.top_k_frames = top_k_frames
        
        # Пороги интенсивности (нормализованные)
        self.intensity_thresholds = intensity_thresholds or {
            "critical": 0.8,   # 80-100% от максимума
            "high": 0.5,       # 50-80%
            "medium": 0.2,     # 20-50%
            "low": 0.05,       # 5-20%
            # < 5% = none
        }
        
        # Шаблоны советов для разных сегментов
        self.segment_templates = {
            0: {  # Начало
                ChangeIntensity.CRITICAL: "В начале видео (первые {duration}с) требуются КРИТИЧЕСКИЕ изменения! Хук не работает - полностью пересмотри первые кадры.",
                ChangeIntensity.HIGH: "Начало видео ({start}-{end}с) нуждается в серьезной доработке. Усиль визуальный хук, добавь динамики.",
                ChangeIntensity.MEDIUM: "В начале ({start}-{end}с) можно улучшить привлечение внимания. Попробуй более яркие цвета или неожиданный ракурс.",
                ChangeIntensity.LOW: "Начало видео неплохое, но можно немного усилить.",
                ChangeIntensity.NONE: "Начало видео отличное, оставь как есть!"
            },
            1: {  # Середина
                ChangeIntensity.CRITICAL: "Середина видео ({start}-{end}с) теряет зрителя! Нужна полная переработка этого сегмента.",
                ChangeIntensity.HIGH: "В середине ({start}-{end}с) падает вовлеченность. Добавь поворот сюжета или смену темпа.",
                ChangeIntensity.MEDIUM: "Середина видео ({start}-{end}с) может быть динамичнее. Сократи паузы, добавь визуальных переходов.",
                ChangeIntensity.LOW: "Середина хорошая, возможны минимальные улучшения.",
                ChangeIntensity.NONE: "Середина видео работает отлично!"
            },
            2: {  # Конец (или общий для остальных сегментов)
                ChangeIntensity.CRITICAL: "Концовка ({start}-{end}с) провальная! Нужен сильный CTA или неожиданный финал.",
                ChangeIntensity.HIGH: "Конец видео ({start}-{end}с) слабый. Добавь запоминающийся момент или призыв к действию.",
                ChangeIntensity.MEDIUM: "Концовку ({start}-{end}с) можно сделать ярче. Усиль финальный акцент.",
                ChangeIntensity.LOW: "Концовка неплохая, можно немного усилить финал.",
                ChangeIntensity.NONE: "Отличная концовка!"
            }
        }

    def analyze(
        self,
        delta_z: torch.Tensor,
        per_frame_delta_norm: Optional[torch.Tensor] = None,
        score_improvement: float = 0.0,
        video_duration_sec: Optional[float] = None
    ) -> VideoAdvice:
        """
        Анализирует delta_z и генерирует советы по таймкодам.
        
        Args:
            delta_z: [Batch, Time, Dim] или [Time, Dim] - вектор изменений
            per_frame_delta_norm: [Batch, Time] или [Time] - нормы дельты (опционально)
            score_improvement: Улучшение скора из оптимизатора
            video_duration_sec: Длительность видео (если известна)
            
        Returns:
            VideoAdvice с полным набором советов
        """
        # Убираем batch dimension если он есть
        if delta_z.dim() == 3:
            delta_z = delta_z[0]  # [Time, Dim]
        if per_frame_delta_norm is not None and per_frame_delta_norm.dim() == 2:
            per_frame_delta_norm = per_frame_delta_norm[0]  # [Time]
        
        num_frames = delta_z.shape[0]
        
        # Вычисляем нормы если не переданы
        if per_frame_delta_norm is None:
            per_frame_delta_norm = torch.norm(delta_z, dim=-1)
        
        # Определяем длительность видео
        if video_duration_sec is None:
            video_duration_sec = num_frames / self.fps
        
        # 1. Анализ по сегментам
        segments = self._analyze_segments(
            delta_z=delta_z,
            per_frame_delta_norm=per_frame_delta_norm,
            num_frames=num_frames,
            video_duration_sec=video_duration_sec
        )
        
        # 2. Топ кадров с наибольшими изменениями
        top_frames = self._get_top_frames(
            per_frame_delta_norm=per_frame_delta_norm,
            num_frames=num_frames,
            video_duration_sec=video_duration_sec
        )
        
        # 3. Генерация текстовых советов
        detailed_advice = self._generate_segment_advice(segments)
        summary = self._generate_summary(segments, score_improvement)
        
        return VideoAdvice(
            video_duration_sec=video_duration_sec,
            total_frames=num_frames,
            overall_score_improvement=score_improvement,
            segments=segments,
            top_frames=top_frames,
            summary=summary,
            detailed_advice=detailed_advice
        )

    def _analyze_segments(
        self,
        delta_z: torch.Tensor,
        per_frame_delta_norm: torch.Tensor,
        num_frames: int,
        video_duration_sec: float
    ) -> List[SegmentAnalysis]:
        """Разбивает видео на сегменты и анализирует каждый."""
        segments = []
        frames_per_segment = num_frames // self.num_segments
        
        # Нормализуем нормы для классификации интенсивности
        max_norm = per_frame_delta_norm.max().item()
        if max_norm < 1e-8:
            max_norm = 1.0  # Избегаем деления на ноль
        
        total_delta_norm = per_frame_delta_norm.sum().item()
        
        for seg_idx in range(self.num_segments):
            start_frame = seg_idx * frames_per_segment
            # Последний сегмент забирает все оставшиеся кадры
            end_frame = (seg_idx + 1) * frames_per_segment if seg_idx < self.num_segments - 1 else num_frames
            
            # Временные метки
            start_time = start_frame / self.fps
            end_time = end_frame / self.fps
            
            # Дельты для этого сегмента
            segment_norms = per_frame_delta_norm[start_frame:end_frame]
            segment_delta = delta_z[start_frame:end_frame]
            
            # Статистики
            delta_norm = segment_norms.mean().item()
            max_delta_norm = segment_norms.max().item()
            
            # Вклад сегмента в общую дельту
            segment_total = segment_norms.sum().item()
            contribution = (segment_total / total_delta_norm * 100) if total_delta_norm > 1e-8 else 0.0
            
            # Направление изменений (усредненный вектор)
            direction = segment_delta.mean(dim=0)
            direction = F.normalize(direction, dim=0)
            
            # Классификация интенсивности
            normalized_norm = delta_norm / max_norm
            intensity = self._classify_intensity(normalized_norm)
            
            segments.append(SegmentAnalysis(
                segment_id=seg_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_sec=start_time,
                end_time_sec=end_time,
                delta_norm=delta_norm,
                max_delta_norm=max_delta_norm,
                intensity=intensity,
                direction_vector=direction,
                contribution_percent=contribution
            ))
        
        return segments

    def _classify_intensity(self, normalized_norm: float) -> ChangeIntensity:
        """Классифицирует интенсивность изменений по порогам."""
        if normalized_norm >= self.intensity_thresholds["critical"]:
            return ChangeIntensity.CRITICAL
        elif normalized_norm >= self.intensity_thresholds["high"]:
            return ChangeIntensity.HIGH
        elif normalized_norm >= self.intensity_thresholds["medium"]:
            return ChangeIntensity.MEDIUM
        elif normalized_norm >= self.intensity_thresholds["low"]:
            return ChangeIntensity.LOW
        else:
            return ChangeIntensity.NONE

    def _get_top_frames(
        self,
        per_frame_delta_norm: torch.Tensor,
        num_frames: int,
        video_duration_sec: float
    ) -> List[FrameAdvice]:
        """Находит топ-K кадров с наибольшими изменениями."""
        k = min(self.top_k_frames, num_frames)
        
        # Топ-K индексов по норме дельты
        top_values, top_indices = torch.topk(per_frame_delta_norm, k)
        
        top_frames = []
        for rank, (idx, norm) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1):
            time_sec = idx / self.fps
            
            # Генерируем совет для кадра
            advice_text = self._generate_frame_advice(idx, time_sec, norm, rank, num_frames)
            
            top_frames.append(FrameAdvice(
                frame_idx=idx,
                time_sec=time_sec,
                delta_norm=norm,
                rank=rank,
                advice_text=advice_text
            ))
        
        return top_frames

    def _generate_frame_advice(
        self,
        frame_idx: int,
        time_sec: float,
        delta_norm: float,
        rank: int,
        total_frames: int
    ) -> str:
        """Генерирует совет для конкретного кадра."""
        # Определяем позицию в видео
        position_ratio = frame_idx / total_frames
        
        if position_ratio < 0.2:
            position_name = "в самом начале"
            action_hint = "Усиль хук"
        elif position_ratio < 0.4:
            position_name = "в начале"
            action_hint = "Добавь динамики"
        elif position_ratio < 0.6:
            position_name = "в середине"
            action_hint = "Удерживай внимание"
        elif position_ratio < 0.8:
            position_name = "ближе к концу"
            action_hint = "Готовь к финалу"
        else:
            position_name = "в конце"
            action_hint = "Усиль концовку"
        
        return f"[{time_sec:.1f}с] {position_name}: {action_hint}. Сила изменения: {delta_norm:.2f}"

    def _generate_segment_advice(self, segments: List[SegmentAnalysis]) -> List[str]:
        """Генерирует текстовые советы для каждого сегмента."""
        advice_list = []
        
        for seg in segments:
            # Получаем шаблон для сегмента
            template_idx = min(seg.segment_id, 2)  # 0, 1, или 2
            templates = self.segment_templates.get(template_idx, self.segment_templates[2])
            template = templates.get(seg.intensity, "")
            
            if template:
                advice = template.format(
                    start=f"{seg.start_time_sec:.1f}",
                    end=f"{seg.end_time_sec:.1f}",
                    duration=f"{seg.end_time_sec - seg.start_time_sec:.1f}"
                )
                
                # Добавляем информацию о вкладе
                if seg.contribution_percent > 40:
                    advice += f" (вклад в общую дельту: {seg.contribution_percent:.0f}% - основной фокус!)"
                
                advice_list.append(advice)
        
        return advice_list

    def _generate_summary(
        self,
        segments: List[SegmentAnalysis],
        score_improvement: float
    ) -> str:
        """Генерирует краткое резюме."""
        # Находим сегмент с наибольшими изменениями
        worst_segment = max(segments, key=lambda s: s.delta_norm)
        
        # Подсчитываем критичные сегменты
        critical_count = sum(1 for s in segments 
                           if s.intensity in [ChangeIntensity.CRITICAL, ChangeIntensity.HIGH])
        
        if score_improvement > 0.5:
            improvement_text = f"Потенциальное улучшение скора: +{score_improvement:.2f} (значительное!)"
        elif score_improvement > 0.1:
            improvement_text = f"Потенциальное улучшение скора: +{score_improvement:.2f} (хорошее)"
        else:
            improvement_text = f"Потенциальное улучшение скора: +{score_improvement:.2f} (небольшое)"
        
        if critical_count == 0:
            status = "Видео в хорошем состоянии, минимальные правки."
        elif critical_count == 1:
            status = f"Основной фокус: {self._get_segment_name(worst_segment.segment_id)} ({worst_segment.start_time_sec:.1f}-{worst_segment.end_time_sec:.1f}с)."
        else:
            status = f"Требуется переработка {critical_count} из {len(segments)} сегментов."
        
        return f"{improvement_text} {status}"

    def _get_segment_name(self, segment_id: int) -> str:
        """Возвращает название сегмента."""
        names = {0: "начало", 1: "середина", 2: "конец"}
        return names.get(segment_id, f"сегмент {segment_id}")

    def format_advice(self, advice: VideoAdvice, verbose: bool = True) -> str:
        """
        Форматирует советы в читаемый текст.
        
        Args:
            advice: Результат анализа
            verbose: Включать детальную информацию
            
        Returns:
            Отформатированный текст с советами
        """
        lines = [
            "=" * 60,
            "АНАЛИЗ ВИДЕО - СОВЕТЫ ПО ТАЙМКОДАМ",
            "=" * 60,
            "",
            f"Длительность: {advice.video_duration_sec:.1f}с ({advice.total_frames} кадров)",
            "",
            "РЕЗЮМЕ:",
            advice.summary,
            "",
        ]
        
        if advice.detailed_advice:
            lines.append("ДЕТАЛЬНЫЕ СОВЕТЫ ПО СЕГМЕНТАМ:")
            for i, adv in enumerate(advice.detailed_advice, 1):
                lines.append(f"  {i}. {adv}")
            lines.append("")
        
        if verbose and advice.top_frames:
            lines.append("ТОП МОМЕНТОВ ДЛЯ ИЗМЕНЕНИЙ:")
            for frame in advice.top_frames:
                lines.append(f"  #{frame.rank}: {frame.advice_text}")
            lines.append("")
        
        if verbose:
            lines.append("СТАТИСТИКА ПО СЕГМЕНТАМ:")
            for seg in advice.segments:
                lines.append(
                    f"  [{seg.start_time_sec:.1f}-{seg.end_time_sec:.1f}с] "
                    f"Интенсивность: {seg.intensity.value.upper()}, "
                    f"Норма: {seg.delta_norm:.3f}, "
                    f"Вклад: {seg.contribution_percent:.1f}%"
                )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
