import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal, Optional
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Результат оптимизации с дополнительной метаинформацией."""
    delta_z: torch.Tensor          # [Batch, Time, Dim] - вектор изменений
    final_score: torch.Tensor      # [Batch, 1] - итоговый скор
    score_improvement: float       # Улучшение скора (final - initial)
    per_frame_delta_norm: torch.Tensor  # [Batch, Time] - норма дельты для каждого кадра


class ViralOptimizer:
    """
    Блок C: Оптимизатор (Симулятор улучшений).
    Использует Projected Gradient Ascent для поиска идеального вектора.
    
    Поддерживает 3D-оптимизацию:
    - Вход: [Batch, Time, Dim] - последовательность эмбеддингов видео-кадров
    - Внутри: пулинг для получения скора от предиктора (Блок B)
    - Выход: 3D delta_z для анализа по таймкодам (Блок D)
    """
    
    def __init__(
        self,
        predictor_model: nn.Module,
        steps: int = 50,
        lr: float = 0.05,
        lambda_cos: float = 10.0,
        lambda_smooth: float = 2.0,
        pooling_mode: Literal["mean", "last", "attention_weighted"] = "mean"
    ):
        """
        Args:
            predictor_model: MLP-голова (viral_head) из ViralPredictorModel
            steps: Количество шагов оптимизации
            lr: Learning rate для Adam
            lambda_cos: Вес штрафа за семантическое отклонение
            lambda_smooth: Вес штрафа за резкие скачки между кадрами
            pooling_mode: Метод пулинга 3D -> 2D:
                - "mean": Mean Pooling по временной оси
                - "last": Last Token Pooling (как в модели)
                - "attention_weighted": Взвешенный пулинг
        """
        self.predictor = predictor_model
        self.steps = steps
        self.lr = lr
        self.lambda_cos = lambda_cos
        self.lambda_smooth = lambda_smooth
        self.pooling_mode = pooling_mode
        
        # Для attention_weighted пулинга
        if pooling_mode == "attention_weighted":
            hidden_size = self._get_hidden_size()
            self.attention_weights = nn.Linear(hidden_size, 1)

    def _get_hidden_size(self) -> int:
        """Определяет размер hidden из predictor."""
        # Предполагаем что predictor это Sequential, первый слой - Linear
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 3584  # Fallback для Qwen2.5-7B

    def _pool_sequence(
        self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Пулинг 3D тензора [Batch, Time, Dim] -> 2D [Batch, Dim].
        
        Args:
            z: Входной тензор [Batch, Time, Dim]
            mask: Опциональная маска [Batch, Time] для игнорирования паддинга
            
        Returns:
            pooled: [Batch, Dim]
        """
        if z.dim() == 2:
            # Уже 2D, просто возвращаем
            return z
        
        batch_size, time_steps, dim = z.shape
        
        if self.pooling_mode == "mean":
            # Mean Pooling: среднее по всем временным шагам
            if mask is not None:
                # Маскированное среднее
                mask_expanded = mask.unsqueeze(-1).expand_as(z)
                z_masked = z * mask_expanded
                pooled = z_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = z.mean(dim=1)
                
        elif self.pooling_mode == "last":
            # Last Token Pooling: берем последний токен
            if mask is not None:
                # Находим индекс последнего валидного токена
                lengths = mask.sum(dim=1).long() - 1
                pooled = z[torch.arange(batch_size, device=z.device), lengths]
            else:
                pooled = z[:, -1, :]
                
        elif self.pooling_mode == "attention_weighted":
            # Attention-weighted Pooling
            # Считаем веса внимания для каждого временного шага
            attn_scores = self.attention_weights(z).squeeze(-1)  # [Batch, Time]
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)  # [Batch, Time]
            pooled = torch.einsum('bt,btd->bd', attn_weights, z)
            
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
        
        return pooled

    def optimize(
        self, 
        z_orig: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Оптимизирует 3D тензор видео-эмбеддингов.
        
        Args:
            z_orig: Тензор [Batch, Time, Dim] - последовательность видео-токенов
            mask: Опциональная маска [Batch, Time] для игнорирования паддинга
            verbose: Выводить логи прогресса
            
        Returns:
            OptimizationResult с:
                - delta_z: [Batch, Time, Dim] - вектор изменений для каждого кадра
                - final_score: [Batch, 1] - итоговый предсказанный скор
                - score_improvement: улучшение скора
                - per_frame_delta_norm: [Batch, Time] - сила изменений по кадрам
        """
        # Гарантируем 3D формат
        if z_orig.dim() == 2:
            # [Batch, Dim] -> [Batch, 1, Dim]
            z_orig = z_orig.unsqueeze(1)
        
        batch_size, time_steps, dim = z_orig.shape
        device = z_orig.device
        
        # 1. Подготовка клона для оптимизации
        z_opt = z_orig.clone().detach().requires_grad_(True)
        
        # Оптимизатор для входных данных
        optimizer = torch.optim.Adam([z_opt], lr=self.lr)
        
        # 2. Запоминаем исходную норму векторов (для PGD)
        with torch.no_grad():
            orig_norm = torch.norm(z_orig, dim=-1, keepdim=True)  # [Batch, Time, 1]
            
            # Начальный скор
            z_pooled_init = self._pool_sequence(z_orig, mask)
            initial_score = self.predictor(z_pooled_init)
        
        if verbose:
            print(f"[Optimizer] Start. Shape: {z_orig.shape}, Avg Norm: {orig_norm.mean().item():.2f}")
            print(f"[Optimizer] Initial Score: {initial_score.mean().item():.3f}")
        
        for i in range(self.steps):
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # Пулинг 3D -> 2D для предиктора
            z_pooled = self._pool_sequence(z_opt, mask)
            
            # Предиктор оценивает текущее состояние
            score = self.predictor(z_pooled)
            
            # Мы хотим МАКСИМИЗИРОВАТЬ score -> МИНИМИЗИРУЕМ -score
            loss_viral = -score.mean()
            
            # --- Штраф 1: Семантическая близость (Cosine) ---
            # Применяется к каждому временному шагу отдельно
            cos_sim = F.cosine_similarity(z_opt, z_orig, dim=-1)  # [Batch, Time]
            if mask is not None:
                cos_sim = cos_sim * mask
                loss_cos = (1.0 - cos_sim).sum() / mask.sum().clamp(min=1)
            else:
                loss_cos = (1.0 - cos_sim).mean()
            
            # --- Штраф 2: Временная плавность (Smoothness) ---
            # Штраф за резкие скачки между соседними кадрами
            if time_steps > 1:
                # Разница между соседними кадрами: [Batch, Time-1, Dim]
                diff_time = z_opt[:, 1:, :] - z_opt[:, :-1, :]
                
                # L2 норма разницы для каждого перехода
                diff_norm = torch.norm(diff_time, dim=-1)  # [Batch, Time-1]
                
                if mask is not None:
                    # Маска для переходов (оба кадра должны быть валидны)
                    transition_mask = mask[:, 1:] * mask[:, :-1]
                    loss_smooth = (diff_norm * transition_mask).sum() / transition_mask.sum().clamp(min=1)
                else:
                    loss_smooth = diff_norm.mean()
            else:
                loss_smooth = torch.tensor(0.0, device=device)
            
            # --- Total Loss ---
            total_loss = loss_viral + (self.lambda_cos * loss_cos) + (self.lambda_smooth * loss_smooth)
            
            # --- Update ---
            total_loss.backward()
            optimizer.step()
            
            # --- Projected Gradient Descent (Нормализация) ---
            # Возвращаем каждый вектор к исходной длине
            with torch.no_grad():
                new_norm = torch.norm(z_opt, dim=-1, keepdim=True)
                scale = orig_norm / (new_norm + 1e-8)
                z_opt.data.mul_(scale)
            
            # Логирование
            if verbose and i % 10 == 0:
                print(f"Step {i}: Score={score.mean().item():.3f} | "
                      f"CosDist={loss_cos.item():.4f} | Smooth={loss_smooth.item():.4f}")
        
        # 3. Финальные вычисления
        with torch.no_grad():
            # Финальный скор
            z_pooled_final = self._pool_sequence(z_opt, mask)
            final_score = self.predictor(z_pooled_final)
            
            # Дельта для каждого кадра
            delta_z = (z_opt - z_orig).detach()
            
            # Норма дельты для каждого кадра (для Блока D)
            per_frame_delta_norm = torch.norm(delta_z, dim=-1)  # [Batch, Time]
            
            # Улучшение скора
            score_improvement = (final_score.mean() - initial_score.mean()).item()
        
        if verbose:
            print(f"[Optimizer] Done. Final Score: {final_score.mean().item():.3f}, "
                  f"Improvement: {score_improvement:+.3f}")
        
        return OptimizationResult(
            delta_z=delta_z,
            final_score=final_score,
            score_improvement=score_improvement,
            per_frame_delta_norm=per_frame_delta_norm
        )