import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ViralOptimizer:
    """
    Блок В: Оптимизатор (Симулятор улучшений).
    Использует Projected Gradient Ascent для поиска идеального вектора.
    """
    def __init__(
        self,
        predictor_model: nn.Module,
        steps: int = 50,
        lr: float = 0.05,
        lambda_cos: float = 10.0,
        lambda_smooth: float = 2.0
    ):
        self.predictor = predictor_model
        self.steps = steps
        self.lr = lr
        self.lambda_cos = lambda_cos
        self.lambda_smooth = lambda_smooth

    def optimize(self, z_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_orig: Тензор [Batch, Time, Dim] (видео) или [Batch, 1, Dim] (аудио)
        Returns:
            delta_z: Вектор изменений [Batch, Time, Dim]
            final_score: Итоговый предсказанный скор
        """
        # 1. Подготовка клона для оптимизации
        # detach() - чтобы не тянуть графы из прошлого
        z_opt = z_orig.clone().detach().requires_grad_(True)

        # Оптимизатор для входных данных
        optimizer = torch.optim.Adam([z_opt], lr=self.lr)

        # 2. Запоминаем исходную норму (длину) векторов
        # dim=-1 значит считаем длину каждого вектора 4096 отдельно
        with torch.no_grad():
            orig_norm = torch.norm(z_orig, dim=-1, keepdim=True)

        print(f"[Optimizer] Start. Avg Norm: {orig_norm.mean().item():.2f}")

        for i in range(self.steps):
            optimizer.zero_grad()

            # --- Forward Pass ---
            # Предиктор оценивает текущее состояние
            score = self.predictor(z_opt)

            # Мы хотим МАКСИМИЗИРОВАТЬ score -> МИНИМИЗИРУЕМ -score
            loss_viral = -score.mean()

            # --- Штраф 1: Семантическая близость (Cosine) ---
            # 1.0 - сходство. Чем меньше сходство, тем больше штраф.
            # Сохраняет "Смысл" (Кошка остается кошкой)
            cos_sim = F.cosine_similarity(z_opt, z_orig, dim=-1)
            loss_cos = (1.0 - cos_sim).mean()

            # --- Штраф 2: Временная плавность (Smoothness) ---
            # Только если есть временное измерение (Time > 1)
            if z_opt.shape[1] > 1:
                # Разница между соседними кадрами
                diff_time = z_opt[:, 1:, :] - z_opt[:, :-1, :]
                # L2 норма разницы
                loss_smooth = torch.norm(diff_time, dim=-1).mean()
            else:
                loss_smooth = torch.tensor(0.0, device=z_opt.device)

            # --- Total Loss ---
            total_loss = loss_viral + (self.lambda_cos * loss_cos) + (self.lambda_smooth * loss_smooth)

            # --- Update ---
            total_loss.backward()
            optimizer.step()

            # --- ХАК: Projected Gradient Descent (Нормализация) ---
            # Возвращаем вектор к исходной длине, чтобы не ломать Attention
            with torch.no_grad():
                new_norm = torch.norm(z_opt, dim=-1, keepdim=True)
                # scale_factor = old / new
                scale = orig_norm / (new_norm + 1e-8)
                # Применяем масштабирование in-place
                z_opt.data.mul_(scale)

            # Логирование (опционально, можно убрать для скорости)
            if i % 10 == 0:
                print(f"Step {i}: Score={score.mean().item():.3f} | CosDist={loss_cos.item():.4f}")

        # 3. Финиш: считаем Дельту
        # detach(), чтобы разорвать граф, так как дальше это пойдет в Блок Г (как данные)
        delta_z = (z_opt - z_orig).detach()

        return delta_z, score.detach()