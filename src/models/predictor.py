import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Стандарт нормализации для LLaMA/Qwen.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Обучаемый параметр масштаба (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class ViralPredictor(nn.Module):
    """
    Regression Head ("Судья").
    Принимает эмбеддинги от LLM и выдает одно число (Viral Index).
    """
    def __init__(self, hidden_dim: int = 4096, dropout_rate: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            # 1. Нормализация входа (критично для стабильности)
            RMSNorm(hidden_dim),
            
            # 2. Проекция (Hidden -> 1024)
            nn.Linear(hidden_dim, 1024),
            
            # 3. Активация
            nn.GELU(),
            
            # 4. Регуляризация
            nn.Dropout(dropout_rate),
            
            # 5. Финальное предсказание (1024 -> 1)
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Hidden_Dim] -> [Batch, 1]
        return self.net(x)