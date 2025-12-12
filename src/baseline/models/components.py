# src/models/components.py
import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    """
    MLP-адаптер (projector) для преобразования эмбеддингов CLAP (512) → Qwen (4096).
    Архитектура:
        Linear(512 → hidden_dim)
        GELU
        Linear(hidden_dim → hidden_dim)
    По умолчанию hidden_dim = 4096 (как у Qwen-3).
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 4096):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        # Инициализация весов (как в трансформерных моделях)
        self._init_weights()

    def _init_weights(self):
        for module in self.projector:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[float] of shape [Batch, 512]
        Returns:
            Tensor[float] of shape [Batch, 4096] (или hidden_dim)
        """
        return self.projector(x)