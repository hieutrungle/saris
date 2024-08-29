import torch.nn as nn
import torch
import numpy as np


class Fourier(nn.Module):
    """Fourier features for encoding the input signal."""

    def __init__(
        self, in_features: int, out_features: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.fourier = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier(x)
        x = 2 * np.pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x
