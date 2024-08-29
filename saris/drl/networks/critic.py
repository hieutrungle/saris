from typing import Sequence
import torch
import torch.nn as nn
from saris.drl.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)
from saris.drl.networks.mlp import MLP, ResidualMLP
from saris.drl.networks.common_blocks import Fourier


class Crtic(nn.Module):
    """Crtic module for Pytorch."""

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        features: Sequence[int],
        activation: Activation,
        dtype: DType,
    ):
        super().__init__()
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        self.fourier = Fourier(num_observations + num_actions, features[0] // 2)
        self.mlp = MLP(
            features[0], features[-1], features[1:-1], activation, nn.Identity()
        )
        # self.mlp = ResidualMLP(
        #     features[0], features[-1], features[1:-1], activation, nn.Identity()
        # )
        self.q_value = nn.Linear(features[-1], 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False
    ) -> torch.Tensor:
        mixed = torch.cat([observations, actions], axis=-1)
        mixed = self.fourier(mixed)
        mixed = self.mlp(mixed)
        q_values = self.q_value(mixed)
        return q_values
