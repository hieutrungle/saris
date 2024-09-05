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


class Actor(nn.Module):
    """Actor module for Pytorch."""

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

        self.fourier = Fourier(num_observations, features[0] // 2)
        # self.mlp = MLP(features[0], features[-1], features[1:-1], activation, nn.Tanh())
        self.mlp = ResidualMLP(
            features[0], features[-1], features[1:-1], activation, nn.Tanh()
        )
        self.ac_means = nn.Linear(features[-1], num_actions)
        self.ac_log_stds = nn.Linear(features[-1], num_actions)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        mapped = self.fourier(observations)
        actions = self.mlp(mapped)
        ac_means: torch.Tensor = self.ac_means(actions)
        ac_log_stds: torch.Tensor = self.ac_log_stds(actions)
        return ac_means, ac_log_stds
