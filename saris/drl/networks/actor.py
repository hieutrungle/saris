from typing import Sequence
import torch
import torch.nn as nn
from saris.drl.networks.network_utils import Activation, _str_to_activation
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
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        self.device_entries = 6
        num_observations = num_observations - self.device_entries
        self.device_embed = nn.Linear(self.device_entries, features[-1] // 2)

        self.fourier = Fourier(num_observations, features[0] // 2)
        self.mlp = ResidualMLP(
            features[0], features[-1] // 2, features[1:-1], activation, nn.Tanh()
        )

        self.ac_means = nn.Linear(features[-1], num_actions)
        self.ac_log_stds = nn.Linear(features[-1], num_actions)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:

        device_embed = observations[..., : self.device_entries]
        device_embed = self.device_embed(device_embed)

        observations = observations[..., self.device_entries :]
        mixed = self.fourier(observations)
        mixed = self.mlp(mixed)

        mixed = torch.cat([mixed, device_embed], axis=-1)

        ac_means: torch.Tensor = self.ac_means(mixed)
        ac_log_stds: torch.Tensor = self.ac_log_stds(mixed)
        return ac_means, ac_log_stds
