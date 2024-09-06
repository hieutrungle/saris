from typing import Sequence
import torch
import torch.nn as nn
from saris.drl.networks.network_utils import Activation, _str_to_activation
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
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]

        self.device_entries = 6
        num_observations = num_observations - self.device_entries
        self.device_embed = nn.Linear(self.device_entries, features[-1] // 2)

        self.obs_fourier = Fourier(num_observations, features[0] // 4)
        self.act_fourier = Fourier(num_actions, features[0] // 4)
        self.act = nn.Linear(features[0] // 2, features[0] // 2)
        self.obs = nn.Linear(features[0] // 2, features[0] // 2)
        self.mlp = ResidualMLP(
            features[0], features[-1] // 2, features[1:-1], activation, nn.Identity()
        )

        self.q_value = nn.Linear(features[-1], 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False
    ) -> torch.Tensor:

        device_embed = observations[..., : self.device_entries]
        device_embed = self.device_embed(device_embed)

        observations = observations[..., self.device_entries :]
        observations = self.obs_fourier(observations)
        observations = self.obs(observations)
        actions = self.act_fourier(actions)
        actions = self.act(actions)
        mixed = torch.cat([observations, actions], axis=-1)
        mixed = self.mlp(mixed)

        mixed = torch.cat([mixed, device_embed], axis=-1)

        q_values = self.q_value(mixed)
        return q_values
