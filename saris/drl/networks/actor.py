from typing import Sequence, Union
import torch
import torch.nn as nn
from saris.drl.networks.network_utils import (
    Activation,
    _str_to_activation,
    DType,
    _str_to_dtype,
)
from saris.drl.networks.mlp import MLP
from saris.drl.networks.common_blocks import Fourier


class Actor(nn.Module):
    """Actor module for Flax."""

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        features: Sequence[int],
        activation: Activation,
        dtype: DType,
    ):
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.features = features
        self.activation = activation
        self.dtype = dtype
        if isinstance(dtype, str):
            self.dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            self.activation = _str_to_activation[activation]

        self.fourier = Fourier(self.num_observations, self.features[0] // 2, self.dtype)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mapped = Fourier(self.features[0] // 2)(observations)
        actions = MLP(self.features, self.activation, self.dtype)(mapped)
        ac_means = nn.Dense(self.num_actions, name="means")(actions)
        ac_log_stds = nn.Dense(self.num_actions, name="log_stds")(actions)
        return ac_means.astype(jnp.float32), ac_log_stds.astype(jnp.float32)
