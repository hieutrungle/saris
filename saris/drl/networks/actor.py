from typing import Sequence
from jax import numpy as jnp
from flax import linen as nn
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

    num_actions: int
    features: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        mapped = Fourier(self.features[0] // 2)(observations)
        actions = MLP(self.features, self.activation, self.dtype)(mapped)
        ac_means = nn.Dense(self.num_actions, name="means")(actions)
        ac_log_stds = nn.Dense(self.num_actions, name="log_stds")(actions)
        return ac_means.astype(jnp.float32), ac_log_stds.astype(jnp.float32)

    @staticmethod
    def create(
        num_actions: int,
        features: Sequence[int],
        activation: Activation,
        dtype: DType,
    ) -> "Actor":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = Actor(num_actions, features, activation, dtype)
        return model
