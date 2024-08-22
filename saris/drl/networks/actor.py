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


class Actor(nn.Module):
    """Actor module for Flax."""

    num_actions: int
    features: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        actions = MLP(self.features, self.activation, self.dtype)(observations)
        ac_means = nn.Dense(self.num_actions)(actions)
        ac_stds = nn.Dense(self.num_actions)(actions)
        return ac_means.astype(jnp.float32), ac_stds.astype(jnp.float32)

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
