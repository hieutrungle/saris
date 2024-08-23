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


class Crtic(nn.Module):
    """Crtic module for Flax."""

    features: Sequence[int]
    activation: nn.activation
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = True
    ) -> jnp.ndarray:

        mixed = jnp.concatenate([observations, actions], axis=-1)
        mixed = Fourier(self.features[0] // 2)(mixed)
        mixed = MLP(self.features, self.activation, self.dtype)(mixed)
        q_values = nn.Dense(1)(mixed)
        return q_values.astype(jnp.float32)

    @staticmethod
    def create(
        features: Sequence[int],
        activation: Activation,
        dtype: DType,
    ) -> "Crtic":
        if isinstance(dtype, str):
            dtype = _str_to_dtype[dtype]
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        model = Crtic(features, activation, dtype)
        return model
