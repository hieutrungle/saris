from typing import Tuple
from jax import numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Multi-layer perceptron module for Flax."""

    features: Tuple[int]
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for feat in self.features:
            x = nn.Dense(feat, dtype=self.dtype)(x)
            x = self.activation(x)
        return x


class ResidualDense(nn.Module):
    """ResidualMLP module for Flax."""

    feature: int
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.Dense(self.feature, dtype=self.dtype)(x)
        x = nn.Dense(self.feature * 2, dtype=self.dtype)(x)
        x = self.activation(x)
        x = nn.Dense(self.feature, dtype=self.dtype)(x)
        x = x + residual
        return x
