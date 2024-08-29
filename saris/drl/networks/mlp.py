from typing import Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron module for Flax."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Tuple[int],
        activation: nn.Module = nn.GELU,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.features = features
        self.activation = getattr(nn, activation)
        self.dtype = getattr(torch, dtype)

        mlp = nn.ModuleList()
        mlp.append(nn.Linear(in_features=in_features, out_features=features[0]))
        for i in range(len(features) - 1):
            mlp.append(self.activation())
            mlp.append(nn.Linear(in_features=features[i], out_features=features[i]))
        mlp.append(nn.Linear(in_features=features[-1], out_features=out_features))
        for i, feat in enumerate(self.features):
            mlp.append(nn.Linear(in_features=feat, out_features=feat))
            mlp.append(self.activation())

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
