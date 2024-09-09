from typing import Tuple, Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron module for Flax."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Sequence[int],
        activation: nn.Module = nn.GELU(),
        output_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()

        features = [in_features] + list(features)
        mlp = nn.ModuleList()
        for i in range(len(features) - 1):
            mlp.append(nn.Linear(in_features=features[i], out_features=features[i + 1]))
            mlp.append(activation)
        mlp.append(nn.Linear(in_features=features[-1], out_features=out_features))
        mlp.append(output_activation)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)
        return x


class ResidualDense(nn.Module):
    """ResidualMLP module for Flax."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.GELU(),
        output_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.residual = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features * 2)
        self.linear2 = nn.Linear(in_features=in_features * 2, out_features=out_features)
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = x + residual
        x = self.output_activation(x)
        return x


class ResidualMLP(nn.Module):
    """ResidualMLP module for Flax."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        features: Sequence[int],
        activation: nn.Module = nn.GELU(),
        output_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()

        features = [in_features] + list(features)
        mlp = nn.ModuleList()
        for i in range(len(features) - 1):
            mlp.append(
                ResidualDense(features[i], features[i + 1], activation, activation)
            )
        mlp.append(nn.Linear(in_features=features[-1], out_features=out_features))
        mlp.append(output_activation)
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)
        return x
