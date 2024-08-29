from typing import Callable, Union

import torch
import torch.nn as nn

Activation = Union[str, Callable]
DType = Union[str, torch.dtype]


class Identity(nn.Module):
    """Identity module for Flax."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


_str_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}
