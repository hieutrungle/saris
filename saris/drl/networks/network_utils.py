from typing import Callable, Union
import jax.numpy as jnp
from flax import linen as nn

Activation = Union[str, Callable]
DType = Union[str, jnp.dtype]


class Identity(nn.Module):
    """Identity module for Flax."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


_str_to_dtype = {
    "float32": jnp.float32,
    "float64": jnp.float64,
    "bfloat16": jnp.bfloat16,
}

_str_to_activation = {
    "relu": nn.activation.relu,
    "tanh": nn.activation.tanh,
    "sigmoid": nn.activation.sigmoid,
    "swish": nn.activation.hard_swish,
    "gelu": nn.activation.gelu,
    "elu": nn.activation.elu,
    "identity": Identity(),
}
