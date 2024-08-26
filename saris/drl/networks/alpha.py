import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
import jax
from jax._src import core
from saris.drl.networks.network_utils import DType
from jax._src.typing import Array
from jax._src import dtypes


def alpha_init(
    key: jax.random.PRNGKey,
    shape: core.Shape,
    temperature: float,
    dtype: DType = jnp.bfloat16,
) -> Array:
    """An initializer that returns a constant array full of ones.

    The ``key`` argument is ignored.

    >>> import jax, jax.numpy as jnp
    >>> jax.nn.initializers.ones(jax.random.key(42), (3, 2), jnp.float32)
    Array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)
    """
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * temperature


class Alpha(nn.Module):
    """
    Temperature parameter for entropy.
    """

    temperature: float = 0.05
    alpha_init: Callable = alpha_init

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        alpha = self.param(
            "alpha",  # parametar name (as it will appear in the FrozenDict)
            self.alpha_init,  # initialization function, RNG passed implicitly through init fn
            (x.shape[-1], 1),
            self.temperature,
        )  # shape info

        x = jnp.dot(x, alpha)
        x = jnp.mean(x)
        x = jnp.clip(x, 0.0001, 0.15)
        return x
