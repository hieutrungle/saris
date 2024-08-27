import jax.numpy as jnp
from flax import linen as nn


class Alpha(nn.Module):
    """
    Temperature parameter for entropy.
    """

    temperature: float = 0.05

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(
            features=1,
            kernel_init=nn.initializers.constant(self.temperature),
            use_bias=False,
        )(x)
        x = jnp.mean(x)
        x = jnp.clip(x, 0.0001, 0.15)
        return x
