import chex
import jax
import jax.numpy as jnp
import numpy as np
from saris.distributions import distribution
from saris.distributions import bijector
from saris.distributions import lambda_bijector
from saris.distributions import sigmoid
from saris.distributions import tanh

Array = chex.Array
Numeric = chex.Numeric
BijectorLike = bijector.BijectorLike
DistributionLike = distribution.DistributionLike


def as_distribution(obj: DistributionLike) -> distribution.DistributionT:
    """Converts a distribution-like object to a Distrax distribution.

    Distribution-like objects are: Distrax distributions and TFP distributions.
    Distrax distributions are returned unchanged. TFP distributions are converted
    to a Distrax equivalent.

    Args:
      obj: A distribution-like object to be converted.

    Returns:
      A Distrax distribution.
    """
    if isinstance(obj, distribution.Distribution):
        return obj
    else:
        raise TypeError(
            f"A distribution-like object can be a `distrax.Distribution` or a"
            f" `tfd.Distribution`. Got type `{type(obj)}`."
        )


def as_bijector(obj: BijectorLike) -> bijector.BijectorT:
    """Converts a bijector-like object to a Distrax bijector.

    Bijector-like objects are: Distrax bijectors, TFP bijectors, and callables.
    Distrax bijectors are returned unchanged. TFP bijectors are converted to a
    Distrax equivalent. Callables are wrapped by `distrax.Lambda`, with a few
    exceptions where an explicit implementation already exists and is returned.

    Args:
      obj: The bijector-like object to be converted.

    Returns:
      A Distrax bijector.
    """
    if isinstance(obj, bijector.Bijector):
        return obj
    elif obj is jax.nn.sigmoid:
        return sigmoid.Sigmoid()
    elif obj is jnp.tanh:
        return tanh.Tanh()
    elif callable(obj):
        return lambda_bijector.Lambda(obj)
    else:
        raise TypeError(
            f"A bijector-like object can be a `distrax.Bijector`, a `tfb.Bijector`,"
            f" or a callable. Got type `{type(obj)}`."
        )


def as_float_array(x: Numeric) -> Array:
    """Converts input to an array with floating-point dtype.

    If the input is already an array with floating-point dtype, it is returned
    unchanged.

    Args:
      x: input to convert.

    Returns:
      An array with floating-point dtype.
    """
    if not isinstance(x, (jax.Array, np.ndarray)):
        x = jnp.asarray(x)

    if jnp.issubdtype(x.dtype, jnp.floating):
        return x
    elif jnp.issubdtype(x.dtype, jnp.integer):
        return x.astype(jnp.float_)
    else:
        raise ValueError(f"Expected either floating or integer dtype, got {x.dtype}.")
