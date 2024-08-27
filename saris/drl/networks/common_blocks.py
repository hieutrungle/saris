from typing import Tuple
from jax import numpy as jnp
from flax import linen as nn
import functools
import numpy as np


class DownResidualBlock(nn.Module):
    """ResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str = "SAME"
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            self.features * 4, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = self.activation(x)
        x = nn.Conv(self.features, (1, 1), (1, 1), padding="SAME", dtype=self.dtype)(x)
        x = x + residual
        return x


class UpResidualBlock(nn.Module):
    """UpResidualBlock module for Flax."""

    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str = "SAME"
    activation: nn.activation = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = nn.ConvTranspose(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = nn.ConvTranspose(
            self.features,
            self.kernel_size,
            self.strides,
            padding=self.padding,
            dtype=self.dtype,
        )(x)
        x = self.activation(x)
        x = nn.ConvTranspose(
            self.features * 4, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = self.activation(x)
        x = nn.ConvTranspose(
            self.features, (1, 1), (1, 1), padding="SAME", dtype=self.dtype
        )(x)
        x = x + residual
        return x


class EvoPositionalEmbedding(nn.Module):
    """EvoPositionalEmbedding module for Flax."""

    hidden_size: int
    max_seq_len: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size, dtype=self.dtype, name="input_layer")(x)
        pos = jnp.arange(x.shape[1], dtype=jnp.int16)
        pos_emb = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.hidden_size,
            dtype=self.dtype,
            name="pos_emb",
        )(pos)
        pos_emb = pos_emb.astype(self.dtype)
        x = x + pos_emb[None, : x.shape[1]]
        return x


class TransformerEncoderBlock(nn.Module):
    """TransformerEncoder module for Flax."""

    hidden_size: int
    num_heads: int
    causal_mask: bool
    dtype: jnp.dtype
    dropout_rate: float = 0.05
    mask: jnp.ndarray | None = None
    train: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        attn_outs = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1] * 2,
            out_features=x.shape[-1],
            dropout_rate=self.dropout_rate,
            deterministic=not self.train,
            dtype=self.dtype,
            force_fp32_for_softmax=True,
            # normalize_qk=True,
        )(x, x, x, mask=self.mask)

        x = x + nn.Dropout(rate=self.dropout_rate)(
            attn_outs, deterministic=not self.train
        )
        x = nn.LayerNorm(dtype=self.dtype)(x)

        # MLP block
        linear_outs = nn.Dense(
            self.hidden_size * 4, dtype=self.dtype, name="mlp_expand"
        )(x)
        linear_outs = nn.gelu(linear_outs)
        linear_outs = nn.Dense(self.hidden_size, dtype=self.dtype, name="mlp_contract")(
            linear_outs
        )
        x = x + nn.Dropout(rate=self.dropout_rate)(
            linear_outs, deterministic=not self.train
        )
        x = nn.LayerNorm(dtype=self.dtype)(x)

        return x


class Fourier(nn.Module):
    """Fourier features for encoding the input signal."""

    num_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.num_features, use_bias=False)(x)
        x = 2 * np.pi * x
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x
