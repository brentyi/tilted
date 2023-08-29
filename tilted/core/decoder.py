from __future__ import annotations

from typing import Optional

import jax
from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float


class DecoderMlp(nn.Module):
    """MLP for computing decoding quantities from latent features. When layers=0, this
    reduces to a simple affine transformation."""

    output_sigmoid: bool
    output_dim: int

    units: int = 16
    layers: int = 2  # When layers=0, we reduce to a single linear transform.

    basis_dim: int = 16

    pos_enc_freqs: int = 4

    @nn.compact
    def __call__(
        self, x: Float[Array, "... c"], barf_alpha: Optional[Float[Array, ""]]
    ) -> Float[Array, "... output_dim"]:
        # Squash input features.
        if self.pos_enc_freqs > 0:
            x = nn.Dense(
                features=self.basis_dim,
                kernel_init=nn.initializers.glorot_normal(),
                use_bias=False,
            )(x)
            encoded_x = fourier_encode(x, self.pos_enc_freqs, barf_alpha)
            x = jnp.concatenate([x, encoded_x], axis=-1)

        for i in range(self.layers):
            x = nn.Dense(
                features=self.units,
                use_bias=True,
                kernel_init=nn.initializers.kaiming_normal(),
            )(x)
            x = nn.relu(x)

        x = nn.Dense(
            features=self.output_dim,
            use_bias=False,
            kernel_init=nn.initializers.glorot_normal(),
        )(x)
        if self.output_sigmoid:
            x = nn.sigmoid(x)

        return x


def fourier_encode(
    coords: jax.Array, n_freqs: int, low_pass_alpha: Optional[Float[Array, ""]]
) -> jax.Array:
    """Fourier feature helper.

    Args:
        coords: Coordinates of shape (*, D).
        n_freqs: Number of fourier frequencies.
        low_pass_alpha: Should be within [0, n_freqs - 1]

    Returns:
        jnp.ndarray: Shape (*, n_freqs * 2).
    """
    *batch_axes, D = coords.shape
    coeffs = 2.0 ** jnp.arange(n_freqs)
    inputs = coords[..., None] * coeffs
    assert inputs.shape == (*batch_axes, D, n_freqs)

    out = jnp.sin(
        jnp.stack(
            [inputs, inputs + 0.5 * jnp.pi],
            axis=-1,
        )
    )

    if low_pass_alpha is not None:
        k = jnp.arange(n_freqs)
        barf_weights = jnp.select(
            [low_pass_alpha < k, low_pass_alpha >= k + 1, jnp.array(True)],
            [0.0, 1.0, (1.0 - jnp.cos((low_pass_alpha - k) * jnp.pi)) / 2.0],
        )
        assert barf_weights.shape == (n_freqs,)
        assert out.shape[-2:] == (n_freqs, 2)

        out = barf_weights[:, None] * out

    return rearrange(
        out,
        "... D freqs P -> ... (D freqs P)",
        D=D,
        freqs=n_freqs,
        P=2,
    )
