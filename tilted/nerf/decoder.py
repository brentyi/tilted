"""NeRF-specific decoder implementation."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import jax
import jax.scipy.special
from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..core.decoder import fourier_encode


class ShallowDensityDecoder(nn.Module):
    """Single linear dimensionality reduction + rectifying activation."""

    density_activation: Literal["trunc_exp", "softplus"] = "trunc_exp"

    @nn.compact
    def __call__(self, x: Float[Array, "... c"]) -> Float[Array, "..."]:
        x = nn.Dense(1, use_bias=True)(x)

        if self.density_activation == "softplus":
            x = nn.softplus(x + 10.0)
        else:
            x = trunc_exp(x)

        return x.squeeze(axis=-1)


class NerfDecoderMlp(nn.Module):
    """MLP for regressing color."""

    viewdir_sph_harm_levels: Literal[0, 1, 2, 3, 4, 5] = 4
    feature_n_freqs: int = 0
    viewdir_n_freqs: int = 0
    camera_embed_dim: int = 0

    color_layers: int = 2
    color_units: int = 128

    density_activation: Literal["trunc_exp", "softplus"] = "trunc_exp"

    @nn.compact
    def __call__(
        self,
        features: Float[Array, "... c"],
        viewdirs: Float[Array, "... 3"],
        camera_indices: Int[Array, "..."],
        num_cameras: int,
        low_pass_alpha: Optional[Float[Array, ""]] = None,
    ) -> Tuple[Float[Array, "... 3"], Optional[Float[Array, "... 1"]]]:
        batch_axes = features.shape[:-1]

        assert (camera_indices is None and num_cameras is None) or (
            camera_indices is not None and num_cameras is not None
        ), "If `camera_indices` is provided, so too should `total_cameras`."
        assert viewdirs.shape == (*batch_axes, 3)
        assert camera_indices.shape == batch_axes

        # Base layers.
        features = features
        features = nn.Dense(
            16, use_bias=True, kernel_init=nn.initializers.kaiming_normal()
        )(features)
        features = nn.relu(features)

        # Decode density.
        sigmas = ShallowDensityDecoder(self.density_activation)(features)

        # Make input for RGB.
        color_feat = [features]
        if self.viewdir_sph_harm_levels > 0:
            # Encode view directions with spherical harmonics.
            color_feat.append(
                components_from_spherical_harmonics(
                    viewdirs, levels=self.viewdir_sph_harm_levels
                )
            )
        if self.feature_n_freqs > 0:
            # Fourier encoding for features.
            color_feat.append(
                fourier_encode(
                    features, self.feature_n_freqs, low_pass_alpha=low_pass_alpha
                ),
            )
        if self.viewdir_n_freqs > 0:
            # Fourier encoding for view directions.
            color_feat.append(
                fourier_encode(
                    viewdirs, self.viewdir_n_freqs, low_pass_alpha=low_pass_alpha
                ),
            )

        if self.camera_embed_dim > 0:
            # Camera embedding.
            camera_embed = nn.Embed(
                num_embeddings=num_cameras, features=self.camera_embed_dim
            )(camera_indices)
            color_feat.append(camera_embed)

        # Apply color MLP.
        rgb = jnp.concatenate(color_feat, axis=-1)
        for i in range(self.color_layers):
            rgb = nn.Dense(
                features=self.color_units,
                use_bias=True,
                kernel_init=nn.initializers.kaiming_normal(),
            )(rgb)
            rgb = nn.relu(rgb)
        rgb = nn.Dense(
            features=3,
            use_bias=True,
            kernel_init=nn.initializers.glorot_normal(),
        )(rgb)
        rgb = nn.sigmoid(rgb)

        assert rgb.shape[-1] == 3
        return rgb, sigmas


@jax.custom_jvp
def trunc_exp(x: jax.Array) -> jax.Array:
    """Exponential with a clipped gradients."""
    return jnp.exp(x)


@trunc_exp.defjvp
def trunc_exp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = trunc_exp(x)
    tangent_out = x_dot * jnp.exp(jnp.clip(x, -15, 15))
    return primal_out, tangent_out


def components_from_spherical_harmonics(
    directions: Float[Array, "... 3"],
    levels: Literal[0, 1, 2, 3, 4, 5],
) -> Float[Array, "... levels_sq"]:
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    components = []
    components.append(
        jnp.full(x.shape, fill_value=0.28209479177387814, dtype=directions.dtype)
    )
    if levels > 1:
        components.extend(
            [
                0.4886025119029199 * y,
                0.4886025119029199 * z,
                0.4886025119029199 * x,
            ]
        )
    if levels > 2:
        components.extend(
            [
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.9461746957575601 * zz - 0.31539156525251999,
                1.0925484305920792 * x * z,
                0.5462742152960396 * (xx - yy),
            ]
        )
    if levels > 3:
        components.extend(
            [
                0.5900435899266435 * y * (3 * xx - yy),
                2.890611442640554 * x * y * z,
                0.4570457994644658 * y * (5 * zz - 1),
                0.3731763325901154 * z * (5 * zz - 3),
                0.4570457994644658 * x * (5 * zz - 1),
                1.445305721320277 * z * (xx - yy),
                0.5900435899266435 * x * (xx - 3 * yy),
            ]
        )
    if levels > 4:
        components.extend(
            [
                2.5033429417967046 * x * y * (xx - yy),
                1.7701307697799304 * y * z * (3 * xx - yy),
                0.9461746957575601 * x * y * (7 * zz - 1),
                0.6690465435572892 * y * (7 * zz - 3),
                0.10578554691520431 * (35 * zz * zz - 30 * zz + 3),
                0.6690465435572892 * x * z * (7 * zz - 3),
                0.47308734787878004 * (xx - yy) * (7 * zz - 1),
                1.7701307697799304 * x * z * (xx - 3 * yy),
                0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)),
            ]
        )
    out = jnp.stack(components, axis=-1)
    assert out.shape == (*x.shape, levels**2)
    return out


def fourier_encode(
    coords: jax.Array, n_freqs: int, low_pass_alpha: Optional[Float[Array, ""]]
) -> jax.Array:
    """Fourier feature helper.

    Args:
        coords: Coordinates of shape (*, D).
        n_freqs: Number of fourier frequencies.
        low_pass_alpha: Should be within [0, n_freqs - 1]

    Returns:
        jax.Array: Shape (*, n_freqs * 2).
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
