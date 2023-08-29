from __future__ import annotations

import dataclasses
import functools
from typing import Any, Literal, Optional, Tuple, Union

import flax.core
import jax
import jax_dataclasses as jdc
import jaxlie
import matplotlib as mpl
import numpy as onp
from einops import einsum, reduce
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import assert_never

from ..core.factored_grid import FactoredGrid
from . import cameras, interlevel, segments


@jdc.pytree_dataclass
class HybridField:
    grid: FactoredGrid
    decoder: jdc.Static[nn.Module]
    decoder_params: flax.core.FrozenDict[str, Any]


@jdc.pytree_dataclass
class AABBSceneContraction:
    ...


@jdc.pytree_dataclass
class LearnableParams:
    # Ideally, we can support a few approaches for factoring density + RGB:
    #
    # - TensoRF style:
    #       One density grid, one RGB grid.
    # - K-Planes style:
    #       Proposal grids, one RGB+density grid.
    # - Hybrid of the two:
    #       Proposal grids, one RGB grid.
    density_fields: Tuple[HybridField, ...]
    primary_field: HybridField
    num_cameras: jdc.Static[int]
    camera_deltas: Optional[Tuple[jaxlie.SO3, Float[Array, "num_cameras 3"]]]
    """Camera extrinsic optimization parameter. Applied a little bit weirdly, see implementation."""

    def make_optax_mask(
        self, label: Literal["factors", "projections", "decoders", "camera_deltas"]
    ) -> LearnableParams:
        with jdc.copy_and_mutate(self, validate=False) as mask:
            for f in mask.density_fields:
                f.grid.factors = label == "factors"  # type: ignore
                f.grid.projecters = label == "projections"  # type: ignore
                f.decoder_params = label == "decoders"  # type: ignore

            mask.primary_field.grid.factors = label == "factors"  # type: ignore
            mask.primary_field.grid.projecters = label == "projections"  # type: ignore
            mask.primary_field.decoder_params = label == "decoders"  # type: ignore
            mask.camera_deltas = label == "camera_deltas"  # type: ignore
        return mask


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    proposal_samples: Tuple[int, ...]
    final_samples: int
    background: Literal["white", "last_sample"]

    near: float
    far: float

    initial_samples: Literal["aabb_collider", "linear", "linear_disparity"]
    sample_aabb_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    sample_aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    histogram_padding: float = 0.01

    # Used to globally rotate the representation.
    global_rotate_seed: Optional[int] = None

    # Scale gradients quadratically based on distance. Not working well.
    # https://arxiv.org/pdf/2305.02756.pdf
    gradient_scaling: bool = False

    def sample_initial(
        self,
        rays_wrt_world: cameras.Rays3D,
        samples_per_ray: int,
        prng: jax.random.KeyArray,
    ) -> segments.Segments:
        """Returns segments."""
        # TODO: this returns a batch axis along the ray count, but we rarely need it.
        # Should remove.
        ray_count = rays_wrt_world.origins.shape[0]
        if self.initial_samples == "aabb_collider":
            # Sample with a ray collider, similar to the original TensoRF implementation.
            # With the proposal network, the benefits of this are pretty small.
            return jax.vmap(
                lambda ray: segments.collide_ray_with_aabb(
                    ray_wrt_world=ray,
                    aabb=jnp.array([self.sample_aabb_min, self.sample_aabb_max]),
                    samples_per_ray=samples_per_ray,
                    prng=prng,
                ),
            )(rays_wrt_world)
        elif self.initial_samples == "linear":
            boundaries = jnp.arange(samples_per_ray + 1)
            boundaries = boundaries + jax.random.uniform(
                key=prng, shape=boundaries.shape
            )
            boundaries = (
                boundaries / (samples_per_ray + 1) * (self.far - self.near) + self.near
            )
            return jax.tree_map(
                lambda a: jnp.broadcast_to(a[None, :], (ray_count, *a.shape)),
                segments.Segments.from_boundaries(boundaries),
            )
        elif self.initial_samples == "linear_disparity":
            # Contracted scene: sample linearly for close samples, then start spacing
            # samples out.
            close_samples_per_ray = samples_per_ray // 2
            far_samples_per_ray = samples_per_ray - close_samples_per_ray + 1

            close_ts = jnp.linspace(self.near, self.near + 1.0, close_samples_per_ray)

            # Some heuristics for sampling far points, which should be close to sampling
            # linearly in disparity when k=1. This is probably reasonable, but it'd be a
            # good idea to look at what others do...
            #
            # TODO revisit!
            far_start = self.near + 1.0 + 1.0 / close_samples_per_ray
            k = 10.0
            far_deltas = (
                1.0
                / (
                    1.0
                    - jnp.linspace(  # onp here is important for float64.
                        0.0,
                        1.0 - 1 / ((self.far - far_start) / k + 1),
                        far_samples_per_ray,
                    )
                )
                - 1.0
            ) * jnp.linspace(1.0, k, far_samples_per_ray)
            far_ts = far_start + far_deltas

            boundaries = jnp.broadcast_to(
                jnp.concatenate([close_ts, far_ts])[None, :],
                (ray_count, samples_per_ray + 1),
            )
            return segments.Segments.from_boundaries(boundaries)

        else:
            assert_never(self.initial_samples)

    def compute_sdist(self, boundaries: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient((boundaries - self.near) / (self.far - self.near))
        # return (boundaries - boundaries[..., :1]) / (
        #     boundaries[..., -1:] - boundaries[..., :1]
        # )


@jdc.pytree_dataclass
class RenderOutputs:
    rgb: Float[Array, "... 3"]
    transform_feature_norm: Float[Array, "... transform"]
    features: Float[Array, "... channels"]
    proposal_dist_maps: Tuple[Float[Array, "..."], ...]
    dist: Float[Array, "... 3"]
    interlevel_loss: Float[Array, ""]
    distortion_loss: Float[Array, ""]


@jdc.jit
def render_rays(
    rays_wrt_world: cameras.Rays3D,
    params: LearnableParams,
    config: jdc.Static[RenderConfig],
    prng: jax.random.KeyArray,
    anneal_factor: Union[float, Float[Array, ""]],
    low_pass_alpha: Optional[Float[Array, ""]],
) -> RenderOutputs:
    prng, prng_ = jax.random.split(prng)

    # Support arbitrary leading batch axes.
    *batch_axes, d = rays_wrt_world.origins.shape
    if config.global_rotate_seed is not None:
        global_rotate = jaxlie.SO3.sample_uniform(
            jax.random.PRNGKey(config.global_rotate_seed)
        ).as_matrix()
        rays_wrt_world = cameras.Rays3D(
            origins=rays_wrt_world.origins.reshape((-1, 3)) @ global_rotate,
            directions=rays_wrt_world.directions.reshape((-1, 3)) @ global_rotate,
            camera_indices=rays_wrt_world.camera_indices.reshape((-1,)),
        )
    else:
        rays_wrt_world = cameras.Rays3D(
            origins=rays_wrt_world.origins.reshape((-1, 3)),
            directions=rays_wrt_world.directions.reshape((-1, 3)),
            camera_indices=rays_wrt_world.camera_indices.reshape((-1,)),
        )

    num_rays = rays_wrt_world.origins.shape[0]

    if params.camera_deltas is not None:
        camera_rot, camera_trans = jax.tree_map(
            lambda a: a[rays_wrt_world.camera_indices], params.camera_deltas
        )
        assert isinstance(camera_rot, jaxlie.SO3)
        assert camera_rot.get_batch_axes() == (num_rays,)
        assert camera_trans.shape == (num_rays, 3)

        rays_wrt_world = cameras.Rays3D(
            origins=rays_wrt_world.origins + camera_trans,
            directions=einsum(
                rays_wrt_world.directions,
                jax.vmap(jaxlie.SO3.as_matrix)(camera_rot),
                "batch j, batch i j -> batch i",
            ),
            camera_indices=rays_wrt_world.camera_indices,
        )

    ray_segments = config.sample_initial(
        rays_wrt_world,
        samples_per_ray=config.proposal_samples[0],
        # samples_per_ray=config.candidate_samples,
        prng=prng_,
    )
    points = rays_wrt_world.points_from_ts(ray_segments.ts)
    assert points.shape[-1] == 3

    # Proposal sampling.
    proposal_dist_maps = []
    ray_history = []
    for i, proposal_field in enumerate(params.density_fields):
        density_feat, _ = proposal_field.grid.interpolate(points)
        sigmas = proposal_field.decoder.apply(
            proposal_field.decoder_params, density_feat
        )
        assert isinstance(sigmas, Array)
        assert sigmas.shape == points.shape[:-1]

        # Pad and re-normalize weights.
        probs = segments.SegmentProbabilities.compute(
            sigmas=sigmas, ray_segments=ray_segments
        )
        proposal_dist_maps.append(probs.render_distance())
        assert ray_segments.boundaries is not None
        ray_history.append(
            {
                "sdist": config.compute_sdist(ray_segments.boundaries),
                "weights": probs.p_terminates,
            }
        )

        weights = probs.p_terminates
        weights = weights + config.histogram_padding
        assert weights.shape == (num_rays, config.proposal_samples[i])
        weights /= jnp.sum(weights, axis=-1, keepdims=True)

        prng, prng_ = jax.random.split(prng)
        boundaries = ray_segments.weighted_sample_stratified(
            weights=weights**anneal_factor,
            prng=prng_,
            num_samples=(config.proposal_samples[1:] + (config.final_samples,))[i] + 1,
        )
        boundaries = jax.lax.stop_gradient(boundaries)
        ray_segments = segments.Segments.from_boundaries(boundaries)
        points = rays_wrt_world.points_from_ts(ray_segments.ts)

    # TensoRF-style rendering.
    #
    # density_feat = params.density_field.grid.interpolate(points / 1.5)
    # sigmas = params.density_field.decoder.apply(
    #     params.density_field.decoder_params, density_feat
    # )
    # assert isinstance(sigmas, Array)
    # probs = segments.SegmentProbabilities.compute(sigmas=sigmas, segments=segments)
    # prng, prng_ = jax.random.split(prng)
    # probs, render_indices = probs.resample_subset(
    #     num_samples=config.final_samples, prng=prng, anneal_factor=1.0
    # )
    # points = jnp.take_along_axis(arr=points, indices=render_indices[:, :, None], axis=1)

    num_rays_, samples_per_ray, d = points.shape
    assert num_rays_ == num_rays
    assert d == 3
    #  assert samples_per_ray == config.final_samples

    # Regress RGB + sigma values.
    primary_feat, primary_components = params.primary_field.grid.interpolate(points)
    assert len(rays_wrt_world.directions.shape) == 2  # (ray_count, 3)
    viewdirs = jnp.tile(rays_wrt_world.directions[:, None, :], (1, samples_per_ray, 1))
    camera_indices = rays_wrt_world.camera_indices
    assert camera_indices.shape == (num_rays,)
    camera_indices = jnp.tile(camera_indices[:, None], (1, samples_per_ray))
    assert camera_indices.shape == (
        num_rays,
        samples_per_ray,
    )
    rgb, sigmas = params.primary_field.decoder.apply(
        params.primary_field.decoder_params,
        features=primary_feat,
        viewdirs=viewdirs,
        camera_indices=camera_indices,
        num_cameras=params.num_cameras,
        low_pass_alpha=low_pass_alpha,
    )
    assert isinstance(rgb, Array)
    assert isinstance(sigmas, Array)

    # Map to [0.0, 1.0].
    # https://arxiv.org/pdf/2305.02756.pdf
    if config.gradient_scaling:
        # Gradient scaling.
        rgb, sigmas = gradient_scaling(ray_segments.ts.clip(0.0, 1.0), rgb, sigmas)

    probs = segments.SegmentProbabilities.compute(
        sigmas=sigmas, ray_segments=ray_segments
    )

    assert ray_segments.boundaries is not None
    ray_history.append(
        {
            "sdist": config.compute_sdist(ray_segments.boundaries),
            "weights": probs.p_terminates,
        }
    )

    # Transform norm visualization.
    transform_count = getattr(
        params.primary_field.grid.projecters, "transform_count", 1
    )
    component_norms = functools.reduce(
        jnp.add,
        map(
            # Each component has shape (groups, rays, samples, channels)
            lambda a: reduce(
                a**2,
                "(g transform_count) rays samples channels -> rays samples transform_count",
                reduction="sum",
                transform_count=transform_count,
                rays=num_rays,
            ),
            primary_components,
        ),
    )
    assert (
        component_norms.shape
        == probs.p_terminates.shape + (transform_count,)
        == rgb.shape[:-1] + (transform_count,)
    )
    transform_feature_norm = einsum(
        component_norms,
        probs.p_terminates,
        "rays samples transform_count, rays samples -> rays transform_count",
    )

    # Feature visualization. Used for PCA.
    rendered_features = einsum(
        primary_feat,
        probs.p_terminates,
        "rays samples channels, rays samples -> rays channels",
    )

    expected_rgb = jnp.sum(rgb * probs.p_terminates[:, :, None], axis=-2)
    assert expected_rgb.shape == (num_rays, 3)

    if config.background == "white":
        background_color = jnp.ones(3)
    elif config.background == "last_sample":
        background_color = rgb[..., -1, :]
    else:
        assert_never(config.background)

    expected_rgb_with_background = (
        expected_rgb + probs.p_exits[:, -1:] * background_color
    )
    assert expected_rgb_with_background.shape == (num_rays, 3)

    return RenderOutputs(
        rgb=expected_rgb_with_background.reshape((*batch_axes, 3)),
        transform_feature_norm=transform_feature_norm.reshape((*batch_axes, -1)),
        features=rendered_features,
        proposal_dist_maps=tuple(d.reshape(batch_axes) for d in proposal_dist_maps),
        dist=probs.render_distance().reshape(batch_axes),
        interlevel_loss=interlevel.interlevel_loss(ray_history),
        distortion_loss=probs.distortion_loss(),
    )


def viz_dist(image: onp.ndarray, cmap: str = "plasma") -> onp.ndarray:
    # Visualization heuristics for "depths".
    out = 1.0 / onp.maximum(image, 1e-4)

    min_invdist = out.min()
    max_invdist = out.max() * 0.9

    out -= min_invdist
    out /= max_invdist - min_invdist
    out = (mpl.colormaps[cmap](out) * 255).astype(onp.uint8)
    assert out.shape[-1] == 4
    # out = onp.clip(out * 255.0, 0.0, 255.0).astype(onp.uint8)
    # out = onp.tile(out[:, :, None], reps=(1, 1, 3))

    return out


@jax.custom_jvp
def gradient_scaling(sdist, rgb, density):
    """https://arxiv.org/pdf/2305.02756.pdf"""
    assert sdist.shape == density.shape == rgb.shape[:-1]
    return rgb, density


@gradient_scaling.defjvp
def lgradient_scaling_jvp(primals, tangents):
    sdist, rgb, density = primals
    assert sdist.shape == density.shape == rgb.shape[:-1]

    _, rgb_dot, density_dot = tangents

    ans = gradient_scaling(sdist, rgb, density)
    scaling = jnp.square(sdist)
    ans_dot = (rgb_dot * scaling[..., None], density_dot * scaling)
    return ans, ans_dot
