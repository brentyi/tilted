from __future__ import annotations

from typing import Literal, Optional, Tuple

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from . import cameras


@jdc.pytree_dataclass
class SegmentProbabilities:
    p_exits: Float[Array, "rays samples"]
    """P(ray exits segment s).

    Note that this also implies that the ray has exited (and thus entered) all previous
    segments."""

    p_terminates: Float[Array, "rays samples"]
    """P(ray terminates at s, ray exits s - 1).

    For a ray to terminate in a segment, it must first pass through (and 'exit') all
    previous segments."""

    segments: Segments

    def distortion_loss(self) -> Float[Array, ""]:
        # The loss incurred between all pairs of intervals.
        # ut = (t[..., 1:] + t[..., :-1]) / 2
        ut = self.segments.ts
        weights = self.p_terminates
        dut = jnp.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = jnp.sum(
            weights * jnp.sum(weights[..., None, :] * dut, axis=-1), axis=-1
        )

        # The loss incurred within each individual interval with itself.
        loss_intra = jnp.sum(weights**2 * self.segments.step_sizes, axis=-1) / 3

        return jnp.mean(loss_inter + loss_intra)

    def get_num_rays(self) -> int:
        # The batch axes should be (num_rays, num_samples).
        assert self.p_exits.shape == self.p_terminates.shape
        shape = self.p_exits.shape
        assert len(shape) == 2
        return shape[0]

    def resample_subset(
        self,
        num_samples: int,
        prng: jax.random.KeyArray,
        anneal_factor: float = 1.0,
    ) -> Tuple[SegmentProbabilities, Int[Array, "rays samples"]]:
        """Hierarchical resampling.

        Returns resampled probabilities and a selected index array.

        anneal_factor=0.0 => uniform distribution.
        anneal_factor=1.0 => samples weighted by termination probability.
        """
        probs = self.p_terminates**anneal_factor
        ray_count, orig_num_samples = probs.shape

        sampled_indices = jax.vmap(
            lambda p, prng: jax.random.choice(
                key=prng,
                a=orig_num_samples,
                shape=(num_samples,),
                replace=False,
                p=p,
            )
        )(
            probs,
            jax.random.split(prng, num=ray_count),
        )
        return self._sample_subset(sampled_indices), sampled_indices

    def _sample_subset(
        self, sample_indices: Int[Array, "rays new_samples"]
    ) -> SegmentProbabilities:
        """Pull out a subset of the sample probabilities from an index array of shape
        (num_rays, new_number_of_samples)."""
        num_rays, new_sample_count = sample_indices.shape
        assert num_rays == self.get_num_rays()

        # Extract subsets of probabilities.
        #
        # For more accurate background rendering, we match the exit probablity of the
        # last sample to the original.
        sub_p_exits = jnp.take_along_axis(
            self.p_exits, sample_indices.at[:, -1].set(-1), axis=-1
        )
        sub_p_terminates = jnp.take_along_axis(
            self.p_terminates, sample_indices, axis=-1
        )
        sub_bins = Segments(
            ts=jnp.take_along_axis(self.segments.ts, sample_indices, axis=-1),
            step_sizes=jnp.take_along_axis(
                self.segments.step_sizes, sample_indices, axis=-1
            ),
            boundaries=None,
            starts=jnp.take_along_axis(self.segments.starts, sample_indices, axis=-1),
            ends=jnp.take_along_axis(self.segments.ends, sample_indices, axis=-1),
            contiguous=False,
        )
        assert sub_bins.ts.shape == sub_p_exits.shape

        # Coefficients for unbiasing the expected RGB values using the sampling
        # probabilities. This is helpful because RGB values for points that are not chosen
        # by our appearance sampler are zeroed out.
        #
        # As an example: if the sum of weights* for all samples is 0.95** but the sum of
        # weights for our subset is only 0.7, we can correct the subset weights by
        # (0.95/0.7).
        #
        # *weight at a segment = termination probability at that segment
        # **equivalently: p=0.05 of the ray exiting the last segment and rendering the
        # background.
        eps = 1e-10
        unbias_coeff = (
            # The 0.95 term in the example.
            1.0
            - self.p_exits[:, -1]
            + eps
        ) / (
            # The 0.7 term in the example.
            jnp.sum(sub_p_terminates, axis=1)
            + eps
        )
        assert unbias_coeff.shape == (num_rays,)

        # TODO: uncomment this out?
        # unbias_coeff = jax.lax.stop_gradient(unbias_coeff)
        sub_p_terminates = sub_p_terminates * unbias_coeff[:, None]

        out = SegmentProbabilities(
            p_exits=sub_p_exits,
            p_terminates=sub_p_terminates,
            segments=sub_bins,
        )
        assert (
            sub_p_exits.shape == sub_p_terminates.shape == (num_rays, new_sample_count)
        )
        return out

    def render_distance(
        self, mode: Literal["mean", "median"] = "median"
    ) -> Float[Array, "*ray_dims"]:
        """Render distances. Useful for depth maps, etc."""
        if mode == "mean":
            # Compute distance via expected value.
            sample_distances = jnp.concatenate(
                [self.segments.ts, self.segments.ts[:, -1:]], axis=-1
            )
            p_terminates_padded = jnp.concatenate(
                [self.p_terminates, self.p_exits[:, -1:]], axis=-1
            )
            assert sample_distances.shape == p_terminates_padded.shape
            return jnp.sum(p_terminates_padded * sample_distances, axis=-1)
        elif mode == "median":
            dtype = self.segments.ts.dtype
            (*batch_axes, _num_samples) = self.p_exits.shape

            # Compute distance via median.
            sample_distances = jnp.concatenate(
                [self.segments.ts, jnp.full((*batch_axes, 1), jnp.inf, dtype=dtype)],
                axis=-1,
            )
            p_not_alive_padded = jnp.concatenate(
                [1.0 - self.p_exits, jnp.ones((*batch_axes, 1), dtype=dtype)], axis=-1
            )
            assert sample_distances.shape == p_not_alive_padded.shape

            median_mask = p_not_alive_padded > 0.5
            median_mask = (
                jnp.zeros_like(median_mask)
                .at[..., 1:]
                .set(jnp.logical_xor(median_mask[..., :-1], median_mask[..., 1:]))
            )

            # Output is medians.
            dists = jnp.sum(median_mask * sample_distances, axis=-1)
            return dists

    @staticmethod
    def compute(
        sigmas: jax.Array,
        ray_segments: Segments,
    ) -> SegmentProbabilities:
        r"""Compute some probabilities needed for rendering rays. Expects sigmas of shape
        (*, sample_count) and a per-ray step size of shape (*,).

        Each of the ray segments we're rendering is broken up into samples. We can treat the
        densities as piecewise constant and use an exponential distribution and compute:

          1. P(ray exits s) = exp(\sum_{i=1}^s -(sigma_i * l_i)
          2. P(ray terminates in s | ray exits s-1) = 1.0 - exp(-sigma_s * l_s)
          3. P(ray terminates in s, ray exits s-1)
             = P(ray terminates at s | ray exits s-1) * P(ray exits s-1)

        where l_i is the length of segment i.
        """

        # Support arbitrary leading batch axes.
        (*batch_axes, sample_count) = sigmas.shape
        assert (
            ray_segments.step_sizes.shape
            == sigmas.shape[-len(ray_segments.step_sizes.shape) :]
        )

        # Equation 1.
        neg_scaled_sigmas = -sigmas * ray_segments.step_sizes
        p_exits = jnp.exp(jnp.cumsum(neg_scaled_sigmas, axis=-1))
        assert p_exits.shape == (*batch_axes, sample_count)

        # Equation 2. Not used outside of this function, and not returned.
        p_terminates_given_exits_prev = 1.0 - jnp.exp(neg_scaled_sigmas)
        assert p_terminates_given_exits_prev.shape == (*batch_axes, sample_count)

        # Equation 3.
        p_terminates = jnp.multiply(
            p_terminates_given_exits_prev,
            # We prepend 1 because the ray is always alive initially.
            jnp.concatenate(
                [
                    jnp.ones((*batch_axes, 1), dtype=neg_scaled_sigmas.dtype),
                    p_exits[..., :-1],
                ],
                axis=-1,
            ),
        )
        assert p_terminates.shape == (*batch_axes, sample_count)

        return SegmentProbabilities(
            p_exits=p_exits,
            p_terminates=p_terminates,
            segments=ray_segments,
        )


@jdc.pytree_dataclass
class Segments:
    ts: jax.Array
    step_sizes: jax.Array

    boundaries: Optional[jax.Array]  # None if `contiguous` is False.
    starts: jax.Array
    ends: jax.Array

    contiguous: jdc.Static[bool]
    """True if there are no gaps between bins."""

    @staticmethod
    def from_boundaries(boundaries: jax.Array) -> Segments:
        starts = boundaries[..., :-1]
        ends = boundaries[..., 1:]
        step_sizes = ends - starts
        ts = starts + step_sizes / 2.0

        return Segments(
            ts=ts,
            step_sizes=step_sizes,
            boundaries=boundaries,
            starts=starts,
            ends=ends,
            contiguous=True,
        )

    def weighted_sample_stratified(
        self,
        weights: jax.Array,
        prng: jax.random.KeyArray,
        num_samples: int,
    ) -> jax.Array:
        *batch_dims, old_num_samples = weights.shape
        batch_dims = tuple(batch_dims)

        assert self.contiguous, "Currently only contiguous bins are supported."
        assert self.boundaries is not None

        # Accumulate weights, and scale from 0 to 1.
        accumulated_weights = jnp.cumsum(weights, axis=-1)
        accumulated_weights = jnp.concatenate(
            [jnp.zeros(batch_dims + (1,)), accumulated_weights],
            axis=-1,
        )
        accumulated_weights = accumulated_weights / (
            accumulated_weights[..., -1:] + 1e-4
        )
        assert accumulated_weights.shape == batch_dims + (old_num_samples + 1,)

        batch_dim_flattened = int(onp.prod(batch_dims))

        x = _sample_quasi_uniform_ordered(
            prng,
            min_bound=0.0,
            max_bound=1.0,
            bins=num_samples,
            batch_dims=(batch_dim_flattened,),
        )
        samples = jax.vmap(jnp.interp)(
            x=x,
            xp=accumulated_weights.reshape((batch_dim_flattened, -1)),
            fp=self.boundaries.reshape((batch_dim_flattened, -1)),
        ).reshape(batch_dims + (num_samples,))

        return samples


def _sample_quasi_uniform_ordered(
    prng: jax.random.KeyArray,
    min_bound: float,
    max_bound: float,
    bins: int,
    batch_dims: Tuple[int, ...],
) -> jax.Array:
    """Quasi-uniform sampling. Separates the sampling range into a specified number of
    bins, and selects one sample from each bin.
    Output is in ascending order."""
    sampling_bin_size = (max_bound - min_bound) / bins
    sampling_bin_starts = jnp.arange(0, bins) * sampling_bin_size

    # Add some batch axes; these two lines are totally unnecessary.
    for i in range(len(batch_dims)):
        sampling_bin_starts = sampling_bin_starts[None, ...]

    samples = sampling_bin_starts + jax.random.uniform(
        key=prng,
        shape=batch_dims + (bins,),
        minval=0.0,
        maxval=sampling_bin_size,
    )
    assert samples.shape == batch_dims + (bins,)
    return samples


def collide_ray_with_aabb(
    ray_wrt_world: cameras.Rays3D,
    aabb: jax.Array,
    samples_per_ray: int,
    prng: Optional[jax.random.KeyArray],
) -> Segments:
    assert ray_wrt_world.get_batch_axes() == ()
    assert ray_wrt_world.origins.shape == ray_wrt_world.directions.shape == (3,)

    # Get segment of ray that's within the bounding box.
    segment = _ray_segment_from_bounding_box(
        ray_wrt_world, aabb=aabb, min_segment_length=1e-3
    )
    assert segment.t_min.shape == segment.t_max.shape == ()

    intermediate_boundaries_per_ray = samples_per_ray - 1
    step_size = (segment.t_max - segment.t_min) / intermediate_boundaries_per_ray

    # Get sample points along ray.
    boundary_ts = jnp.arange(intermediate_boundaries_per_ray)
    if prng is not None:
        # Jitter if a PRNG key is passed in.
        boundary_ts = boundary_ts + jax.random.uniform(
            key=prng,
            shape=boundary_ts.shape,
            dtype=step_size.dtype,
        )
    boundary_ts = boundary_ts * step_size
    boundary_ts = segment.t_min + boundary_ts

    return Segments.from_boundaries(
        boundaries=jnp.concatenate(
            [segment.t_min[None], boundary_ts, segment.t_max[None]]
        )
    )


@jdc.pytree_dataclass
class _RaySegmentSpecification:
    t_min: jax.Array
    t_max: jax.Array


def _ray_segment_from_bounding_box(
    ray_wrt_world: cameras.Rays3D,
    aabb: jax.Array,
    min_segment_length: float,
) -> _RaySegmentSpecification:
    """Given a ray and bounding box, compute the near and far t values that define a
    segment that lies fully in the box."""
    assert ray_wrt_world.origins.shape == ray_wrt_world.directions.shape == (3,)
    assert aabb.shape == (2, 3)

    # Find t for per-axis collision with the bounding box.
    #     origin + t * direction = bounding box
    #     t = (bounding box - origin) / direction
    offsets = aabb - ray_wrt_world.origins[None, :]
    eps = 1e-10
    t_intersections = offsets / (ray_wrt_world.directions + eps)

    # Compute near/far distances.
    t_min_per_axis = jnp.min(t_intersections, axis=0)
    t_max_per_axis = jnp.max(t_intersections, axis=0)
    assert t_min_per_axis.shape == t_max_per_axis.shape == (3,)

    # Clip.
    t_min = jnp.maximum(0.0, jnp.max(t_min_per_axis))
    t_max = jnp.min(t_max_per_axis)
    t_max_clipped = jnp.maximum(t_max, t_min + min_segment_length)

    # TODO: this should likely be returned as well, and used as a mask for supervision.
    # Currently our loss includes rays outside of the bounding box.
    valid_mask = t_min < t_max

    return _RaySegmentSpecification(
        t_min=jnp.where(valid_mask, t_min, 0.0),
        t_max=jnp.where(valid_mask, t_max_clipped, min_segment_length),
    )
