"""Proposal loss implementation.

This is copied almost exactly from multinerf, which is released under an Apache license.

https://github.com/google-research/multinerf
"""


import jax
from jax import numpy as jnp


def interlevel_loss(ray_history) -> jax.Array:
    """Computes the interlevel loss defined in mip-NeRF 360."""
    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    last_ray_results = ray_history[-1]
    c = jax.lax.stop_gradient(last_ray_results["sdist"])
    w = jax.lax.stop_gradient(last_ray_results["weights"])
    loss_interlevel = 0.0
    for ray_results in ray_history[:-1]:
        cp = ray_results["sdist"]
        wp = ray_results["weights"]
        loss_interlevel += jnp.mean(lossfun_outer(c, w, cp, wp))
    assert not isinstance(loss_interlevel, float)
    return loss_interlevel


def searchsorted(a, v):
    """Find indices where v should be inserted into a to maintain order.
    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.
    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.
    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = jnp.arange(a.shape[-1])
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = jnp.max(jnp.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)
    idx_hi = jnp.min(jnp.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)
    return idx_lo, idx_hi


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = jnp.concatenate(
        [jnp.zeros_like(y1[..., :1]), jnp.cumsum(y1, axis=-1)], axis=-1
    )
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = jnp.take_along_axis(cy1, idx_lo, axis=-1)
    cy1_hi = jnp.take_along_axis(cy1, idx_hi, axis=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
    y0_inner = jnp.where(
        idx_hi[..., :-1] <= idx_lo[..., 1:], cy1_lo[..., 1:] - cy1_hi[..., :-1], 0
    )
    return y0_inner, y0_outer


def lossfun_outer(t, w, t_env, w_env, eps=jnp.finfo(jnp.float32).eps):
    """The proposal weight should be an upper envelope on the nerf weight."""
    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return jnp.maximum(0, w - w_outer) ** 2 / (w + eps)
