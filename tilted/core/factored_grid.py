"""Core implementation for factored latent grids. This can be used to implement various
factorized latent grid architectures: CP or VM decompositions as per TensoRF,
tri-planes, k-Planes, HexPlane, etc.

There's a lot of duplicate code here that could be cleaned up!
"""

from __future__ import annotations

import functools
from typing import (
    Callable,
    Generic,
    Literal,
    NewType,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import jax
import jax.core
import jax_dataclasses as jdc
import jaxlie
from einops import einsum, rearrange, reduce
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import TypeVar, assert_never

LatentComponent = NewType("LatentComponent", Array)

ReduceFunction = Callable[
    [Tuple[Float[LatentComponent, "groups batch channels"], ...]],
    Float[Array, "batch final_channels"],
]


# Generic factored grid implementation.


class SceneContraction(Protocol):
    def __call__(self, coords: Float[Array, "... 3"]) -> Float[Array, "... 3"]:
        ...


def make_unbounded_scene_contraction(ord: Literal["2", "inf"]) -> SceneContraction:
    """Scene contraction for unbounded scenes. We should generally use l-infinity."""

    def inner(coords: Float[Array, "... 3"]) -> Float[Array, "... 3"]:
        # Contract all points to [-2, 2].
        norm = jnp.linalg.norm(
            coords, ord=2 if ord == "2" else jnp.inf, axis=-1, keepdims=True
        )
        contracted_points = jnp.where(
            norm <= 1.0, coords, (2.0 - 1.0 / norm) * coords / norm
        )
        assert isinstance(contracted_points, jax.Array)
        assert contracted_points.shape[-1] == 3

        # [-2, 2] => [-1, 1]
        return contracted_points / 2.0

    return inner


def make_scale_scene_contraction(scale: float) -> SceneContraction:
    """Simple scene contraction that scales a set of coordinates."""
    return lambda coords: coords * scale


def chain_contractions(*contractions: SceneContraction) -> SceneContraction:
    """Apply a sequence of scene contraction functions. Left-most arg is applied first."""
    return lambda coords: functools.reduce(
        lambda c, f: f(c), contractions, initial=coords
    )


class Projecters(Protocol):
    """A set of projections, which map input coordinates to factor coordinates, one per
    group per factor."""

    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        ...


@jdc.pytree_dataclass
class Factor:
    values: Float[Array, "groups *coords channels"]
    groups: jdc.Static[int]
    channels: jdc.Static[int]

    @staticmethod
    def make(
        groups: int,
        channels_per_group: int,
        spatial_dims: Tuple[int, ...],
        prng: jax.random.KeyArray,
        initializer: jax.nn.initializers.Initializer,
    ) -> Factor:
        return Factor(
            values=initializer(
                prng, shape=(groups,) + spatial_dims + (channels_per_group,)
            ),
            groups=groups,
            channels=channels_per_group,
        )

    def get_spatial_dims(self) -> Tuple[int, ...]:
        return self.values.shape[1:-1]

    def resize(self, new_spatial_dims: Tuple[int, ...]) -> Factor:
        assert len(new_spatial_dims) == len(self.get_spatial_dims())

        with jdc.copy_and_mutate(self, validate=False) as out:
            out.values = resize_with_aligned_corners(
                self.values,
                (self.groups, *new_spatial_dims, self.channels),
                method="linear",
                antialias=True,  # Only matters when downsampling.
            )
        return out

    def total_variation_cost(self, mode: Literal["l1", "l2"]) -> Float[Array, ""]:
        spatial_dims = self.values.shape[1:-1]

        terms = []

        slice_all = slice(None)
        slice_prev = slice(None, -1)
        slice_next = slice(1, None)

        for i in range(len(spatial_dims)):
            # Note offsets by 1 to account for group and channel axes.
            after_dims = len(spatial_dims) - i
            ind_prev = [slice_all] * (i + 1) + [slice_prev] + [slice_all] * after_dims
            ind_next = [slice_all] * (i + 1) + [slice_next] + [slice_all] * after_dims

            val_prev = self.values[tuple(ind_prev)]
            val_next = self.values[tuple(ind_next)]

            if mode == "l1":
                terms.append(jnp.mean(jnp.abs(val_next - val_prev)))
            elif mode == "l2":
                terms.append(jnp.mean((val_next - val_prev) ** 2))

        # Conv-based implementation. Much faster JIT (avoids a const folding warning),
        # but much slower at runtime.
        #
        # spatial_dims = self.values.shape[1:-1]
        # terms = []
        # finite_difference_filter = jnp.array([-1, 1])
        #
        # for i in range(len(spatial_dims)):
        #     compute_gradient = lambda v: jnp.convolve(
        #         v, finite_difference_filter, mode="valid"
        #     )
        #
        #     # Note that we add 1 to skip the group axis.
        #     for i in range(i + 1):
        #         compute_gradient = jax.vmap(compute_gradient, in_axes=0, out_axes=0)
        #
        #     after_dims = len(spatial_dims) - i
        #     for i in range(after_dims):
        #         compute_gradient = jax.vmap(compute_gradient, in_axes=-1, out_axes=-1)
        #
        #     diff = compute_gradient(self.values)
        #
        #     if mode == "l1":
        #         terms.append(jnp.mean(jnp.abs(diff)))
        #     elif mode == "l2":
        #         terms.append(jnp.mean(diff**2))
        #     else:
        #         assert_never(mode)

        return functools.reduce(jnp.add, terms) / len(terms)

    def interpolate(
        self, coords: Float[Array, "groups batch dim"]
    ) -> Float[LatentComponent, "groups batch channels"]:
        """Interpolate a latent component from some coordinates. Coordinates should be
        in the range of [-1, 1]."""
        assert coords.shape[-1] == len(self.values.shape) - 2

        def interp(
            grid: Float[Array, "batch dim"],
            coords: Float[Array, "batch dim"],
        ) -> Array:
            return jax.scipy.ndimage.map_coordinates(
                grid,
                coordinates=tuple(
                    (coords[..., i] + 1.0) / 2.0 * (grid.shape[i] - 1.0)
                    for i in range(coords.shape[-1])
                ),
                # Toroidal boundary conditions: this is grid-wrap in standard
                # scipy.ndimage.map_coordinates.
                #
                # TODO
                #
                # mode="wrap",
                order=1,
            )

        interp = jax.vmap(interp)  # Add group axis.
        interp = jax.vmap(interp, in_axes=(-1, None), out_axes=-1)  # Add channel axis.

        out = interp(self.values, coords)
        assert out.shape == (self.groups, *coords.shape[1:-1], self.channels)

        return LatentComponent(out)


ProjectersT = TypeVar("ProjectersT", bound=Projecters, default=Projecters)


@jdc.pytree_dataclass
class FactoredGrid(Generic[ProjectersT]):
    # Why don't we make this a Flax module?
    # -> Parameters don't go in dictionaries, so masked learning rates, regularization,
    # etc become easier.
    # -> Being type-safe is easier outside of Flax.
    # -> It's nice for this to be self-contained.
    factors: Tuple[Factor, ...]
    projecters: ProjectersT
    reduce_fn: jdc.Static[ReduceFunction]

    def total_groups(self) -> int:
        return sum(f.groups for f in self.factors)

    def interpolate(
        self, coords: Float[Array, "*batch dim"]
    ) -> Tuple[Float[Array, "*batch latent_dim"], Tuple[Array, ...]]:
        *batch, d = coords.shape
        coords = coords.reshape((-1, d))

        projected_coords = self.projecters.project(coords)
        assert len(projected_coords) == len(self.factors)

        # Interpolate a latent vector for each component.
        latent_components = []
        for component, proj_coords in zip(self.factors, projected_coords):
            latent_components.append(component.interpolate(proj_coords))

        # Reduce components.
        out = self.reduce_fn(tuple(latent_components))
        out = out.reshape((*batch, out.shape[-1]))
        return out, tuple(
            comp.reshape((comp.shape[0], *batch, comp.shape[-1]))
            for comp in latent_components
        )

    def l12_cost(self) -> Float[Array, ""]:
        return jnp.mean(
            jnp.array(
                [
                    jnp.sqrt(
                        jnp.mean(
                            factor.values.reshape((factor.groups, -1)) ** 2, axis=1
                        )
                    )
                    for factor in self.factors
                ]
            )
        )

    def total_variation_cost(self, mode: Literal["l1", "l2"]) -> Float[Array, ""]:
        return functools.reduce(
            jnp.add, [factor.total_variation_cost(mode) for factor in self.factors]
        ) / len(self.factors)


@jdc.pytree_dataclass
class Learnable2dProjecters:
    tau: jaxlie.SO2
    """Learnable transform. We should have one transform per 2 groups."""

    transforms_per_res: jdc.Static[int]
    res_count: jdc.Static[int]
    orthogonal_uv: jdc.Static[bool]

    @staticmethod
    def make(
        prng: jax.random.KeyArray,
        transforms_per_res: int,
        res_count: int,
        tau_init: Literal["random", "zeros"],
        orthogonal_uv: bool,
    ) -> Learnable2dProjecters:
        if tau_init == "random":
            tau = jax.vmap(jaxlie.SO2.sample_uniform)(
                jax.random.split(prng, num=transforms_per_res * res_count)
            )
        elif tau_init == "zeros":
            tau = jax.vmap(jaxlie.SO2.from_radians)(
                jnp.zeros(transforms_per_res * res_count)
            )

        if not orthogonal_uv:
            # If we are not enforcing orthogonality for u/v pairs, we initialize pairs
            # to be orthogonal.
            with jdc.copy_and_mutate(tau) as tau:
                tau.unit_complex = tau.unit_complex.at[1::2, :].set(
                    tau.unit_complex[::2, :]
                    @ jaxlie.SO2.from_radians(jnp.pi / 2.0).as_matrix()
                )

        return Learnable2dProjecters(
            tau,
            transforms_per_res=transforms_per_res,
            res_count=res_count,
            orthogonal_uv=orthogonal_uv,
        )

    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 2

        if self.orthogonal_uv:
            R = jax.vmap(jaxlie.SO2.as_matrix)(self.tau)
            assert R.shape == (self.transforms_per_res * self.res_count, 2, 2)
            out = einsum(R, coords, "pair_groups i j, batch j -> pair_groups i batch")
            out = rearrange(
                out,
                "pair_groups uv batch -> (pair_groups uv) batch 1",
                pair_groups=self.transforms_per_res * self.res_count,
                uv=2,
                batch=batch,
            )
            assert out.shape == (self.transforms_per_res * self.res_count * 2, batch, 1)
        else:
            # Switching to jnp.eisum because einops was giving a singleton axes error.
            out = jnp.einsum("ti,bi->tb", self.tau.unit_complex, coords)[:, :, None]
            assert out.shape == (self.transforms_per_res * self.res_count, batch, 1)

        return tuple(jnp.split(out, self.res_count, axis=0))


def make_2d_grid(
    prng: jax.random.KeyArray,
    output_channels: int,
    transforms_per_res: int,
    resolutions: Tuple[int, ...],
    tau_init: Literal["random", "zeros"],
    orthogonal_uv: bool,
) -> FactoredGrid[Learnable2dProjecters]:
    assert output_channels % (transforms_per_res * len(resolutions)) == 0
    if orthogonal_uv:
        # Each group is a separate u or v vector.
        # In this case, we have a single transformation for every pair.
        channels_per_group = output_channels // (transforms_per_res * len(resolutions))
        groups = 2 * transforms_per_res
    else:
        # Each group is a separate u or v vector.
        # In this case, we have a separate transformation for every vector.
        channels_per_group = (
            2 * output_channels // (transforms_per_res * len(resolutions))
        )
        groups = transforms_per_res
        assert groups % 2 == 0

    def reduce_fn(components: Tuple[LatentComponent, ...]) -> Array:
        out_parts = []
        for comp in components:
            batch = comp.shape[1]
            assert comp.shape == (groups, batch, channels_per_group)
            comp = rearrange(
                comp,
                "(pairs uv) b c -> pairs uv b c",
                pairs=groups // 2,
                uv=2,
                b=batch,
                c=channels_per_group,
            )

            # Take the product of uv pairs.
            comp = reduce(comp, "pairs uv b c -> pairs b c", "prod")

            # Concatenate and return.
            comp = rearrange(comp, "pairs b c -> b (c pairs)")
            out_parts.append(comp)

        out = jnp.concatenate(out_parts, axis=-1)
        assert out.shape[-1] == output_channels
        return out

    prng, prng_proj = jax.random.split(prng)
    prngs_factor = jax.random.split(prng, num=len(resolutions))
    return FactoredGrid(
        factors=tuple(
            Factor.make(
                groups=groups,
                channels_per_group=channels_per_group,
                spatial_dims=(res,),
                prng=prngs_factor[i],
                initializer=jax.nn.initializers.normal(1.0),
            )
            for i, res in enumerate(resolutions)
        ),
        projecters=Learnable2dProjecters.make(
            prng_proj,
            transforms_per_res=transforms_per_res,
            res_count=len(resolutions),
            tau_init=tau_init,
            orthogonal_uv=orthogonal_uv,
        ),
        reduce_fn=reduce_fn,
    )


# 3D grid implementations.


def make_3d_grid(
    prng: jax.random.KeyArray,
    output_channels: int,
    resolutions: Tuple[int, ...],
    grid_type: Literal["kplane", "kplane_add", "vm", "cp"],
    transform_count: Optional[int],
    tau_init: Literal["random", "zeros"],
    scene_contract: SceneContraction,
) -> FactoredGrid:
    out = {
        "kplane": functools.partial(_make_kplane_grid, reduce="multiply"),
        "kplane_add": functools.partial(_make_kplane_grid, reduce="add"),
        "vm": _make_vm_grid,
        "cp": _make_cp_grid,
    }[grid_type](
        prng, output_channels, resolutions, transform_count, tau_init, scene_contract
    )

    return out


T = TypeVar("T")


@jdc.pytree_dataclass
class Learnable3DProjectersBase:
    res_count: jdc.Static[int]
    """# of resolutions."""

    transform_count: jdc.Static[int]
    tau: jaxlie.SO3
    scene_contract: jdc.Static[SceneContraction]

    @classmethod
    def make(
        cls: Type[T],
        prng: jax.random.KeyArray,
        res_count: int,
        transform_count: int,
        tau_init: Literal["random", "zeros"],
        scene_contract: SceneContraction,
    ) -> T:
        if tau_init == "random":
            tau = jax.vmap(jaxlie.SO3.sample_uniform)(
                jax.random.split(prng, transform_count)
            )
        elif tau_init == "zeros":
            tau = jax.vmap(jaxlie.SO3.exp)(jnp.zeros((transform_count, 3)))
        return cls(
            res_count=res_count,
            transform_count=transform_count,
            tau=tau,
            scene_contract=scene_contract,
        )


@jdc.pytree_dataclass
class LearnableKPlanesProjecters(Learnable3DProjectersBase):
    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3

        R = jax.vmap(jaxlie.SO3.as_matrix)(self.tau)
        transformed_coords = einsum(
            R, coords, "transform i j, batch j -> transform batch i"
        )
        transformed_coords = self.scene_contract(transformed_coords)

        ij = transformed_coords[:, :, :2]
        ki = transformed_coords[:, :, jnp.array([2, 0])]
        jk = transformed_coords[:, :, jnp.array([1, 2])]
        matrix_coords = jnp.concatenate([ij, ki, jk], axis=0)

        assert matrix_coords.shape == (3 * self.transform_count, batch, 2)
        return (matrix_coords,) * self.res_count


@jdc.pytree_dataclass
class KPlaneProjecters:
    res_count: jdc.Static[int]
    """# of resolutions."""
    scene_contract: jdc.Static[SceneContraction]

    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3
        coords = self.scene_contract(coords)
        ij = coords[:, :2]
        ki = coords[:, jnp.array([2, 0])]
        jk = coords[:, jnp.array([1, 2])]
        matrix_coords = jnp.stack([ij, ki, jk], axis=0)
        assert matrix_coords.shape == (3, batch, 2)
        return (matrix_coords,) * self.res_count


def _make_kplane_grid(
    prng: jax.random.KeyArray,
    output_channels: int,
    resolutions: Tuple[int, ...],
    transform_count: Optional[int],
    tau_init: Literal["random", "zeros"],
    scene_contract: SceneContraction,
    reduce: Literal["multiply", "add"],
) -> FactoredGrid:
    """Construct a latent grid as specified in k-Planes. This is a tri-plane
    architecture with terms multiplied at the end instead of summed."""

    def reduce_fn(components: Tuple[LatentComponent, ...]) -> Array:
        # Collect latent vectors at each resolution.
        latents = []
        for comp in components:
            groups = comp.shape[0]
            assert groups % 3 == 0

            p0, p1, p2 = jnp.array_split(comp, 3, axis=0)
            latents.append(
                rearrange(
                    p0 * p1 * p2 if reduce == "multiply" else p0 + p1 + p2,
                    "plane_groups b c -> b (plane_groups c)",
                    plane_groups=groups // 3,
                )
            )

        out = jnp.concatenate(latents, axis=-1)
        assert out.shape[-1] == output_channels
        return out

    no_transform = False
    if transform_count is None:
        transform_count = 1
        no_transform = True

    assert output_channels % (len(resolutions) * transform_count) == 0
    channels_per_group = output_channels // len(resolutions) // transform_count

    projecter_prng, factor_prng = jax.random.split(prng, num=2)

    def weight_init(key, shape, dtype=jnp.float32) -> Array:
        if reduce == "multiply":
            # As per k-Planes paper.
            return 0.1 + 0.4 * jax.random.uniform(key, shape, dtype)
        else:
            # Same as TensoRF.
            return 0.1 * jax.random.normal(key, shape, dtype)

    return FactoredGrid(
        factors=tuple(
            # Matrix components.
            Factor.make(
                groups=3 * transform_count,
                channels_per_group=channels_per_group,
                spatial_dims=(res, res),
                prng=jax.random.fold_in(factor_prng, i),
                # initializer=jax.nn.initializers.normal(1.0),
                initializer=weight_init,
                # def init(key: KeyArray,
                #          shape: core.Shape,
                #          dtype: DTypeLikeInexact = dtype) -> Array:
                #   dtype = dtypes.canonicalize_dtype(dtype)
                #   return random.uniform(key, shape, dtype) * scale
            )
            for i, res in enumerate(resolutions)
        ),
        projecters=KPlaneProjecters(
            res_count=len(resolutions), scene_contract=scene_contract
        )
        if no_transform
        else LearnableKPlanesProjecters.make(
            # projecters=else LearnableKPlanesProjecters.make(
            prng=projecter_prng,
            res_count=len(resolutions),
            transform_count=transform_count,
            tau_init=tau_init,
            scene_contract=scene_contract,
        ),
        reduce_fn=reduce_fn,
    )


@jdc.pytree_dataclass
class LearnableVmProjecters(Learnable3DProjectersBase):
    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3

        R = jax.vmap(jaxlie.SO3.as_matrix)(self.tau)
        transformed_coords = einsum(
            R, coords, "transform i j, batch j -> transform batch i"
        )
        transformed_coords = self.scene_contract(transformed_coords)

        ij = transformed_coords[:, :, :2]
        ki = transformed_coords[:, :, jnp.array([2, 0])]
        jk = transformed_coords[:, :, jnp.array([1, 2])]
        matrix_coords = jnp.concatenate([ij, ki, jk], axis=0)
        vector_coords = rearrange(
            transformed_coords, "transform batch i -> (transform i) batch 1"
        )

        assert matrix_coords.shape == (3 * self.transform_count, batch, 2)
        assert vector_coords.shape == (3 * self.transform_count, batch, 1)
        return (matrix_coords, vector_coords) * self.res_count


@jdc.pytree_dataclass
class VmProjecters:
    res_count: jdc.Static[int]
    """# of resolutions."""
    scene_contract: jdc.Static[SceneContraction]

    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3
        coords = self.scene_contract(coords)
        ij = coords[:, :2]
        ki = coords[:, jnp.array([2, 0])]
        jk = coords[:, jnp.array([1, 2])]
        matrix_coords = jnp.stack([ij, ki, jk], axis=0)
        vector_coords = jnp.moveaxis(coords, -1, 0)[:, :, None]
        assert matrix_coords.shape == (3, batch, 2)
        assert vector_coords.shape == (3, batch, 1)
        return (matrix_coords, vector_coords) * self.res_count


def _make_vm_grid(
    prng: jax.random.KeyArray,
    output_channels: int,
    resolutions: Tuple[int, ...],
    transform_count: Optional[int],
    tau_init: Literal["random", "zeros"],
    scene_contract: SceneContraction,
) -> FactoredGrid:
    """Construct a VM-decomposed latent grid, a la TensoRF."""

    def reduce_fn(components: Tuple[LatentComponent, ...]) -> Array:
        # Collect latent vectors at each resolution.
        assert len(components) % 2 == 0
        latents = []
        for i in range(0, len(components), 2):
            matrix = components[i]
            vector = components[i + 1]

            latents.append(
                rearrange(
                    matrix * vector,
                    "vm_groups b c -> b (vm_groups c)",
                )
            )

        out = jnp.concatenate(latents, axis=-1)
        assert out.shape[-1] == output_channels
        return out

    no_transform = False
    if transform_count is None:
        transform_count = 1
        no_transform = True

    assert output_channels % (3 * transform_count * len(resolutions)) == 0
    channels_per_group = output_channels // 3 // len(resolutions) // transform_count
    factors = []
    for res in resolutions:
        prng, prng0, prng1 = jax.random.split(prng, num=3)
        factors.extend(
            [
                # Matrix components.
                Factor.make(
                    groups=3 * transform_count,
                    channels_per_group=channels_per_group,
                    spatial_dims=(res, res),
                    prng=prng0,
                    initializer=jax.nn.initializers.normal(0.1),
                ),
                # Vector components.
                Factor.make(
                    groups=3 * transform_count,
                    channels_per_group=channels_per_group,
                    spatial_dims=(res,),
                    prng=prng1,
                    initializer=jax.nn.initializers.normal(0.1),
                ),
            ]
        )

    return FactoredGrid(
        factors=tuple(factors),
        projecters=VmProjecters(
            res_count=len(resolutions), scene_contract=scene_contract
        )
        if no_transform
        else LearnableVmProjecters.make(
            prng,
            res_count=len(resolutions),
            transform_count=transform_count,
            tau_init=tau_init,
            scene_contract=scene_contract,
        ),
        reduce_fn=reduce_fn,
    )


@jdc.pytree_dataclass
class LearnableCpProjecters(Learnable3DProjectersBase):
    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3

        R = jax.vmap(jaxlie.SO3.as_matrix)(self.tau)
        transformed_coords = einsum(
            R, coords, "transform i j, batch j -> transform batch i"
        )
        transformed_coords = self.scene_contract(transformed_coords)
        vector_coords = rearrange(
            transformed_coords, "transform batch i -> (i transform) batch 1"
        )
        assert vector_coords.shape == (3 * self.transform_count, batch, 1)
        return (vector_coords,) * self.res_count


@jdc.pytree_dataclass
class CpProjecters:
    res_count: jdc.Static[int]
    """# of resolutions."""
    scene_contract: jdc.Static[SceneContraction]

    def project(
        self, coords: Float[Array, "batch dim"]
    ) -> Tuple[Float[Array, "groups batch out_dim"], ...]:
        batch, dim = coords.shape
        assert dim == 3
        coords = self.scene_contract(coords)

        vector_coords = jnp.moveaxis(coords, -1, 0)[:, :, None]
        assert vector_coords.shape == (3, batch, 1)
        return (vector_coords,) * self.res_count


def _make_cp_grid(
    prng: jax.random.KeyArray,
    output_channels: int,
    resolutions: Tuple[int, ...],
    transform_count: Optional[int],
    tau_init: Literal["random", "zeros"],
    scene_contract: SceneContraction,
) -> FactoredGrid:
    """Construct a CP-decomposed latent grid."""

    def reduce_fn(components: Tuple[LatentComponent, ...]) -> Array:
        # Collect latent vectors at each resolution.
        latents = []
        for comp in components:
            p0, p1, p2 = jnp.array_split(comp, 3, axis=0)
            latents.append(
                rearrange(
                    p0 * p1 * p2,
                    "vector_groups b c -> b (vector_groups c)",
                )
            )

        out = jnp.concatenate(latents, axis=-1)
        assert out.shape[-1] == output_channels
        return out

    no_transform = False
    if transform_count is None:
        transform_count = 1
        no_transform = True

    assert output_channels % (len(resolutions) * transform_count) == 0
    channels_per_group = output_channels // len(resolutions) // transform_count
    factors = []
    for res in resolutions:
        prng, prng0 = jax.random.split(prng, num=2)
        factors.extend(
            [
                # Vector component.
                Factor.make(
                    groups=3 * transform_count,
                    channels_per_group=channels_per_group,
                    spatial_dims=(res,),
                    prng=prng0,
                    initializer=jax.nn.initializers.normal(0.1),
                ),
            ]
        )

    return FactoredGrid(
        factors=tuple(factors),
        # TODO sketchy
        projecters=CpProjecters(
            res_count=len(resolutions), scene_contract=scene_contract
        )
        if no_transform
        else LearnableCpProjecters.make(
            prng=prng,
            res_count=len(resolutions),
            transform_count=transform_count,
            tau_init=tau_init,
            scene_contract=scene_contract,
        ),
        reduce_fn=reduce_fn,
    )


def resize_with_aligned_corners(
    image: jax.Array,
    shape: Tuple[int, ...],
    method: Union[str, jax.image.ResizeMethod],
    antialias: bool,
):
    """Alternative to jax.image.resize(), which emulates align_corners=True in PyTorch's
    interpolation functions."""
    spatial_dims = tuple(
        i
        for i in range(len(shape))
        if not jax.core.symbolic_equal_dim(image.shape[i], shape[i])
    )
    scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])
    translation = -(scale / 2.0 - 0.5)
    return jax.image.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )
