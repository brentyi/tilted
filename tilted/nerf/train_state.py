from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, overload

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import optax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from typing_extensions import assert_never

from ..core.factored_grid import (
    make_3d_grid,
    make_scale_scene_contraction,
    make_unbounded_scene_contraction,
)
from . import cameras, data, render
from .decoder import NerfDecoderMlp, ShallowDensityDecoder


@dataclasses.dataclass(frozen=True)
class OptimizationConfig:
    half_resolution_steps: int = 0
    low_pass_steps: int = 512
    """Steps to anneal low-pass alpha parameter; see BARF, Nerfies, etc."""
    minibatch_size: int = 4096
    warmup_steps: int = 512
    total_steps: int = 30_000

    factor_lr: float = 0.01
    projection_lr: float = 0.01
    decoder_lr: float = 0.01
    camera_delta_lr: float = 0.0
    """Learning rate for camera pose refinement."""

    l12_reg_coeff: float = 0.0000
    tv_reg_l1_coeff: float = 0.00
    tv_reg_l2_coeff: float = 0.01

    interlevel_loss_coeff: float = 1.0
    distortion_loss_coeff: float = 0.001

    projection_decay_start: int = 500
    projection_decay_steps: int = 4500
    camera_delta_steps: int = 5_000

    proposal_anneal_iters: int = 1_000
    proposal_anneal_slope: float = 10.0


@dataclasses.dataclass(frozen=True)
class FieldConfig:
    # Factorization to use for grids.
    grid_type: Literal["cp", "vm", "kplane", "kplane_add"] = "kplane"

    # Decoders used for density and primary fields.
    density_decoder: ShallowDensityDecoder = jdc.field(
        default_factory=ShallowDensityDecoder
    )
    primary_decoder: NerfDecoderMlp = jdc.field(
        default_factory=lambda: NerfDecoderMlp()
    )

    # Grid dimensions.
    primary_resolutions: Tuple[int, ...] = (64, 128, 256, 512)
    primary_channels: int = 32

    # Density field parameters for proposal sampling.
    proposal_resolutions: Tuple[int, ...] = (64, 128)
    proposal_channels: Tuple[int, ...] = (8, 8)


@dataclasses.dataclass(frozen=True)
class NerfConfig:
    # Input data directory.
    dataset_path: Path
    dataset_type: Literal["blender", "nerfstudio"]

    # Per-dataset grid bound for synthetic scenes.
    # General guide: 1.3 for Blender, but 1.5 for ship.
    grid_bound: float = 1.0

    # Norm function to use for scene contraction, for real scenes.
    unbounded_contraction_norm: Literal["2", "inf"] = "inf"

    # Optimization configuration.
    optim: OptimizationConfig = OptimizationConfig()

    # Field resolution config.
    field: FieldConfig = FieldConfig()

    # TILTED grid transforms.
    proposal_transform_count: Optional[int] = None
    primary_transform_count: Optional[int] = None
    tau_init: Literal[
        "random", "zeros"
    ] = "random"  # Only active if transform counts are >0.

    # Default rendering configuration. Used for training.
    render_config: render.RenderConfig = render.RenderConfig(
        proposal_samples=(256, 128),
        final_samples=48,
        background="white",
        initial_samples="linear",
        near=0.0,
        far=300.0,
    )

    # Rendering overrides for evaluation.
    eval_proposal_samples: Tuple[int, ...] = (512, 256)
    eval_final_samples: int = 96

    # Render at some frequency. Helps with debugging.
    debug_render_interval: Optional[int] = None

    seed: int = 0

    def get_dataset(
        self,
        split: Literal["train", "test", "val"],
    ) -> data.NerfDataset:
        return data.make_dataset(self.dataset_type, split, self.dataset_path)


@jdc.pytree_dataclass
class TrainState:
    config: jdc.Static[NerfConfig]
    params: render.LearnableParams
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState
    prng: jax.random.KeyArray
    step: Int[Array, ""]

    @staticmethod
    def make(
        config: NerfConfig,
    ) -> TrainState:
        prng = jax.random.PRNGKey(config.seed)

        prng, prng_ = jax.random.split(prng)

        # Initialize density/proposal networks.
        assert len(config.field.proposal_resolutions) == len(
            config.field.proposal_channels
        )

        assert config.optim.half_resolution_steps >= 0
        res_div = 2 if config.optim.half_resolution_steps != 0 else 1

        density_fields: List[render.HybridField] = []

        if config.dataset_type == "blender":
            scene_contract = make_scale_scene_contraction(scale=1.0 / config.grid_bound)
        elif config.dataset_type == "nerfstudio":
            assert config.grid_bound == 1.0
            scene_contract = make_unbounded_scene_contraction(
                config.unbounded_contraction_norm
            )
        else:
            assert_never(config.dataset_type)

        for res, chan in zip(
            config.field.proposal_resolutions, config.field.proposal_channels
        ):
            # Make density grid.
            prng_, prng = jax.random.split(prng)
            grid = make_3d_grid(
                prng=prng_,
                output_channels=chan,
                resolutions=(res // res_div,),
                grid_type=config.field.grid_type,
                transform_count=config.proposal_transform_count,
                tau_init=config.tau_init,
                scene_contract=scene_contract,
            )

            # Make density decoder..
            prng_, prng = jax.random.split(prng)
            decoder_params = config.field.density_decoder.init(
                prng_, jnp.zeros((1, chan))
            )

            # Create proposal field.
            density_fields.append(
                render.HybridField(
                    grid=grid,
                    decoder=config.field.density_decoder,
                    decoder_params=decoder_params,  # type: ignore
                )
            )

        # Make primary grid.
        prng_, prng = jax.random.split(prng)
        primary_grid = make_3d_grid(
            prng=prng_,
            output_channels=config.field.primary_channels,
            resolutions=tuple(
                res // res_div for res in config.field.primary_resolutions
            ),
            grid_type=config.field.grid_type,
            transform_count=config.primary_transform_count,
            tau_init=config.tau_init,
            scene_contract=scene_contract,
        )
        prng_, prng = jax.random.split(prng)
        num_cameras = len(config.get_dataset("train").get_cameras())
        primary_decoder_params = config.field.primary_decoder.init(
            prng_,
            features=jnp.zeros((1, config.field.primary_channels)),
            viewdirs=jnp.zeros((1, 3)),
            camera_indices=jnp.zeros((1,), dtype=jnp.int32),
            num_cameras=num_cameras,
            low_pass_alpha=None,
        )
        primary_field = render.HybridField(
            primary_grid,
            config.field.primary_decoder,
            primary_decoder_params,  # type: ignore
        )

        params = render.LearnableParams(
            # proposal_fields=tuple(proposal_fields),
            density_fields=tuple(density_fields),
            primary_field=primary_field,
            num_cameras=num_cameras,
            camera_deltas=None
            if config.optim.camera_delta_lr == 0.0
            else (
                jax.vmap(jaxlie.SO3.exp)(
                    jnp.zeros((num_cameras, 3)),
                ),
                jnp.zeros((num_cameras, 3)),
            ),
        )

        # Initialize optimizer, optimizer state.
        optimizer = optax.chain(
            optax.scale_by_adam(),
            # Factor learning rate.
            optax.masked(
                optax.scale(-config.optim.factor_lr),
                params.make_optax_mask("factors"),
            ),
            # Factor learning rate.
            optax.masked(
                optax.scale_by_schedule(
                    optax.cosine_decay_schedule(
                        init_value=-config.optim.camera_delta_lr,
                        decay_steps=int(config.optim.camera_delta_steps),
                    )
                ),
                params.make_optax_mask("camera_deltas"),
            ),
            # Factor learning rate.
            optax.masked(
                optax.scale_by_schedule(
                    optax.linear_schedule(
                        init_value=-config.optim.projection_lr,
                        end_value=0.0,
                        transition_steps=int(config.optim.projection_decay_steps),
                        transition_begin=int(config.optim.projection_decay_start),
                    )
                ),
                params.make_optax_mask("projections"),
            ),
            # Network learning rate.
            optax.masked(
                optax.scale(-config.optim.decoder_lr),
                params.make_optax_mask("decoders"),
            ),
            # Overall learning rate decay.
            optax.scale_by_schedule(
                optax.warmup_cosine_decay_schedule(
                    init_value=0.01,
                    peak_value=1.0,
                    warmup_steps=config.optim.warmup_steps,
                    decay_steps=config.optim.total_steps,
                    end_value=0.0001,
                )
            ),
        )
        optimizer_state = optimizer.init(jaxlie.manifold.zero_tangents(params))

        return TrainState(
            config=config,
            params=params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            prng=prng,
            step=jnp.array(0),
        )

    # Rendering helpers.

    @overload
    def render_camera(
        self,
        camera: cameras.Camera,
        camera_index: int,
        eval_mode: bool,
        mode: Literal["rgb", "dist"],
        chunk_size: int = 16384,
    ) -> onp.ndarray:
        ...

    @overload
    def render_camera(
        self,
        camera: cameras.Camera,
        camera_index: int,
        eval_mode: bool,
        mode: Literal["all"],
        chunk_size: int = 16384,
    ) -> render.RenderOutputs:
        ...

    def render_camera(
        self,
        camera: cameras.Camera,
        camera_index: int,
        eval_mode: bool,
        mode: Literal["rgb", "dist", "all"],
        chunk_size: int = 16384,
    ) -> Union[render.RenderOutputs, onp.ndarray]:
        """Render an image from a camera. Should not be JITed due to chunking."""
        return self.render_rays_chunked(
            camera.pixel_rays_wrt_world(camera_index),
            eval_mode=eval_mode,
            mode=mode,
            prng=self.prng,
            chunk_size=chunk_size,
        )

    @overload
    def render_rays_chunked(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: bool,
        mode: Literal["rgb", "dist"],
        prng: jax.random.KeyArray,
        chunk_size: int = 16384,
    ) -> onp.ndarray:
        ...

    @overload
    def render_rays_chunked(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: bool,
        mode: Literal["all"],
        prng: jax.random.KeyArray,
        chunk_size: int = 16384,
    ) -> render.RenderOutputs:
        ...

    def render_rays_chunked(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: bool,
        mode: Literal["rgb", "dist", "all"],
        prng: jax.random.KeyArray,
        chunk_size: int = 16384,
    ) -> Union[render.RenderOutputs, onp.ndarray]:
        """Split a batch of rays into chunks, loop to render each chunk, and return the
        concatenated result. Should not be JITed."""
        batch_axes = rays_wrt_world.get_batch_axes()
        rays_wrt_world = (
            cameras.Rays3D(  # TODO: feels like this could be done less manually!
                origins=rays_wrt_world.origins.reshape((-1, 3)),
                directions=rays_wrt_world.directions.reshape((-1, 3)),
                camera_indices=rays_wrt_world.camera_indices.reshape((-1,)),
            )
        )
        (total_rays,) = rays_wrt_world.get_batch_axes()
        out_parts = []
        for start in range(0, total_rays, chunk_size):
            end = min(start + chunk_size, total_rays)
            chunk = jax.tree_map(lambda x: x[start:end], rays_wrt_world)
            out_parts.append(
                jax.tree_map(
                    onp.array,
                    self.render_rays(chunk, mode=mode, eval_mode=eval_mode, prng=prng),
                )
            )

        out = jax.tree_map(
            lambda *p: onp.concatenate(p, axis=0) if p[0].shape != () else onp.mean(p),
            *out_parts,
        )

        def _reshape(arr: onp.ndarray) -> onp.ndarray:
            if arr.shape == ():
                return arr
            assert arr.shape[0] == total_rays
            return arr.reshape((*batch_axes, *arr.shape[1:]))

        return jax.tree_map(_reshape, out)

    @overload
    def render_rays(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: jdc.Static[bool],
        mode: jdc.Static[Literal["rgb", "dist", "transform_feature_norm", "features"]],
        prng: Optional[jax.random.KeyArray] = None,
    ) -> Array:
        ...

    @overload
    def render_rays(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: jdc.Static[bool],
        mode: jdc.Static[Literal["all"]],
        prng: Optional[jax.random.KeyArray] = None,
    ) -> render.RenderOutputs:
        ...

    @jdc.jit
    def render_rays(
        self,
        rays_wrt_world: cameras.Rays3D,
        eval_mode: jdc.Static[bool],
        mode: jdc.Static[
            Literal["rgb", "dist", "all", "transform_feature_norm", "features"]
        ],
        prng: Optional[jax.random.KeyArray] = None,
    ) -> Union[render.RenderOutputs, Array]:
        """Render a batch of rays."""
        render_config = self.config.render_config

        if prng is None:
            prng = self.prng

        if eval_mode:
            render_config = dataclasses.replace(
                render_config,
                proposal_samples=self.config.eval_proposal_samples,
                final_samples=self.config.eval_final_samples,
            )

        render_outs = render.render_rays(
            rays_wrt_world,
            params=self.params,
            config=render_config,
            prng=prng,
            anneal_factor=self.get_anneal_factor(),
            low_pass_alpha=self.get_low_pass_alpha(),
        )
        if mode == "rgb":
            return render_outs.rgb
        elif mode == "dist":
            return render_outs.dist
        elif mode == "all":
            return render_outs
        elif mode == "transform_feature_norm":
            return render_outs.transform_feature_norm
        elif mode == "features":
            return render_outs.features
        else:
            assert_never(mode)

    @jdc.jit
    def resize_to_final_size(self) -> TrainState:
        """For coarse-to-fine / to help speed up training. Does not seem to do much, could likely be removed."""

        def resize_params(
            params: render.LearnableParams,
        ) -> render.LearnableParams:
            with jdc.copy_and_mutate(params, validate=False) as params:
                for i in range(len(params.density_fields)):
                    target_res = self.config.field.proposal_resolutions[i]
                    factors = params.density_fields[i].grid.factors
                    new_factors = []
                    for j in range(len(factors)):
                        current_spatial_dims = factors[j].get_spatial_dims()
                        assert all(
                            dim == target_res // 2 for dim in current_spatial_dims
                        )
                        new_factors.append(
                            factors[j].resize((target_res,) * len(current_spatial_dims))
                        )
                    params.density_fields[i].grid.factors = tuple(new_factors)

                factors = params.primary_field.grid.factors
                new_factors = []
                res_map = {
                    res // 2: res for res in self.config.field.primary_resolutions
                }
                for f in factors:
                    current_spatial_dims = f.get_spatial_dims()
                    new_factors.append(
                        f.resize(tuple(res_map[r] for r in current_spatial_dims))
                    )
                params.primary_field.grid.factors = tuple(new_factors)
            return params

        with jdc.copy_and_mutate(self, validate=False) as out:
            # Resize parameters.
            out.params = resize_params(out.params)

            # Resize ADAM moments.
            assert isinstance(out.optimizer_state, tuple)
            adam_state = out.optimizer_state[0]
            assert isinstance(adam_state, optax.ScaleByAdamState)
            nu = adam_state.nu
            mu = adam_state.mu
            assert isinstance(nu, render.LearnableParams)
            assert isinstance(mu, render.LearnableParams)
            out.optimizer_state = (
                adam_state._replace(
                    nu=resize_params(nu), mu=resize_params(mu)
                ),  # NamedTuple `_replace()`.
            ) + out.optimizer_state[1:]

        return out

    def get_anneal_factor(self) -> Float[Array, ""]:
        step = self.step

        # anneal the weights of the proposal network before doing PDF sampling
        N = self.config.optim.proposal_anneal_iters
        # https://arxiv.org/pdf/2111.12077.pdf eq. 18
        train_frac = jnp.clip(step / N, 0, 1)

        def bias(x, b):
            return b * x / ((b - 1) * x + 1)

        anneal = bias(train_frac, self.config.optim.proposal_anneal_slope)
        return anneal

    def get_low_pass_alpha(self) -> Optional[Float[Array, ""]]:
        """Get alpha constant a la BARF, Nerfies, etc. Should be in the range of [0,
        num_freqs]."""
        if self.config.optim.low_pass_steps is None:
            return None

        # Clipping here has no functional benefit, but is nice for being rigorous with
        # the paper definition.
        return jnp.minimum(
            self.step
            / self.config.optim.low_pass_steps
            * self.config.field.primary_decoder.feature_n_freqs,
            self.config.field.primary_decoder.feature_n_freqs,
        )
