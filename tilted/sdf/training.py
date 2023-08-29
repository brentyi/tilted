from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, cast

import fifteen
import flax
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import optax
from einops import rearrange
from fifteen.experiments import TensorboardLogData
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..core.decoder import DecoderMlp
from ..core.factored_grid import FactoredGrid, make_3d_grid
from . import meshing


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    exp_name: Optional[str] = None
    mesh_path: Path = Path("./data/meshes/armadillo.hdf5")
    mesh_rotate_seed: Optional[int] = 0

    # Grid parameters.
    grid_type: Literal["vm", "kplane", "cp"] = "kplane"
    grid_output_channels: int = 30
    grid_resolutions: Tuple[int, ...] = (32, 128)

    transform_count: int = 0
    tau_init: Literal["random", "zeros"] = "random"  # Only active if tranform count >0.

    # Learning rates.
    factor_lr: float = 0.005
    projection_lr: float = 0.01
    decoder_lr: float = 0.002

    # Optimizer settings.
    train_steps: int = 3_000
    minibatch_size: int = 2**18
    l12_reg_coeff: float = 0.001
    tv_reg_coeff: float = 0.001

    # Steps to completion for BARF-style coarse-to-fine schedule. Set to None to disable
    # BARF.
    barf_steps: Optional[int] = 500

    decoder_mlp: DecoderMlp = dataclasses.field(
        default_factory=lambda: DecoderMlp(
            output_sigmoid=False,
            output_dim=1,
            units=64,
            layers=3,
            pos_enc_freqs=4,
        )
    )
    loss: Literal["l1", "l2", "mape"] = "mape"
    seed: int = 0

    def auto_experiment_name(self) -> str:
        diff = fifteen.utils.diff_dict_from_dataclasses(ExperimentConfig(), self)
        return "_".join(
            [fifteen.utils.timestamp()]
            + [f"{k}={str(v).replace('/', '_')}" for k, v in diff.items()]
        )


@jdc.pytree_dataclass
class LearnableParams:
    latent_grid: FactoredGrid
    decoder_params: flax.core.FrozenDict[str, Any]

    def make_optax_mask(
        self, label: Literal["factors", "projections", "decoder"]
    ) -> LearnableParams:
        return LearnableParams(
            latent_grid=FactoredGrid(
                factors=label == "factors",  # type: ignore
                projecters=label == "projections",  # type: ignore
                reduce_fn=self.latent_grid.reduce_fn,
            ),
            decoder_params=label == "decoder",  # type: ignore
        )


@jdc.pytree_dataclass
class TrainState:
    params: LearnableParams
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState
    config: jdc.Static[ExperimentConfig]
    global_rotate: jaxlie.SO3
    step: Int[Array, ""]

    @staticmethod
    @jdc.jit
    def make(config: jdc.Static[ExperimentConfig]) -> TrainState:
        prng = jax.random.PRNGKey(config.seed)
        prng_grid, prng_mlp = jax.random.split(prng, num=2)

        # Initialize learnable parameters.
        latent_grid = make_3d_grid(
            prng_grid,
            output_channels=config.grid_output_channels,
            resolutions=config.grid_resolutions,
            grid_type=config.grid_type,
            transform_count=config.transform_count,
            tau_init=config.tau_init,
            # No contraction.
            scene_contract=lambda coords: coords,
        )
        params = LearnableParams(
            latent_grid=latent_grid,
            decoder_params=config.decoder_mlp.init(  # type: ignore
                prng_mlp,
                jnp.zeros((1, config.grid_output_channels)),
                barf_alpha=0.0,
            ),
        )

        # Initialize optimizer, optimizer state.
        optimizer = optax.chain(
            optax.scale_by_adam(),
            # Factor learning rate.
            optax.masked(
                optax.scale(-config.factor_lr),
                params.make_optax_mask("factors"),
            ),
            # Factor learning rate.
            optax.masked(
                optax.scale_by_schedule(
                    optax.linear_schedule(
                        init_value=-config.projection_lr,
                        end_value=0.0,
                        transition_steps=int(config.train_steps * 0.5),
                        transition_begin=int(config.train_steps * 0.2),
                    )
                ),
                params.make_optax_mask("projections"),
            ),
            # Network learning rate.
            optax.masked(
                optax.scale(-config.decoder_lr),
                params.make_optax_mask("decoder"),
            ),
            # Overall learning rate decay.
            optax.scale_by_schedule(
                optax.linear_schedule(
                    1.0,
                    0.1,
                    transition_steps=config.train_steps // 2,
                    transition_begin=config.train_steps // 2,
                )
            ),
        )
        optimizer_state = optimizer.init(jaxlie.manifold.zero_tangents(params))

        # Rotate the whole mesh?
        if config.mesh_rotate_seed is not None:
            mesh_rotate = jaxlie.SO3.sample_uniform(
                jax.random.PRNGKey(config.mesh_rotate_seed)
            )
        else:
            mesh_rotate = jaxlie.SO3.identity()

        return TrainState(
            params=params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            config=config,
            global_rotate=mesh_rotate,
            step=jnp.array(0),
        )

    def export_mesh(
        self, out_path: Path, resolution: int = 256, chunk_size=2**20
    ) -> None:
        points = (
            onp.mgrid[:resolution, :resolution, :resolution] / (resolution - 1.0) - 0.5
        ) * 2.0
        points = rearrange(
            points,
            "dim res0 res1 res2 -> (res0 res1 res2) dim",
            dim=3,
            res0=resolution,
            res1=resolution,
            res2=resolution,
        )

        sdf_grid = rearrange(
            self.compute_sdf_chunked(points, chunk_size=chunk_size),
            "(res0 res1 res2) 1 -> res0 res1 res2",
            res0=resolution,
            res1=resolution,
            res2=resolution,
        )

        meshing.export_sdf_as_mesh(
            sdf_grid,
            out_path=out_path,
        )

    def compute_sdf_chunked(
        self,
        points: Float[onp.ndarray, "batch 3"],
        chunk_size: int = 2**20,
    ) -> Float[onp.ndarray, "batch 1"]:
        assert len(points.shape) == 2
        chunk_count = max(1, points.shape[0] // chunk_size)
        return onp.concatenate(
            [
                self.compute_sdf(cast(Array, p))
                for p in onp.array_split(points, chunk_count, axis=0)
            ],
            axis=0,
        )

    @jdc.jit
    def compute_sdf(
        self,
        points: Float[Array, "*batch_dim 3"],
        params: Optional[LearnableParams] = None,
    ) -> Float[Array, "*batch_dim 1"]:
        if params is None:
            params = self.params
        *batch_dims, d = points.shape
        assert d == 3

        points = points @ self.global_rotate.as_matrix().T

        latents = params.latent_grid.interpolate(points)
        sdfs_pred = self.config.decoder_mlp.apply(
            params.decoder_params,
            latents,
            barf_alpha=self.get_barf_alpha(),
        )
        assert isinstance(sdfs_pred, Array)
        assert sdfs_pred.shape == (*batch_dims, 1)

        return sdfs_pred

    @jdc.jit(donate_argnums=(0,))
    def train_step(
        self,
        points: Float[Array, "batch_dim 3"],
        sdfs: Float[Array, "batch_dim 1"],
    ) -> Tuple[TrainState, TensorboardLogData]:
        def compute_loss(
            params: LearnableParams,
        ) -> Tuple[Float[Array, ""], TensorboardLogData]:
            sdfs_pred = self.compute_sdf(points, params)
            assert sdfs_pred.shape == sdfs.shape

            # Group sparsity + TV regularization.
            l12_reg_cost = params.latent_grid.l12_cost()
            tv_reg_cost = params.latent_grid.total_variation_cost("l1")

            # MAPE loss.
            delta = sdfs - sdfs_pred
            metrics = {
                "mape": jnp.mean(
                    jnp.abs(delta) / (jnp.abs(sdfs) + 1e-2),
                ),
                "l2": jnp.mean(delta**2),
                "l1": jnp.mean(jnp.abs(delta)),
                "l12_reg_cost": l12_reg_cost,
                "tv_reg_cost": tv_reg_cost,
            }

            loss = (
                metrics[self.config.loss]
                + self.config.l12_reg_coeff * l12_reg_cost
                + self.config.tv_reg_coeff * tv_reg_cost
            )
            return loss, TensorboardLogData(
                # Type here is invariant but should be covariant?
                scalars=metrics,  # type: ignore
                histograms={"sdfs": sdfs[::100, :], "sdfs_pred": sdfs_pred[::100, :]},
            )

        (loss, log_data), grads = jaxlie.manifold.value_and_grad(
            compute_loss, has_aux=True
        )(self.params)
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        params = jaxlie.manifold.rplus(self.params, updates)

        with jdc.copy_and_mutate(self, validate=True) as new_state:
            new_state.params = params
            new_state.optimizer_state = optimizer_state
            new_state.step = new_state.step + 1

        return (
            new_state,
            cast(TensorboardLogData, log_data).merge_scalars(
                {
                    "grad_norm": optax.global_norm(grads),  # type: ignore
                }
            ),
        )

    def get_barf_alpha(self) -> Optional[Float[Array, ""]]:
        """Get alpha constant from BARF. Should be in the range of [0, num_freqs]."""
        if self.config.barf_steps is None:
            return None

        # Clipping here has no functional benefit, but is nice for being rigorous with
        # the paper definition.
        return jnp.minimum(
            self.step / self.config.barf_steps * self.config.decoder_mlp.pos_enc_freqs,
            self.config.decoder_mlp.pos_enc_freqs,
        )
