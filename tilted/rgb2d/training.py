from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, cast

import einops
import fifteen
import flax.core
import jax
import jax.scipy.ndimage
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import optax
from einops.einops import rearrange
from fifteen.experiments import TensorboardLogData
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from typing_extensions import assert_never

from ..core.decoder import DecoderMlp
from ..core.factored_grid import FactoredGrid, Learnable2dProjecters, make_2d_grid
from . import data


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    exp_name: Optional[str]
    exp_name_prefix: Optional[str]
    target_image: Path
    target_rotate_deg: float = 30.0

    # Representation size.
    grid_transforms_per_res: int = 8
    grid_output_channels: int = 64
    grid_resolutions: Tuple[int, ...] = (16,)  # Should be overridden...!

    # Projecters.
    decoder: DecoderMlp = dataclasses.field(
        default_factory=lambda: DecoderMlp(
            output_dim=3,
            output_sigmoid=True,
            units=32,
            layers=2,
            pos_enc_freqs=4,
        )
    )

    # Ratio of pixels to hold out for evaluation.
    holdout_ratio: float = 0.2

    # Transform optimization.
    tau_init: Literal["random", "zeros"] = "random"
    orthogonal_uv: bool = True
    """Set to False to learn separate transforms for each u/v vector, as opposed to
    learning one transform per orthogonal pair. Does not appear to help much."""

    # Blur the target image for the first X steps to improve convergence.
    blur_size: int = 13
    blur_steps: int = 100

    # Steps to completion for BARF-style coarse-to-fine schedule. Set to None to disable
    # BARF.
    barf_steps: Optional[int] = 500

    # Learning rates.
    factor_lr: float = 0.005
    projection_lr: float = 0.01
    decoder_lr: float = 0.002

    # Optimizer settings.
    train_steps: int = 5_000  # Could bump to 10_000 for final experiments.
    minibatch_size: Optional[int] = (
        2**14
    )  # If `None`, we use all pixels at every step.
    loss: Literal["l1", "l2"] = "l2"
    l12_reg_coeff: float = 0.05
    tv_reg_coeff: float = 0.001

    text_on_vis: bool = True
    gif_until_step: int = 2500

    seed: int = 94709

    def auto_experiment_name(self) -> str:
        diff = fifteen.utils.diff_dict_from_dataclasses(
            ExperimentConfig(self.exp_name, self.exp_name_prefix, self.target_image),
            self,
        )

        parts = []
        if self.exp_name_prefix is not None:
            parts.append(self.exp_name_prefix)
        parts.append(fifteen.utils.timestamp())
        parts.extend([f"{k}={str(v).replace('/', '_')}" for k, v in diff.items()])
        return "_".join(parts)


def _mask_outside_circle(
    image: Float[Union[Array, onp.ndarray], "h w c"]
) -> Float[Array, "h w c"]:
    """Given an image, make all pixels black except those inside a center circle."""
    h, w, c = image.shape
    assert c == 3
    mask = (
        jnp.linalg.norm(
            einops.rearrange(jnp.mgrid[:h, :w], "d h w -> h w d")
            / jnp.array([h - 1.0, w - 1.0])
            - 0.5,
            axis=-1,
        )
        < 0.5
    )
    return jnp.where(mask[:, :, None], image, 1.0)


@jdc.pytree_dataclass
class LearnableParams:
    latent_grid: FactoredGrid[Learnable2dProjecters]
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

    @staticmethod
    def make(config: ExperimentConfig, prng: jax.random.KeyArray) -> LearnableParams:
        prng_latent, prng_decoder = jax.random.split(prng)
        latent_grid = make_2d_grid(
            prng_latent,
            output_channels=config.grid_output_channels,
            transforms_per_res=config.grid_transforms_per_res,
            resolutions=config.grid_resolutions,
            tau_init=config.tau_init,
            orthogonal_uv=config.orthogonal_uv,
        )
        dummy_feat = jnp.zeros((1, config.grid_output_channels))
        return LearnableParams(
            latent_grid=latent_grid,
            decoder_params=config.decoder.init(  # type: ignore
                prng_decoder, dummy_feat, barf_alpha=0.0
            ),
        )


@jdc.pytree_dataclass
class TrainState:
    params: LearnableParams
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState
    config: jdc.Static[ExperimentConfig]
    global_rotate: jaxlie.SO2
    step: Int[Array, ""]

    @staticmethod
    @partial(jax.jit, static_argnums=0)
    def make(config: jdc.Static[ExperimentConfig]) -> TrainState:
        params = LearnableParams.make(config, prng=jax.random.PRNGKey(config.seed))
        # optimizer = optax.adam(False=1e-4)
        optimizer = optax.chain(
            optax.scale_by_adam(),
            # Factor learning rate.
            optax.masked(
                optax.scale(-config.factor_lr), params.make_optax_mask("factors")
            ),
            # Transform learning rate.
            optax.masked(
                optax.scale_by_schedule(
                    optax.linear_schedule(
                        init_value=-config.projection_lr,
                        end_value=0.0,
                        transition_steps=int(config.train_steps * 0.3),
                        transition_begin=int(config.train_steps * 0.2),
                    )
                ),
                params.make_optax_mask("projections"),
            ),
            # MLP learning rate.
            optax.masked(
                optax.scale(-config.decoder_lr), params.make_optax_mask("decoder")
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
        return TrainState(
            params,
            optimizer,
            optimizer_state,
            config,
            global_rotate=jaxlie.SO2.from_radians(
                jnp.deg2rad(config.target_rotate_deg)
            ),
            step=jnp.array(0),
        )

    def bilerp(
        self,
        coords: Float[Array, "*batch 2"],
        params: Optional[LearnableParams] = None,
    ) -> Float[Array, "*batch 3"]:
        if params is None:
            params = self.params

        coords = coords @ self.global_rotate.as_matrix().T

        latent_vectors = params.latent_grid.interpolate(coords)
        out = self.config.decoder.apply(
            params.decoder_params, latent_vectors, barf_alpha=self._get_barf_alpha()
        )
        assert isinstance(out, Array)
        assert out.shape[-1] == 3
        return out

    @jdc.jit
    def image(self, h: jdc.Static[int], w: jdc.Static[int]) -> Float[Array, "h w 3"]:
        # Get coordinates in the range [-1.0, 1.0].
        coords = (
            rearrange(jnp.mgrid[:h, :w], "d h w -> h w d")
            / jnp.array([h - 1.0, w - 1.0])
            * 2.0
            - 1.0
        )
        return _mask_outside_circle(self.bilerp(coords))

    @jdc.jit
    def train_step(
        self: TrainState, target: data.ReconstructionData
    ) -> Tuple[Float[Array, ""], TrainState, TensorboardLogData]:
        """Returns loss and updated state."""

        def compute_loss(
            params: LearnableParams,
        ) -> Tuple[Float[Array, ""], TensorboardLogData]:
            rgb_pred = self.bilerp(target.coords, params=params)
            assert rgb_pred.shape == target.rgb.shape

            l1_error = jnp.mean(jnp.abs(rgb_pred - target.rgb))
            l2_error = jnp.mean((rgb_pred - target.rgb) ** 2)

            if self.config.loss == "l1":
                loss = l1_error
            elif self.config.loss == "l2":
                loss = l2_error
            else:
                assert_never(self.config.loss)

            # Group sparsity + TV regularization.
            l12_reg_cost = params.latent_grid.l12_cost()
            tv_reg_cost = params.latent_grid.total_variation_cost("l1")

            loss = (
                loss
                + self.config.l12_reg_coeff * l12_reg_cost
                + self.config.tv_reg_coeff * tv_reg_cost
            )

            return (
                loss,
                TensorboardLogData(
                    scalars={
                        "train/l1": l1_error,
                        "train/l2": l2_error,
                        "train/l12_reg_cost": l12_reg_cost,
                        "train/tv_reg_cost": tv_reg_cost,
                        "train/psnr": -10.0 * jnp.log(l2_error) / jnp.log(10.0),
                    },
                    histograms={
                        "image_pred": rgb_pred.flatten()[::20],
                    },
                ),
            )

        (loss, log_data), grads = jaxlie.manifold.value_and_grad(
            compute_loss, has_aux=True
        )(self.params)
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        params = jaxlie.manifold.rplus(self.params, updates)

        return (
            loss,
            TrainState(
                params,
                self.optimizer,
                optimizer_state,
                self.config,
                self.global_rotate,
                step=self.step + 1,
            ),
            cast(TensorboardLogData, log_data).merge_scalars(
                {
                    "grad_norm": optax.global_norm(grads),  # type: ignore
                }
            ),
        )

    def _get_barf_alpha(self) -> Optional[Float[Array, ""]]:
        """Get alpha constant from BARF. Should be in the range of [0, num_freqs]."""
        if self.config.barf_steps is None:
            return None

        # Clipping here has no functional benefit, but is nice for being rigorous with
        # the paper definition.
        return jnp.minimum(
            self.step / self.config.barf_steps * self.config.decoder.pos_enc_freqs,
            self.config.decoder.pos_enc_freqs,
        )
