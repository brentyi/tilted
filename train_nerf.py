"""Training script for radiance fields with hybrid neural field architectures."""

from __future__ import annotations

import os

# Don't use all the GPU memory. Helps us run visualization in parallel if we want.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import fifteen
import jax_dataclasses as jdc
import termcolor
import tyro
from typing_extensions import Annotated

from tilted.core.factored_grid import Learnable3DProjectersBase
from tilted.nerf.decoder import NerfDecoderMlp
from tilted.nerf.render import RenderConfig
from tilted.nerf.train_loop import train_loop
from tilted.nerf.train_state import (
    FieldConfig,
    NerfConfig,
    OptimizationConfig,
    TrainState,
)


@dataclasses.dataclass(frozen=True)
class BottleneckConfig:
    """Configuration for bottleneck phase for training TILTED feature volumes."""

    enable: bool
    optim: OptimizationConfig = OptimizationConfig(camera_delta_lr=0.0)
    field: FieldConfig = FieldConfig(
        grid_type="cp",
        primary_decoder=NerfDecoderMlp(color_units=32),
        primary_resolutions=(1024,),
        primary_channels=32,
        proposal_resolutions=(64,),
        proposal_channels=(8,),
    )


@dataclasses.dataclass(frozen=True)
class RunConfig:
    # Note that we erase the name of the config argument, this lets us pass in --dataset-path instead
    # of --config.dataset-path.
    config: Annotated[NerfConfig, tyro.conf.arg(name="")]

    bottleneck: BottleneckConfig = BottleneckConfig(enable=False)
    """Configuration for a bottleneck phase, which helps initialize transformations."""

    exp_name: Optional[str] = None
    outputs_dir: Path = Path("./outputs_nerf")

    def __post_init__(self) -> None:
        # Make sure the parameters make sense.
        if self.config.primary_transform_count is not None:
            assert (
                self.config.field.primary_channels
                % len(self.config.field.primary_resolutions)
            ) == 0
            channels_per_res = self.config.field.primary_channels // len(
                self.config.field.primary_resolutions
            )
            if self.config.field.grid_type == "kplane":
                # In K-Planes, we divide the total channel account evenly across each
                # resolution, then evenly across each transform.
                assert channels_per_res % self.config.primary_transform_count == 0

            elif self.config.field.grid_type == "vm":
                # In TensoRF (Vector-Matrix), we concatenate across three planes, so we
                # need to divide again by 3.
                assert channels_per_res % 3 == 0
                assert channels_per_res // 3 % self.config.primary_transform_count == 0

            elif self.config.field.grid_type == "cp":
                # In CP, we divide the total channel account evenly across each
                # resolution, then evenly across each transform.
                assert channels_per_res % self.config.primary_transform_count == 0


def make_config(
    dataset_type: Literal["blender", "nerfstudio"],
    grid_type: Literal["kplane", "vm", "cp"],
    primary_channels: int,
    transform_count: Optional[int],
    # Next two arguments are mutated.
    descriptions: Dict[str, str],
    configs: Dict[str, RunConfig],
) -> None:
    """Helper for populating base configurations, which can then be overridden via the commandline.

    `descriptions` and `configs` arguments are mutated."""
    if transform_count is None:
        config_name = f"{dataset_type}-{grid_type}-{primary_channels}c-axis-aligned"
    else:
        config_name = (
            f"{dataset_type}-{grid_type}-{primary_channels}c-tilted-{transform_count}t"
        )

    is_blender_data = dataset_type == "blender"

    assert config_name not in descriptions
    assert config_name not in configs
    descriptions[config_name] = (
        termcolor.colored("**Recommended** ", attrs=["bold"])
        if grid_type == "kplane" and primary_channels == 64
        else ""
    ) + f"{'Synthetic' if dataset_type == 'blender' else 'Real-world'} data, with a" f" {grid_type} decomposition, {primary_channels} channels," + (
        f" {transform_count} transform(s)."
        if transform_count is not None
        else " axis-aligned."
    )
    configs[config_name] = RunConfig(
        NerfConfig(
            # Dataset path needs to be explicitly specified.
            dataset_path=tyro.MISSING,
            dataset_type=dataset_type,
            optim=OptimizationConfig(
                # Turn off pose optimization for synthetic datasets.
                camera_delta_lr=0.0 if is_blender_data else 6e-4,
                projection_lr=0.0,
                l12_reg_coeff=0.001 if grid_type == "kplane" else 0.0,
                tv_reg_l1_coeff=0.00 if grid_type == "vm" else 0.0,
                tv_reg_l2_coeff=0.01 if grid_type == "kplane" else 0.0,
            ),
            field=FieldConfig(
                grid_type=grid_type,
                primary_resolutions=(64, 128, 256, 512 if is_blender_data else 1024),
                primary_channels=primary_channels,
                # Note that channel count needs to be divisible by 3 for Vector-Matrix
                # decompositions.
                proposal_channels={"kplane": (8, 8), "vm": (48, 48)}[grid_type],
            ),
            # Note that we need to make bounding boxes larger than usual to account for
            # scene rotation.
            grid_bound=1.6 if is_blender_data else 1.0,
            render_config=(
                # Sampling strategies are slightly different if we're doing synthetic
                # (bounded) scenes vs real ones.
                RenderConfig(
                    proposal_samples=(256, 128),
                    final_samples=48,
                    background="white",
                    initial_samples="linear",
                    near=2.0,
                    far=6.0,
                    # Apply a global random rotation.
                    global_rotate_seed=0,
                )
                if is_blender_data
                else RenderConfig(
                    proposal_samples=(256, 128),
                    final_samples=48,
                    background="last_sample",
                    initial_samples="linear_disparity",
                    near=0.001,
                    far=300.0,
                    # Apply a global random rotation.
                    global_rotate_seed=0,
                )
            ),
            eval_final_samples=48,
            eval_proposal_samples=(256, 128),
            primary_transform_count=transform_count,
        ),
        bottleneck=BottleneckConfig(enable=transform_count is not None),
    )


def make_configs() -> Tuple[Dict[str, RunConfig], Dict[str, str]]:
    configs: Dict[str, RunConfig] = {}
    descriptions: Dict[str, str] = {}

    # Make some configurations. These will be exposed as subcommands.
    for dataset_type in ("blender", "nerfstudio"):
        for transform_count in (None, 4, 8):
            make_config(
                dataset_type=dataset_type,
                grid_type="kplane",
                primary_channels=32,
                transform_count=transform_count,
                descriptions=descriptions,
                configs=configs,
            )
            make_config(
                dataset_type=dataset_type,
                grid_type="kplane",
                primary_channels=64,
                transform_count=transform_count,
                descriptions=descriptions,
                configs=configs,
            )
            make_config(
                dataset_type=dataset_type,
                grid_type="vm",
                primary_channels=96,
                transform_count=transform_count,
                descriptions=descriptions,
                configs=configs,
            )
            make_config(
                dataset_type=dataset_type,
                grid_type="vm",
                primary_channels=192,
                transform_count=transform_count,
                descriptions=descriptions,
                configs=configs,
            )
    return configs, descriptions


def main(args: RunConfig) -> None:
    """Run training."""

    exp_name = args.exp_name
    if exp_name is None:
        exp_name = "-".join(
            [
                args.config.dataset_type,
                args.config.dataset_path.name,
                args.config.field.grid_type,
                str(args.config.primary_transform_count) + "t"
                if args.config.primary_transform_count is not None
                else "axis-aligned",
                str(args.config.field.primary_channels) + "c",
                "seed=" + str(args.config.seed),
                "global_rotate_seed="
                + str(args.config.render_config.global_rotate_seed),
            ]
        )
        if config.bottleneck.enable:
            exp_name += "-bottleneck"

    # TILTED bottleneck stage.
    bottleneck_train_state = None
    if args.bottleneck.enable:
        # Initialize bottleneck train state.
        bottleneck_config = dataclasses.replace(
            args.config, optim=args.bottleneck.optim, field=args.bottleneck.field
        )
        bottleneck_train_state = TrainState.make(bottleneck_config)
        assert hasattr(
            bottleneck_train_state.params.primary_field.grid.projecters, "tau"
        )

        # Train bottleneck model.
        bottleneck_train_state = train_loop(
            args.outputs_dir / ("_bneck-" + exp_name),
            bottleneck_train_state,
            early_stop_steps=bottleneck_config.optim.projection_decay_start
            + bottleneck_config.optim.projection_decay_steps,
            do_eval=False,
        )

    # Train primary model.
    train_state = TrainState.make(args.config)
    if bottleneck_train_state is not None:
        with jdc.copy_and_mutate(train_state) as train_state:
            cast(
                Learnable3DProjectersBase,
                train_state.params.primary_field.grid.projecters,
            ).tau = cast(
                Learnable3DProjectersBase,
                bottleneck_train_state.params.primary_field.grid.projecters,
            ).tau
    train_state = train_loop(args.outputs_dir / exp_name, train_state)


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()

    configs, descriptions = make_configs()
    RunConfigChoices = tyro.extras.subcommand_type_from_defaults(
        defaults=configs,
        descriptions=descriptions,
        prefix_names=False,
    )
    config = tyro.cli(
        tyro.conf.FlagConversionOff[RunConfigChoices], description=__doc__
    )
    main(config)
