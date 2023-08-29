from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import fifteen
import imageio.v3 as iio
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import optax
from jax import numpy as jnp
from jaxtyping import Array, Float
from tqdm.auto import tqdm

from . import data, render
from .train_state import NerfConfig, TrainState


def train_loop(
    experiment_path: Path,
    state: TrainState,
    early_stop_steps: Optional[int] = None,
    do_eval: bool = True,
) -> TrainState:
    """Run training from an output directory and initial training state."""
    exp = fifteen.experiments.Experiment(experiment_path)
    exp.assert_new()

    config = state.config
    config_dict = fifteen.utils.flattened_dict_from_dataclass(config)

    # Load dataset.
    train_dataset = config.get_dataset("train")
    train_cameras = train_dataset.get_cameras()
    dataloader = data.CachedNerfDataloader(
        dataset=train_dataset, minibatch_size=config.optim.minibatch_size
    )
    train_minibatches = fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0)

    # Save some metadata!
    exp.write_metadata("num_cameras", len(train_cameras))
    exp.write_metadata("config", config)
    exp.write_metadata("config_dict", config_dict)
    (exp.data_dir / "git_diff").write_text(fifteen.utils.get_git_diff())
    (exp.data_dir / "git_hash").write_text(fifteen.utils.get_git_commit_hash())

    # Load eval set.
    test_dataset = None
    test_minibatches = None
    if config.dataset_type == "blender":
        test_dataset = config.get_dataset("test")
        test_dataloader = data.CachedNerfDataloader(
            dataset=test_dataset, minibatch_size=4096 * 4
        )
        test_minibatches = fifteen.data.cycled_minibatches(
            test_dataloader, shuffle_seed=0
        )

    # Run!
    print("Training with config:", config)
    final_train_metrics: List[Dict[str, float]] = []
    for loop_metrics in tqdm(
        fifteen.utils.range_with_metrics(config.optim.total_steps),
        desc="Training",
    ):
        # Load minibatch.
        minibatch = next(train_minibatches)
        assert minibatch.get_batch_axes() == (config.optim.minibatch_size,)
        assert minibatch.colors.shape == (config.optim.minibatch_size, 3)

        # Training step.
        state, log_data = _train_step(state, minibatch)
        log_data = log_data.merge_scalars(
            {"train/iterations_per_sec": loop_metrics.iterations_per_sec}
        )

        # Log & checkpoint.
        step = int(state.step)
        exp.log(log_data, step=step, log_scalars_every_n=5, log_histograms_every_n=100)

        if loop_metrics.counter > config.optim.total_steps - 100 or (
            early_stop_steps is not None
            and loop_metrics.counter > early_stop_steps - 100
        ):
            final_train_metrics.append(
                dict(
                    **jax.tree_map(float, log_data.scalars),
                    iterations_per_sec=loop_metrics.iterations_per_sec,
                )
            )

        # Upsample grids.
        if step == config.optim.half_resolution_steps:
            state = state.resize_to_final_size()

        # Eval PSNR.
        if config.dataset_type == "blender" and step % 100 == 0:
            assert test_minibatches is not None
            test_minibatch = next(test_minibatches)
            pred = state.render_rays(
                test_minibatch.rays_wrt_world,
                eval_mode=True,
                mode="rgb",
                prng=state.prng,
            )

            eval_mse = onp.mean((pred - test_minibatch.colors) ** 2)
            eval_psnr = -10.0 * onp.log(eval_mse) / jnp.log(10.0)

            exp.summary_writer.add_scalar("eval/psnr", eval_psnr, global_step=step)

        if (
            config.debug_render_interval is not None
            and step % config.debug_render_interval == 0
        ):
            for i in range(0, len(train_cameras), len(train_cameras) // 3):
                outputs = state.render_camera(
                    train_cameras[i].resize_with_fixed_fov(400, 400, fixed_axis="x"),
                    camera_index=i,
                    # This is the training render config! Not the eval!
                    eval_mode=False,
                    mode="all",
                )
                exp.summary_writer.add_image(
                    f"rendered_500/rgb{i}",
                    outputs.rgb,
                    global_step=step,
                    dataformats="HWC",
                )
                exp.summary_writer.add_image(
                    f"rendered_500/dist{i}",
                    render.viz_dist(onp.array(outputs.dist)),
                    global_step=step,
                    dataformats="HWC",
                )
                for j, prop_dist in enumerate(outputs.proposal_dist_maps):
                    exp.summary_writer.add_image(
                        f"rendered_500/dist{i}_prop_{j}",
                        render.viz_dist(onp.array(prop_dist)),
                        global_step=step,
                        dataformats="HWC",
                    )

        if step % 5000 == 0:
            exp.save_checkpoint(state, step=int(state.step), keep=1)
        if early_stop_steps is not None and step >= early_stop_steps:
            break

    # Wrap up! We save a checkpoint, and for synthetic scene generate some eval metrics.
    # We (redundantly) overwrite to prevent errors if we just checkpointed; could also
    # use a try/except.
    exp.save_checkpoint(state, step=int(state.step), keep=1, overwrite=True)

    # Wrap up with any dataset-specific evaluation we want to do.
    if do_eval:
        _do_eval(config, state, test_dataset, exp)

    # Record mean train metrics, from last 100 steps.
    train_metrics = jax.tree_map(
        lambda *args: float(onp.mean(args)), *final_train_metrics
    )
    exp.write_metadata("train_metrics", train_metrics)

    return state


@jdc.jit(donate_argnums=0)
def _train_step(
    state: TrainState, minibatch: data.RenderedRays
) -> Tuple[TrainState, fifteen.experiments.TensorboardLogData]:
    prng, prng_render = jax.random.split(state.prng)

    def compute_loss(
        params: render.LearnableParams,
    ) -> Tuple[Float[Array, ""], fifteen.experiments.TensorboardLogData]:
        render_out = render.render_rays(
            minibatch.rays_wrt_world,
            params=params,
            config=state.config.render_config,
            prng=prng_render,
            anneal_factor=state.get_anneal_factor(),
            low_pass_alpha=state.get_low_pass_alpha(),
        )
        label = minibatch.colors
        assert render_out.rgb.shape == label.shape

        mse = jnp.mean((render_out.rgb - label) ** 2)

        l12_reg_cost = (
            sum(f.grid.l12_cost() for f in params.density_fields)
            + params.primary_field.grid.l12_cost()
        )
        tv_reg_l1_cost = sum(
            f.grid.total_variation_cost("l1") for f in params.density_fields
        ) + params.primary_field.grid.total_variation_cost("l1")
        tv_reg_l2_cost = sum(
            f.grid.total_variation_cost("l2") for f in params.density_fields
        ) + params.primary_field.grid.total_variation_cost("l2")

        loss = (
            mse
            + state.config.optim.l12_reg_coeff * l12_reg_cost
            + state.config.optim.tv_reg_l1_coeff * tv_reg_l1_cost
            + state.config.optim.tv_reg_l2_coeff * tv_reg_l2_cost
            + state.config.optim.interlevel_loss_coeff * render_out.interlevel_loss
            + state.config.optim.distortion_loss_coeff * render_out.distortion_loss
        )
        log_data = fifteen.experiments.TensorboardLogData(
            scalars={
                "loss": loss,
                "mse": mse,
                "psnr": -10.0 * jnp.log(mse) / jnp.log(10.0),
                "l12_reg": l12_reg_cost,
                "tv_reg_l1": tv_reg_l1_cost,
                "tv_reg_l2": tv_reg_l2_cost,
                "interlevel_loss": render_out.interlevel_loss,
                "distortion_loss": render_out.distortion_loss,
            },
        )

        return loss, log_data

    # Compute gradients.
    log_data: fifteen.experiments.TensorboardLogData
    grads: render.LearnableParams
    (loss, log_data), grads = jaxlie.manifold.value_and_grad(
        compute_loss, has_aux=True
    )(state.params)
    del loss

    # Propagate gradients through ADAM, learning rate scheduler, etc.
    updates, new_optimizer_state = state.optimizer.update(
        grads, state.optimizer_state, state.params  # type: ignore
    )

    # Add gradient norm to Tensorboard logs.
    #
    # Cast for invariance annoyingness, we should fix this in `fifteen`. Would be
    # easier with an immutable type.
    log_data = log_data.merge_scalars(
        cast(
            Dict[str, Union[float, onp.ndarray, jax.Array]],
            {
                "grad_norm": optax.global_norm(grads),
            },
        )
    )

    with jdc.copy_and_mutate(state, validate=True) as new_state:
        new_state.optimizer_state = new_optimizer_state
        new_state.params = jaxlie.manifold.rplus(state.params, updates)
        new_state.prng = prng
        new_state.step = new_state.step + 1
    return new_state, log_data.prefix("train/")


def _do_eval(
    config: NerfConfig,
    state: TrainState,
    test_dataset: Optional[data.NerfDataset],
    exp: fifteen.experiments.Experiment,
) -> None:
    if config.dataset_type == "blender":
        print("Evaluating...")
        assert test_dataset is not None
        eval_metrics, eval_images = data.compute_evaluation_metrics(test_dataset, state)
        del eval_images
        print(eval_metrics)
        exp.write_metadata("eval_metrics_fixed", eval_metrics)
        exp.log(
            fifteen.experiments.TensorboardLogData(
                scalars=eval_metrics,  # type: ignore
            ).prefix("final_eval_fixed/"),
            step=int(state.step),
        )

        # # Save some of the eval images.
        # for i, image in enumerate(eval_images[::5]):
        #     image = (image * 255).astype(onp.uint8)
        #     iio.imwrite(exp.data_dir / f"render_eval_{i:03d}.png", image)
        #     exp.summary_writer.add_image(
        #         "render_eval", image, global_step=i, dataformats="HWC"
        #     )
    else:
        print("Skipping eval!")
