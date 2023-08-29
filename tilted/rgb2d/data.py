from typing import List, Tuple, cast

import cv2
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from einops import rearrange
from fifteen.experiments import TensorboardLogData
from jaxtyping import Array, Float

from ..core.factored_grid import FactoredGrid, Learnable2dProjecters


@jdc.pytree_dataclass
class ReconstructionData:
    """Reconstruction data. Both RGB and coords should be in the range [0, 1]."""

    rgb: Float[Array, "holdout_count 3"]
    coords: Float[Array, "train_count 2"]


def generate_data(
    target_image: onp.ndarray,
    holdout_ratio: float,
) -> Tuple[ReconstructionData, ReconstructionData]:
    """Returns a tuple of (train data, holdout data)."""
    h, w = target_image.shape[:2]
    assert target_image.shape == (h, w, 3)

    # Make the training and eval sets.
    # Coordinates will all be in the range [-1.0, 1.0].
    pixels = target_image.reshape((-1, 3))
    coords = rearrange(onp.mgrid[:h, :w], "d h w -> (h w) d", d=2)
    coords = coords / onp.array([h - 1.0, w - 1.0]) * 2.0 - 1.0

    circle_mask = onp.linalg.norm(coords, axis=-1) <= 1.0
    pixels = pixels[circle_mask, :]
    coords = coords[circle_mask, :]
    num_pixels = pixels.shape[0]

    # Make a binary mask that marks holdout pixels.
    holdout_mask = onp.zeros(num_pixels, dtype=onp.bool_)
    holdout_mask[: int(num_pixels * holdout_ratio)] = True
    onp.random.default_rng(0).shuffle(holdout_mask)

    return ReconstructionData(
        rgb=cast(Array, pixels[~holdout_mask]),
        coords=cast(Array, coords[~holdout_mask]),
    ), ReconstructionData(
        rgb=cast(Array, pixels[holdout_mask]),
        coords=cast(Array, coords[holdout_mask]),
    )


def logging_overlay_on_frame(
    frame: onp.ndarray,
    text_on_vis: bool,
    step: int,
    hparam_override_lines: List[str],
    log_data: TensorboardLogData,
    latent_grid: FactoredGrid[Learnable2dProjecters],
    target_rotate_deg: float,
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Given a rendered frame, we:
    - Visualize transforms using arrows.
    - Add a text overlay with hyperparameters and metrics.
    """
    viz_h, viz_w, d = frame.shape
    assert d == 3

    frame_with_labels = frame.copy()
    if text_on_vis:
        font = cv2.FONT_HERSHEY_PLAIN  # type: ignore
        for line_number, line_text in enumerate(
            [
                f"{step=}",
                *[
                    f"{k}={float(log_data.scalars[k]):0.4f}"
                    for k in log_data.scalars.keys()
                ],
            ]
            + hparam_override_lines
        ):
            cv2.putText(  # type: ignore
                frame_with_labels,
                text=line_text,
                org=(2, 1 + 15 * (line_number + 1)),
                fontFace=font,
                fontScale=1.0,
                color=(0, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA,  # type: ignore
            )
            cv2.putText(  # type: ignore
                frame_with_labels,
                text=line_text,
                org=(2, 1 + 15 * (line_number + 1)),
                fontFace=font,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,  # type: ignore
            )

    tau = latent_grid.projecters.tau
    num_transforms = tau.unit_complex.shape[0]
    transforms_per_res = latent_grid.projecters.transforms_per_res
    assert num_transforms % transforms_per_res == 0
    assert tau.unit_complex.shape == (num_transforms, 2)
    magnitudes = []
    for i in range(num_transforms):
        res_idx = i // transforms_per_res
        factor = latent_grid.factors[res_idx]

        if latent_grid.projecters.orthogonal_uv:
            u = factor.values[(i % transforms_per_res) * 2]
            v = factor.values[(i % transforms_per_res) * 2 + 1]
            assert u.shape == v.shape
            assert len(u.shape) == 2
            magnitudes.append(onp.mean((u[None, :, :] * v[:, None, :]) ** 2))
        else:
            magnitudes.append(onp.mean(factor.values[i % transforms_per_res]))

    magnitudes_norm = onp.array(magnitudes)
    magnitudes_norm /= onp.max(magnitudes_norm)  # type: ignore

    viz_deltas = (
        magnitudes_norm[:, None]
        * onp.array(
            tau.unit_complex
            @ jaxlie.SO2.from_radians(onp.deg2rad(target_rotate_deg)).as_matrix().T
        )
        * onp.array([viz_h / 2.0, viz_w / 2.0])
    ).astype(onp.int64)
    for rank, i in enumerate(onp.argsort(magnitudes_norm)):
        cv2.arrowedLine(  # type: ignore
            frame_with_labels,  # type: ignore
            (viz_h // 2, viz_w // 2),
            (viz_h // 2 + viz_deltas[i, 0], viz_w // 2 + viz_deltas[i, 1]),
            color=(0, 0, 0),
            thickness=3,
            line_type=cv2.LINE_AA,  # type: ignore
        )
        cv2.arrowedLine(  # type: ignore
            frame_with_labels,  # type: ignore
            (viz_h // 2, viz_w // 2),
            (viz_h // 2 + viz_deltas[i, 0], viz_w // 2 + viz_deltas[i, 1]),
            color=(255, 0, 0) if rank == len(magnitudes_norm) - 1 else (127, 127, 127),
            thickness=1,
            line_type=cv2.LINE_AA,  # type: ignore
        )

    return frame_with_labels, magnitudes_norm
