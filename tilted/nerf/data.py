from __future__ import annotations

import concurrent.futures
import dataclasses
import fcntl
import functools
import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import cv2
import dacite
import dm_pix as pix
import fifteen
import flax
import h5py
import imageio.v3 as iio
import jax
import jax_dataclasses as jdc
import jaxlie
import lpips_jax
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated, assert_never

from . import cameras

if TYPE_CHECKING:
    from .train_state import TrainState


class NerfDataset(Protocol):
    dataset_root: Path

    def load_all_training_rays(self) -> RenderedRays:
        ...

    def get_cameras(self) -> List[cameras.Camera]:
        ...


def _assert_dataclass_no_type_guard(x: Any) -> None:
    assert dataclasses.is_dataclass(x)


class CachedNerfDataloader:
    """Dataloader that caches preprocessing and minibatch shuffling to disk. This helps
    reduce memory usage + startup times.

    Note that we implement the protocol defined by fifteen.util.DataloaderProtocol.
    """

    def __init__(self, dataset: NerfDataset, minibatch_size: int):
        cache_filename = ["_tilted"]

        for f in dataclasses.fields(dataset):  # type: ignore
            key = f.name
            val = getattr(dataset, key)
            if isinstance(val, Path):
                continue
            cache_filename.append(f"{key}={val}".replace("/", "-"))

        _assert_dataclass_no_type_guard(dataset)
        cache_filename.append("cache.hdf5")
        cache_path = dataset.dataset_root / "_".join(cache_filename)

        # Create h5py cache file if one doesn't exist yet.
        cache_chunk_size = 512
        assert (
            minibatch_size % cache_chunk_size == 0
        ), f"Ideally, minibatch size should be divisible by {cache_chunk_size}"

        lock_path = dataset.dataset_root / (cache_path.name + ".lock")
        with open(lock_path, "w") as lock_fd:
            # Try to grab the cache lock. The goal is to prevent multiple training
            # processes from writing the cache at the same time.
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            if not cache_path.exists():
                print(f"{cache_path} does not exist. Creating...")

                # Load all training rays.
                training_rays = dataset.load_all_training_rays()
                (num_training_rays,) = training_rays.get_batch_axes()

                # Cull out training rays beyond some threshold.
                shuffled_indices = onp.random.default_rng(0).permutation(
                    num_training_rays
                )
                max_training_rays = 100_000 * 4096
                if num_training_rays > max_training_rays:
                    print(
                        f"Culling out rays for cache: {num_training_rays} =>"
                        f" {max_training_rays}, a"
                        f" {(1.0 - max_training_rays / num_training_rays) * 100.0:.02f}%"
                        " decrease"
                    )
                    shuffled_indices = shuffled_indices[:max_training_rays]

                # Shuffle training rays.
                training_rays = jax.tree_map(
                    lambda x: x[shuffled_indices], training_rays
                )

                with h5py.File(cache_path, "w", libver="latest") as f:
                    for k, v in flax.traverse_util.flatten_dict(
                        jdc.asdict(training_rays), sep="."
                    ).items():
                        assert isinstance(v, onp.ndarray)
                        print(f"\tWriting {k} with shape {v.shape}")
                        chunk_shape = (cache_chunk_size,) + v.shape[1:]
                        f.create_dataset(name=k, data=v, chunks=chunk_shape)
                del training_rays

            fcntl.flock(lock_fd, fcntl.LOCK_UN)

        self.cache_path = cache_path
        self.hdf5_file = h5py.File(self.cache_path, "r")
        self.ray_count = self.hdf5_file["colors"].shape[0]  # type: ignore
        self.minibatch_size = minibatch_size

    def minibatch_count(self) -> int:
        return self.ray_count // self.minibatch_size

    def minibatches(
        self, shuffle_seed: Optional[int]
    ) -> fifteen.data.SizedIterable[RenderedRays]:
        class _Inner:
            def __iter__(_self):
                minibatch_indices = onp.random.default_rng(shuffle_seed).permutation(
                    self.minibatch_count()
                )
                for i in minibatch_indices:
                    yield self._index(
                        slice(i * self.minibatch_size, (i + 1) * self.minibatch_size)
                    )

            def __len__(_self):
                return self.minibatch_count()

        return _Inner()

    def _index(self, index: slice) -> RenderedRays:
        contents = {k: v[index] for k, v in self.hdf5_file.items()}
        return dacite.from_dict(
            RenderedRays,
            flax.traverse_util.unflatten_dict(contents, sep="."),
            config=dacite.Config(check_types=False),
        )

    def __len__(self) -> int:
        return self.ray_count


def make_dataset(
    dataset_type: Literal["blender", "nerfstudio"],
    split: Literal["train", "test", "val"],
    dataset_root: Path,
) -> NerfDataset:
    if dataset_type == "blender":
        return BlenderDataset(dataset_root, split)
    elif dataset_type == "nerfstudio":
        assert split == "train"
        return NerfstudioDataset(dataset_root)
    else:
        assert_never(dataset_type)


@dataclasses.dataclass(frozen=True)
class NerfstudioDataset:
    dataset_root: Path

    def load_all_training_rays(self) -> RenderedRays:
        metadata = self._get_metadata()

        camera_model = metadata.get("camera_model", "OPENCV")
        if camera_model == "OPENCV":
            dist_coeffs = onp.array(
                [metadata.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")]
            )
        elif camera_model == "OPENCV_FISHEYE":
            dist_coeffs = onp.array([metadata[k] for k in ("k1", "k2", "k3", "k4")])
        else:
            assert False, f"Unsupported camera model {camera_model}."

        image_paths = tuple(
            map(
                lambda frame: self.dataset_root / frame["file_path"],
                metadata["frames"],
            )
        )
        out: List[RenderedRays] = []
        for i, image, camera in zip(
            range(len(image_paths)),
            _threaded_image_fetcher(image_paths),
            tqdm(
                self.get_cameras(),
                desc=f"Loading {self.dataset_root.stem}",
            ),
        ):
            h, w = image.shape[:2]
            orig_h, orig_w = h, w

            # Resize image to some heuristic target dimension.
            # - We don't want iamges any smaller than 800 x 600.
            # - We aim for having enough pixels for 30k iterations with batch size 4096.
            target_pixels = max(600 * 800, 30_000 * 4_096 / len(image_paths))
            scale = 1.0
            if h * w > target_pixels:
                scale = onp.sqrt(target_pixels / (h * w))
                h = int(h * scale)
                w = int(w * scale)
                image = cv2.resize(image, (w, h))  # type: ignore

            image = (image / 255.0).astype(onp.float32)

            # (2, w, h) => (h, w, 2) => (h * w, 2)
            orig_image_points = (
                onp.mgrid[:w, :h].T.reshape((h * w, 2))
                / onp.array([h, w])
                * onp.array([orig_h, orig_w])
            )

            if camera_model == "OPENCV":
                ray_directions = cv2.undistortPoints(  # type: ignore
                    src=orig_image_points,
                    cameraMatrix=camera.K,
                    distCoeffs=dist_coeffs,
                ).squeeze(axis=1)
            elif camera_model == "OPENCV_FISHEYE":
                ray_directions = cv2.fisheye.undistortPoints(  # type: ignore
                    distorted=orig_image_points[:, None, :],
                    K=camera.K,
                    D=dist_coeffs,
                ).squeeze(axis=1)
            else:
                assert False

            assert ray_directions.shape == (h * w, 2)
            ray_directions = onp.concatenate(
                [ray_directions, onp.ones((h * w, 1))], axis=-1
            )
            ray_directions /= onp.linalg.norm(ray_directions, axis=-1, keepdims=True)
            assert ray_directions.shape == (h * w, 3)

            T_world_camera = camera.T_camera_world.inverse()
            out.append(
                RenderedRays(
                    colors=image.reshape((-1, 3)),  # type: ignore
                    rays_wrt_world=cameras.Rays3D(
                        origins=onp.tile(  # type: ignore
                            T_world_camera.translation()[None, :], (h * w, 1)
                        ),
                        directions=ray_directions @ onp.array(T_world_camera.rotation().as_matrix().T),  # type: ignore
                        camera_indices=onp.full(
                            shape=(h * w),
                            fill_value=i,
                            dtype=onp.uint32,
                        ),  # type: ignore
                    ),
                )
            )

        return jax.tree_map(lambda *leaves: onp.concatenate(leaves, axis=0), *out)

    def get_cameras(self) -> List[cameras.Camera]:
        # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
        # convention.
        T_blendercam_camera = jaxlie.SE3.from_rotation(
            jaxlie.SO3.from_x_radians(onp.pi)
        )

        metadata = self._get_metadata()

        @dataclasses.dataclass
        class Frame:
            transform_matrix: onp.ndarray
            fl_x: float
            fl_y: float
            cx: float
            cy: float
            w: float
            h: float

        frames = []
        for frame in metadata["frames"]:
            # Expected keys in each frame.
            assert "file_path" in frame.keys()
            assert "transform_matrix" in frame.keys()

            if "fl_x" in frame:
                frame = Frame(
                    transform_matrix=onp.array(
                        frame["transform_matrix"], dtype=onp.float32
                    ),
                    **{k: frame[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
                )
            else:
                frame = Frame(
                    transform_matrix=onp.array(
                        frame["transform_matrix"], dtype=onp.float32
                    ),
                    **{k: metadata[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
                )
            frames.append(frame)
            assert frames[-1].transform_matrix.shape == (4, 4)  # Should be in SE(3).

        # Re-orient poses based on up direction.
        up = onp.mean([frame.transform_matrix[:3, 1] for frame in frames], axis=0)
        up /= onp.linalg.norm(up)
        rot_axis = onp.cross(up, onp.array([0.0, 0.0, 1.0]))
        rot_angle = onp.arccos(onp.dot(up, onp.array([0.0, 0.0, 1.0])))
        T_worldfix_world = onp.array(
            jaxlie.SE3.from_rotation(jaxlie.SO3.exp(rot_angle * rot_axis)).as_matrix()
        )
        for frame in frames:
            frame.transform_matrix = T_worldfix_world @ frame.transform_matrix

        # Compute pose bounding box.
        positions = onp.array([f.transform_matrix for f in frames])[:, :3, 3]
        aabb_min = positions.min(axis=0)
        aabb_max = positions.max(axis=0)
        del positions
        assert aabb_min.shape == aabb_max.shape == (3,)

        out = []
        for frame in frames:
            # Center and scale scene. In the future we might also auto-orient it.
            transform_matrix = frame.transform_matrix.copy()
            transform_matrix[:3, 3] -= aabb_min
            transform_matrix[:3, 3] -= (aabb_max - aabb_min) / 2.0
            transform_matrix[:3, 3] /= (aabb_max - aabb_min).max() / 2.0

            camera_matrix = onp.eye(3)
            camera_matrix[0, 0] = frame.fl_x
            camera_matrix[1, 1] = frame.fl_y
            camera_matrix[0, 2] = frame.cx
            camera_matrix[1, 2] = frame.cy

            # Compute extrinsics.
            T_world_blendercam = jaxlie.SE3.from_matrix(transform_matrix)
            T_camera_world = (T_world_blendercam @ T_blendercam_camera).inverse()

            out.append(
                cameras.Camera(
                    K=camera_matrix,  # type: ignore
                    T_camera_world=T_camera_world,
                    image_width=frame.w,
                    image_height=frame.h,
                )
            )

        return out

    def _get_metadata(self) -> Dict[str, Any]:
        """Read metadata: image paths, transformation matrices, FOV."""
        with open(self.dataset_root / "transforms.json") as f:
            metadata: dict = json.load(f)
        return metadata


@dataclasses.dataclass(frozen=True)
class BlenderDataset:
    dataset_root: Path
    split: Literal["train", "test", "val"]

    def load_all_training_rays(self) -> RenderedRays:
        return rendered_rays_from_views(self.registered_views)

    def get_cameras(self) -> List[cameras.Camera]:
        return self._cameras

    @functools.cached_property
    def _cameras(self) -> List[cameras.Camera]:
        metadata = self._get_metadata()

        transform_matrices: List[onp.ndarray] = []
        for frame in metadata["frames"]:
            assert frame.keys() == {"file_path", "rotation", "transform_matrix"}

            transform_matrices.append(
                onp.array(frame["transform_matrix"], dtype=onp.float32)
            )
            assert transform_matrices[-1].shape == (4, 4)  # Should be in SE(3).
        fov_x_radians: float = metadata["camera_angle_x"]

        # Image dimensions. We assume all images are the same.
        height, width = iio.imread(
            self.dataset_root / (metadata["frames"][0]["file_path"] + ".png")
        ).shape[:2]
        del metadata

        # Transformation from Blender camera coordinates to OpenCV ones. We like the OpenCV
        # convention.
        T_blendercam_camera = jaxlie.SE3.from_rotation(
            jaxlie.SO3.from_x_radians(onp.pi)
        )

        out = []
        for transform_matrix in transform_matrices:
            T_world_blendercam = jaxlie.SE3.from_matrix(transform_matrix)
            T_camera_world = (T_world_blendercam @ T_blendercam_camera).inverse()
            out.append(
                cameras.Camera.from_fov(
                    T_camera_world=T_camera_world,
                    image_width=width,
                    image_height=height,
                    fov_x_radians=fov_x_radians,
                )
            )
        return out

    @functools.cached_property
    def registered_views(self) -> List[RegisteredRgbaView]:
        metadata = self._get_metadata()

        image_paths: List[Path] = []
        for frame in metadata["frames"]:
            assert frame.keys() == {"file_path", "rotation", "transform_matrix"}
            image_paths.append(self.dataset_root / f"{frame['file_path']}.png")
        del metadata

        out = []
        for image, camera in zip(
            _threaded_image_fetcher(image_paths),
            tqdm(
                self.get_cameras(),
                desc=f"Loading {self.dataset_root.stem}",
            ),
        ):
            assert image.dtype == onp.uint8
            height, width = image.shape[:2]

            # Note that this is RGBA!
            assert image.shape == (height, width, 4)

            # [0, 255] => [0, 1]
            image = (image / 255.0).astype(onp.float32)

            # Compute extrinsics.
            out.append(
                RegisteredRgbaView(
                    image_rgba=image,  # type: ignore
                    camera=camera,
                )
            )
        return out

    def _get_metadata(self) -> Dict[str, Any]:
        """Read metadata: image paths, transformation matrices, FOV."""
        with open(self.dataset_root / f"transforms_{self.split}.json") as f:
            metadata: dict = json.load(f)
        return metadata


# Helpers.


@jdc.pytree_dataclass
class RegisteredRgbaView(jdc.EnforcedAnnotationsMixin):
    """Structure containing 2D image + camera pairs."""

    image_rgba: Annotated[
        jax.Array,
        jnp.floating,  # Range of contents is [0, 1].
    ]
    camera: cameras.Camera


@jdc.pytree_dataclass
class RenderedRays(jdc.EnforcedAnnotationsMixin):
    """Structure containing individual 3D rays in world space + colors."""

    colors: Annotated[jax.Array, (3,), jnp.floating]
    rays_wrt_world: cameras.Rays3D


def rendered_rays_from_views(views: List[RegisteredRgbaView]) -> RenderedRays:
    """Convert a list of registered 2D views into a pytree containing individual
    rendered rays."""

    out = []
    for i, view in enumerate(views):
        height = view.camera.image_height
        width = view.camera.image_width

        rays = jax.tree_map(
            onp.array,
            view.camera.pixel_rays_wrt_world(camera_index=i),
        )
        assert rays.get_batch_axes() == (height, width)

        rgba = view.image_rgba
        assert rgba.shape == (height, width, 4)

        # Add white background color; this is what the standard alpha compositing over
        # operator works out to with an opaque white background.
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4])

        out.append(
            RenderedRays(
                colors=rgb.reshape((-1, 3)),
                rays_wrt_world=cameras.Rays3D(
                    origins=rays.origins.reshape((-1, 3)),
                    directions=rays.directions.reshape((-1, 3)),
                    camera_indices=rays.camera_indices.reshape((-1,)),
                ),
            )
        )

    out_concat: RenderedRays = jax.tree_map(
        lambda *children: onp.concatenate(children, axis=0), *out
    )

    # Shape of rays should (N,3), colors should be (N,4), etc.
    assert len(out_concat.rays_wrt_world.get_batch_axes()) == 1
    return out_concat


T = TypeVar("T", bound=Iterable)


def _threaded_image_fetcher(paths: Iterable[Path]) -> Iterable[onp.ndarray]:
    """Maps an iterable over image paths to an iterable over image arrays, which are
    opened via PIL.

    Helpful for parallelizing IO."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for image in executor.map(
            iio.imread,
            paths,
            chunksize=4,
        ):
            yield image


def compute_evaluation_metrics(
    test_dataset: NerfDataset, state: TrainState
) -> Tuple[Dict[str, float], List[onp.ndarray]]:
    assert isinstance(test_dataset, BlenderDataset)

    # Render.
    images_pred = []
    images_label = []
    for i, view in enumerate(tqdm(test_dataset.registered_views)):
        images_pred.append(
            state.render_camera(view.camera, camera_index=0, eval_mode=True, mode="rgb")
        )
        images_label.append(
            view.image_rgba[..., :3] * view.image_rgba[..., 3:4]
            + (1.0 - view.image_rgba[..., 3:4])
        )

    # Compute metrics.
    lpips = lpips_jax.LPIPSEvaluator(replicate=False, net="alexnet")
    eval_mses = []
    eval_psnrs = []
    eval_lpips = []
    eval_ssims = []
    for pred, label in zip(images_pred, images_label):
        mse = onp.mean((onp.array(pred) - onp.asarray(label)) ** 2)
        eval_mses.append(mse)
        eval_psnrs.append(-10.0 * onp.log(mse) / onp.log(10.0))
        eval_lpips.append(jax.jit(lpips)(pred[None], label[None]).squeeze(axis=0))
        eval_ssims.append(jax.jit(pix.ssim)(pred, label))

    return {
        "mse": float(onp.mean(eval_mses)),
        "psnr": float(onp.mean(eval_psnrs)),
        "lpips": float(onp.mean(eval_lpips)),
        "ssim": float(onp.mean(eval_ssims)),
    }, images_pred
