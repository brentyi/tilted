from pathlib import Path
from typing import Optional, TypedDict

import fifteen
import h5py
import numpy as onp
import pysdf
import trimesh
from jaxtyping import Float


class SdfDataDict(TypedDict):
    sdfs: Float[onp.ndarray, "batch 1"]
    points: Float[onp.ndarray, "batch 3"]


class CachedSdfDataloader:
    """Dataloader for SDF files."""

    def __init__(self, hdf5_path: Path, minibatch_size: int) -> None:
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.point_count = self.hdf5_file["train_points"].shape[0]  # type: ignore
        self.minibatch_size = minibatch_size

    def minibatch_count(self) -> int:
        return self.point_count // self.minibatch_size

    def minibatches(
        self, shuffle_seed: Optional[int]
    ) -> fifteen.data.SizedIterable[SdfDataDict]:
        class _Inner:
            def __iter__(_self):
                minibatch_indices = onp.random.default_rng(shuffle_seed).permutation(
                    self.minibatch_count()
                )
                for i in minibatch_indices:
                    indexer = slice(
                        i * self.minibatch_size, (i + 1) * self.minibatch_size
                    )
                    yield {
                        "sdfs": self.hdf5_file["train_sdfs"][indexer],  # type: ignore
                        "points": self.hdf5_file["train_points"][indexer],  # type: ignore
                    }

            def __len__(_self):
                return self.minibatch_count()

        return _Inner()  # type: ignore


def write_hdf5_from_mesh(
    mesh_path: Path,
    output_path: Path,
    num_train_points: int = 8_000_000,
    num_eval_points: int = 16_000_000,
) -> None:
    """Load a mesh, sample points, and write train/eval sets to an hdf5 file."""

    rng = onp.random.default_rng(0)

    mesh = trimesh.load(mesh_path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    assert output_path.suffix == ".hdf5"

    # Normalize to [-1, 1].
    vs = mesh.vertices
    vmin = vs.min(axis=0)
    vmax = vs.max(axis=0)
    v_center = (vmin + vmax) / 2
    v_scale = 2 / onp.sqrt(onp.sum((vmax - vmin) ** 2)) * 0.95
    vs = (vs - v_center[None, :]) * v_scale
    mesh.vertices = vs

    # Initialize SDF.
    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)

    # Compute training data.
    # Note: Factor Fields claims to do 20% surface points, instant-ngp does 1/8.
    print("Computing train data...")
    num_train_points_uniform = num_train_points // 5
    num_train_points_surface = num_train_points - num_train_points_uniform

    train_points_uniform = rng.uniform(
        low=-1.0, high=1.0, size=(num_train_points_uniform, 3)
    )
    train_points_surface = mesh.sample(num_train_points_surface, return_index=False)
    assert isinstance(train_points_surface, onp.ndarray)
    train_points_surface[num_train_points // 2 :, :] += rng.normal(
        loc=0.0, scale=0.01, size=(num_train_points_surface - num_train_points // 2, 3)
    )

    train_points = onp.concatenate([train_points_surface, train_points_uniform])
    train_sdfs = onp.zeros((num_train_points, 1))
    train_sdfs[num_train_points // 2 :] = -sdf_fn(
        train_points[num_train_points // 2 :]
    )[:, None]

    # Shuffle training data.
    indices = rng.permutation(num_train_points)
    train_points = train_points[indices, :]
    train_sdfs = train_sdfs[indices, :]

    # Compute eval data.
    print("Computing eval data...")
    eval_points = rng.uniform(low=-1.0, high=1.0, size=(num_eval_points, 3))
    eval_sdfs = -sdf_fn(eval_points)[:, None]

    # Check shapes.
    assert train_points.shape == (num_train_points, 3)
    assert train_sdfs.shape == (num_train_points, 1)
    assert eval_points.shape == (num_eval_points, 3)
    assert eval_sdfs.shape == (num_eval_points, 1)

    print("Writing to hdf5...")
    with h5py.File(output_path, mode="w") as f:
        f.create_dataset(name="train_points", data=train_points, chunks=(4096, 3))
        f.create_dataset(name="train_sdfs", data=train_sdfs, chunks=(4096, 1))
        f.create_dataset(name="eval_points", data=eval_points, chunks=(4096, 3))
        f.create_dataset(name="eval_sdfs", data=eval_sdfs, chunks=(4096, 1))
