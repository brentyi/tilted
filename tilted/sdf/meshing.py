"""Data utilities for SDFs."""
import os

# Use EGL for offscreen rendering. This will probably break attempts at using pyrender
# for GUI stuff.
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
from pathlib import Path
from typing import Optional, Tuple

import jaxlie
import matplotlib as mpl
import mcubes
import numpy as onp
import pyrender
import trimesh
import trimesh.repair
from jaxtyping import Float


def export_sdf_as_mesh(
    sdf_grid: Float[onp.ndarray, "res res res"],
    out_path: Path,
    aabb_min: Float[onp.ndarray, "3"] = -onp.ones(3),
    aabb_max: Float[onp.ndarray, "3"] = onp.ones(3),
) -> None:
    # Run marching cube on zero level set.
    assert len(sdf_grid.shape) == 3
    threshold = 0.0
    vertices, triangles = mcubes.marching_cubes(sdf_grid, threshold)
    assert vertices.shape[-1] == 3
    vertices = (
        vertices / (onp.array(sdf_grid.shape) - 1.0) * (aabb_max - aabb_min) + aabb_min
    )

    # Export as mesh.
    mesh = trimesh.Trimesh(vertices, triangles, process=True)

    # Repair mesh. Blender is perfectly happy with the original meshes, but pyrender
    # previews have some clear issues if we don't repair the mesh.
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    mesh.export(out_path)


@functools.lru_cache
def _get_default_camera_pose() -> onp.ndarray:
    return onp.array(
        (
            jaxlie.SE3.from_rotation(
                jaxlie.SO3.exp(onp.array([onp.pi / 16.0, onp.pi / 8.0, 0.0]))
            )
            @ jaxlie.SE3.from_rotation(jaxlie.SO3.exp(onp.array([0.0, onp.pi, 0.0])))
            @ jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.identity(),
                translation=onp.array([0.0, 0.0, 1.75]),
            )
        ).as_matrix()
    )


def render_mesh(
    mesh_path: Path,
    resolution: Tuple[int, int] = (400, 400),
    depth_cmap: Optional[str] = "hot",
    normalize: bool = False,
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Returns rendered color + depth maps."""

    mesh = trimesh.load(mesh_path)
    assert isinstance(mesh, trimesh.Trimesh)

    if normalize:
        # Normalize to [-1, 1].
        vs = mesh.vertices
        vmin = vs.min(axis=0)
        vmax = vs.max(axis=0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / onp.sqrt(onp.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        mesh.vertices = vs

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=onp.pi / 3.0, aspectRatio=1.0)
    camera_pose = _get_default_camera_pose()

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(
        color=onp.array([1.0, 0.8, 0.8]),
        intensity=10.0,
        innerConeAngle=onp.pi / 16.0,
        outerConeAngle=onp.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(*resolution)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)  # type: ignore

    if depth_cmap is not None:
        depth -= depth.min()
        depth /= depth.max() + 1e-3
        depth = (mpl.colormaps[depth_cmap](depth) * 255).astype(onp.uint8)

    return color, depth
