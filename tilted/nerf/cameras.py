from typing import Literal, Union

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from jax import numpy as jnp
from typing_extensions import Annotated


@jdc.pytree_dataclass
class Rays3D(jdc.EnforcedAnnotationsMixin):
    """Structure defining some rays in 3D space. Should contain origin and direction
    arrays of the same shape; `(*, 3)`."""

    origins: Annotated[jax.Array, (3,), jnp.floating]
    directions: Annotated[jax.Array, (3,), jnp.floating]
    camera_indices: Annotated[
        jax.Array, (), jnp.uint32
    ]  # Used for per-camera appearance embeddings.

    def points_from_ts(self, ts: jax.Array) -> jax.Array:
        """Given an array of scalar distances of shape (*batch_axes, num_samples),
        compute a set of points of shape (*batch_axes, num_samples, 3)."""

        num_samples = ts.shape[-1]
        #  assert ts.shape == (*self.get_batch_axes(), num_samples)
        points = (
            self.origins[..., None, :]
            + ts[..., :, None] * self.directions[..., None, :]
        )
        assert points.shape == (*self.get_batch_axes(), num_samples, 3)
        return points


@jdc.pytree_dataclass
class Camera(jdc.EnforcedAnnotationsMixin):
    K: Annotated[jax.Array, (3, 3), jnp.floating]
    """Intrinsics. alpha * [u v 1]^T = K @ [x_c y_c z_c]^T"""

    T_camera_world: jaxlie.SE3
    """Extrinsics."""

    image_width: jdc.Static[int]
    image_height: jdc.Static[int]

    @staticmethod
    def from_fov(
        T_camera_world: jaxlie.SE3,
        image_width: int,
        image_height: int,
        fov_x_radians: Union[float, jax.Array, None] = None,
        fov_y_radians: Union[float, jax.Array, None] = None,
    ) -> "Camera":
        """Initialize camera parameters from FOV. At least one of `fov_x_radians` or
        `fov_y_radians` must be passed in."""
        # Offset by 1/2 pixel because (0,0) in pixel space corresponds actually to a
        # square whose upper-left corner is (0,0), and bottom-right corner is (1,1).
        cx = image_width / 2.0 - 0.5
        cy = image_height / 2.0 - 0.5

        fx = None
        fy = None

        if fov_x_radians is not None:
            fx = (image_width / 2.0) / jnp.tan(fov_x_radians / 2.0)
        if fov_y_radians is not None:
            fy = (image_height / 2.0) / jnp.tan(fov_y_radians / 2.0)

        if fx is None:
            assert fy is not None
            fx = fy
        if fy is None:
            assert fx is not None
            fy = fx

        K = jnp.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )
        return Camera(
            K=K,
            T_camera_world=T_camera_world,
            image_width=image_width,
            image_height=image_height,
        )

    @jdc.jit
    def compute_fov_x_radians(self) -> jax.Array:
        fx = self.K[0, 0]
        return 2.0 * jnp.arctan((self.image_width / 2.0) / fx)

    @jdc.jit
    def compute_fov_y_radians(self) -> jax.Array:
        fy = self.K[1, 1]
        return 2.0 * jnp.arctan((self.image_height / 2.0) / fy)

    @jdc.jit
    def resize_with_fixed_fov(
        self,
        image_width: jdc.Static[int],
        image_height: jdc.Static[int],
        fixed_axis: jdc.Static[Literal["x", "y", "both"]] = "both",
    ) -> "Camera":
        return Camera.from_fov(
            self.T_camera_world,
            image_width=image_width,
            image_height=image_height,
            fov_x_radians=self.compute_fov_x_radians()
            if fixed_axis in ("x", "both")
            else None,
            fov_y_radians=self.compute_fov_y_radians()
            if fixed_axis in ("y", "both")
            else None,
        )

    @jdc.jit
    def ray_wrt_world_from_uv(self, u: float, v: float, camera_index: int) -> Rays3D:
        """Input is a scalar u/v coordinate. Output is a Rays struct, with origin and
        directions of shape (3,).,"""

        # 2D -> 3D projection: `R_world_camera @ K^-1 @ [u v 1]^T`.
        uv_coord_homog = jnp.array([u, v, 1.0])
        T_world_camera = self.T_camera_world.inverse()
        ray_direction_wrt_world = (
            T_world_camera.rotation().as_matrix()
            @ jnp.linalg.inv(self.K)
            @ uv_coord_homog
        )
        assert ray_direction_wrt_world.shape == (3,)

        ray_direction_wrt_world /= jnp.linalg.norm(ray_direction_wrt_world) + 1e-8
        rays_wrt_world = Rays3D(
            origins=T_world_camera.translation(),  # type: ignore
            directions=ray_direction_wrt_world,
            camera_indices=jnp.array(camera_index, dtype=jnp.uint32),
        )
        return rays_wrt_world

    @jdc.jit
    def pixel_rays_wrt_world(self, camera_index: int) -> Rays3D:
        # Get width and height of image.
        image_width = self.image_width
        image_height = self.image_height

        # Get image-space uv coordinates.
        v, u = onp.mgrid[:image_height, :image_width]
        assert u.shape == v.shape == (image_height, image_width)

        # Compute world-space rays.
        rays_wrt_world = jax.vmap(
            jax.vmap(lambda u, v: self.ray_wrt_world_from_uv(u, v, camera_index))
        )(u, v)
        assert rays_wrt_world.get_batch_axes() == (image_height, image_width)
        return rays_wrt_world
