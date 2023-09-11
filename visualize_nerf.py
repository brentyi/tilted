"""Visualization utility for rendering from a NeRF and previewing in our web browser."""

from __future__ import annotations

import os
import time

import matplotlib as mpl
import viser.transforms as tf
from typing_extensions import assert_never

from render_panel import populate_render_tab

# Visualization is pretty lightweight, and only runs when we move the camera. Let's make
# sure we can run multiple visualizers on the same GPU.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import pathlib
import threading
from typing import Dict, List, Literal, Optional, Tuple, cast

import fifteen
import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import tyro
import viser
from jax import numpy as jnp

from tilted.core.factored_grid import Learnable3DProjectersBase
from tilted.nerf.cameras import Camera, Rays3D
from tilted.nerf.render import viz_dist
from tilted.nerf.train_state import NerfConfig, TrainState


def get_transform_count(state: TrainState) -> int:
    """Get number of learned transforms for a TILTED model. Returns 1 for axis-aligned
    decompositions."""
    projecters = state.params.primary_field.grid.projecters
    if isinstance(projecters, Learnable3DProjectersBase):
        return projecters.transform_count
    else:
        return 1


train_state_cache: Dict[str, TrainState] = {}


@dataclasses.dataclass
class ClientRenderState:
    client: viser.ClientHandle

    state: Literal["ready", "rendering", "done"]
    mode: Literal["rgb", "dist", "transform_feature_norm", "feature_pca"]
    cmap: str
    gui_status: viser.GuiInputHandle[str]
    transform_viz_index: int
    stage: int

    ray_queue: List[Rays3D]
    render_hw: Tuple[int, int]
    rendered_chunks: List[jax.Array]

    last_update_timestamp: float

    sharded_train_state: Optional[TrainState]
    devices: List[jax.Device]

    @staticmethod
    def setup(
        client: viser.ClientHandle,
        mode: Literal["rgb", "dist", "transform_feature_norm", "feature_pca"],
        cmap: str,
        gui_status: viser.GuiInputHandle[str],
    ) -> ClientRenderState:
        out = ClientRenderState(
            client,
            "ready",
            mode=mode,
            cmap=cmap,
            gui_status=gui_status,
            transform_viz_index=0,
            stage=0,
            ray_queue=[],
            render_hw=(0, 0),
            rendered_chunks=[],
            last_update_timestamp=0.0,
            sharded_train_state=None,
            devices=[],
        )
        return out

    def reset(self) -> None:
        self.last_update_timestamp = self.client.camera.update_timestamp
        self.state = "ready"
        self.stage = 0
        self.ray_queue = []
        self.rendered_chunks = []

    def step(self) -> None:
        camera = self.client.camera
        sharded_train_state = self.sharded_train_state
        devices = self.devices

        if sharded_train_state is None:
            return

        # How many rays to render.
        stage_ray_counts = (4096 * 8, 4096 * 64, 4096 * 256)
        chunk_size = 2048 * len(devices)

        if camera.update_timestamp != self.last_update_timestamp and self.stage > 0:
            self.reset()

        if self.state == "ready":
            self.gui_status.value = "Preparing rays"

            # Ready to start rendering. We'll load a queue of rays that need to be
            # rendered for a particular resolution.
            assert len(self.ray_queue) == 0

            # How many rays should we render for this stage?
            target_rays = stage_ray_counts[self.stage]
            scale = onp.sqrt(target_rays / camera.aspect)
            image_width = int(camera.aspect * scale)
            image_height = int(scale)
            self.render_hw = (image_height, image_width)

            # Get flattened rays.
            total_rays = image_width * image_height
            rays_wrt_world = jax.tree_map(
                lambda a: onp.array(a.reshape((total_rays, *a.shape[2:]))),
                get_camera(
                    camera.wxyz,
                    camera.position,
                    image_width,
                    image_height,
                    camera.fov,
                ).pixel_rays_wrt_world(camera_index=0),
            )

            self.ray_queue = []
            start = 0
            while start < total_rays:
                ray_batch = jax.tree_map(
                    lambda a: self.slice_padded(a, start=start, chunk_size=chunk_size),
                    rays_wrt_world,
                )
                ray_batch = jax.tree_map(
                    lambda a: jax.device_put_sharded(
                        list(
                            a.reshape(
                                (len(devices), a.shape[0] // len(devices), *a.shape[1:])
                            )
                        ),
                        devices,
                    ),
                    ray_batch,
                )
                self.ray_queue.append(ray_batch)
                start += chunk_size

            self.state = "rendering"

        elif self.state == "rendering":
            self.gui_status.value = "Rendering"

            # Render one batch of rays.
            ray_batch = self.ray_queue.pop()
            self.rendered_chunks.append(
                jax.pmap(
                    TrainState.render_rays,
                    static_broadcasted_argnums=(2, 3),
                )(
                    sharded_train_state,
                    ray_batch,
                    True,
                    "features" if self.mode == "feature_pca" else self.mode,
                ).reshape((ray_batch.origins.shape[0] * ray_batch.origins.shape[1], -1))
            )
            if len(self.ray_queue) == 0:
                # Done!
                rendered_rays = onp.concatenate(self.rendered_chunks[::-1], axis=0)
                assert rendered_rays.shape[0] >= self.render_hw[0] * self.render_hw[1]

                # Trim padding.
                image = self.image_from_rendered_rays(rendered_rays, *self.render_hw)

                # Send the image along.
                self.client.set_background_image(image, "png")

                # When done.
                self.stage += 1
                if self.stage < len(stage_ray_counts):
                    self.state = "ready"
                else:
                    self.state = "done"
        elif self.state == "done":
            # Nothing to do!
            self.gui_status.value = "Done"
            pass
        else:
            assert_never(self.state)

    def image_from_rendered_rays(
        self, rendered_rays: onp.ndarray, height: int, width: int
    ) -> onp.ndarray:
        assert rendered_rays.shape[0] >= height * width

        # Trim padding.
        rendered_rays = rendered_rays[: height * width]
        image = rendered_rays.reshape((height, width, -1))

        # Color mapping for distance and norm maps.
        if self.mode == "dist":
            assert image.shape[-1] == 1
            image = image.squeeze(axis=-1)
            assert image.shape == (height, width)
            image = viz_dist(image, self.cmap)

        if self.mode == "transform_feature_norm":
            image = image - image.min()
            image /= image.max()
            image = image[
                ...,
                onp.argsort(
                    -onp.linalg.norm(image.reshape((-1, image.shape[-1])), axis=0)
                )[self.transform_viz_index % image.shape[-1]],
            ]
            image = (mpl.colormaps[self.cmap](image) * 255.0).astype(onp.uint8)

        if self.mode == "feature_pca":
            X = image.reshape((height * width, -1))
            X = X - onp.mean(X, axis=0, keepdims=True)  # type: ignore
            eigenvalues, eigenvectors = onp.linalg.eigh(onp.cov(X.T))
            ind = onp.argsort(-eigenvalues)[:3]
            top_3 = eigenvectors[:, ind]
            assert top_3.shape == (X.shape[-1], 3)

            image = X @ top_3
            image = image / onp.sqrt(eigenvalues[ind[0]]) / 3.0
            image = onp.clip(image + 0.5, 0.0, 1.0)
            # image = image[
            #     ..., onp.argsort(onp.mean(image, axis=0, keepdims=True))
            # ]
            image = (image * 255.0).astype(onp.uint8)
            image = image.reshape((height, width, 3))

        return image

    @staticmethod
    def slice_padded(a: onp.ndarray, start: int, chunk_size: int) -> onp.ndarray:
        """Slice an array and pad to match chunk size. Padding minimizes JIT
        overhead."""
        items_until_end = a.shape[0] - start

        if items_until_end >= chunk_size:
            out = a[start : start + chunk_size]
        else:
            # We shouldn't be padding too much...!
            # assert items_until_end < 2048, items_until_end
            out = onp.concatenate(
                [
                    a[start : start + items_until_end],
                    onp.zeros(
                        (chunk_size - items_until_end,) + a.shape[1:],
                        dtype=a.dtype,
                    ),
                ],
                axis=0,
            )
        assert out.shape[0] == chunk_size
        return out

    def render_one_image(
        self, T_world_camera: tf.SE3, height: int, width: int, fov: float
    ) -> onp.ndarray:
        # Get flattened rays.
        total_rays = width * height
        rays_wrt_world = jax.tree_map(
            lambda a: onp.array(a.reshape((total_rays, *a.shape[2:]))),
            get_camera(
                T_world_camera.rotation().wxyz,
                T_world_camera.translation(),
                width,
                height,
                fov,
            ).pixel_rays_wrt_world(camera_index=0),
        )

        devices = self.devices
        sharded_train_state = self.sharded_train_state

        ray_queue = []
        start = 0
        chunk_size = 2048 * len(devices)
        while start < total_rays:
            ray_batch = jax.tree_map(
                lambda a: self.slice_padded(a, start=start, chunk_size=chunk_size),
                rays_wrt_world,
            )
            ray_batch = jax.tree_map(
                lambda a: jax.device_put_sharded(
                    list(
                        a.reshape(
                            (len(devices), a.shape[0] // len(devices), *a.shape[1:])
                        )
                    ),
                    devices,
                ),
                ray_batch,
            )
            ray_queue.append(ray_batch)
            start += chunk_size

        rendered_chunks = []

        while len(ray_queue) > 0:
            ray_batch = ray_queue.pop()
            rendered_chunks.append(
                jax.pmap(
                    TrainState.render_rays,
                    static_broadcasted_argnums=(2, 3),
                )(
                    sharded_train_state,
                    ray_batch,
                    True,
                    "features" if self.mode == "feature_pca" else self.mode,
                ).reshape((ray_batch.origins.shape[0] * ray_batch.origins.shape[1], -1))
            )
        rendered_rays = onp.concatenate(rendered_chunks[::-1], axis=0)
        return self.image_from_rendered_rays(rendered_rays, height, width)


def main(experiment_dir: pathlib.Path, /, port: int = 8080) -> None:
    server = viser.ViserServer(port=port)
    server.world_axes.visible = True

    devices = jax.devices()

    train_state_lock = threading.Lock()
    sharded_train_states: Dict[int, TrainState] = {}
    render_states: Dict[int, ClientRenderState] = {}

    sync_cameras = server.add_gui_checkbox("Sync client cameras", False)
    client_in_control: Optional[int] = None
    client_in_control_reset_time = time.time()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        tab_group = client.add_gui_tab_group()

        main_tab = tab_group.add_tab("Main")
        render_tab = tab_group.add_tab("Render")

        gui_status = client.add_gui_text("Status", initial_value="", disabled=True)
        render_state = ClientRenderState.setup(
            client, mode="rgb", cmap="hot", gui_status=gui_status
        )
        render_states[client.client_id] = render_state

        # Make render tab.
        with render_tab:
            populate_render_tab(server, client, render_state.render_one_image)

        # Put everything else into the main tab.
        main_tab.__enter__()

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            """When the client camera updates..."""

            nonlocal client_in_control
            nonlocal client_in_control_reset_time

            if not sync_cameras.value:
                return

            if (
                client_in_control != client.client_id
                and time.time() < client_in_control_reset_time
            ):
                return

            wxyz = client.camera.wxyz
            position = client.camera.position

            client_in_control = client.client_id
            client_in_control_reset_time = time.time() + 0.1

            # Sync all of the other camera.
            for other_id, other_handle in server.get_clients().items():
                if other_id == client.client_id:
                    # (but not ourselves)
                    continue

                with other_handle.atomic():
                    other_handle.camera.wxyz = wxyz
                    other_handle.camera.position = position

        gui_reset_up = client.add_gui_button("Reset Up Direction")
        gui_experiment_filter = client.add_gui_text("Experiment contains", "")
        gui_experiment = client.add_gui_dropdown(
            "Experiment", tuple(sorted([exp.name for exp in experiment_dir.iterdir()]))
        )
        gui_output_type = client.add_gui_dropdown(
            "Output type",
            ("rgb", "dist", "transform_feature_norm", "feature_pca"),
            initial_value="rgb",
        )
        gui_cmap = client.add_gui_dropdown(
            "Colormap",
            (
                "plasma",
                "viridis",
                "pink",
                "spring",
                "summer",
                "autumn",
                "winter",
                "cool",
                "hot",
                "copper",
            ),
            initial_value="hot",
            disabled=True,
        )
        gui_transform_index: Optional[
            viser.GuiInputHandle[int]
        ] = None  # Set on train state load.

        @gui_experiment_filter.on_update
        def _(_) -> None:
            ...
            gui_experiment.options = tuple(
                sorted(
                    [
                        exp.name
                        for exp in experiment_dir.iterdir()
                        if gui_experiment_filter.value in exp.name
                    ]
                )
            )

        @gui_output_type.on_update
        def _(_) -> None:
            with train_state_lock:
                render_states[client.client_id].mode = gui_output_type.value
                render_states[client.client_id].reset()

                if gui_transform_index is not None:
                    gui_transform_index.disabled = (
                        gui_output_type.value != "transform_feature_norm"
                    )
                gui_cmap.disabled = gui_output_type.value == "rgb"

        @gui_cmap.on_update
        def _(_) -> None:
            with train_state_lock:
                render_states[client.client_id].cmap = gui_cmap.value
                render_states[client.client_id].reset()

        @gui_reset_up.on_click
        def _(_) -> None:
            print(f"Setting up direction for client {client.client_id}!")
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
                [0.0, -1.0, 0.0]
            )

        # Checkpoint loading.
        @gui_experiment.on_update
        def load_train_state(_) -> None:
            with train_state_lock:
                experiment = gui_experiment.value
                # Use cached train state if possible.
                train_state = train_state_cache.get(experiment, None)

                gui_status.value = "Restoring checkpoint..."
                if train_state is None:
                    # Load training state.
                    exp = fifteen.experiments.Experiment(experiment_dir / experiment)
                    config = exp.read_metadata("config", NerfConfig)

                    # Overwrite the near/far bounds...
                    config = dataclasses.replace(
                        config,
                        render_config=dataclasses.replace(
                            config.render_config,
                            near=min(0.05, config.render_config.near),
                            far=max(12.0, config.render_config.far),
                        ),
                    )

                    train_state = TrainState.make(config)
                    train_state = exp.restore_checkpoint(train_state)

                    # Cache train state.
                    train_state_cache[experiment] = train_state
                    if len(train_state_cache) > 30:
                        train_state_cache.pop(next(iter(train_state_cache.keys())))
                else:
                    print("Cache hit!")

                sharded_train_state = cast(
                    TrainState, jax.device_put_replicated(train_state, devices)
                )
                del train_state

                sharded_train_states[client.client_id] = sharded_train_state
                render_states[
                    client.client_id
                ].sharded_train_state = sharded_train_state
                render_states[client.client_id].devices = devices
                render_states[client.client_id].reset()

                nonlocal gui_transform_index
                if gui_transform_index is not None:
                    gui_transform_index.remove()
                gui_transform_index = client.add_gui_slider(
                    "Transform #",
                    min=0,
                    max=get_transform_count(sharded_train_state) - 1,
                    initial_value=0,
                    step=1,
                    disabled=gui_output_type.value != "transform_feature_norm",
                )

                @gui_transform_index.on_update
                def _(_) -> None:
                    with train_state_lock:
                        assert gui_transform_index is not None
                        render_states[
                            client.client_id
                        ].transform_viz_index = gui_transform_index.value
                        render_states[client.client_id].reset()

                gui_status.value = "Ready"

        load_train_state(None)  # type: ignore

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        sharded_train_states.pop(client.client_id)
        render_states.pop(client.client_id)

    while True:
        # Step each render state of each client.
        for id in server.get_clients().keys():
            render_state = render_states.get(id, None)
            if render_state is None:
                continue

            with train_state_lock:
                assert sharded_train_states[id] is not None
                render_state.step()

        time.sleep(1e-2)


@jdc.jit
def get_camera(
    wxyz: onp.ndarray,
    position: onp.ndarray,
    image_width: jdc.Static[int],
    image_height: jdc.Static[int],
    fov: float,
) -> Camera:
    T_world_cam = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.array(wxyz)),
        position / 4.0,
    )
    return Camera.from_fov(
        T_camera_world=T_world_cam.inverse(),
        image_width=image_width,
        image_height=image_height,
        fov_y_radians=fov,
    )


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    tyro.cli(main)
