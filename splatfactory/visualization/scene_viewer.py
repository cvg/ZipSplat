"""Interactive viser viewer for a Gaussian scene (a saved Gaussians tensor or a gsplat checkpoint).

Usage:
    python -m splatfactory.visualization.scene_viewer --file gaussians.pt
"""

import atexit
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import viser
from nerfview import CameraState, RenderTabState, Viewer, apply_float_colormap
from nerfview._renderer import Renderer, RenderTask

from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.utils import tools


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    render_mode: Literal["rgb", "depth(accumulated)", "depth(expected)", "alpha"] = "rgb"
    colormap: Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"] = "turbo"

    min_opacity: float = 0.0
    max_scale: float = 10.0
    scale_factor: float = 1.0


class GsplatViewer(Viewer):
    """Viewer for gsplat."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        show_trajectory_options: bool = False,
    ):
        self.show_trajectory_options = show_trajectory_options

        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _connect_client(self, client: viser.ClientHandle):
        """Override to use custom renderer and set initial camera pose."""
        client.camera.position = np.array([0.0, 0.0, -2.0])
        client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        client_id = client.client_id
        self._renderers[client_id] = Renderer(viewer=self, client=client, lock=self.lock)
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                # slider for min opacity
                min_opacity_slider = server.gui.add_slider(
                    "Min Opacity",
                    min=0.0,
                    max=1.0,
                    initial_value=self.render_tab_state.min_opacity,
                    step=0.01,
                    hint="Minimum opacity threshold for rendering splats.",
                )

                @min_opacity_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.min_opacity = min_opacity_slider.value
                    self.rerender(_)

                # slider for max scale
                max_scale_slider = server.gui.add_slider(
                    "Max Scale",
                    min=0.1,
                    max=10.0,
                    initial_value=self.render_tab_state.max_scale,
                    step=0.1,
                    hint="Maximum scale threshold for rendering splats.",
                )

                @max_scale_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.max_scale = max_scale_slider.value
                    self.rerender(_)

                # slider for scale factor
                scale_factor_slider = server.gui.add_slider(
                    "Scale Factor",
                    min=0.1,
                    max=3.0,
                    initial_value=self.render_tab_state.scale_factor,
                    step=0.05,
                    hint="Rescale Gaussians. >1 = uniform scale up, <1 = shrink towards sphere.",
                )

                @scale_factor_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.scale_factor = scale_factor_slider.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "render_mode_dropdown": render_mode_dropdown,
                "colormap_dropdown": colormap_dropdown,
                "min_opacity_slider": min_opacity_slider,
                "max_scale_slider": max_scale_slider,
                "scale_factor_slider": scale_factor_slider,
            }
        )
        super()._populate_rendering_tab()

        if not self.show_trajectory_options:
            handles = [
                "add_keyframe_button",
                "clear_keyframes_button",
                "loop_checkbox",
                "tension_slider",
                "move_checkbox",
                "show_keyframe_checkbox",
                "show_spline_checkbox",
                "transition_sec_number",
                "framerate_number",
                "duration_number",
                "trajectory_name_text",
                "preview_frame_slider",
                "load_camera_path_button",
                "save_camera_path_button",
                "play_button",
                "pause_button",
                "preview_save_camera_path_button",
                "preview_render_stop_button",
                "dump_video_button",
            ]

            for handel in handles:
                if handel in self._rendering_tab_handles:
                    self._rendering_tab_handles[handel].disabled = False
                    self._rendering_tab_handles[handel].visible = False

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles["total_gs_count_number"].value = (
            self.render_tab_state.total_gs_count
        )
        self._rendering_tab_handles["rendered_gs_count_number"].value = (
            self.render_tab_state.rendered_gs_count
        )


class SceneViewer:
    def __init__(self, scene: Gaussians):
        self.device = tools.get_device()
        self.scene = scene.to(self.device)

    def _get_scene_bounds(self):
        """Get the bounding box of the scene to set a good initial camera pose."""
        if hasattr(self.scene, "means"):
            means = self.scene.means.detach().cpu().numpy()
            min_bounds = means.min(axis=0)
            max_bounds = means.max(axis=0)
            center = (min_bounds + max_bounds) / 2
            size = np.linalg.norm(max_bounds - min_bounds)
            return center, size
        else:
            # Default fallback
            return np.array([0.0, 0.0, 0.0]), 5.0

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, pin_to_cpu: bool = False):
        """Load a gsplat checkpoint (log scales, logit opacities, sh0 | shN)."""
        checkpoint = torch.load(checkpoint_path)["model"]
        sh_coeffs = torch.cat([checkpoint["sh0"], checkpoint["shN"]], dim=-2)
        scene = Gaussians.from_parameters(
            means=checkpoint["means"],
            scales=checkpoint["scales"].exp(),
            quats=checkpoint["quats"],
            opacities=checkpoint["opacities"].sigmoid(),
            sh_coeffs=sh_coeffs,
        )

        print(scene)
        return cls(scene)

    def setup_scene(self, render_tab_state: GsplatRenderTabState):
        """Setup scene parameters based on render tab state."""
        # Apply min opacity and max scale thresholds
        min_opacity = render_tab_state.min_opacity
        max_scale = render_tab_state.max_scale

        opacities = self.scene.opacities
        scales = self.scene.scales

        mask = (opacities >= min_opacity) & (scales <= max_scale).all(dim=-1)
        filtered_scene = self.scene[mask]

        # Apply scale factor
        scale_factor = render_tab_state.scale_factor
        if scale_factor != 1.0:
            scales = filtered_scene.scales
            if scale_factor > 1.0:
                # Uniform scaling
                new_scales = scales * scale_factor
            else:
                # Interpolate towards sphere (mean scale per Gaussian)
                mean_scale = scales.mean(dim=-1, keepdim=True)
                new_scales = scales * scale_factor + mean_scale * (1 - scale_factor)

            # Create new Gaussians with modified scales
            filtered_scene = Gaussians.from_parameters(
                means=filtered_scene.means,
                scales=new_scales,
                quats=filtered_scene.quats,
                opacities=filtered_scene.opacities,
                sh_coeffs=filtered_scene.sh_coeffs,
            )

        return filtered_scene

    @torch.no_grad()
    def viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        camera = Camera.from_calibration_matrix(K)[None]
        pose = Pose.from_4x4mat(c2w)[None]

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "RGB+D",
            "depth(expected)": "RGB+ED",
            "alpha": "RGB",
        }

        scene = self.setup_scene(render_tab_state)
        results = scene.render_view(
            cameras=camera,
            poses=pose,
            rendering_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
        )

        render_colors, render_alphas, infos = (
            results["rendering"].moveaxis(-3, -1),
            results["alphas"].moveaxis(-3, -1),
            results["info"],
        )

        render_tab_state.total_gs_count = scene.num_gaussians
        render_tab_state.rendered_gs_count = (infos["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 3:4]
            near_plane = depth.min()
            far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            renders = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()

        return renders

    def run(
        self,
        output_dir: str = None,
        port: int = 8080,
        mode: str = "rendering",
        show_trajectory_options: bool = False,
    ):
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self.viewer_render_fn,
            output_dir=output_dir,
            mode=mode,
            show_trajectory_options=show_trajectory_options,
        )
        # Pass reference to scene viewer for camera pose initialization
        self.viewer._scene_viewer = self

    def wait_for_shutdown(self):
        should_exit = False

        def signal_handler(sig, frame):
            nonlocal should_exit
            should_exit = True
            print("\nShutting down gracefully...")
            try:
                self.server.stop()
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    raise
            try:
                atexit.unregister(self.server._websock_server.stop)
            except (AttributeError, ValueError):
                pass

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Viewer running... Ctrl+C to exit.")
        try:
            while not should_exit:
                time.sleep(1)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

        print("Exited.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Scene Viewer")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--file", type=str, help="Path to saved gaussian tensor.")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viewer server")
    parser.add_argument("--pin_cpu", action="store_true", help="Pin model features to CPU")
    args = parser.parse_args()

    if args.file:
        data = torch.load(args.file)
        scene = Gaussians(data)
        print(scene)
        viewer = SceneViewer(scene)
    elif args.checkpoint:
        viewer = SceneViewer.from_checkpoint(args.checkpoint, args.pin_cpu)
    else:
        raise ValueError("Either --checkpoint or --file must be provided.")

    viewer.run(output_dir=args.output_dir, port=args.port)
    viewer.wait_for_shutdown()
