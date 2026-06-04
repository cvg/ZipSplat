"""Turntable / orbit video rendering for ZipSplat Gaussians.

Pose-free: the orbit geometry is derived from the predicted Gaussian cloud alone (center =
median of means, radius = a high quantile of the spread). The trajectory orbits the canonical
front view using the prediction frame's own up direction (OpenCV y-down, so up = (0, -1, 0)).
Provide a `Camera` for the render intrinsics.

Author: Alexander Veicht
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.pose import Pose


def scene_center_radius(
    gaussians: Gaussians, quantile: float = 0.9, margin: float = 1.2
) -> Tuple[torch.Tensor, float]:
    """Scene center and radius from the Gaussian positions alone (no poses needed).

    center = median(means); radius = quantile(||means - center||) * margin.
    """
    means = gaussians.means.detach()
    center = means.median(dim=0).values
    radius = float((means - center).norm(dim=-1).quantile(quantile) * margin)
    return center, radius


def orbit_poses(
    center: torch.Tensor,
    radius: float,
    num_frames: int,
    *,
    sweep_deg: Optional[float] = 360.0,
    base_azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    azimuth_amp_deg: float = 16.0,
    elevation_amp_deg: float = 4.0,
) -> Pose:
    """Orbit trajectory around `center` at `radius` (up = (0, -1, 0), OpenCV y-down frame).

    azimuth/elevation 0 places the camera on the canonical front (-z side, looking toward +z),
    matching the reference-view orientation of the prediction frame.

    Args:
        center: orbit pivot, shape (3,).
        radius: distance from the pivot.
        num_frames: number of poses along the path.
        sweep_deg: if set, sweep this many degrees of azimuth (e.g. 360 = full turntable).
            If ``None``, use an amplitude-limited wiggle instead (better for sparse,
            front-facing scenes where a full revolution would expose empty backsides).
        base_azimuth_deg / elevation_deg: starting azimuth and (constant) elevation.
        azimuth_amp_deg / elevation_amp_deg: wiggle amplitudes (only used when sweep_deg is None).
    """
    t = torch.linspace(0.0, 1.0, num_frames)
    if sweep_deg is None:
        azimuths = base_azimuth_deg + azimuth_amp_deg * torch.sin(2 * math.pi * t)
        elevations = elevation_deg + elevation_amp_deg * torch.sin(4 * math.pi * t) * 0.5
    else:
        azimuths = base_azimuth_deg + sweep_deg * t
        elevations = torch.full_like(t, float(elevation_deg))
    center = torch.as_tensor(center, dtype=torch.float32).detach().cpu()
    return Pose.orbit(center, radius=radius, azimuths=azimuths, elevations=elevations)


def render_video(
    gaussians: Gaussians,
    cameras: Camera,
    poses: Pose,
    *,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk: int = 16,
) -> np.ndarray:
    """Render a camera path through gsplat. Returns (T, H, W, 3) uint8 frames.

    `cameras` and `poses` must share batch shape (T,). Rendering is chunked over frames to
    bound memory; all frames are assumed to share the same image size.
    """
    device = gaussians.device
    cameras, poses = cameras.to(device), poses.to(device)
    T = poses.shape[0]
    bg_t = torch.tensor(bg, device=device, dtype=torch.float32)
    out = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        rgb, _ = gaussians.render(
            cameras[s:e], poses[s:e], mode="RGB", backgrounds=bg_t.expand(e - s, 3)
        )
        rgb = rgb.float().clamp(0, 1).cpu()  # (n, 3, H, W)
        out.append((rgb.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))
    return np.concatenate(out, 0)


def save_video(
    frames: Union[np.ndarray, torch.Tensor], path: Union[str, Path], fps: int = 30
) -> None:
    """Write (T, H, W, 3) frames (uint8, or float in [0, 1]) to an mp4 (libx264, yuv420p)."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def turntable(
    gaussians: Gaussians,
    path: Union[str, Path],
    *,
    fov_deg: float = 55.0,
    render_size: int = 512,
    num_frames: int = 180,
    fps: int = 30,
    sweep_deg: Optional[float] = 360.0,
    elevation_deg: float = 0.0,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk: int = 16,
) -> np.ndarray:
    """Render and save a turntable mp4 orbiting the predicted scene; returns the frames.

    Args:
        gaussians: unbatched predicted scene (shape (N,)).
        path: output mp4 path.
        fov_deg: horizontal field of view of the render camera, in degrees.
        render_size: output frame size in pixels (square).
        num_frames / fps: trajectory length and playback rate.
        sweep_deg: azimuth sweep (360 = full turntable); None = amplitude-limited wiggle.
        elevation_deg: constant elevation of the orbit.
        bg: background RGB in [0, 1].
        chunk: frames rendered per gsplat call.
    """
    fov = torch.tensor(math.radians(fov_deg))
    camera = Camera.from_fov(fov, w=render_size, h=render_size).to(gaussians.device)
    center, radius = scene_center_radius(gaussians)
    poses = orbit_poses(
        center, radius, num_frames, sweep_deg=sweep_deg, elevation_deg=elevation_deg
    )
    cameras = Camera(camera.data_.unsqueeze(0).expand(num_frames, -1).clone())
    frames = render_video(gaussians, cameras, poses, bg=bg, chunk=chunk)
    save_video(frames, path, fps=fps)
    return frames
