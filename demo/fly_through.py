"""Fly-through video of a ZipSplat scene for the demo.

Left: the render along a smooth path through the input cameras (and back, so it loops). Right: a
fixed overview of the Gaussians from behind the input cameras, with the input views pasted on
their image planes, the path as a line and the moving camera as a frustum.

Author: Alexander Veicht
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline

from zipsplat import Camera, Gaussians, Pose
from zipsplat.viz import orbit_poses

BG = (249 / 255, 248 / 255, 244 / 255)  # the project page background
ACCENT = (58 / 255, 122 / 255, 69 / 255)  # the project page green
DARK = (30 / 255, 33 / 255, 28 / 255)  # the project page text color
FPS = 30
HEIGHT = 720  # of each panel
THUMB = 252  # long edge of the pasted input views
FRUSTUM_DEPTH = 0.16  # image plane distance / camera ring radius
OVERVIEW = (0.0, 20.0, 2.8)  # azimuth deg, elevation deg, distance / ring radius


@torch.no_grad()
def look_targets(gaussians: Gaussians, cameras: Camera, poses: Pose) -> torch.Tensor:
    """World points (V, 3) the views look at: the optical axes at the median depth of the centre."""
    rgbd, _ = gaussians.render(cameras, poses, mode="RGB+ED")
    h, w = rgbd.shape[-2:]
    depth = rgbd[:, 3, 2 * h // 5 : 3 * h // 5, 2 * w // 5 : 3 * w // 5].flatten(1).median(1).values
    return poses.t + depth[:, None] * poses.R[..., 2]


def scene_frame(poses: Pose, targets: torch.Tensor) -> Dict:
    """Look-at point, up axis, ring radius and horizontal basis of the input cameras (V,)."""
    center = targets.mean(0)
    up = F.normalize(-poses.R[..., 1].mean(0), dim=-1)
    h = (poses.t - center) - ((poses.t - center) @ up)[:, None] * up
    e1 = F.normalize(h[0], dim=-1) if h[0].norm() > 1e-6 else poses.R[0, :, 0]
    e2 = torch.linalg.cross(up, e1)
    return dict(center=center, up=up, radius=float(h.norm(dim=-1).median()), e1=e1, e2=e2)


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> Pose:
    """c2w pose (OpenCV axes) at eye looking at target."""
    f = F.normalize(target - eye, dim=-1)
    r = F.normalize(torch.linalg.cross(f, up), dim=-1)
    return Pose.from_Rt(torch.stack([r, torch.linalg.cross(f, r), f], -1), eye)


def view_pose(frame: Dict, az_deg: float, el_deg: float, dist: float) -> Pose:
    """Pose at an azimuth and elevation around the scene, `dist` ring radii from its centre."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    horiz = np.cos(az) * frame["e1"] + np.sin(az) * frame["e2"]
    eye = frame["center"] + dist * frame["radius"] * (np.cos(el) * horiz + np.sin(el) * frame["up"])
    return look_at(eye, frame["center"], frame["up"])


def project(points: torch.Tensor, pose: Pose, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels (..., 2) and depths (...) of world points (..., 3) in one view."""
    pc = pose.inv().transform(points.reshape(-1, 3)).reshape(points.shape)
    uv = pc[..., :2] / pc[..., 2:].clamp_min(1e-6) * camera.f + camera.c
    return uv.numpy(), pc[..., 2].numpy()


def frustum_corners(poses: Pose, cameras: Camera, depth: float) -> torch.Tensor:
    """World corners (V, 4, 3) of the image planes at `depth`: tl, tr, br, bl."""
    w, h = cameras.size.unbind(-1)
    z = torch.zeros_like(w)
    uv = torch.stack([torch.stack(c, -1) for c in [(z, z), (w, z), (w, h), (z, h)]], 1)
    xy = (uv - cameras.c[:, None]) / cameras.f[:, None] * depth
    return poses.transform(torch.cat([xy, torch.full_like(xy[..., :1], depth)], -1))


def spline_path(poses: Pose, scale: float, n: int) -> Pose:
    """n c2w poses (n,) on a spline through the poses (V,), easing in and out.

    Positions follow a shape-preserving cubic (PCHIP) and rotations a rotation spline, on knots
    spaced by the distance (in units of `scale`) plus the angle (rad) between consecutive poses.
    """
    rot = Rotation.from_matrix(poses.R.double().numpy())
    pos = poses.t.double().numpy()
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / scale
    step += (rot[:-1].inv() * rot[1:]).magnitude() + 1e-6
    knots = np.r_[0.0, np.cumsum(step)]
    s = knots[-1] * (0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n)))
    R = torch.from_numpy(RotationSpline(knots, rot)(s).as_matrix()).float()
    return Pose.from_Rt(R, torch.from_numpy(PchipInterpolator(knots, pos)(s)).float())


def paste_quad(canvas: np.ndarray, img: np.ndarray, quad: np.ndarray) -> None:
    """Warp img (h, w, 3) onto the pixel quad (4, 2) of canvas (H, W, 3), in place."""
    H, W = canvas.shape[:2]
    x0, y0 = np.floor(quad.min(0)).astype(int).clip(0, [W, H])
    x1, y1 = np.ceil(quad.max(0)).astype(int).clip(0, [W, H])
    if x1 <= x0 or y1 <= y0:
        return
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad - [x0, y0]))
    warped = cv2.warpPerspective(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (x1 - x0, y1 - y0))[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - mask) + warped * mask


def draw_frustum(canvas: np.ndarray, quad: np.ndarray, apex: np.ndarray, color, width) -> None:
    """Wireframe of one camera: its image plane quad (4, 2) and the edges to the apex (2,)."""
    q = quad.round().astype(np.int32)
    cv2.polylines(canvas, [q], True, color, width, cv2.LINE_AA)
    apex = tuple(int(v) for v in apex.round())
    for c in q:
        cv2.line(canvas, apex, (int(c[0]), int(c[1])), color, max(width - 1, 1), cv2.LINE_AA)


@torch.no_grad()
def render(gaussians: Gaussians, camera: Camera, poses: Pose) -> np.ndarray:
    """Float frames (n, H, W, 3) of the scene at poses (n,) through one camera, over BG."""
    n = poses.shape[0]
    cameras = Camera(camera.data_.expand(n, -1))
    bg = torch.tensor(BG, device=gaussians.device).expand(n, 3)
    rgb, _ = gaussians.render(cameras, poses, mode="RGB", backgrounds=bg)
    return rgb.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return (frame.clip(0, 1) * 255).round().astype(np.uint8)


@torch.no_grad()
def fly_through(
    gaussians: Gaussians,
    cameras: Camera,
    poses: Pose,
    images: torch.Tensor,
    path: Path,
) -> None:
    """Write the fly-through mp4: the render along the path | the overview with the cameras.

    Args:
        gaussians: The scene (N,).
        cameras: Cameras (V,) of the model inputs.
        poses: c2w poses (V,) of the input views.
        images: The model inputs (V, 3, H, W), shown on their image planes in the overview.
        path: Output mp4 path.
    """
    targets = look_targets(gaussians, cameras, poses).cpu()
    cameras, poses = cameras.cpu(), poses.cpu()
    frame = scene_frame(poses, targets)
    radius, V = frame["radius"], poses.shape[0]

    n = int(np.clip(FPS * (V - 1), 3 * FPS, 16 * FPS))  # about a second per input view
    if V > 1:
        traj = spline_path(poses, radius, n)
    else:
        traj = orbit_poses(targets[0], radius, n, sweep_deg=None)
    traj = Pose(torch.cat([traj.data_, traj.data_.flip(0)[1:-1]]))  # forth and back, so it loops

    w, h = cameras[0].size.tolist()
    W, H = 2 * round(w * HEIGHT / h / 2), HEIGHT
    camera = cameras[0].scale((W / w, H / h))

    # the overview: the Gaussians, the path, then the input cameras far to near
    overview = view_pose(frame, *OVERVIEW)
    base = render(gaussians, camera, overview[None])[0]
    uv, z = project(traj.t, overview, camera)
    if (z > 0).all():
        cv2.polylines(base, [uv.round().astype(np.int32)], False, ACCENT, 2, cv2.LINE_AA)
    depth = FRUSTUM_DEPTH * radius
    quads, zq = project(frustum_corners(poses, cameras, depth), overview, camera)
    apexes, za = project(poses.t, overview, camera)
    visible = (zq.min(1) > 0.05 * depth) & (za > 0.05 * depth)
    thumbs = images.permute(0, 2, 3, 1).float().cpu().numpy()
    size = (THUMB, round(THUMB * h / w)) if w >= h else (round(THUMB * w / h), THUMB)
    thumbs = [cv2.resize(t, size, interpolation=cv2.INTER_AREA) for t in thumbs]
    for i in np.argsort(-za):
        if visible[i]:
            paste_quad(base, thumbs[i], quads[i])
            draw_frustum(base, quads[i], apexes[i], ACCENT, 2)

    # the moving camera
    T = traj.shape[0]
    cur_quads, _ = project(
        frustum_corners(traj, Camera(cameras[0].data_.expand(T, -1)), depth), overview, camera
    )
    cur_apex, cur_z = project(traj.t, overview, camera)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        for start in range(0, T, 16):
            rgb = render(gaussians, camera, traj[start : start + 16])
            for j, left in enumerate(rgb):
                t, right = start + j, base.copy()
                if cur_z[t] > 0.05 * depth:
                    draw_frustum(right, cur_quads[t], cur_apex[t], DARK, 3)
                writer.append_data(np.concatenate([to_uint8(left), to_uint8(right)], 1))
