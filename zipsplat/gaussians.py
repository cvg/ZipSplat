"""3D Gaussian Splatting scene as a packed tensor.

Layout per Gaussian (last dim D = SH_START + K*3 where K = (max_sh_degree + 1)^2):
    [0:3]               means (xyz, world coords)
    [3:6]               scales (linear, sigma)
    [6:10]              quats (wxyz)
    [10:11]             opacities (in [0, 1])
    [11:11+K*3]         SH coefficients (K rows of 3 RGB channels, flattened)

Rendering uses gsplat.

Author: Alexander Veicht
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from gsplat.rendering import rasterization
from plyfile import PlyData, PlyElement
from zipsplat.camera import Camera
from zipsplat.pose import Pose
from zipsplat.utils import TensorWrapper

# DC-term constant for spherical-harmonics RGB encoding.
_SH_C0 = 0.28209479177387814


def _sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * _SH_C0 + 0.5


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / _SH_C0


class Gaussians(TensorWrapper):
    """3D Gaussian Splatting scene."""

    MEANS_START, MEANS_END = 0, 3
    SCALES_START, SCALES_END = 3, 6
    QUATS_START, QUATS_END = 6, 10
    OPACITY_START, OPACITY_END = 10, 11
    SH_START = 11

    def __init__(self, data_: torch.Tensor):
        assert data_.shape[-1] >= self.SH_START
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def from_parameters(
        cls,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
    ) -> "Gaussians":
        """Build from separate parameter tensors.

        Args:
            means: (..., N, 3) world-coords positions.
            scales: (..., N, 3) linear scales (sigma).
            quats: (..., N, 4) wxyz unit quaternions.
            opacities: (..., N) in [0, 1].
            sh_coeffs: (..., N, K, 3) SH coefficients.
        """
        N, K = means.shape[-2], sh_coeffs.shape[-2]
        D = cls.SH_START + K * 3
        shape = means.shape[:-2] + (N, D)
        data = torch.zeros(shape, device=means.device, dtype=means.dtype)
        data[..., cls.MEANS_START : cls.MEANS_END] = means
        data[..., cls.SCALES_START : cls.SCALES_END] = scales
        data[..., cls.QUATS_START : cls.QUATS_END] = quats
        data[..., cls.OPACITY_START : cls.OPACITY_END] = opacities.unsqueeze(-1)
        data[..., cls.SH_START : cls.SH_START + K * 3] = rearrange(
            sh_coeffs, "... n k c -> ... n (k c)"
        )
        return cls(data)

    # ------------------------- properties -------------------------

    @property
    def num_gaussians(self) -> int:
        return self.data_.shape[-2]

    @property
    def max_sh_degree(self) -> int:
        n_coeffs = (self.data_.shape[-1] - self.SH_START) // 3
        return int(n_coeffs**0.5) - 1

    @property
    def means(self) -> torch.Tensor:
        return self.data_[..., self.MEANS_START : self.MEANS_END]

    @property
    def scales(self) -> torch.Tensor:
        return self.data_[..., self.SCALES_START : self.SCALES_END]

    @property
    def quats(self) -> torch.Tensor:
        return self.data_[..., self.QUATS_START : self.QUATS_END]

    @property
    def opacities(self) -> torch.Tensor:
        return self.data_[..., self.OPACITY_START : self.OPACITY_END].squeeze(-1)

    @property
    def sh_coeffs(self) -> torch.Tensor:
        n_coeffs = (self.max_sh_degree + 1) ** 2
        return self.data_[..., self.SH_START : self.SH_START + n_coeffs * 3].reshape(
            *self.shape, n_coeffs, 3
        )

    @property
    def sh0(self) -> torch.Tensor:
        return self.data_[..., self.SH_START : self.SH_START + 3].reshape(*self.shape, 1, 3)

    @property
    def shN(self) -> torch.Tensor:
        n_coeffs = (self.max_sh_degree + 1) ** 2
        return self.data_[..., self.SH_START + 3 : self.SH_START + n_coeffs * 3].reshape(
            *self.shape, n_coeffs - 1, 3
        )

    @property
    def rgb(self) -> torch.Tensor:
        """Per-Gaussian RGB (DC-only SH evaluation), shape (..., N, 3)."""
        return _sh_to_rgb(self.sh0).squeeze(-2)

    # ------------------------- methods -------------------------

    def scale(self, factor: Union[float, torch.Tensor]) -> "Gaussians":
        """Uniformly scale the scene (means and scales) by a factor."""
        factor = factor if not hasattr(factor, "shape") else factor[..., None, None]
        return Gaussians.from_parameters(
            means=self.means * factor,
            scales=self.scales * factor,
            quats=self.quats,
            opacities=self.opacities,
            sh_coeffs=self.sh_coeffs,
        )

    def color_by_group(self, group_size: int, cmap: str = "tab20") -> "Gaussians":
        """Replace SH colors with group-distinct colors from a matplotlib colormap.

        Assumes Gaussians come in consecutive groups of `group_size`, each sharing one color.
        Useful e.g. for visualizing how per-token Gaussian sets are laid out.
        """
        n = self.num_gaussians // group_size
        perm = torch.randperm(n, device=self.device)
        ids = perm.repeat_interleave(group_size)  # (N,)
        cm = plt.get_cmap(cmap)
        rgb = torch.tensor(
            cm(ids.cpu().numpy() % cm.N)[:, :3], device=self.device, dtype=self.dtype
        )
        sh = _rgb_to_sh(rgb).unsqueeze(-2)
        return self.from_parameters(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh_coeffs=sh,
        )

    def render(
        self,
        cameras: Camera,
        poses: Pose,
        mode: str = "RGB",
        sh_degree: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Render the scene at the given cameras / poses (gsplat).

        Args:
            cameras: Camera — scalar (single view), [V], or [B, V]. Moved to the scene's device.
            poses: Pose with matching shape.
            mode: "RGB" (3 ch), "D" (1 ch), "ED" (1 ch), "RGB+D" (4 ch), "RGB+ED" (4 ch).
            sh_degree: SH degree to use. Defaults to ``self.max_sh_degree``.
            **kwargs: Forwarded to gsplat.rasterization.

        Returns:
            renderings: (..., V, C, H, W) tensor where C depends on `mode`.
            infos: dict with "alphas" (..., V, 1, H, W) and gsplat's per-batch "info" dict.
        """
        assert len(self.shape) <= 2, f"Gaussians shape {self.shape} not supported (max 2 dims)."
        cameras, poses = cameras.to(self.device), poses.to(self.device)
        if len(cameras.shape) == 0:  # scalar camera/pose -> single view
            cameras, poses = cameras[None], poses[None]
        if len(self.shape) == 1:
            return self._render_one(cameras, poses, mode, sh_degree, **kwargs)
        renderings, alphas, info = [], [], []
        for b in range(self.shape[0]):
            r, i = self[b]._render_one(cameras[b], poses[b], mode, sh_degree, **kwargs)
            renderings.append(r)
            alphas.append(i["alphas"])
            info.append(i["info"])
        return (
            torch.stack(renderings, 0),
            {"alphas": torch.stack(alphas, 0), "info": info},
        )

    def save_ply(self, path: Path) -> None:
        """Save to a 3DGS-format PLY (compatible with SuperSplat, gsplat-viewer, etc.).

        Opacity is stored in logit space, scales in log space, quats as wxyz, SH split into
        DC (f_dc_*) + rest (f_rest_*) using the original 3DGS channel-first transpose.
        Requires unbatched Gaussians (shape (N,)).
        """
        assert len(self.shape) == 1, f"save_ply requires unbatched Gaussians, got {self.shape}"
        N = self.num_gaussians
        K = (self.max_sh_degree + 1) ** 2

        means = self.means.detach().cpu().numpy()
        normals = np.zeros_like(means)
        # 3DGS PLY format stores SH as channel-first (3, K), then flattens.
        f_dc = self.sh0.detach().transpose(-1, -2).reshape(N, -1).cpu().numpy()  # (N, 3)
        f_rest = self.shN.detach().transpose(-1, -2).reshape(N, -1).cpu().numpy()  # (N, 3*(K-1))
        opacity = (
            torch.logit(self.opacities.clamp(1e-6, 1 - 1e-6)).reshape(N, 1).detach().cpu().numpy()
        )
        scale = torch.log(self.scales.clamp_min(1e-6)).detach().cpu().numpy()
        rotation = self.quats.detach().cpu().numpy()

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"f_dc_{i}" for i in range(3)]
        attrs += [f"f_rest_{i}" for i in range(3 * (K - 1))]
        attrs += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        dtype = [(name, "f4") for name in attrs]

        data = np.concatenate([means, normals, f_dc, f_rest, opacity, scale, rotation], axis=1)
        elements = np.empty(N, dtype=dtype)
        elements[:] = list(map(tuple, data))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))

    def _render_one(
        self,
        cameras: Camera,
        poses: Pose,
        mode: str,
        sh_degree: Optional[int],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        width, height = cameras[0].size.round().unbind(-1)
        sh_degree = self.max_sh_degree if sh_degree is None else sh_degree
        rendering, alphas, info = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=torch.cat([self.sh0, self.shN], -2),
            viewmats=poses.Rt_inv,
            Ks=cameras.K,
            width=width,
            height=height,
            render_mode=mode,
            sh_degree=sh_degree,
            packed=False,
            **kwargs,
        )
        return rendering.moveaxis(-1, -3), {"alphas": alphas.moveaxis(-1, -3), "info": info}

    def __str__(self) -> str:
        return f"Gaussians({tuple(self.shape)} sh={self.max_sh_degree} {self.dtype} {self.device})"
