"""3D Gaussians packed into one tensor and rendered with gsplat.

Author: Alexander Veicht
"""

import inspect
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from gsplat.rendering import rasterization

from splatfactory import get_logger
from splatfactory.geometry import Camera, Pose
from splatfactory.utils import conversions, mappings, tensor, tools

logger = get_logger(__name__)


class Gaussians(tensor.TensorWrapper):
    """N Gaussians stored as a (..., N, D) tensor.

    Layout of the last dim, with K = (max_sh_degree + 1)^2:
        [0:3]         means (xyz)
        [3:6]         scales (linear)
        [6:10]        quats (wxyz)
        [10:11]       opacity (in [0, 1])
        [11:11+K*3]   SH coefficients (K x 3, flattened)
    """

    # Dimension indices
    MEANS_START = 0
    MEANS_END = 3
    SCALES_START = 3
    SCALES_END = 6
    QUATS_START = 6
    QUATS_END = 10
    OPACITY_START = 10
    OPACITY_END = 11
    SH_START = 11

    def __init__(self, data_: torch.Tensor):
        """Wrap a (..., N, D) tensor."""
        assert data_.shape[-1] >= self.SH_START, f"Data dimension too small: {data_.shape[-1]}"
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
        """Pack separate parameter tensors.

        Args:
            means: (..., N, 3) xyz positions.
            scales: (..., N, 3) linear scales.
            quats: (..., N, 4) wxyz quaternions.
            opacities: (..., N) opacities in [0, 1].
            sh_coeffs: (..., N, K, 3) SH coefficients.

        Returns:
            Gaussians of shape (..., N).
        """
        N = means.shape[-2]
        K = sh_coeffs.shape[-2]
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

    def color_gaussians_by_prototype(self, n: int, k: int) -> "Gaussians":
        """Give each run of k consecutive Gaussians one shuffled tab20 color (SH degree 0)."""
        n = self.num_gaussians // k
        clusters = torch.tensor(list(range(n)), device=tools.get_device())

        # shuffle cluster-to-color mapping to break spatial patterns
        perm = torch.randperm(n, device=tools.get_device())
        shuffled = perm[clusters]
        shuffled = repeat(shuffled[None], "1 N -> N K", K=k)

        # map clusters to the tab20 colormap
        cmap = plt.get_cmap("tab20")
        cluster_colors = cmap(shuffled.cpu().numpy().flatten() % cmap.N)[:, :3]
        cluster_colors = torch.tensor(cluster_colors, device=tools.get_device())
        cluster_sh = conversions.rgb_to_sh(cluster_colors)[..., None, :]

        return self.from_parameters(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh_coeffs=cluster_sh,
        )

    def color_gaussians_by_rgb(self, colors: torch.Tensor) -> "Gaussians":
        """Recolor with per-Gaussian RGB (..., N, 3) at SH degree 0, geometry kept."""
        sh = conversions.rgb_to_sh(colors.to(self.means))[..., None, :]
        return self.from_parameters(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh_coeffs=sh,
        )

    # ================================
    # ========== Properties ==========
    # ================================
    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians N."""
        return self.data_.shape[-2]

    @property
    def max_sh_degree(self) -> int:
        """SH degree inferred from D."""
        D = self.data_.shape[-1]
        sh_dim = D - self.SH_START
        n_coeffs = sh_dim // 3
        return int(n_coeffs**0.5) - 1

    # ========== Position ==========
    @property
    def means(self) -> torch.Tensor:
        """(..., N, 3) xyz positions."""
        return self.data_[..., self.MEANS_START : self.MEANS_END]

    # ========== Scale ==========
    @property
    def scales(self) -> torch.Tensor:
        """(..., N, 3) linear scales."""
        return self.data_[..., self.SCALES_START : self.SCALES_END]

    # ========== Rotation ==========
    @property
    def quats(self) -> torch.Tensor:
        """(..., N, 4) wxyz quaternions."""
        return self.data_[..., self.QUATS_START : self.QUATS_END]

    # ========== Opacity ==========
    @property
    def opacities(self) -> torch.Tensor:
        """(..., N) opacities in [0, 1]."""
        return self.data_[..., self.OPACITY_START : self.OPACITY_END].squeeze(-1)

    # ===== Spherical Harmonics ====
    @property
    def sh_coeffs(self) -> torch.Tensor:
        """(..., N, K, 3) all SH coefficients."""
        n_coeffs = (self.max_sh_degree + 1) ** 2
        shdata_ = self.data_[..., self.SH_START : self.SH_START + n_coeffs * 3]
        return shdata_.reshape(*self.shape, n_coeffs, 3)

    @property
    def sh0(self) -> torch.Tensor:
        """(..., N, 1, 3) degree-0 SH coefficients."""
        shdata_ = self.data_[..., self.SH_START : self.SH_START + 3]
        return shdata_.reshape(*self.shape, 1, 3)

    @property
    def shN(self) -> torch.Tensor:
        """(..., N, K - 1, 3) higher-degree SH coefficients."""
        n_coeffs = (self.max_sh_degree + 1) ** 2
        shdata_ = self.data_[..., self.SH_START + 3 : self.SH_START + n_coeffs * 3]
        return shdata_.reshape(*self.shape, n_coeffs - 1, 3)

    @property
    def rgb(self) -> torch.Tensor:
        """(..., N, 3) view-independent RGB from the degree-0 SH."""
        colors = conversions.sh_to_rgb(self.sh0)
        return colors.squeeze(-2)

    # ===============================
    # ============ Misc =============
    # ===============================

    def scale(self, factor: float) -> "Gaussians":
        """Scale means and scales by a float or a per-batch (...,) tensor."""
        # adjust for broadcasting
        factor = factor if not hasattr(factor, "shape") else factor[..., None, None]
        means = self.means * factor
        scales = self.scales * factor
        return Gaussians.from_parameters(
            means=means,
            scales=scales,
            quats=self.quats,
            opacities=self.opacities,
            sh_coeffs=self.sh_coeffs,
        )

    # ===============================
    # ========== Rendering ==========
    # ===============================
    def _render_single_scene(
        self,
        cameras: Camera,
        poses: Pose,
        rendering_mode: str = "RGB",
        sh_degree: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Render one unbatched (N, D) scene into V views."""
        width, height = cameras[0].size.round().unbind(-1)

        args = {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "viewmats": poses.Rt_inv,
            "Ks": cameras.K,
            "width": width,
            "height": height,
            "render_mode": rendering_mode,
            "packed": False,
            **kwargs,
        }

        if rendering_mode.upper() in ["RGB", "RGB+D", "RGB+ED", "D", "ED"]:
            args["colors"] = torch.cat([self.sh0, self.shN], -2)
            args["sh_degree"] = sh_degree if sh_degree is not None else self.max_sh_degree
        else:
            raise NotImplementedError(f"Rendering mode {rendering_mode} is not supported.")

        # Reject unknown kwargs. inspect.signature follows __wrapped__, which gsplat >= 1.5
        # needs since it decorates rasterization.
        valid_args = set(inspect.signature(rasterization).parameters)
        for k in args:
            if k not in valid_args:
                raise ValueError(
                    f"Unknown argument for rasterization: {k} - valid args: {valid_args}"
                )

        args = {k: v for k, v in args.items() if k in valid_args}

        rendering, alphas, info = rasterization(**args)

        # "activated" (V, N): the Gaussian survives projection and culling in that view.
        if "radii" in info:
            radii = info["radii"]
            info["activated"] = (radii > 0).all(-1) if radii.dim() == 3 else radii > 0

        out = {
            "rendering": rendering.moveaxis(-1, -3),
            "alphas": alphas.moveaxis(-1, -3),
            "info": info,
        }
        return out

    def render_view(
        self,
        cameras: Camera,
        poses: Pose,
        rendering_mode: str = "RGB",
        sh_degree: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Render (N, D) or (B, N, D) Gaussians into V views.

        Args:
            cameras: (V,) or (B, V) cameras.
            poses: (V,) or (B, V) cam2world poses.
            rendering_mode: gsplat render mode, "RGB", "RGB+D", "RGB+ED", "D" or "ED".
            sh_degree: SH degree to evaluate, defaults to max_sh_degree.
            **kwargs: extra gsplat rasterization arguments.

        Returns:
            "rendering": (..., V, C, H, W) colors and/or depth.
            "alphas": (..., V, 1, H, W) accumulated opacity.
            "info": gsplat meta, only "activated" (..., V, N) when batched.
        """
        assert len(self.shape) <= 2, "Only gaussians of shape (N, D) or (B, N, D) are supported."
        if len(self.shape) == 1:
            return self._render_single_scene(
                cameras,
                poses,
                rendering_mode=rendering_mode,
                sh_degree=sh_degree,
                **kwargs,
            )

        # batched gaussians
        results = []
        for batch_id in range(self.shape[0]):
            result = self[batch_id]._render_single_scene(
                cameras[batch_id],
                poses[batch_id],
                rendering_mode=rendering_mode,
                sh_degree=sh_degree,
                **kwargs,
            )
            # remove dynamic sized tensors from info
            result["info"].pop("isect_ids", None)
            result["info"].pop("flatten_ids", None)

            info = {}
            for k in ["activated"]:
                if k not in result["info"]:
                    continue
                info[k] = result["info"][k]
            result["info"] = info

            results.append(result)

        return mappings.stack_tree(results)

    def __str__(self) -> str:
        return (
            f"Gaussians({self.shape} - max_sh_degree={self.max_sh_degree}"
            f" - {self.device} - {self.dtype})"
        )
