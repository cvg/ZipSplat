"""3D Gaussian Splatting scene: construction, rendering (gsplat), and PLY I/O.

Author: Alexander Veicht
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from gsplat.rendering import rasterization

from splatfactory import get_logger
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.utils import knn
from splatfactory.utils import conversions, mappings, tensor, tools

logger = get_logger(__name__)


class Gaussians(tensor.TensorWrapper):
    """Gaussian splat representation as a unified tensor.

    Tensor layout per Gaussian (dimension D):
    [0:3]       means (xyz positions)
    [3:6]       scales (xyz scales, linear space)
    [6:10]      quats (wxyz quaternion)
    [10:11]     logit_opacity (logit of opacity)
    [11:11+K*3] sh_coefficients (K coefficients x 3 RGB channels, flattened)

    where K = (max_sh_degree + 1)^2
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
        """Initialize Gaussian with raw tensor data."""
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
        """Create Gaussians from separate parameter tensors.

        Args:
            means: (..., N, 3) xyz positions
            scales: (..., N, 3) scales in [0, inf]
            quats: (..., N, 4) quaternions
            opacities: (..., N) opacities in [0, 1]
            sh_coeffs: (..., N, K, 3) spherical harmonics coefficients

        Returns:
            Gaussian: New Gaussian instance
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

    @classmethod
    def random_init(
        cls,
        points: torch.Tensor = None,
        colors: torch.Tensor = None,
        num_gaussians: int = 1000,
        init_opacity: float = 0.1,
        max_sh_degree: int = 3,
    ) -> "Gaussians":
        """Randomly initialize Gaussians in a cube [-3, 3]^3.

        Args:
            points (torch.Tensor, optional): Optional initial xyz positions [N, 3]. Defaults to None.
            colors (torch.Tensor, optional): Optional initial rgb colors [N, 3]. Defaults to None.
            num_gaussians (int, optional): Number of Gaussians to create. Defaults to 1000.
            init_opacity (float, optional): Initial opacity value. Defaults to 0.1.
            max_sh_degree (int, optional): Maximum spherical harmonics degree. Defaults to 3.

        Returns:
            Gaussians: New Gaussian instance
        """
        num_gaussians = num_gaussians if points is None else points.shape[0]
        num_gaussians = num_gaussians if colors is None else colors.shape[0]

        if points is None:
            points = 3 * (torch.rand((num_gaussians, 3)) * 2 - 1).float()

        if colors is None:
            colors = torch.rand((num_gaussians, 3)).float()

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        N = points.shape[0]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        harmonics = torch.zeros((N, (max_sh_degree + 1) ** 2, 3))  # [N, K, 3]
        harmonics[:, 0, :] = conversions.rgb_to_sh(colors)

        return cls.from_parameters(
            means=points,
            scales_log=scales,
            quats=quats,
            opacities_logit=opacities,
            sh_coeffs=harmonics,
        )

    def color_gaussians_by_prototype(self, n: int, k: int) -> "Gaussians":
        """Color gaussians based on their prototype cluster."""
        n = self.num_gaussians // k
        clusters = torch.tensor(list(range(n)), device=tools.get_device())

        # shuffle cluster-to-color mapping to break spatial patterns
        perm = torch.randperm(n, device=tools.get_device())
        shuffled = perm[clusters]
        shuffled = repeat(shuffled[None], "1 N -> N K", K=k)

        # map clusters to Set1 colormap
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

    # ================================
    # ========== Properties ==========
    # ================================
    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians (N)."""
        return self.data_.shape[-2]

    @property
    def max_sh_degree(self) -> int:
        """Infer maximum SH degree from tensor dimensions."""
        D = self.data_.shape[-1]
        sh_dim = D - self.SH_START
        n_coeffs = sh_dim // 3
        return int(n_coeffs**0.5) - 1

    # ========== Position ==========
    @property
    def means(self) -> torch.Tensor:
        """Get xyz positions [N, 3]"""
        return self.data_[..., self.MEANS_START : self.MEANS_END]

    # ========== Scale ==========
    @property
    def scales(self) -> torch.Tensor:
        """Get scales [N, 3]"""
        return self.data_[..., self.SCALES_START : self.SCALES_END]

    # @property
    # def scales_log(self) -> torch.Tensor:
    #     """Get log scales [N, 3]"""
    #     return self.data_[..., self.SCALES_START : self.SCALES_END]

    # ========== Rotation ==========
    @property
    def quats(self) -> torch.Tensor:
        """Get quaternions [N, 4]"""
        return self.data_[..., self.QUATS_START : self.QUATS_END]

    # ========== Opacity ==========
    @property
    def opacities(self) -> torch.Tensor:
        """Get opacities (sigmoided) [N]"""
        return self.data_[..., self.OPACITY_START : self.OPACITY_END].squeeze(-1)
        # return torch.sigmoid(self.data_[..., self.OPACITY_START : self.OPACITY_END].squeeze(-1))

    # @property
    # def opacities_logit(self) -> torch.Tensor:
    #     """Get logit opacities [N]"""
    #     return self.data_[..., self.OPACITY_START : self.OPACITY_END].squeeze(-1)

    # ========= Covariance =========
    @property
    def covariances(self) -> torch.Tensor:
        """Get covariance matrices [N, 3, 3]"""
        return conversions.build_covariance(scale=self.scales, rotation_xyzw=self.quats)

    # ===== Spherical Harmonics ====
    @property
    def sh_coeffs(self) -> torch.Tensor:
        """Get all SH coefficients [N, max_K, 3]"""
        n_coeffs = (self.max_sh_degree + 1) ** 2
        shdata_ = self.data_[..., self.SH_START : self.SH_START + n_coeffs * 3]
        return shdata_.reshape(*self.shape, n_coeffs, 3)

    @property
    def sh0(self) -> torch.Tensor:
        """Get SH 0 coefficients [N, 1, 3]"""
        shdata_ = self.data_[..., self.SH_START : self.SH_START + 3]
        return shdata_.reshape(*self.shape, 1, 3)

    @property
    def shN(self) -> torch.Tensor:
        """Get SH >0 coefficients [N, max_K-1, 3]"""
        n_coeffs = (self.max_sh_degree + 1) ** 2
        shdata_ = self.data_[..., self.SH_START + 3 : self.SH_START + n_coeffs * 3]
        return shdata_.reshape(*self.shape, n_coeffs - 1, 3)

    @property
    def rgb(self) -> torch.Tensor:
        """Get RGB colors by evaluating SH at normal direction [N, 3]"""
        colors = conversions.sh_to_rgb(self.sh0)
        return colors.squeeze(-2)

    # ===============================
    # ============ Misc =============
    # ===============================

    def scale(self, factor: float) -> "Gaussians":
        """Scale the entire scene by a given factor."""
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
        """Render a single unbatched scene.

        This is an internal method that assumes Gaussians have shape [N, D].
        """
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
        elif rendering_mode.lower() == "features":
            if not self.has_features():
                raise ValueError("Cannot render features: scene has no features")
            feature_level = kwargs.pop("feature_level", 0)
            args["colors"] = self.get_features(feature_level)
            args["render_mode"] = "RGB"
        elif rendering_mode.lower() in ["features"]:
            raise NotImplementedError(f"Rendering mode {rendering_mode} is not yet supported.")
        else:
            raise NotImplementedError(f"Rendering mode {rendering_mode} is not supported.")

        # Filter to valid args for rasterization
        args.pop("feature_level", None)
        valid_args = set(rasterization.__code__.co_varnames)
        for k in args:
            if k not in valid_args:
                raise ValueError(
                    f"Unknown argument for rasterization: {k} - valid args: {valid_args}"
                )

        args = {k: v for k, v in args.items() if k in valid_args}

        rendering, alphas, info = rasterization(**args)

        if rendering_mode.lower() == "features":
            rendering = F.normalize(rendering, dim=-1, p=2)

        return {
            "rendering": rendering.moveaxis(-1, -3),
            "alphas": alphas.moveaxis(-1, -3),
            "info": info,
        }

    def render_view(
        self,
        cameras: Camera,
        poses: Pose,
        rendering_mode: str = "RGB",
        sh_degree: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Render the scene for the given camera poses.

        Args:
            cameras (Camera): The camera to use for rendering.
            poses (Pose): The poses of the camera in the scene.
            rendering_mode (str, optional): The rendering mode to use. Defaults to "RGB".
            feature_level (int, optional): The feature level to use. Defaults to 0.

        Returns:
            Dict[str, torch.Tensor]: The rendering results.
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
        return f"Gaussians({self.shape} - max_sh_degree={self.max_sh_degree} - {self.device} - {self.dtype})"


if __name__ == "__main__":
    # Simple test
    N = 10
    K = 3
    gaussians = Gaussians.random_init(num_gaussians=N, max_sh_degree=K)
    print(gaussians)
