"""ZipSplat wrapper: weight loading + clean inference API.

Author: Alexander Veicht
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.pose import Pose
from zipsplat.utils import IMAGE_SIZE, to_square
from zipsplat.zipsplat import ZipSplat as Model

logger = logging.getLogger(__name__)

# Registry of pre-trained checkpoints (HuggingFace Hub resolve URLs).
_WEIGHTS_URLS: Dict[str, str] = {
    "zipsplat": "https://huggingface.co/veichta/zipsplat/resolve/main/zipsplat-da3g-252p.tar",
}


def _load_state_dict(weights: str) -> Dict[str, Any]:
    """Resolve `weights` (registry name / path / URL) to a model state dict.

    Loads `.tar` checkpoints and returns their "model" entry. Uses weights_only=False, so only
    load trusted checkpoints.
    """
    if weights in _WEIGHTS_URLS:
        weights = _WEIGHTS_URLS[weights]
    if weights.startswith(("http://", "https://")):
        ckpt = torch.hub.load_state_dict_from_url(
            weights,
            model_dir=f"{torch.hub.get_dir()}/zipsplat",
            map_location="cpu",
            file_name=Path(weights).name,
            weights_only=False,
        )
    elif Path(weights).exists():
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid weights {weights!r}: not a registry name, path, or URL.")
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


class ZipSplat(nn.Module):
    """Feed-forward 3DGS predictor.

    Loads pre-trained weights from a registry shortname, local path, or http(s) URL.

    Example:
        >>> import math, torch
        >>> from zipsplat import ZipSplat, Camera, Pose, load_image, viz
        >>> model = ZipSplat(weights="zipsplat").cuda().eval()
        >>> images = [load_image(p) for p in paths]  # raw images, any size
        >>> gaussians = model(images)[0]
        >>>
        >>> camera = Camera.from_fov(math.radians(60), w=512, h=512)
        >>> pose = Pose.from_Rt(torch.eye(3), torch.zeros(3))  # identity pose
        >>> rgb, info = gaussians.render(camera, pose)
        >>>
        >>> viz.turntable(gaussians, "turntable.mp4", sweep_deg=None)  # wiggle orbit video
    """

    def __init__(self, weights: str = "zipsplat") -> None:
        super().__init__()
        self.model = Model()
        self.model.flexible_load(_load_state_dict(weights))

    def _prepare_inputs(
        self,
        images: Union[Tensor, List[Tensor]],
        cameras: Optional[Camera],
        poses: Optional[Pose],
    ) -> Tuple[Tensor, Optional[Camera], Optional[Pose]]:
        """Stack/batch views, square-resize to 252 (adjust priors), move to the model device."""
        if isinstance(images, (list, tuple)):
            assert cameras is None and poses is None, (
                "list input is for pose-free use; pass a (V, 3, H, W) tensor when supplying "
                "cameras / poses."
            )
            images = torch.stack([to_square(im) for im in images])

        if images.ndim == 4:
            images = images.unsqueeze(0)
            if cameras is not None:
                cameras = cameras[None]
            if poses is not None:
                poses = poses[None]

        h, w = images.shape[-2:]
        if cameras is not None:
            side = min(h, w)
            cameras = cameras.crop((side - w, side - h)).scale(IMAGE_SIZE / side)
        images = to_square(images)

        device = next(self.model.parameters()).device
        images = images.to(device)
        if cameras is not None:
            cameras = cameras.to(device)
        if poses is not None:
            poses = poses.to(device)
        return images, cameras, poses

    @torch.no_grad()
    def forward(
        self,
        images: Union[Tensor, List[Tensor]],
        cameras: Optional[Camera] = None,
        poses: Optional[Pose] = None,
        use_priors: bool = False,
        compression: float = 1.0,
    ) -> Gaussians:
        """Predict Gaussians from multi-view images.

        Args:
            images: list of (3, H, W) views (any size) or a (V, 3, H, W) / (B, V, 3, H, W) tensor
                in [0, 1].
            cameras / poses: priors with batch shape (V,) or (B, V); required when use_priors=True
                (pass a tensor, not a list, of images).
            use_priors: inject pose + intrinsics tokens into the backbone.
            compression: query-sampling ratio in (0, 1]; 1.0 disables k-means.

        Returns:
            Gaussians with batch shape (B, N_gaussians).
        """
        if not use_priors and (cameras is not None or poses is not None):
            logger.warning("cameras / poses are ignored when use_priors=False.")
        images, cameras, poses = self._prepare_inputs(images, cameras, poses)
        return self.model(
            images, cameras=cameras, poses=poses, use_priors=use_priors, compression=compression
        )
