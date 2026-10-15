"""ZipSplat wrapper: weight loading, input handling and the optional test-time BA.

Author: Alexander Veicht
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from zipsplat.camera import Camera
from zipsplat.token_ba import TokenBA
from zipsplat.utils import add_batch_dim, remove_batch_dim, resize_image
from zipsplat.zipsplat import ZipSplat as Model

# Registry of pre-trained checkpoints (HuggingFace Hub resolve URLs).
_HF = "https://huggingface.co/veichta/zipsplat/resolve/main"
_WEIGHTS_URLS: Dict[str, str] = {
    "zipsplat": f"{_HF}/zipsplat-da3g-504.tar",  # multi-dataset finetune, long edge 504
    "zipsplat-252": f"{_HF}/zipsplat-da3g-252.tar",  # 252 x 252, RE10K + DL3DV
}

# Model input sizes (W, H) the checkpoints trained on, per training resolution: the 252 px
# checkpoints on square crops, the 504 px finetune on seven aspect ratios at long edge 504.
TRAIN_SIZES = {
    252: [(252, 252)],
    504: [(504, 280), (504, 336), (504, 378), (504, 504), (378, 504), (336, 504), (280, 504)],
}


def _load_checkpoint(weights: str) -> Dict[str, Any]:
    """Resolve `weights` (registry name / path / URL) to a release checkpoint {model, conf}.

    Uses weights_only=False, so only load trusted checkpoints.
    """
    weights = _WEIGHTS_URLS.get(weights, weights)
    if weights.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(
            weights,
            model_dir=f"{torch.hub.get_dir()}/zipsplat",
            map_location="cpu",
            file_name=Path(weights).name,
            weights_only=False,
        )
    return torch.load(weights, map_location="cpu", weights_only=False)


class ZipSplat(nn.Module):
    """Feed-forward 3DGS predictor with test-time bundle adjustment, for one scene.

    Input views of any size are center-cropped to the nearest aspect ratio the checkpoint
    trained on and resized to its input size: 252 x 252 for the 252 px checkpoint, long edge
    504 at seven aspect ratios for the 504 px finetune. The returned cameras refer to the
    ORIGINAL images again. Poses and Gaussians are unaffected by the crop.

    Example:
        >>> import math, torch
        >>> from zipsplat import ZipSplat, Camera, Pose, load_image
        >>> model = ZipSplat(weights="zipsplat").cuda().eval()
        >>> images = [load_image(p) for p in paths]  # raw images, any size
        >>> pred = model(images)  # feed-forward + TokenBA
        >>> gaussians = pred["gaussians"]
        >>>
        >>> camera = Camera.from_fov(math.radians(60), w=512, h=512)
        >>> pose = Pose.from_Rt(torch.eye(3), torch.zeros(3))  # the first input view's frame
        >>> rgb, info = gaussians.render(camera, pose)

    Args:
        weights: Registry name ("zipsplat" or "zipsplat-252"), local path or http(s) URL of a
            checkpoint: a release file or the checkpoint_best.tar of a training run.
        amp: Autocast dtype of the feed-forward pass on CUDA, None for full fp32. The Gaussian
            and camera heads always decode in fp32. TokenBA keeps its own precision.
    """

    def __init__(self, weights: str, amp: Optional[torch.dtype] = torch.bfloat16) -> None:
        super().__init__()
        ckpt = _load_checkpoint(weights)
        # The releases share the default architecture. Only the Gaussians per token differ.
        self.model = Model(gaussians_per_token=ckpt["conf"]["model"]["gaussians_per_token"])
        self.model.load_state_dict(ckpt["model"])
        self.input_sizes = TRAIN_SIZES[ckpt["conf"]["data"]["preprocessing"]["resize"]]
        self.amp = amp

    def autocast(self):
        """The feed-forward autocast context (a no-op off CUDA or with amp=None)."""
        device = next(self.model.parameters()).device.type
        enabled = self.amp is not None and device == "cuda"
        return torch.autocast(device, dtype=self.amp, enabled=enabled)

    def prepare_inputs(self, images: Union[Tensor, List[Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        """Center-crop the views to the training input size closest to their mean aspect ratio
        (in log space) and resize them to it.

        Args:
            images: list of (3, h, w) views (any sizes) or a (V, 3, h, w) tensor in [0, 1].

        Returns:
            (V, 3, H, W) model inputs on the model device, and per view the (V, 2) crop offset
            (x, y) in original pixels and the (V, 2) scale (model px per original px).
        """
        log_ar = sum(math.log(im.shape[-1] / im.shape[-2]) for im in images) / len(images)
        W, H = min(self.input_sizes, key=lambda s: abs(math.log(s[0] / s[1]) - log_ar))
        out, offset, scale = [], [], []
        for im in images:
            h, w = im.shape[-2:]
            cw, ch = (round(h * W / H), h) if w * H > W * h else (w, round(w * H / W))
            left, top = (w - cw) // 2, (h - ch) // 2
            out.append(resize_image(im[:, top : top + ch, left : left + cw], H, W))
            offset.append((left, top))
            scale.append((W / cw, H / ch))
        device = next(self.model.parameters()).device
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        return torch.stack(out).to(device), t(offset), t(scale)

    def refine(self, pred: Dict[str, Any], **ba: Any) -> Dict[str, Any]:
        """TokenBA on the input views: refines the Gaussians, scene tokens and poses.

        Args:
            pred: A prediction of forward (it carries the model inputs it was made from).
            **ba: TokenBA settings (see zipsplat/token_ba.py), e.g. joint_steps=192,
                joint_tol=None (no early stop) or verbose=True.

        Returns:
            pred with the refined gaussians, poses and scene_tokens.
        """
        batch = add_batch_dim(pred)
        out = TokenBA(**ba)(
            scene_tokens=batch["scene_tokens"],
            head=self.model.gaussian_head,
            camera=batch["model_cameras"],
            pose=batch["poses"],
            image=batch["model_images"],
            assignments=batch["assignments"],
        )
        return {**pred, **remove_batch_dim(out)}  # refined gaussians, poses, scene_tokens

    @torch.no_grad()
    def forward(
        self,
        images: Union[Tensor, List[Tensor]],
        compression: float = 1.0,
        ba: Union[bool, Dict[str, Any]] = True,
    ) -> Dict[str, Any]:
        """Predict Gaussians and the input cameras from the views of one scene.

        Args:
            images: list of (3, h, w) views (any sizes) or a (V, 3, h, w) tensor in [0, 1].
            compression: Fraction of the tokens kept as scene tokens, in (0, 1]. 1.0 skips
                k-means.
            ba: Refine the prediction with test-time bundle adjustment on the input views
                (TokenBA, needs CUDA). A dict passes TokenBA settings (see `refine`).

        Returns:
            gaussians: Gaussians (N,), in the frame of the first input view.
            poses: c2w poses (V,) of the input views.
            cameras: Cameras (V,) of the original input images.
            scene_tokens, nearest_idx, assignments: The (K, D) scene tokens and the clustering.
            model_images, model_cameras: The (V, 3, H, W) model inputs and their cameras.
        """
        sizes = [(im.shape[-1], im.shape[-2]) for im in images]
        model_images, offset, scale = self.prepare_inputs(images)
        with self.autocast():
            pred = remove_batch_dim(self.model(model_images[None], compression=compression))
        pred["model_images"], pred["model_cameras"] = model_images, pred.pop("cameras")
        if ba:
            pred = self.refine(pred, **(ba if isinstance(ba, dict) else {}))
        cams = pred["model_cameras"]
        size = torch.tensor(sizes, dtype=cams.dtype, device=cams.device)
        pred["cameras"] = Camera(torch.cat([size, cams.f / scale, cams.c / scale + offset], -1))
        return pred
