"""Various metrics for evaluating predictions.

Author: Alexander Veicht
"""

import warnings
from functools import cache

import torch
from einops import reduce
from fused_ssim import FusedSSIMMap
from lpips import LPIPS
from skimage.metrics import structural_similarity

from splatfactory import get_logger

logger = get_logger(__name__)


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, "valid", True)
    return map.mean(dim=(-1, -2, -3))


@torch.no_grad()
def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    predicted = predicted.clip(min=0, max=1)
    ground_truth = ground_truth.clip(min=0, max=1)
    mse = reduce((predicted - ground_truth) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*(pretrained|Arguments other than a weight enum).*"
        )
        return LPIPS(net="vgg", verbose=False).to(device)


@torch.no_grad()
def calculate_lpips(
    predicted: torch.Tensor, ground_truth: torch.Tensor, batch_size: int = None
) -> torch.Tensor:
    lpips_fn = get_lpips(predicted.device)
    if batch_size is None or predicted.shape[0] <= batch_size:
        return lpips_fn.forward(ground_truth, predicted, normalize=True)[..., 0, 0, 0]
    values = []
    for i in range(0, predicted.shape[0], batch_size):
        v = lpips_fn.forward(
            ground_truth[i : i + batch_size], predicted[i : i + batch_size], normalize=True
        )[..., 0, 0, 0]
        values.append(v)
    return torch.cat(values)


@torch.no_grad()
def calculate_ssim(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def calculate_iou(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between two binary masks."""
    assert (
        pred_mask.shape == gt_mask.shape
    ), "Predicted and ground truth masks must have the same shape."
    pred_mask, gt_mask = pred_mask.bool(), gt_mask.bool()
    intersection = (pred_mask & gt_mask).float().sum(dim=(-2, -1))
    union = (pred_mask | gt_mask).float().sum(dim=(-2, -1))
    return (intersection + eps) / (union + eps)
