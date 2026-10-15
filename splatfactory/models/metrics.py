"""Various metrics for evaluating predictions.

Author: Alexander Veicht
"""

import math
import warnings
from functools import cache

import numpy as np
import torch
from einops import reduce
from lpips import LPIPS
from skimage.metrics import structural_similarity

from splatfactory import get_logger
from splatfactory.geometry import Pose
from splatfactory.utils.conversions import matrix_to_quaternion

logger = get_logger(__name__)


@torch.no_grad()
def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    predicted = predicted.clip(min=0, max=1)
    ground_truth = ground_truth.clip(min=0, max=1)
    mse = reduce((predicted - ground_truth) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device, net: str = "vgg", spatial: bool = False) -> LPIPS:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*(pretrained|Arguments other than a weight enum).*"
        )
        return LPIPS(net=net, spatial=spatial, verbose=False).to(device)


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


# --- DyCheck masked metrics (mPSNR / mSSIM / mLPIPS) ---------------------------
# Torch ports of the official implementation (KAIR-BAIR/dycheck, Apache 2.0).


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dims: tuple) -> torch.Tensor:
    mask = mask.expand_as(x)
    return (x * mask).sum(dims) / mask.sum(dims).clamp(min=1e-6)


@torch.no_grad()
def calculate_masked_psnr(
    predicted: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked PSNR (mPSNR). predicted/ground_truth [B,3,H,W], mask [B,H,W] in {0,1}."""
    predicted = predicted.clip(min=0, max=1)
    ground_truth = ground_truth.clip(min=0, max=1)
    mse = _masked_mean((predicted - ground_truth) ** 2, mask[:, None].float(), (-3, -2, -1))
    return -10 * mse.log10()


def _partial_conv(z: torch.Tensor, m: torch.Tensor, kernel: torch.Tensor):
    """One masked 'valid' convolution pass (partial convolution).

    z [B,C,H,W], m [B,1,H,W] in {0,1}, kernel [1,1,kh,kw]. Renormalizes by the
    valid-pixel count times kernel numel, exactly like the original.
    """
    c = z.shape[1]
    z_ = torch.nn.functional.conv2d(z * m, kernel.expand(c, 1, -1, -1), groups=c)
    m_ = torch.nn.functional.conv2d(m, torch.ones_like(kernel))
    z_out = torch.where(m_ != 0, z_ * kernel.numel() / m_, torch.zeros_like(z_))
    return z_out, (m_ != 0).to(z.dtype)


@torch.no_grad()
def calculate_masked_ssim(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Masked SSIM (mSSIM). predicted/ground_truth [B,3,H,W], mask [B,H,W] in {0,1}."""
    if batch_size is not None and predicted.shape[0] > batch_size:
        return torch.cat(
            [
                calculate_masked_ssim(
                    predicted[i : i + batch_size],
                    ground_truth[i : i + batch_size],
                    mask[i : i + batch_size],
                    filter_size,
                    filter_sigma,
                    k1,
                    k2,
                )
                for i in range(0, predicted.shape[0], batch_size)
            ]
        )
    dtype = torch.float64
    img0 = predicted.to(dtype)
    img1 = ground_truth.to(dtype)
    m = mask[:, None].to(dtype)

    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = (
        (torch.arange(filter_size, dtype=dtype, device=img0.device) - hw + shift) / filter_sigma
    ) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt = filt / filt.sum()
    k_h = filt.view(1, 1, 1, -1)
    k_v = filt.view(1, 1, -1, 1)

    def filt_fn(z, m0):
        z1, m1 = _partial_conv(z, m0, k_h)
        return _partial_conv(z1, m1, k_v)[0]

    mu0 = filt_fn(img0, m)
    mu1 = filt_fn(img1, m)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, m) - mu00
    sigma11 = filt_fn(img1**2, m) - mu11
    sigma01 = filt_fn(img0 * img1, m) - mu01

    sigma00 = sigma00.clamp(min=0)
    sigma11 = sigma11.clamp(min=0)
    sigma01 = sigma01.sign() * torch.minimum((sigma00 * sigma11).sqrt(), sigma01.abs())

    c1 = k1**2
    c2 = k2**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    return (numer / denom).mean((-3, -2, -1)).to(predicted.dtype)


@torch.no_grad()
def calculate_masked_lpips(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Masked LPIPS (mLPIPS, alex/spatial). Images [B,3,H,W], mask [B,H,W] in {0,1}."""
    lpips_fn = get_lpips(predicted.device, net="alex", spatial=True)
    m = mask[:, None].float()
    pred_in = 2 * (predicted * m) - 1
    gt_in = 2 * (ground_truth * m) - 1

    if batch_size is None:
        batch_size = predicted.shape[0]
    values = []
    for i in range(0, predicted.shape[0], batch_size):
        d = lpips_fn.forward(gt_in[i : i + batch_size], pred_in[i : i + batch_size])
        values.append(_masked_mean(d, m[i : i + batch_size], (-3, -2, -1)))
    return torch.cat(values)


# --- Relative pose metrics (VGGT protocol) -------------------------------------
# Ported from facebookresearch/vggt (eval/pose_evaluation/test_co3d.py).


@torch.no_grad()
def calculate_rotation_angle(
    predicted: torch.Tensor, ground_truth: torch.Tensor, eps: float = 1e-15
) -> torch.Tensor:
    """Geodesic angle (deg) between rotation matrices [N, 3, 3], via quaternions."""
    q_pred = matrix_to_quaternion(predicted)
    q_gt = matrix_to_quaternion(ground_truth)
    loss_q = (1 - (q_gt * q_pred).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss_q) * 180 / math.pi


@torch.no_grad()
def calculate_translation_angle(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    eps: float = 1e-15,
    default_err: float = 1e6,
) -> torch.Tensor:
    """Angle (deg) between translation directions [N, 3], with 180-deg ambiguity."""
    t = predicted / (predicted.norm(dim=1, keepdim=True) + eps)
    t_gt = ground_truth / (ground_truth.norm(dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - (t * t_gt).sum(dim=1) ** 2, eps)
    err = torch.acos(torch.sqrt(1 - loss_t)) * 180 / math.pi
    err[torch.isnan(err) | torch.isinf(err)] = default_err
    return torch.min(err, (180 - err).abs())


@torch.no_grad()
def calculate_pose_pair_errors(
    predicted: Pose, ground_truth: Pose
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair (rotation, translation) angles (deg) over all C(V, 2) view pairs.

    Both poses are c2w over V views. They are compared relatively in w2c, so any rigid
    transform and scale between the two frames drops out.
    """
    pred_se3 = predicted.inv().Rt.double().reshape(-1, 4, 4)
    gt_se3 = ground_truth.inv().Rt.double().reshape(-1, 4, 4)
    i1, i2 = torch.combinations(torch.arange(pred_se3.shape[0]), 2).unbind(-1)
    rel_gt = gt_se3[i1] @ torch.linalg.inv(gt_se3[i2])
    rel_pred = pred_se3[i1] @ torch.linalg.inv(pred_se3[i2])
    r_deg = calculate_rotation_angle(rel_pred[:, :3, :3], rel_gt[:, :3, :3])
    t_deg = calculate_translation_angle(rel_pred[:, :3, 3], rel_gt[:, :3, 3])
    return r_deg, t_deg


def calculate_pose_auc(r_error, t_error, max_threshold: int) -> float:
    """AUC of the per-pair max(rotation, translation) error, 1-degree bins.

    Pool the errors over ALL pairs of ALL scenes before calling this: a mean of
    per-scene AUCs is a different (and wrong) number.
    """
    max_errors = np.maximum(r_error, t_error)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    return float(np.mean(np.cumsum(histogram) / len(max_errors)))
