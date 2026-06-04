"""Training losses: photometric (MSE/LPIPS) rendering loss, chamfer/depth geometry
losses, and opacity regularization.

Author: Alexander Veicht
"""

import warnings

import torch
import torch.nn as nn
from chamferdist.chamfer import ChamferDistance
from lpips import LPIPS
from omegaconf import OmegaConf

from splatfactory import get_logger

logger = get_logger(__name__)


def apply_reduction(tensor: torch.Tensor, reduction: str | None) -> torch.Tensor:
    """Apply reduction to a loss tensor."""
    if reduction is None:
        return tensor
    if reduction == "mean":
        return tensor.mean()
    if reduction == "sum":
        return tensor.sum()
    raise ValueError(f"Unknown reduction type: {reduction}")


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


class ChamferLoss(nn.Module):
    default_conf = {
        "reduction": None,
        "bidirectional": False,
        "reversed": False,
        "max_coord": 10.0,  # keep in reasonable range to avoid overflow
        "scale_penalty_weight": 0.1,  # weight for penalty on clamped points
    }

    def __init__(self, conf):
        """Efficient Chamfer Distance Loss."""
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)

        self.chamferDist = ChamferDistance()

    @torch._dynamo.disable()
    def forward(self, pred, data):
        """Compute Chamfer Distance Loss between predicted and target point clouds.
        Args:
            pred: {
                "points": [B, N, 3] - predicted 3D points
            }
            data: {
                "points": [B, M, 3] - target 3D points
                "valid_mask": [B, M] - optional mask for valid target points
            }
        """
        # Clamp to reasonable range (prevent overflow in squared distance)
        source_cloud = pred["points"].clamp(-self.conf.max_coord, self.conf.max_coord)
        target_cloud = data["points"].clamp(-self.conf.max_coord, self.conf.max_coord)

        loss = self.chamferDist(
            source_cloud=source_cloud,
            target_cloud=target_cloud,
            bidirectional=self.conf.bidirectional,
            reverse=self.conf.reversed,
            batch_reduction=None,
            point_reduction="mean",
        )

        # Add penalty for clamped points (raw pred, before clamp)
        # sqrt to bound gradient: d(sqrt(x))/dx = 1/(2*sqrt(x)) -> decays for large x
        raw_points = pred["points"]
        excess = torch.relu(torch.abs(raw_points) - self.conf.max_coord + 1e-6)
        location_penalty = torch.sqrt(excess + 1e-8)

        if (location_penalty > 100).any():
            raw_abs = raw_points.abs()
            worst_batch = location_penalty.mean(dim=(1, 2)).argmax().item()
            logger.warning(
                f"Location penalty: max={location_penalty.max().item():.1f} "
                f"mean={location_penalty[location_penalty > 0].mean().item():.1f} "
                f"| raw absmax={raw_abs.max().item():.1f} "
                f"| clamped {(raw_abs > self.conf.max_coord).sum().item()}/{raw_points.numel()} "
                f"| max_coord={self.conf.max_coord} | worst batch={worst_batch}"
            )
            b = worst_batch
            for i, name in enumerate(["x", "y", "z"]):
                logger.warning(
                    f"  [{name}] raw=[{raw_points[b,:,i].min().item():.1f}, "
                    f"{raw_points[b,:,i].max().item():.1f}] "
                    f"target=[{data['points'][b,:,i].min().item():.1f}, "
                    f"{data['points'][b,:,i].max().item():.1f}]"
                )

        loss = loss + location_penalty.mean(dim=(1, 2)) * self.conf.scale_penalty_weight

        return apply_reduction(loss, self.conf.reduction)


class LPIPSLoss(nn.Module):
    default_conf = {
        "net_type": "vgg",
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*(pretrained|Arguments other than a weight enum).*"
            )
            self.loss_fn = LPIPS(net=self.conf.net_type, verbose=False)

        convert_to_buffer(self.loss_fn, persistent=False)

    def forward(self, pred, data):
        predictions = pred["rendering"].clip(0, 1)
        targets = data["image"].clip(0, 1)
        loss = self.loss_fn(predictions, targets, normalize=True).squeeze()
        if loss.ndim == 0:
            loss = loss.unsqueeze(-1)
        return loss


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights
    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)
    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0
