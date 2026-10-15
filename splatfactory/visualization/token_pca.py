"""Token PCA colorization: (N, D) tokens -> (N, 3) RGB.

Author: Alexander Veicht
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pca_colorize(
    tokens: torch.Tensor,
    *,
    l2norm: bool = True,
    whiten: bool = False,
    fit_on: str = "all",
    robust: Optional[Tuple[float, float]] = (2.0, 98.0),
) -> torch.Tensor:
    """PCA-project (N, D) tokens to (N, 3) RGB in [0, 1], deterministic across calls.

    Args:
        tokens: (N, D) tokens.
        l2norm: L2-normalize the tokens before PCA.
        whiten: scale the 3 components to equal variance.
        fit_on: fit the basis on "all" or "unique" tokens.
        robust: (lo, hi) display percentiles, or None for plain min/max.

    Returns:
        (N, 3) colors in [0, 1].
    """
    x = tokens.float()
    if l2norm:
        x = F.normalize(x, dim=-1)

    fit = x
    if fit_on == "unique":
        fit = torch.unique(x, dim=0)

    mean = fit.mean(dim=0, keepdim=True)
    fitc = fit - mean
    cov = (fitc.transpose(-1, -2) @ fitc) / max(1, fitc.shape[0] - 1)  # (D, D)
    with torch.amp.autocast(device_type=cov.device.type, enabled=False):
        _, evecs = torch.linalg.eigh(cov.float())
    comps = evecs[:, -3:]  # top-3 components (eigh returns ascending eigenvalues)

    z = (x - mean) @ comps  # (N, 3)
    if whiten:
        z = z / z.std(dim=0, keepdim=True).clamp_min(1e-6)
    if robust is None:
        lo = z.min(dim=0, keepdim=True).values
        hi = z.max(dim=0, keepdim=True).values
    else:
        lo = torch.quantile(z, robust[0] / 100.0, dim=0, keepdim=True)
        hi = torch.quantile(z, robust[1] / 100.0, dim=0, keepdim=True)
    z = torch.minimum(torch.maximum(z, lo), hi)
    return ((z - lo) / (hi - lo + 1e-8)).clamp(0, 1)
