"""Token clustering for compression: spherical k-means with complexity-seeded centroids.

Author: Alexander Veicht
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def photo_complexity_weights(
    images: Tensor, pH: int, pW: int, alpha: float = 2.0, eps: float = 0.05
) -> Tensor:
    """Per-token seeding weights from 2D photometric complexity: (B, V, 3, H, W) -> (B, V*pH*pW).

    Squared grayscale image-gradient energy, averaged per patch, turned into a per-scene rank
    in [0, 1] and raised to `alpha`: complex regions get more centroids.
    """
    B, V, _, H, W = images.shape
    w = images.new_tensor([0.299, 0.587, 0.114])
    gray = (images * w[:, None, None]).sum(2).reshape(B * V, H, W)
    e = torch.zeros_like(gray)
    e[:, :, :-1] += (gray[:, :, 1:] - gray[:, :, :-1]).square()
    e[:, :-1] += (gray[:, 1:] - gray[:, :-1]).square()
    c = F.adaptive_avg_pool2d(e.unsqueeze(1), (pH, pW)).reshape(B, -1)
    rank = c.argsort(dim=-1).argsort(dim=-1).float() / max(c.shape[-1] - 1, 1)
    return (eps + rank) ** alpha


def kmeans(
    x: Tensor,
    K: int,
    n_iters: int = 50,
    chunk_size: int = 2048,
    init_weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Spherical hard k-means with chunked distances; stops at the assignment fixed point.

    Args:
        x: (B, N, D) features (L2-normalized internally).
        K: Number of clusters.
        n_iters: Iteration cap.
        chunk_size: Tokens per chunk of the distance computation.
        init_weights: (B, N) seeding weights: the K seeds are sampled without replacement in
            proportion to them. None seeds uniformly over the token order.

    Returns:
        centroids (B, K, D) unit-norm, nearest_idx (B, K) the token nearest to each centroid,
        assignments (B, N) the cluster of each token.
    """
    B, N, D = x.shape
    x = F.normalize(x, dim=-1)
    if init_weights is not None:
        idx = torch.multinomial(init_weights.float().clamp_min(1e-12), K, replacement=False)
    else:
        idx = torch.linspace(0, N - 1, K, device=x.device).round().long().unsqueeze(0).expand(B, -1)
    centroids = x[torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K), idx]
    ones = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

    prev = None
    for _ in range(n_iters):
        assignments = torch.empty(B, N, dtype=torch.long, device=x.device)
        for s in range(0, N, chunk_size):
            assignments[:, s : s + chunk_size] = torch.cdist(
                x[:, s : s + chunk_size], centroids
            ).argmin(dim=-1)
        # Converged: the centroids already are the means of exactly these assignments.
        if prev is not None and bool((assignments == prev).all()):
            break
        prev = assignments
        sums = torch.zeros_like(centroids).scatter_add_(
            1, assignments.unsqueeze(-1).expand(-1, -1, D), x
        )
        counts = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, assignments.unsqueeze(-1), ones)
        centroids = F.normalize(sums / counts.clamp(min=1), dim=-1)

    nearest_idx = torch.empty(B, K, dtype=torch.long, device=x.device)
    for s in range(0, K, chunk_size):
        nearest_idx[:, s : s + chunk_size] = torch.cdist(
            centroids[:, s : s + chunk_size], x
        ).argmin(dim=-1)
    return centroids, nearest_idx, assignments


def complexity_kmeans(
    tokens: Tensor, K: int, images: Tensor, patch_size: int
) -> Tuple[Tensor, Tensor]:
    """ZipSplat's token clustering: spherical k-means seeded by photometric complexity.

    Args:
        tokens: (B, N, D) geometry tokens, N = V * pH * pW.
        K: Number of clusters.
        images: (B, V, 3, H, W) context images in [0, 1].
        patch_size: Patch size in pixels.

    Returns:
        nearest_idx (B, K) and assignments (B, N).
    """
    pH, pW = images.shape[-2] // patch_size, images.shape[-1] // patch_size
    weights = photo_complexity_weights(images, pH, pW)
    _, nearest_idx, assignments = kmeans(tokens, K, init_weights=weights)
    return nearest_idx, assignments


def cluster_mean(tokens: Tensor, assignments: Tensor, K: int) -> Tensor:
    """Per-cluster member mean (B, K, D) of (B, N, D) tokens (identity when K == N)."""
    B, N, D = tokens.shape
    if K == N:
        return tokens
    idx = assignments.unsqueeze(-1)
    sums = tokens.new_zeros(B, K, D).scatter_add(1, idx.expand(-1, -1, D), tokens)
    counts = tokens.new_zeros(B, K, 1).scatter_add(1, idx, tokens.new_ones(B, N, 1))
    return sums / counts.clamp(min=1)
