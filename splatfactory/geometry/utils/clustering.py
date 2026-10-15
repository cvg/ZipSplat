"""k-NN and chunked hard k-means for the token clustering.

Author: Alexander Veicht
"""

import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def knn(x: torch.Tensor, K: int = 4, return_ids: bool = False) -> torch.Tensor:
    """Distances [N, K] to the K nearest neighbors of points [N, D], the point itself first.

    `return_ids` returns the neighbor ids instead.
    """
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, cid = model.kneighbors(x_np)
    if return_ids:
        return torch.from_numpy(cid).to(x)

    return torch.from_numpy(distances).to(x)


def photo_complexity_weights(
    images: torch.Tensor, pH: int, pW: int, alpha: float = 2.0, eps: float = 0.05
) -> torch.Tensor:
    """Per-token k-means seeding weights [B, V * pH * pW] of images [B, V, 3, H, W].

    The per-patch gradient energy of the grayscale image, ranked to [0, 1] per scene, shifted
    by `eps` and raised to `alpha`. Complex regions get more centroids.
    """
    B, V, _, H, W = images.shape
    w = images.new_tensor([0.299, 0.587, 0.114])
    gray = (images * w[:, None, None]).sum(2).reshape(B * V, H, W)
    e = torch.zeros_like(gray)
    e[:, :, :-1] += (gray[:, :, 1:] - gray[:, :, :-1]).square()
    e[:, :-1] += (gray[:, 1:] - gray[:, :-1]).square()
    c = F.adaptive_avg_pool2d(e.unsqueeze(1), (pH, pW)).reshape(B, -1)  # [B, V*pH*pW]
    rank = c.argsort(dim=-1).argsort(dim=-1).float() / max(c.shape[-1] - 1, 1)
    return (eps + rank) ** alpha


def hard_kmeans_chunked(
    x: torch.Tensor,
    K: int,
    n_iters: int = 50,
    chunk_size: int = 2048,
    init_mode: str = "uniform",
    metric: str = "cos",
    early_stop: bool = True,
    init_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard k-means with chunked distances, O(chunk * K) memory instead of O(N * K).

    Args:
        x: Features [B, N, D].
        K: Number of clusters.
        n_iters: Iteration cap.
        chunk_size: Tokens per chunk of the distance computation.
        init_mode: Seeding, "uniform", "random" or "complexity" (sampled by `init_weights`).
        metric: "cos" for spherical k-means on L2-normalized features, or "euclid".
        early_stop: Stop once the assignments no longer change.
        init_weights: Non-negative seeding weights [B, N] of "complexity". Default: uniform.

    Returns:
        centroids: Cluster centroids [B, K, D], unit-norm for "cos".
        nearest_idx: Index of the token nearest to each centroid [B, K].
        assignments: Cluster id per token [B, N].
    """
    assert metric in ("euclid", "cos"), metric
    B, N, D = x.shape
    spherical = metric == "cos"
    if spherical:
        x = F.normalize(x, dim=-1)

    if init_mode == "complexity" and init_weights is not None:
        idx = torch.multinomial(init_weights.float().clamp_min(1e-12), K, replacement=False)
    elif init_mode in ("uniform", "complexity"):
        idx = torch.linspace(0, N - 1, K, device=x.device).round().long().unsqueeze(0).expand(B, -1)
    else:
        if N == K:
            idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            idx = torch.stack([torch.randperm(N, device=x.device)[:K] for _ in range(B)])

    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
    centroids = x[batch_idx, idx]

    ones = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

    prev_assignments = None
    for _ in range(n_iters):
        # chunked assignment
        assignments = torch.empty(B, N, dtype=torch.long, device=x.device)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            dists = torch.cdist(x[:, s:e], centroids)  # [B, chunk, K]
            assignments[:, s:e] = dists.argmin(dim=-1)
            del dists

        # converged: the centroids are already the means of these assignments
        if early_stop and prev_assignments is not None:
            if bool((assignments == prev_assignments).all()):
                break
        prev_assignments = assignments

        # scatter-add centroid update
        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, D), x)
        counts = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, assignments.unsqueeze(-1), ones)
        centroids = new_centroids / counts.clamp(min=1)
        if spherical:
            centroids = F.normalize(centroids, dim=-1)

    # nearest token per centroid, chunked
    nearest_idx = torch.empty(B, K, dtype=torch.long, device=x.device)
    for s in range(0, K, chunk_size):
        e = min(s + chunk_size, K)
        dists = torch.cdist(centroids[:, s:e], x)  # [B, chunk, N]
        nearest_idx[:, s:e] = dists.argmin(dim=-1)
        del dists

    return centroids, nearest_idx, assignments


def complexity_kmeans(
    tokens: torch.Tensor,
    K: int,
    images: torch.Tensor,
    patch_size: int,
    n_iters: int = 50,
    chunk_size: int = 2048,
    init_alpha: float = 2.0,
    init_eps: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ZipSplat's token clustering: spherical k-means with complexity-seeded centroids.

    Args:
        tokens: Geometry tokens [B, N, D], N = V * pH * pW.
        K: Number of clusters.
        images: Context images [B, V, 3, H, W], whose complexity seeds the centroids.
        patch_size: Patch size in pixels.
        n_iters: Iteration cap.
        chunk_size: Tokens per chunk of the distance computation.
        init_alpha: Exponent of the seeding weights.
        init_eps: Floor of the seeding weights.

    Returns:
        nearest_idx: Index of the token nearest to each centroid [B, K].
        assignments: Cluster id per token [B, N].
    """
    pH, pW = images.shape[-2] // patch_size, images.shape[-1] // patch_size
    weights = photo_complexity_weights(images, pH, pW, init_alpha, init_eps)
    _, nearest_idx, assignments = hard_kmeans_chunked(
        tokens,
        K,
        n_iters=n_iters,
        chunk_size=chunk_size,
        init_mode="complexity",
        init_weights=weights,
    )
    return nearest_idx, assignments


def cluster_mean(tokens: torch.Tensor, assignments: torch.Tensor, K: int) -> torch.Tensor:
    """Per-cluster mean [B, K, D] of tokens [B, N, D], the identity when K == N.

    Gradients flow to the member tokens, not through the assignments.
    """
    B, N, D = tokens.shape
    if K == N:
        return tokens
    idx = assignments.unsqueeze(-1)
    sums = tokens.new_zeros(B, K, D).scatter_add(1, idx.expand(-1, -1, D), tokens)
    counts = tokens.new_zeros(B, K, 1).scatter_add(1, idx, tokens.new_ones(B, N, 1))
    return sums / counts.clamp(min=1)
