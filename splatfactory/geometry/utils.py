"""Geometry utilities: k-means clustering (incl. chunked-hard variant), k-NN,
farthest-point sampling, PCA token downscaling, and SO(3) exponential map.

Author: Alexander Veicht
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn

from splatfactory.utils.conversions import from_homogeneous, skew_symmetric, to_homogeneous


class PCA:
    def __init__(self, n_components: int = 3):
        """Principal Component Analysis (PCA) implementation."""
        self.n_components = n_components
        self.mean = None
        self.components = None

        self.min_vals = None
        self.max_vals = None

    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1] range per dimension. Input: (.., N, D)."""
        X = X.float()
        min_vals = X.min(dim=0, keepdim=True)[0]
        max_vals = X.max(dim=0, keepdim=True)[0]
        return (X - min_vals) / (max_vals - min_vals + 1e-8)

    def fit(self, X: torch.Tensor) -> "PCA":
        """Fit PCA on data of shape (..., N, D)."""
        # Center the data
        X = X.float()
        self.mean = X.mean(dim=-2, keepdim=True)
        X_centered = X - self.mean

        N = X_centered.shape[-2]
        cov = (X_centered.transpose(-1, -2) @ X_centered) / (N - 1)  # (B, D, D)

        # Eigen decomposition (eigh doesn't support bfloat16, disable autocast)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            _, eigvecs = torch.linalg.eigh(cov.float())
        self.components = eigvecs[..., -self.n_components :]

        self.min_vals, _ = self.transform(X).min(dim=0, keepdim=True)
        self.max_vals, _ = self.transform(X).max(dim=0, keepdim=True)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project data to PCA space. Input: (..., N, D) -> (..., N, n_components)."""
        X = X.float()
        X_centered = X - self.mean
        return torch.einsum("...nd,...dc->...nc", X_centered, self.components)

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Project from PCA space back to original. Input: (..., N, n_components)."""
        Z = Z.float()
        X_reconstructed = torch.einsum("...nc,...dc->...nd", Z, self.components)
        return X_reconstructed + self.mean


class PCADownscale(nn.Module):
    """Differentiable PCA-based downscaling. Computes PCA per forward pass."""

    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downscale tokens via PCA projection.

        Args:
            x: Input tensor of shape [..., D]

        Returns:
            Projected tensor of shape [..., n_components]
        """
        *batch_dims, D = x.shape
        x_flat = x.reshape(-1, D).float()  # [N, D]

        # Center
        mean = x_flat.mean(dim=0, keepdim=True)
        x_centered = x_flat - mean

        # SVD: more stable than eigendecomposition
        # x_centered = U @ S @ Vh, where Vh rows are principal components
        _, _, Vh = torch.linalg.svd(x_centered, full_matrices=False)

        # Take top n_components (first rows of Vh = highest variance)
        # Vh has shape [min(N, D), D], so we may have fewer components available
        n_available = min(self.n_components, Vh.shape[0])
        components = Vh[:n_available].T  # [D, n_available]

        # Project
        projected = x_centered @ components  # [N, n_available]

        # Pad with zeros if we couldn't get enough components
        if n_available < self.n_components:
            padding = torch.zeros(
                projected.shape[0],
                self.n_components - n_available,
                device=projected.device,
                dtype=projected.dtype,
            )
            projected = torch.cat([projected, padding], dim=-1)

        return projected.reshape(*batch_dims, self.n_components)


def knn(x: torch.Tensor, K: int = 4, return_ids: bool = False) -> torch.Tensor:
    """Find K-nearest neighbors for each point in the input tensor."""
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, cid = model.kneighbors(x_np)
    if return_ids:
        return torch.from_numpy(cid).to(x)

    return torch.from_numpy(distances).to(x)


def kMeans(x: torch.Tensor, K: int = 4, return_ids: bool = False) -> torch.Tensor:
    """Perform K-Means clustering on the input tensor."""
    x_np = x.cpu().numpy()
    model = KMeans(n_clusters=K, random_state=0).fit(x_np)
    cid = model.labels_
    if return_ids:
        return torch.from_numpy(cid).to(x)

    return torch.from_numpy(model.cluster_centers_).to(x)


def kmeans_batched(x: torch.Tensor, K: int, n_iters: int = 100, tol: float = 1e-4) -> torch.Tensor:
    """x: [B, N, D] -> assignments: [B, N]"""
    B, N, D = x.shape

    # Random init (k-means++ style would be better but slower)
    indices = torch.stack([torch.randperm(N, device=x.device)[:K] for _ in range(B)])
    indices = torch.sort(indices, dim=-1).values  # [B, K]
    centroids = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))  # [B, K, D]

    for i in range(n_iters):
        dists = torch.cdist(x, centroids)  # [B, N, K]
        assignments = dists.argmin(dim=-1)  # [B, N]

        one_hot = F.one_hot(assignments, K).float()
        counts = one_hot.sum(dim=1, keepdim=True).transpose(-1, -2).clamp(min=1)
        new_centroids = torch.einsum("bnk,bnd->bkd", one_hot, x) / counts

        # Early stopping: check centroid movement
        shift = (new_centroids - centroids).norm(dim=-1).max()
        centroids = new_centroids

        if shift < tol:
            break

    return assignments, centroids


def hard_kmeans_chunked(
    x: torch.Tensor,
    K: int,
    n_iters: int = 5,
    chunk_size: int = 2048,
    init_mode: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient hard k-means with chunked distance computation.

    Uses chunked cdist + scatter_add to avoid materializing full [N, K] matrices.
    Peak memory: O(chunk x K) instead of O(N x K).

    Args:
        x: [B, N, D] input features
        K: number of clusters
        n_iters: number of iterations (converges in 2-3, use 5 for headroom)
        chunk_size: tokens per chunk for distance computation
        init_mode: initialization ('uniform' or 'random')

    Returns:
        centroids: [B, K, D] cluster centroids
        nearest_idx: [B, K] index of nearest original token per centroid
        assignments: [B, N] cluster assignment per token (from last iteration)
    """
    B, N, D = x.shape

    if init_mode == "uniform":
        idx = torch.linspace(0, N - 1, K, device=x.device).round().long().unsqueeze(0).expand(B, -1)
    else:
        if N == K:
            idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            idx = torch.stack([torch.randperm(N, device=x.device)[:K] for _ in range(B)])

    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
    centroids = x[batch_idx, idx]

    ones = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

    for _ in range(n_iters):
        # Chunked assignment - peak mem: O(chunk x K) instead of O(N x K)
        assignments = torch.empty(B, N, dtype=torch.long, device=x.device)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            dists = torch.cdist(x[:, s:e], centroids)  # [B, chunk, K]
            assignments[:, s:e] = dists.argmin(dim=-1)
            del dists

        # Scatter-add centroid update - no [N, K] matrix needed
        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, D), x)
        counts = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, assignments.unsqueeze(-1), ones)
        centroids = new_centroids / counts.clamp(min=1)

    # Find nearest original token per centroid (also chunked)
    nearest_idx = torch.empty(B, K, dtype=torch.long, device=x.device)
    for s in range(0, K, chunk_size):
        e = min(s + chunk_size, K)
        dists = torch.cdist(centroids[:, s:e], x)  # [B, chunk, N]
        nearest_idx[:, s:e] = dists.argmin(dim=-1)
        del dists

    return centroids, nearest_idx, assignments


def soft_kmeans(
    x: torch.Tensor,
    K: int,
    n_iters: int = 10,
    temperature: float = 0.1,
    init_mode: str = "random",  # 'random', 'fps', or 'uniform'
    metric: str = "euclidean",  # 'euclidean' or 'cosine'
) -> torch.Tensor:
    """Differentiable soft k-means clustering.
    Args:
        x: [B, N, D] input features
        K: number of clusters
        n_iters: number of iterations
        temperature: softmax temperature (lower = harder assignments)
        init_mode: initialization mode ('random', 'fps', or 'uniform')
        metric: distance metric ('euclidean' or 'cosine')
    Returns:
        centroids: [B, K, D] cluster centroids
    """
    B, N, D = x.shape
    x_normalized = x if metric == "euclidean" else F.normalize(x, p=2, dim=-1)

    if init_mode == "fps":
        indices = farthest_point_sampling(x_normalized, K)  # [B, K]
    elif init_mode == "uniform":
        indices = (
            torch.linspace(0, N - 1, K, device=x.device).round().long().unsqueeze(0).expand(B, -1)
        )
    else:
        if N == K:
            indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            indices = torch.stack([torch.randperm(N, device=x.device)[:K] for _ in range(B)])

    # sort indices which makes the results more deterministic
    indices = torch.sort(indices, dim=-1).values  # [B, K]

    # Gather initial centroids - [B, K, D]
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
    centroids = x_normalized[batch_indices, indices]
    for _ in range(n_iters):
        if metric == "cosine":
            dists = 1 - torch.einsum("bnd,bkd->bnk", x_normalized, centroids)  # [B, N, K]
        else:
            dists = torch.cdist(x_normalized, centroids, p=2)  # [B, N, K]

        soft_assignments = F.softmax(-dists / temperature, dim=-1)  # [B, N, K]

        weighted_sum = torch.einsum("bnk,bnd->bkd", soft_assignments, x_normalized)
        counts = soft_assignments.sum(dim=1, keepdim=True).clamp(min=1e-6)

        centroids = weighted_sum / counts.transpose(1, 2)

        if metric == "cosine":
            centroids = F.normalize(centroids, p=2, dim=-1)

    return centroids


def farthest_point_sampling(x: torch.Tensor, K: int) -> torch.Tensor:
    """Selects K points that are maximally distant from each other."""
    B, N, D = x.shape
    device = x.device

    # Initialize distances to infinity
    # distance[b, n] stores minimum distance from point n to any selected centroid
    distances = torch.full((B, N), 1e10, device=device)

    # Start from token closest to global mean (deterministic, content-aware)
    mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
    farthest_idx = torch.cdist(mean, x).squeeze(1).argmin(dim=-1)  # [B]
    batch_indices = torch.arange(B, device=device)
    centroids_indices = torch.zeros(B, K, dtype=torch.long, device=device)

    for i in range(K):
        centroids_indices[:, i] = farthest_idx

        # Get the coordinates of the currently selected centroid [B, 1, D]
        centroid = x[batch_indices, farthest_idx].unsqueeze(1)

        # Select the point with the largest minimum distance
        dist = torch.norm(x - centroid, dim=-1)
        distances = torch.min(distances, dist)
        farthest_idx = torch.max(distances, dim=-1)[1]

    return centroids_indices


def transform_points(T, points):
    return from_homogeneous(to_homogeneous(points) @ T.transpose(-1, -2))


def is_inside(pts, shape):
    return (pts > 0).all(-1) & (pts < shape[:, None]).all(-1)


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


@torch.jit.script
def distort_points(pts, dist):
    """Distort normalized 2D coordinates
    and check for validity of the distortion model.
    """
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1**2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1**2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts**2)
            # TODO: handle tangential boundaries

    return undist, valid


@torch.jit.script
def J_distort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        d_radial = 2 * k1 + 4 * k2 * r2
        J_diag += radial + (pts**2) * d_radial
        J_cross += uv * d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2 * p12 * pts.flip(-1) + 6 * p21 * pts
            J_cross += 2 * p12 * pts + 2 * p21 * pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J
