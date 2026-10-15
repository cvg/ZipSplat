"""Principal Component Analysis (PCA) for feature visualization / dimensionality reduction.

Author: Alexander Veicht
"""

import torch


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
