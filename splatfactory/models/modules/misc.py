"""Small shared modules: LayerScale, Mlp, ZipSplat's color embedding and token merge."""

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn

from splatfactory import get_logger

logger = get_logger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ColorEmbedMLP(nn.Module):
    """Per-patch MLP (patchify Conv2d, GELU, Linear) from RGB patches to color tokens.

    Args:
        patch_size: Patch size in pixels.
        in_chans: Input channels.
        embed_dim: Output token dimension.
    """

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """[(B V), 3, H, W] -> [(B V), T, embed_dim]."""
        x = self.proj1(x).flatten(2).transpose(1, 2)
        return self.proj2(self.act(x))


class TokenMerge(nn.Module):
    """Merge N token streams into one: sum_i LS_i(MLP_i(LN_i(t_i))).

    Each stream has its own LayerNorm, a 2-layer MLP (hidden = dim) and a LayerScale gate
    initialized at 1 / N, so the merge starts as an average.

    Args:
        dim: Token dimension of every stream.
        num_streams: Number of merged streams.
    """

    def __init__(self, dim, num_streams):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_streams)])
        self.projs = nn.ModuleList([Mlp(dim, hidden_features=dim) for _ in range(num_streams)])
        self.gates = nn.ModuleList(
            [LayerScale(dim, init_values=1 / num_streams) for _ in range(num_streams)]
        )

    def forward(self, tokens):
        """List of N [..., dim] tensors -> [..., dim]."""
        out = 0
        for norm, proj, gate, t in zip(self.norms, self.projs, self.gates, tokens):
            out = out + gate(proj(norm(t)))
        return out
