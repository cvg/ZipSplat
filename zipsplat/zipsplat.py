"""ZipSplat model: feed-forward 3D Gaussians, camera poses and intrinsics from unposed views.

The training model (splatfactory/models/networks/zipsplat.py) without the losses:
    1. Encode. A DA3 backbone with cross-view attention turns the views into per-layer tokens
       (local and global halves) plus a camera token per view.
    2. Prepare. Each layer's payload is the global half plus a gated LayerNorm of the local
       half. A ColorEmbedMLP turns the RGB patches into color tokens, and a TokenMerge over
       all payloads and the color tokens gives one geometry token per patch.
    3. Compress. k-means on the geometry tokens picks K = r * V * T clusters, skipped at
       r = 1. The scene tokens start as the cluster means.
    4. Fuse. Per layer, deepest first, a cross-attention (queries: scene tokens, keys:
       geometry tokens, values: the layer's payload) and a self-attention. A fourth
       cross-attention reads the color tokens.
    5. Decode. The GaussianHead turns each scene token into a group of Gaussians with free 3D
       offsets, the camera head reads pose and intrinsics off the deepest camera token.

Author: Alexander Veicht
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from zipsplat import modules
from zipsplat.backbone import DAV3Encoder
from zipsplat.camera import Camera
from zipsplat.clustering import cluster_mean, complexity_kmeans
from zipsplat.heads import GaussianHead, decode_pose_encoding
from zipsplat.pose import Pose

# DA3 pretraining normalizes the backbone input with the ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ZipSplat(nn.Module):
    """Feed-forward 3DGS model: images -> Gaussians, poses and intrinsics."""

    def __init__(self, gaussians_per_token: int = 32) -> None:
        super().__init__()
        mean, std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        self.register_buffer("_img_mean", mean.view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", std.view(1, 1, 3, 1, 1), persistent=False)

        self.backbone = DAV3Encoder()  # DA3-Giant, read out at layers 19, 29 and 39
        D = self.embed_dim = self.backbone.embed_dim
        L = self.num_layers = len(self.backbone.backbone.out_layers)
        self.patch_size = self.backbone.backbone.patch_size

        # Prepare: local-half gates, color embedding, geometry merge.
        self.local_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.local_gate = nn.ModuleList([modules.LayerScale(D) for _ in range(L)])
        self.color_embed_mlp = modules.ColorEmbedMLP(self.patch_size, 3, D)
        self.geo_merge = modules.TokenMerge(D, L + 1)

        # Fuse: per-layer cross- and self-attention, plus the color cross-attention.
        H = D // 64
        self.cross_attention = nn.ModuleList([modules.CrossAttentionBlock(D, H) for _ in range(L)])
        self.self_attention = nn.ModuleList([modules.SelfAttentionBlock(D, H) for _ in range(L)])
        self.color_ca = modules.CrossAttentionBlock(D, H)

        # Decode (the camera decoder is part of the DA3 backbone).
        self.gaussian_head = GaussianHead(D, gaussians_per_token)

    # ------------------------- stages -------------------------

    def encode(
        self, images: Tensor, cameras: Optional[Camera] = None, poses: Optional[Pose] = None
    ) -> Dict[str, Tensor]:
        """Encode and prepare, everything that does not depend on the compression.

        Args:
            images: Input images (B, V, 3, H, W) in [0, 1].
            cameras: Optional camera priors (B, V) for the backbone.
            poses: Optional c2w pose priors (B, V) for the backbone.

        Returns:
            layer_tokens: Per-layer payloads (B, V, T, D), shallow to deep.
            geo_tokens: Geometry tokens (B, V, T, D).
            color_tokens: Color tokens (B, V, T, D).
            cam_token: Camera tokens (B, V, 2 * D) of the deepest layer.
        """
        x = (images - self._img_mean) / self._img_std
        tokens = self.backbone(x, pose=poses, camera=cameras)  # (patch, camera) per layer
        D = self.embed_dim
        layer_tokens = [
            raw[..., D:] + self.local_gate[l](self.local_norm[l](raw[..., :D]))
            for l, (raw, _) in enumerate(tokens)
        ]
        col = self.color_embed_mlp(rearrange(images, "B V C H W -> (B V) C H W"))
        col = rearrange(col, "(B V) T D -> B V T D", B=images.shape[0])
        return {
            "layer_tokens": layer_tokens,
            "geo_tokens": self.geo_merge([*layer_tokens, col]),
            "color_tokens": col,
            "cam_token": tokens[-1][1],
        }

    def cluster(self, geo_tokens: Tensor, images: Tensor, compression: float) -> Tuple[Tensor, ...]:
        """Token nearest to each cluster (B, K) and cluster per token (B, V * T) of a k-means.

        Both are arange(V * T) when the compression keeps every token.
        """
        B, V, T, _ = geo_tokens.shape
        if compression >= 1.0:
            idx = torch.arange(V * T, device=geo_tokens.device).unsqueeze(0).expand(B, -1)
            return idx, idx

        K = max(1, int(V * T * compression))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return complexity_kmeans(
                rearrange(geo_tokens, "B V T D -> B (V T) D"), K, images, self.patch_size
            )

    def fuse(self, enc: Dict[str, Tensor], scene_tokens: Tensor) -> Tensor:
        """Refine the scene tokens (B, K, D) by attending into the encoded views."""
        flat = lambda t: rearrange(t, "B V T D -> B (V T) D")  # noqa: E731
        geo = flat(enc["geo_tokens"])
        # deepest layer first: coarse scene layout before fine detail
        for l in reversed(range(self.num_layers)):
            scene_tokens = self.cross_attention[l](scene_tokens, geo, flat(enc["layer_tokens"][l]))
            scene_tokens = self.self_attention[l](scene_tokens)
        return self.color_ca(scene_tokens, geo, flat(enc["color_tokens"]))

    def decode(
        self, enc: Dict[str, Tensor], images: Tensor, compression: float = 1.0
    ) -> Dict[str, object]:
        """Compress, fuse and decode the encoded views (see forward for the outputs)."""
        nearest_idx, assignments = self.cluster(enc["geo_tokens"], images, compression)
        geo = rearrange(enc["geo_tokens"], "B V T D -> B (V T) D")
        scene_tokens = cluster_mean(tokens=geo, assignments=assignments, K=nearest_idx.shape[1])
        scene_tokens = self.fuse(enc, scene_tokens)
        # fp32 heads under any autocast, bf16 rounding shows in the positions and the poses
        with torch.autocast(images.device.type, enabled=False):
            gaussians = self.gaussian_head(scene_tokens.float())
            pose_enc = self.backbone.cam_dec(enc["cam_token"].float())
        poses, cameras = decode_pose_encoding(pose_enc, images.shape[-2:])
        return {
            "gaussians": gaussians,
            "poses": poses,
            "cameras": cameras,
            "scene_tokens": scene_tokens,
            "nearest_idx": nearest_idx,
            "assignments": assignments,
        }

    # ------------------------- forward -------------------------

    def forward(
        self,
        images: Tensor,
        cameras: Optional[Camera] = None,
        poses: Optional[Pose] = None,
        compression: float = 1.0,
    ) -> Dict[str, object]:
        """Predict Gaussians and the input cameras from multi-view images.

        Args:
            images: Input images (B, V, 3, H, W) in [0, 1], sides divisible by the patch size.
            cameras: Optional camera priors (B, V) for the backbone.
            poses: Optional c2w pose priors (B, V) for the backbone.
            compression: Fraction of the V * T tokens kept as scene tokens, in (0, 1]. 1.0
                skips k-means.

        Returns:
            gaussians: Gaussians (B, K * G).
            poses: Predicted c2w poses (B, V), the first view near identity.
            cameras: Predicted cameras (B, V) at the input resolution.
            scene_tokens: Scene tokens (B, K, D), the GaussianHead input.
            nearest_idx: Token nearest to each cluster (B, K).
            assignments: Cluster per token (B, V * T).
        """
        return self.decode(self.encode(images, cameras, poses), images, compression)
