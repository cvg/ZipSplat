"""ZipSplat: feed-forward 3D Gaussian Splatting from multi-view images.

Pipeline:
    1. A vision-transformer backbone (DA3) with cross-view attention encodes the
       input views into per-layer feature tokens.
    2. Optional k-means clustering selects a subset of tokens (the compression knob),
       with the same cluster indices shared across layers.
    3. Per-layer cross-attention gathers queries from each backbone layer's tokens
       and accumulates them into a scene-token residual stream, followed by
       self-attention over the scene tokens.
    4. A color skip connection re-injects high-frequency image detail.
    5. The Gaussian head decodes each scene token into a small group of 3D Gaussians
       with free 3D offsets (means, scales, quats, opacity, SH).

Author: Alexander Veicht
"""

import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint as ckpt

from splatfactory import get_logger
from splatfactory.geometry.utils import hard_kmeans_chunked
from splatfactory.models import BaseModel
from splatfactory.models.encoders.dav3_encoder import DAV3Encoder
from splatfactory.models.modules import patch_embed, transformer_block
from splatfactory.utils import mappings
from splatfactory.visualization.visualize_batch import visualize_batch

logger = get_logger(__name__)


class ZipSplat(BaseModel):
    """Feed-forward 3DGS model."""

    default_conf = {
        "backbone": {
            "vit_name": "vitg",
            "embed_dim": 1536,
            "out_layers": [19, 29, 39],
            "alt_start": 13,
            "qknorm_start": 13,
            "rope_start": 13,
            "cat_token": True,
            "with_camera_enc": True,
            "freeze_camera_enc": True,
            "weights": "weights/da3/da3-giant.pth",
            "trainable": True,
            "use_checkpoint": False,
        },
        "img_size": 252,
        "eval_use_priors": False,
        # Layer roles: indices into layer_tokens (0 = deepest backbone layer)
        "clustering_layer": 0,
        "scene_init_layer": 0,
        # Attention blocks (num_heads=24 -> head_dim=64 for embed_dim=1536)
        "num_heads": 24,
        "mlp_ratio": 4.0,
        "fuse_layer_scale": 0.01,
        # Query clustering (compression knob)
        "query_sample_ratio": [1.0, 1.0],
        "query_ratio_schedule": 0.5,
        "query_scale_with_views": 0.5,
        "knn": {
            "init_mode": "uniform",
            "chunk_size": 2048,
            "n_iters": 5,
        },
        "token_selection_mode": "kmeans",  # kmeans | stride | random (eval ablation)
        # Color skip
        "with_color_skip": True,
        "color_skip_dim": 128,
        # Runtime flags
        "use_checkpoint": False,
        "return_attention": True,  # eval-time attention capture for viz
        "skip_head_render": False,  # eval: benchmark renders separately
        # Gaussian head
        "gaussians_per_prototype": 32,
        "gaussian_head": {
            "sh_degree": 1,
            "random_background": False,
            "coupled_init": True,
            "use_l1_loss": True,
            "use_chamfer": True,
            "loss_on_context": False,
            "mse_weight": 1.0,
            "lpips_weight": 0.05,
            "chamfer_weight": 0.1,
            "depth_weight": 0.01,
        },
    }

    required_data_keys = ["context"]

    # ============================================================
    # Setup
    # ============================================================

    def _init(self, conf):
        # Sets self.embed_dim, self.num_layers, self.patch_size.
        self._setup_backbone(conf)

        # Mutable list so the trainer's query_ratio_schedule can anneal ratio_min in place
        self.query_sample_ratio = list(conf.query_sample_ratio)

        assert (
            0 <= conf.clustering_layer < self.num_layers
        ), f"clustering_layer={conf.clustering_layer} out of range [0, {self.num_layers})"
        assert (
            0 <= conf.scene_init_layer < self.num_layers
        ), f"scene_init_layer={conf.scene_init_layer} out of range [0, {self.num_layers})"

        self._setup_fuse(conf, self.embed_dim, self.num_layers)
        self._setup_attention(conf, self.embed_dim, self.num_layers)
        self._setup_color(conf, self.embed_dim)
        self._setup_head(conf, self.embed_dim)

        tokens_per_view = (conf.img_size // self.patch_size) ** 2
        gs_per_view_min = int(
            tokens_per_view * conf.query_sample_ratio[0] * conf.gaussians_per_prototype
        )
        gs_per_view_max = int(
            tokens_per_view * conf.query_sample_ratio[1] * conf.gaussians_per_prototype
        )
        logger.info(
            f"ZipSplat: tokens/view={tokens_per_view}, gaussians/view={gs_per_view_min}-{gs_per_view_max} "
            f"(ratio {conf.query_sample_ratio[0]}-{conf.query_sample_ratio[1]} x {conf.gaussians_per_prototype})"
        )
        logger.info(
            f"  clustering_layer={conf.clustering_layer}, scene_init_layer={conf.scene_init_layer}, "
            f"fuse_layer_scale={conf.fuse_layer_scale}"
        )

    def _setup_backbone(self, conf):
        self.backbone = DAV3Encoder(conf.backbone)
        self.embed_dim = self.backbone.embed_dim
        self.num_layers = len(conf.backbone.out_layers)
        self.patch_size = self.backbone.backbone.patch_size

    def _setup_fuse(self, conf, embed_dim, num_layers):
        # Per-layer LayerNorm on local and global halves (backbone returns cat_token=True -> 2*D)
        self.pre_norm_local = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.pre_norm_global = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.downscale = nn.ModuleList(
            [nn.Linear(2 * embed_dim, embed_dim) for _ in range(num_layers)]
        )

    def _setup_attention(self, conf, embed_dim, num_layers):
        self.cross_attention = nn.ModuleList(
            [
                transformer_block.CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=conf.num_heads,
                    mlp_ratio=conf.mlp_ratio,
                    qk_norm=True,
                    attn_mode="softmax",
                    init_values=conf.fuse_layer_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attention = nn.ModuleList(
            [
                transformer_block.SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=conf.num_heads,
                    mlp_ratio=conf.mlp_ratio,
                    qk_norm=True,
                    init_values=conf.fuse_layer_scale,
                )
                for _ in range(num_layers)
            ]
        )

    def _setup_color(self, conf, embed_dim):
        self.color_embed = None
        self.color_cross_attention = None
        if not conf.with_color_skip:
            return
        self.color_embed = patch_embed.PatchEmbed(
            img_size=conf.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=conf.color_skip_dim,
        )
        logger.info("ZipSplat: using PatchEmbed for color skip.")

        # Color CA operates entirely in color_skip_dim - symmetric 4th "layer"
        self.color_cross_attention = transformer_block.CrossAttentionBlock(
            dim=conf.color_skip_dim,
            num_heads=max(1, conf.color_skip_dim // 64),
            mlp_ratio=1.0,
            qk_norm=True,
            init_values=conf.fuse_layer_scale,
        )

    def _setup_head(self, conf, embed_dim):
        from splatfactory.models.decoders.gaussian_head import GaussianHead

        head_dim = embed_dim + (conf.color_skip_dim if conf.with_color_skip else 0)
        self.gaussian_head = GaussianHead(
            {
                "embed_dim": head_dim,
                "gaussians_per_token": conf.gaussians_per_prototype,
                **conf.gaussian_head,
            }
        )
        if conf.skip_head_render:
            # Eval: the benchmark renders novel views itself, so skip the in-head render pass.
            self.gaussian_head._render_results = lambda *a, **k: {}

    def _compile(self, *args, **kwargs):
        for i in range(self.num_layers):
            self.cross_attention[i] = torch.compile(self.cross_attention[i], *args, **kwargs)
            self.self_attention[i] = torch.compile(self.self_attention[i], *args, **kwargs)
            self.downscale[i] = torch.compile(self.downscale[i], *args, **kwargs)
        if self.color_embed is not None:
            self.color_embed = torch.compile(self.color_embed, *args, **kwargs)
        if self.color_cross_attention is not None:
            self.color_cross_attention = torch.compile(self.color_cross_attention, *args, **kwargs)

    # ============================================================
    # Backbone + prep
    # ============================================================

    def _backbone_features(self, data):
        """Extract per-layer features from DAV3Encoder. Returns list where index 0 = deepest layer."""
        backbone_data = {"image": data["context"]["image"]}
        if self.backbone.conf.with_camera_enc:
            use_priors = (
                (torch.rand(1).item() < 0.2) if self.training else self.conf.eval_use_priors
            )
            if use_priors and "pose" in data["context"] and "camera" in data["context"]:
                backbone_data["use_priors"] = True
                backbone_data["pose"] = data["context"]["pose"].Rt
                backbone_data["camera"] = data["context"]["camera"].K
        tokens = self.backbone(backbone_data)["feats"]
        # tokens is a tuple of (patch_tokens, camera_token) per layer.
        # Reverse so features[0] = deepest layer, drop camera token.
        return [t[0] for t in tokens[::-1]]

    def _prepare(self, features):
        """Per-layer prep: pre_norm + downscale + fuse_skip residual.

        Returns a list of [B, V, T, D] tensors, one per backbone layer, in the same order
        as `features` (index 0 = deepest = scene_init_layer default).
        """
        D = self.embed_dim
        layer_tokens = []
        for l in range(self.num_layers):
            raw = features[l]  # [B, V, T, 2*D]
            local = self.pre_norm_local[l](raw[..., :D])
            global_ = self.pre_norm_global[l](raw[..., D:])
            cat_feat = torch.cat([local, global_], dim=-1)
            down = self.downscale[l](cat_feat)
            tok = down + (local + global_) / 2  # fuse_skip residual (always on)
            layer_tokens.append(tok)
        return layer_tokens

    # ============================================================
    # Clustering
    # ============================================================

    def _assign_to_centroids(self, tokens_flat, nearest_idx):
        """One-pass nearest-centroid assignment for non-kmeans selection modes.

        For stride/random, after selecting K token indices we still need a [B, V*T]
        assignment vector (used for token-coloring viz). We use a single argmin over
        cdist(tokens, tokens[nearest_idx]) - i.e., the first iteration of kmeans
        starting from the chosen centroids.
        """
        B, _VT, D = tokens_flat.shape
        with torch.no_grad():
            centroids = torch.gather(tokens_flat, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, D))
            dists = torch.cdist(tokens_flat, centroids)  # [B, VT, K]
            assignments = dists.argmin(dim=-1)
        return assignments

    def _cluster(self, layer_tokens):
        """Select K token indices on `clustering_layer` per `token_selection_mode`.

        Returns (nearest_idx [B, K], assignments [B, V*T]). At r=1.0 (identity) both
        equal arange(V*T). For mode=kmeans, hard_kmeans_chunked produces both. For
        mode=stride/random, K indices are picked deterministically/uniformly and
        assignments come from a single nearest-centroid pass.
        """
        tokens = layer_tokens[self.conf.clustering_layer]
        B, V, T, D = tokens.shape
        tokens_flat = rearrange(tokens, "B V T D -> B (V T) D")
        VT = V * T

        # Read from the mutable attribute (may have been annealed by the trainer's schedule)
        ratio_min, ratio_max = self.query_sample_ratio
        # view-scaled min ratio: more views -> smaller min ratio
        if self.conf.query_scale_with_views > 0:
            ratio_min = min(ratio_min, ratio_min * (2 / V) ** self.conf.query_scale_with_views)
        if ratio_min == ratio_max or not self.training:
            ratio = ratio_max
        else:
            ratio = random.uniform(ratio_min, ratio_max)

        num_queries = max(1, int(VT * ratio))

        # r=1.0 identity case - skip selection entirely (all modes equivalent)
        if num_queries >= VT:
            idx = torch.arange(VT, device=tokens.device).unsqueeze(0).expand(B, -1)
            return idx, idx

        mode = self.conf.token_selection_mode
        if mode == "kmeans":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, nearest_idx, assignments = hard_kmeans_chunked(
                    tokens_flat,
                    num_queries,
                    n_iters=self.conf.knn.n_iters,
                    chunk_size=self.conf.knn.chunk_size,
                    init_mode=self.conf.knn.init_mode,
                )
            return nearest_idx, assignments

        if mode == "stride":
            idx_1d = torch.linspace(0, VT - 1, num_queries, device=tokens.device).round().long()
            nearest_idx = idx_1d.unsqueeze(0).expand(B, -1).contiguous()
            assignments = self._assign_to_centroids(tokens_flat, nearest_idx)
            return nearest_idx, assignments

        if mode == "random":
            nearest_idx = torch.stack(
                [torch.randperm(VT, device=tokens.device)[:num_queries] for _ in range(B)]
            )
            assignments = self._assign_to_centroids(tokens_flat, nearest_idx)
            return nearest_idx, assignments

        raise ValueError(
            f"Unknown token_selection_mode: {mode!r}. Supported: kmeans, stride, random."
        )

    # ============================================================
    # Fusion
    # ============================================================

    def _fuse(self, features):
        """Main geometry fusion with layer-specific queries.

        Returns dict with:
          - scene_tokens: [B, K, D] final geometry stream
          - nearest_idx: [B, K] cluster indices for downstream reuse (color CA, viz)
          - assignments: [B, V*T] full per-token assignments (for token-coloring viz)
          - attention: list of per-layer attention weights (or None)
          - block_stats: dict of per-block residual contribution ratios for monitoring
        """
        layer_tokens = self._prepare(features)
        nearest_idx, assignments = self._cluster(layer_tokens)
        D = self.embed_dim

        # Initialize scene_tokens by gathering from scene_init_layer at the cluster indices.
        init_flat = rearrange(layer_tokens[self.conf.scene_init_layer], "B V T D -> B (V T) D")
        scene_tokens = torch.gather(init_flat, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, D))

        return_attn = (not self.training) and self.conf.return_attention
        use_ckpt = self.training and self.conf.use_checkpoint
        attn_weights = [] if return_attn else None
        block_stats = {}

        for l in range(self.num_layers):
            keys_l = rearrange(layer_tokens[l], "B V T D -> B (V T) D")
            queries_l = torch.gather(keys_l, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, D))

            # CA returns the gamma-gated delta added to the scene-token residual stream.
            if use_ckpt:
                update = ckpt(
                    self._ca_delta_step,
                    l,
                    queries_l,
                    keys_l,
                    use_reentrant=False,
                )
            elif return_attn:
                update, attn = self.cross_attention[l](
                    queries_l,
                    keys_l,
                    return_delta=True,
                    return_attention=True,
                )
                attn_weights.append(attn)
            else:
                update = self.cross_attention[l](queries_l, keys_l, return_delta=True)

            # Track contribution ratio (||update|| / ||scene_tokens||).
            with torch.no_grad():
                block_stats[f"ca_{l}_delta_ratio"] = update.detach().norm(
                    dim=-1
                ).mean() / scene_tokens.detach().norm(dim=-1).mean().clamp(min=1e-8)

            scene_tokens = scene_tokens + update

            # Track SA update ratio (||sa_out - scene|| / ||scene||)
            with torch.no_grad():
                pre_sa_norm = scene_tokens.detach().norm(dim=-1).mean()
            pre_sa = scene_tokens
            scene_tokens = self.self_attention[l](scene_tokens)
            with torch.no_grad():
                block_stats[f"sa_{l}_update_ratio"] = (scene_tokens - pre_sa).detach().norm(
                    dim=-1
                ).mean() / pre_sa_norm.clamp(min=1e-8)

        return {
            "scene_tokens": scene_tokens,
            "attention": attn_weights,
            "nearest_idx": nearest_idx,
            "assignments": assignments,
            "block_stats": block_stats,
        }

    def _ca_delta_step(self, l, queries_l, keys_l):
        """Checkpointable wrapper for a single CA delta pass."""
        return self.cross_attention[l](queries_l, keys_l, return_delta=True)

    # ============================================================
    # Forward
    # ============================================================

    def _forward(self, data):
        context_images = data["context"]["image"]  # [B, S, 3, H, W]
        B, S, C, H, W = context_images.shape

        features = self._backbone_features(data)

        # Color tokens (if enabled)
        color_tokens_full = None
        if self.conf.with_color_skip and self.color_embed is not None:
            imgs = rearrange(context_images, "B S C H W -> (B S) C H W")
            color_tokens_full = self.color_embed(imgs)
            color_tokens_full = rearrange(color_tokens_full, "(B S) T D -> B (S T) D", B=B)

        fused = self._fuse(features)
        scene_tokens = fused["scene_tokens"]
        nearest_idx = fused["nearest_idx"]
        assignments = fused["assignments"]
        attn_weights = fused["attention"]
        block_stats = fused["block_stats"]

        # Color CA as symmetric 4th "layer" - uses color features only at shared indices
        if self.conf.with_color_skip and self.color_cross_attention is not None:
            color_dim = color_tokens_full.shape[-1]
            color_queries = torch.gather(
                color_tokens_full,
                1,
                nearest_idx.unsqueeze(-1).expand(-1, -1, color_dim),
            )
            return_attn = (not self.training) and self.conf.return_attention
            if return_attn:
                color_feats, color_attn = self.color_cross_attention(
                    color_queries,
                    color_tokens_full,
                    return_attention=True,
                )
                if attn_weights is not None:
                    attn_weights.append(color_attn)
            else:
                color_feats = self.color_cross_attention(color_queries, color_tokens_full)
            # Track color CA delta ratio (how much the full-block color CA changed the queries)
            with torch.no_grad():
                block_stats["color_ca_delta_ratio"] = (color_feats - color_queries).detach().norm(
                    dim=-1
                ).mean() / color_queries.detach().norm(dim=-1).mean().clamp(min=1e-8)
            final_tokens = torch.cat([scene_tokens, color_feats], dim=-1)
        else:
            final_tokens = scene_tokens

        with torch.autocast("cuda", enabled=False):
            result = self.gaussian_head({"tokens": final_tokens.float()} | data)
        result["scene_tokens"] = scene_tokens
        result["nearest_idx"] = nearest_idx
        result["assignments"] = assignments
        result["block_stats"] = block_stats
        if attn_weights is not None:
            result["attention"] = attn_weights

        return result

    # ============================================================
    # Loss / metrics / viz
    # ============================================================

    def loss(self, pred, data):
        """Return (losses, metrics). Mirrors gaussian_head.loss() shape so the trainer's
        loss_metrics() (which just calls self.loss) gets the model-level health metrics
        merged in alongside the gaussian head's PSNR/LPIPS/etc.
        """
        losses, metrics = self.gaussian_head.loss(pred, data)
        health = self.metrics(pred, data)
        if health:
            metrics = {**metrics, **health}
        return losses, metrics

    def metrics(self, pred, data):
        """Return per-block diagnostics as [B]-shaped tensors for the trainer's AverageMetric.

        - `block_ratio/*`: per-layer residual contribution ratios from the latest forward pass
          (CA delta / scene norm; SA update / scene norm; color CA delta / color query norm).
        - `block_ls/*`: LayerScale gamma norms for each ls1 / ls2 in CA, SA, and color CA.
          Static params, but still broadcast to [B] so AverageMetric accumulates them uniformly.
        """
        metrics: dict[str, torch.Tensor] = {}

        # Batch size and device from scene_tokens (always present)
        scene_tokens = pred["scene_tokens"]
        B = scene_tokens.shape[0]
        device = scene_tokens.device

        def _scalar_to_batch(value) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value.detach().to(device).reshape(()).expand(B).clone()
            return torch.full((B,), float(value), device=device)

        # Dynamic per-block residual ratios from forward pass
        block_stats = pred.get("block_stats", {}) or {}
        for name, value in block_stats.items():
            metrics[f"block_ratio/{name}"] = _scalar_to_batch(value)

        # Static LayerScale gamma |mean| per block - directly comparable to init value
        # (e.g., init=0.01 means "alive" channels have |gamma|>0.01, "dead" <0.01).
        def _gamma_norm(module: nn.Module) -> float | None:
            g = getattr(module, "gamma", None)
            return g.detach().abs().mean().item() if g is not None else None

        for l in range(self.num_layers):
            ca = self.cross_attention[l]
            sa = self.self_attention[l]
            for prefix, block in ((f"ca_{l}", ca), (f"sa_{l}", sa)):
                for suffix in ("ls1", "ls2"):
                    v = _gamma_norm(getattr(block, suffix))
                    if v is not None:
                        metrics[f"block_ls/{prefix}_{suffix}"] = _scalar_to_batch(v)

        if self.color_cross_attention is not None:
            for suffix in ("ls1", "ls2"):
                v = _gamma_norm(getattr(self.color_cross_attention, suffix))
                if v is not None:
                    metrics[f"block_ls/color_ca_{suffix}"] = _scalar_to_batch(v)

        return metrics

    def visualize(self, pred, data, **kwargs):
        device = pred["gaussians"].means.device

        total_gaussians = pred["gaussians"].num_gaussians
        actual_num_prototypes = total_gaussians // self.conf.gaussians_per_prototype

        prototype_ids = torch.arange(actual_num_prototypes, device=device)
        prototype_ids = prototype_ids[torch.randperm(actual_num_prototypes)]
        prototype_ids = repeat(prototype_ids, "N_p -> (N_p G)", G=self.conf.gaussians_per_prototype)
        pred["prototype_ids"] = prototype_ids

        clustered_gaussians = pred["gaussians"].color_gaussians_by_prototype(
            actual_num_prototypes, self.conf.gaussians_per_prototype
        )

        if device != torch.device("cuda"):
            clustered_gaussians = clustered_gaussians.to("cuda")
            data = mappings.batch_to_device(data, "cuda", non_blocking=False)

        pred["cluster_rgb_context"] = clustered_gaussians.render_view(
            cameras=data["context"]["camera"], poses=data["context"]["pose"]
        )["rendering"].to(device)
        if "target" in data:
            pred["cluster_rgb_target"] = clustered_gaussians.render_view(
                cameras=data["target"]["camera"], poses=data["target"]["pose"]
            )["rendering"].to(device)

        data = mappings.batch_to_device(data, device, non_blocking=False)
        return visualize_batch(pred, data, **kwargs)
