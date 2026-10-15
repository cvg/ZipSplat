"""ZipSplat: feed-forward 3D Gaussian Splatting from unposed multi-view images.

Pipeline:
    1. A DA3 backbone with cross-view attention encodes the input views into per-layer
       tokens (local | global halves) plus a camera token per view.
    2. Prepare: each layer's payload is the global half plus a gated LayerNorm of the local
       half. A ColorEmbedMLP turns the RGB patches into color tokens, and a TokenMerge over
       all payloads and the color tokens gives one geometry token per patch.
    3. Compress: k-means on the geometry tokens picks K = r * V * T clusters (skipped at
       r = 1). The scene tokens start as the per-cluster member means.
    4. Fuse: per layer, deepest first, a cross-attention (q = scene tokens, k = geometry
       tokens, v = that layer's payload) and a self-attention. A 4th cross-attention reads
       the color tokens.
    5. Decode: the GaussianHead turns each scene token into G Gaussians with free 3D
       offsets. DA3's camera decoder reads pose and intrinsics off the deepest camera token.

Training adds GeoCE on the geometry tokens, the cone loss on the Gaussian means and the
pose loss on top of the head's rendering loss (models/losses.py).

Usage:
    model = get_model("networks.zipsplat")(conf)
    pred = model({"context": {"image": images}})  # images [B, V, 3, H, W] in [0, 1]
    pred["gaussians"], pred["pose_pred"], pred["camera_pred"]

Author: Alexander Veicht
"""

import random

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint as ckpt

from splatfactory import get_logger
from splatfactory.geometry.utils.clustering import cluster_mean, complexity_kmeans
from splatfactory.geometry.utils.overlap import context_self_overlap_degree, pixel_overlap_degree
from splatfactory.models import BaseModel
from splatfactory.models.decoders.cam_dec import decode_pose_encoding
from splatfactory.models.decoders.gaussian_head import GaussianHead
from splatfactory.models.encoders.dav3_encoder import IMAGENET_MEAN, IMAGENET_STD, DAV3Encoder
from splatfactory.models.losses import ConeLoss, GeoCELoss, PoseLoss
from splatfactory.models.modules import transformer_block
from splatfactory.models.modules.attention import set_attention_backend
from splatfactory.models.modules.misc import ColorEmbedMLP, LayerScale, TokenMerge
from splatfactory.utils import mappings
from splatfactory.visualization.token_pca import pca_colorize
from splatfactory.visualization.visualize_batch import visualize_batch

logger = get_logger(__name__)


class ZipSplat(BaseModel):
    """Feed-forward 3DGS model: images -> Gaussians, poses and intrinsics."""

    head_class = GaussianHead

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
            "weights": "weights/da3/da3-giant-1.1.pth",
            "trainable": True,  # false: frozen trunk (the camera head still trains)
            "use_checkpoint": False,
        },
        "weights": None,  # full-model checkpoint, loaded by BaseModel after _init
        "compile_loss": False,  # LPIPS and GeoCE compile themselves
        # Attention blocks (num_heads = embed_dim / 64)
        "num_heads": 24,
        "mlp_ratio": 4.0,
        "fuse_layer_scale": 0.01,  # LayerScale init of the fusion blocks
        "prep_layer_scale": 0.1,  # LayerScale init of the local-half gates
        # Compression: training samples r in [min, max] per step, eval uses max. The min shrinks
        # as (2 / V) ** query_scale_with_views, so the token floor stays constant over V.
        "query_sample_ratio": [0.1, 1.0],
        "query_scale_with_views": 1.0,
        "knn": {},  # complexity_kmeans overrides (geometry/utils/clustering.py)
        # Losses (models/losses.py): the weights here, the other settings in each loss's conf
        "geoce_loss": {"weight": 0.05},
        "cone_loss": {"weight": 0.01},
        "pose_loss": {"weight": 0.05},
        # Runtime
        "attn_backend": "sdpa",  # sdpa | fa4 (FlashAttention-4, Hopper+, bf16 only)
        "use_checkpoint": False,  # activation checkpointing of the fusion blocks
        "return_attention": True,  # eval: return the fusion attention maps for viz
        "gaussians_per_token": 32,
        "gaussian_head": {},  # GaussianHead overrides (decoders/gaussian_head.py)
    }

    required_data_keys = ["context"]

    # ============================================================
    # Setup
    # ============================================================

    def _init(self, conf):
        mean, std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        self.register_buffer("_img_mean", mean.view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", std.view(1, 1, 3, 1, 1), persistent=False)

        set_attention_backend(conf.attn_backend)

        # Sets self.embed_dim, self.num_layers, self.patch_size.
        self._setup_backbone(conf)
        # The inference ratio: callers may set it on the model without rebuilding.
        self.query_sample_ratio = list(conf.query_sample_ratio)

        self._setup_prepare(conf)
        self._setup_fusion(conf)
        self._setup_pose(conf)
        self._setup_head(conf)

        # Losses
        self.geoce_loss = GeoCELoss(conf.geoce_loss)
        self.cone_loss = ConeLoss(conf.cone_loss)
        self.pose_loss = PoseLoss(conf.pose_loss)

        r_min, r_max = conf.query_sample_ratio
        logger.info(
            f"ZipSplat: {conf.gaussians_per_token} Gaussians per scene token, query ratio "
            f"{r_min}-{r_max}."
        )

    def _setup_backbone(self, conf):
        self.backbone = DAV3Encoder(conf.backbone)
        self.embed_dim = self.backbone.embed_dim
        self.num_layers = len(conf.backbone.out_layers)
        self.patch_size = self.backbone.backbone.patch_size

    def _setup_prepare(self, conf):
        """Local-half gates, the color embedding and the geometry TokenMerge."""
        embed_dim, num_layers = self.embed_dim, self.num_layers
        self.local_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.local_gate = nn.ModuleList(
            [LayerScale(embed_dim, init_values=conf.prep_layer_scale) for _ in range(num_layers)]
        )
        self.color_embed_mlp = ColorEmbedMLP(self.patch_size, 3, embed_dim)
        self.geo_merge = TokenMerge(embed_dim, num_layers + 1)

    def _setup_fusion(self, conf):
        """Per-layer cross- and self-attention, plus the color cross-attention."""
        num_layers = self.num_layers
        block = lambda cls: cls(  # noqa: E731
            dim=self.embed_dim,
            num_heads=conf.num_heads,
            mlp_ratio=conf.mlp_ratio,
            qk_norm=True,
            init_values=conf.fuse_layer_scale,
        )
        self.cross_attention = nn.ModuleList(
            [block(transformer_block.CrossAttentionBlock) for _ in range(num_layers)]
        )
        self.self_attention = nn.ModuleList(
            [block(transformer_block.SelfAttentionBlock) for _ in range(num_layers)]
        )
        self.color_ca = block(transformer_block.CrossAttentionBlock)

    def _setup_pose(self, conf):
        """No-op: DA3's camera decoder ships with the backbone (DAV3Encoder.cam_dec)."""

    def _setup_head(self, conf):
        self.gaussian_head = self.head_class(
            {
                "embed_dim": self.embed_dim,
                "gaussians_per_token": conf.gaussians_per_token,
                **conf.gaussian_head,
            }
        )

    def _compile(self, *args, **kwargs):
        """Compile the head decode and the fusion + DA3 blocks (Trainer.prepare_model).

        Runs after checkpoint loading, and wraps block.forward instead of replacing the
        module: torch.compile(module) prefixes state_dict keys with _orig_mod.
        """
        self.gaussian_head.compile(*args, **kwargs)
        if self.conf.use_checkpoint:
            return  # compiled callables inside torch.utils.checkpoint are unvalidated
        for blk in (*self.cross_attention, *self.self_attention, *self.backbone.backbone.blocks):
            blk.forward = torch.compile(blk.forward, dynamic=True, *args, **kwargs)

    # ============================================================
    # Backbone
    # ============================================================

    def _use_priors(self):
        """Inject the GT cameras on 20% of the training steps (ZipSplatEval decides at eval)."""
        return self.training and torch.rand(1).item() < 0.2

    def _backbone_data(self, data):
        """Backbone input dict: the normalized images, plus the camera priors when injected."""
        image = (data["context"]["image"] - self._img_mean) / self._img_std
        backbone_data = {"image": image}
        if self.backbone.conf.with_camera_enc:
            if self._use_priors() and "pose" in data["context"] and "camera" in data["context"]:
                backbone_data["use_priors"] = True
                # c2w, the encoder converts to the DA3 convention
                backbone_data["pose"] = data["context"]["pose"].Rt
                backbone_data["camera"] = data["context"]["camera"].K
        return backbone_data

    def _backbone_features(self, data):
        """Per-layer patch tokens [B, V, T, 2 * D] (shallow to deep) and the camera token.

        Only the deepest camera token [B, V, 2 * D] is read, by the camera head as in DA3.
        """
        tokens = self.backbone(self._backbone_data(data))["feats"]
        return [patch for patch, _ in tokens], tokens[-1][1]

    def _predict_pose(self, cam_token, H, W):
        """Camera head readout -> (c2w Pose, Camera, raw c2w encoding [B, V, 9]).

        Runs in fp32, bf16 rounding of the readout shows in the pose error.
        """
        with torch.autocast("cuda", enabled=False):
            pose_enc = self.backbone.cam_dec(cam_token.float())  # t | quat_xyzw | fov
        return (*decode_pose_encoding(pose_enc, (H, W)), pose_enc)

    # ============================================================
    # Prepare, compress, fuse
    # ============================================================

    def _prepare(self, features, images):
        """Payload, geometry and color tokens.

        payload_l = global_l + LS_l(LN_l(local_l))
        color = ColorEmbedMLP(rgb)
        geo = TokenMerge(payload_0, ..., payload_{L-1}, color)

        Args:
            features: Per-layer [B, V, T, 2 * D] backbone tokens, shallow to deep.
            images: Context images [B, V, 3, H, W].

        Returns:
            (layer_tokens, geo_tokens, color_tokens), each [B, V, T, D] (layer_tokens a list
            over layers).
        """
        D = self.embed_dim
        layer_tokens = [
            raw[..., D:] + self.local_gate[l](self.local_norm[l](raw[..., :D]))
            for l, raw in enumerate(features)
        ]
        col = self.color_embed_mlp(rearrange(images, "B V C H W -> (B V) C H W"))
        col = rearrange(col, "(B V) T D -> B V T D", B=images.shape[0])
        return layer_tokens, self.geo_merge([*layer_tokens, col]), col

    def _cluster(self, geo_tokens, images):
        """k-means on the geometry tokens.

        Args:
            geo_tokens: [B, V, T, D] geometry tokens.
            images: Context images [B, V, 3, H, W] for the complexity seeding.

        Returns:
            (nearest_idx [B, K], assignments [B, V * T]), both arange(V * T) when the sampled
            ratio keeps every token.
        """
        B, V, T, D = geo_tokens.shape
        VT = V * T
        ratio_min, ratio_max = self.query_sample_ratio
        if self.conf.query_scale_with_views > 0:
            ratio_min = min(ratio_min, ratio_min * (2 / V) ** self.conf.query_scale_with_views)
        if ratio_min == ratio_max or not self.training:
            ratio = ratio_max
        else:
            ratio = random.uniform(ratio_min, ratio_max)
        K = max(1, int(VT * ratio))
        if K >= VT:
            idx = torch.arange(VT, device=geo_tokens.device).unsqueeze(0).expand(B, -1)
            return idx, idx

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return complexity_kmeans(
                rearrange(geo_tokens, "B V T D -> B (V T) D"),
                K,
                images,
                self.patch_size,
                **self.conf.knn,
            )

    def _scene_tokens(self, geo_tokens, images):
        """k-means cluster means of the geometry tokens [B, K, D], nearest_idx, assignments."""
        nearest_idx, assignments = self._cluster(geo_tokens, images)
        geo_flat = rearrange(geo_tokens, "B V T D -> B (V T) D")
        scene_tokens = cluster_mean(geo_flat, assignments, nearest_idx.shape[1])
        return scene_tokens, nearest_idx, assignments

    def _ca_step(self, block, queries, keys, values, use_ckpt=False, return_attention=False):
        """One full cross-attention block (LayerScale-gated residual on the queries)."""
        if use_ckpt:
            return ckpt(block, queries, keys, values, use_reentrant=False), None
        if return_attention:
            return block(query=queries, key=keys, value=values, return_attention=True)
        return block(query=queries, key=keys, value=values), None

    def _fuse(self, features, images):
        """Prepare, compress and fuse the backbone tokens into scene tokens.

        Returns:
            scene_tokens: [B, K, D].
            nearest_idx: [B, K].
            assignments: [B, V * T].
            attention: list of attention maps, or None.
            layer_tokens: [geo_tokens], the GeoCE input.
        """
        layer_tokens, geo_tokens, color_tokens = self._prepare(features, images)
        scene_tokens, nearest_idx, assignments = self._scene_tokens(geo_tokens, images)
        # The attention keys / values: every token of every view, [B, V * T, D].
        geo_flat = rearrange(geo_tokens, "B V T D -> B (V T) D")
        layer_flat = [rearrange(t, "B V T D -> B (V T) D") for t in layer_tokens]
        color_flat = rearrange(color_tokens, "B V T D -> B (V T) D")

        return_attn = (not self.training) and self.conf.return_attention
        use_ckpt = self.training and self.conf.use_checkpoint
        attn_weights = [] if return_attn else None

        # Deepest layer first: coarse scene layout before fine detail.
        for l in reversed(range(self.num_layers)):
            scene_tokens, attn = self._ca_step(
                self.cross_attention[l],
                scene_tokens,
                geo_flat,
                layer_flat[l],
                use_ckpt=use_ckpt,
                return_attention=return_attn,
            )
            if return_attn:
                attn_weights.append(attn)
            scene_tokens = self.self_attention[l](scene_tokens)

        # 4th CA: color values on the geometry keys, no trailing SA.
        scene_tokens, attn = self._ca_step(
            self.color_ca,
            scene_tokens,
            geo_flat,
            color_flat,
            use_ckpt=use_ckpt,
            return_attention=return_attn,
        )
        if return_attn:
            attn_weights.append(attn)

        return {
            "scene_tokens": scene_tokens,
            "attention": attn_weights,
            "nearest_idx": nearest_idx,
            "assignments": assignments,
            "layer_tokens": [geo_tokens],  # live refs: GeoCE input, viz token PCA
        }

    # ============================================================
    # Forward
    # ============================================================

    def _forward(self, data):
        images = data["context"]["image"]  # [B, V, 3, H, W]
        H, W = images.shape[-2:]

        features, cam_token = self._backbone_features(data)
        fused = self._fuse(features, images)
        scene_tokens = fused["scene_tokens"]

        # Training skips the context renders.
        render_context = not self.training or self.gaussian_head.conf.loss_on_context
        with torch.autocast("cuda", enabled=False):
            result = self.gaussian_head(
                {"tokens": scene_tokens.float(), "render_context": render_context} | data
            )
        result["scene_tokens"] = scene_tokens  # live tokens TokenBA refits
        result["layer_tokens"] = fused["layer_tokens"]
        pose_pred, camera_pred, pose_enc = self._predict_pose(cam_token, H, W)
        result |= {"pose_pred": pose_pred, "camera_pred": camera_pred, "pose_enc": pose_enc}
        result["nearest_idx"] = fused["nearest_idx"]
        result["assignments"] = fused["assignments"]
        if fused["attention"] is not None:
            result["attention"] = fused["attention"]
        return result

    # ============================================================
    # Loss, metrics, visualization
    # ============================================================

    def loss(self, pred, data):
        """Rendering loss + GeoCE + pose loss + cone loss. Returns (losses, metrics), [B] each."""
        losses, metrics = self.gaussian_head.loss(pred, data)
        context = data["context"]

        # GeoCE: token similarity follows 3D proximity (GT depth + pose).
        geoce_weight = self.geoce_loss.conf.weight
        if self.training and geoce_weight > 0 and "layer_tokens" in pred:
            with torch.autocast("cuda", enabled=False):
                losses["geoce"] = self.geoce_loss(pred["layer_tokens"], context)
            losses["total"] = losses["total"] + geoce_weight * losses["geoce"]

        # Pose loss (also on prior-injected steps).
        if "pose" in context:
            with torch.autocast("cuda", enabled=False):
                pose_losses, pose_metrics = self.pose_loss(
                    pred["pose_enc"],
                    pred["pose_pred"],
                    pred["camera_pred"],
                    context["pose"],
                    context["camera"],
                )
            losses.update(pose_losses)
            losses["total"] = losses["total"] + self.pose_loss.conf.weight * pose_losses["pose"]
            metrics = {**metrics, **pose_metrics}

        # Cone loss: each token's Gaussians stay on its source patches.
        cone_weight = self.cone_loss.conf.weight
        if cone_weight > 0 and "pose" in context:
            H, W = context["image"].shape[-2:]
            means = pred["gaussians"].means
            means = means.reshape(means.shape[0], -1, self.conf.gaussians_per_token, 3)
            with torch.autocast("cuda", enabled=False):
                losses["cone"] = self.cone_loss(
                    means.float(),
                    pred["assignments"],
                    context["pose"],
                    context["camera"],
                    (H // self.patch_size, W // self.patch_size),
                    float(self.patch_size),
                )
            losses["total"] = losses["total"] + cone_weight * losses["cone"]

        return losses, metrics

    def visualize(self, pred, data, **kwargs):
        """Cluster-colored and token-PCA renders plus the GT overlap maps, then the batch figure."""
        device = pred["gaussians"].means.device
        G = self.conf.gaussians_per_token
        num_tokens = pred["gaussians"].num_gaussians // G

        token_ids = torch.randperm(num_tokens, device=device)
        pred["prototype_ids"] = repeat(token_ids, "N -> (N G)", G=G)
        clustered = pred["gaussians"].color_gaussians_by_prototype(num_tokens, G)
        if device != torch.device("cuda"):
            clustered = clustered.to("cuda")
            data = mappings.batch_to_device(data, "cuda", non_blocking=False)

        show_context = kwargs.get("show_context", False)

        def render(gaussians, view):
            return gaussians.render_view(cameras=data[view]["camera"], poses=data[view]["pose"])[
                "rendering"
            ].to(device)

        if show_context:
            pred["cluster_rgb_context"] = render(clustered, "context")
        if "target" in data:
            pred["cluster_rgb_target"] = render(clustered, "target")

        # Token PCA: each query's Gaussians take the PCA color of its geometry token.
        if pred.get("layer_tokens") and pred.get("nearest_idx") is not None:
            tok = rearrange(pred["layer_tokens"][0], "B V T D -> B (V T) D")
            rgb_tok = pca_colorize(tok[0].float().to("cuda"), l2norm=True, whiten=True)
            g_rgb = repeat(rgb_tok[pred["nearest_idx"][0]], "K C -> (K G) C", G=G)
            pca = clustered.color_gaussians_by_rgb(g_rgb)
            if show_context:
                pred["token_pca_context"] = render(pca, "context")
            if "target" in data:
                pred["token_pca_target"] = render(pca, "target")

        if "depth" in data.get("target", {}) and "depth" in data["context"]:
            pred["overlap_target"] = pixel_overlap_degree(data)
            if show_context:
                pred["overlap_context"] = context_self_overlap_degree(data)

        data = mappings.batch_to_device(data, device, non_blocking=False)
        return visualize_batch(pred, data, **kwargs)
