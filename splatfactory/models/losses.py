"""ZipSplat training losses, named as in the paper.

    LPIPSLoss: perceptual term of the rendering loss (GaussianHead).
    GeoCELoss: token similarities follow 3D proximity (geometry tokens).
    ConeLoss: a token's Gaussians stay on its source patch (Gaussian means).
    PoseLoss: camera head against the GT cameras (pose encoding).

The weight of each loss lives in its own default_conf, so a model conf block like
`geoce_loss: {}` turns it on at the paper setting. All losses return one value per scene [B].

Author: Alexander Veicht
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from omegaconf import OmegaConf

from splatfactory.models.utils.buffers import convert_to_buffer
from splatfactory.models.utils.geoce import (
    key_pixel_points,
    patch_footprint,
    patch_points,
    photometric_sigma,
)
from splatfactory.utils.conversions import matrix_to_quaternion


class LPIPSLoss(nn.Module):
    """LPIPS between the renders and the GT images. The frozen VGG adds no state_dict keys."""

    default_conf = {
        "net_type": "vgg",
        "compile": True,
        # inductor autotunes the VGG convs (cudagraphs off: batch shape varies per step)
        "compile_mode": "default",
    }

    def __init__(self, conf=None):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf or {})
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*(pretrained|Arguments other than a weight enum).*"
            )
            self.loss_fn = LPIPS(net=self.conf.net_type, verbose=False)
        convert_to_buffer(self.loss_fn, persistent=False)
        if self.conf.compile:
            self.forward = torch.compile(self.forward, mode=self.conf.compile_mode)

    def forward(self, pred, data):
        """pred["rendering"], data["image"]: [N, 3, H, W] in [0, 1] -> [N]."""
        predictions = pred["rendering"].clip(0, 1)
        targets = data["image"].clip(0, 1)
        loss = self.loss_fn(predictions, targets, normalize=True).squeeze()
        if loss.ndim == 0:
            loss = loss.unsqueeze(-1)
        return loss


class GeoCELoss(nn.Module):
    """GeoCE: geometry tokens that see nearby surfaces should be similar.

    Each patch is unprojected to a 3D anchor (its median-depth pixel, see models/utils/geoce.py).
    The soft target of anchor i over key j is a Gaussian kernel of their 3D distance,

        A_ij = mean_p exp(-||x_i - x_jp||^2 / (2 s_i s_j)),

    averaged over a pixel grid p inside key patch j and normalized per row. The bandwidth s is
    the scene's patch footprint, shrunk per token by its photometric complexity. The loss is
    the cross-entropy between the target rows and softmax(cos(z_i, z_j) / tau) over all valid
    tokens, self excluded. Anchors are subsampled per scene, keys stay complete.
    """

    default_conf = {
        "weight": 0.05,
        "tau": 0.1,  # softmax temperature on the cosine similarity
        "anchors": 2048,  # anchor rows sampled per scene, keys are all valid tokens
        "min_valid_frac": 0.5,  # a patch is valid if this fraction of its pixels has depth
        "key_pixels": 16,  # pixel grid per key patch (4 x 4)
        "sigma_min_scale": 0.4,  # bandwidth factor of the most complex token
        "sigma_max_scale": 1.0,  # bandwidth factor of the flattest token
        # torch.compile the per-scene loss (dynamic: the token count varies with V and depth)
        "compile": True,
    }

    def __init__(self, conf=None):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf or {})
        if self.conf.compile:
            self._scene_loss = torch.compile(self._scene_loss, dynamic=True)

    def _scene_loss(self, feats, pts, vidx, a_sel, sig_tok, pk, mk):
        """Soft cross-entropy of one scene, all layers in one batched matmul.

        Sampling stays outside (eager) to keep the compiled region deterministic.

        Args:
            feats: Tokens [L, V * T, D] (fp32).
            pts: Anchor points [V * T, 3].
            vidx: Valid token indices [n].
            a_sel: Anchor rows, indices into vidx [A].
            sig_tok: Per-token bandwidth [V * T].
            pk: Key pixel points [n, m, 3].
            mk: Key pixel validity [n, m].

        Returns:
            Scalar loss, or None when no anchor has a geometric neighbor.
        """
        n = vidx.numel()
        pa = pts[vidx].float()[a_sel]
        s = sig_tok[vidx].float()

        # Target: kernel over the m pixels of each key patch, averaged (chunked over anchors).
        pk2 = pk.square().sum(-1)  # [n, m]
        cnt = mk.sum(-1).clamp_min(1)  # [n]
        rows = []
        for st in range(0, pa.shape[0], 512):
            en = min(st + 512, pa.shape[0])
            dot = torch.einsum("cd,nmd->cnm", pa[st:en], pk)
            d2 = (pa[st:en].square().sum(-1)[:, None, None] + pk2[None] - 2 * dot).clamp_min(0)
            w = torch.exp(-d2 / (2.0 * (s[a_sel[st:en], None] * s[None, :])[..., None]))
            rows.append((w * mk[None]).sum(-1) / cnt[None])
        target = torch.cat(rows, dim=0)  # [A, n]
        target[torch.arange(a_sel.numel(), device=feats.device), a_sel] = 0.0  # self excluded
        row_sum = target.sum(dim=-1)
        keep = row_sum > 1e-6  # drop isolated anchors
        if not keep.any():
            return None
        target = target[keep] / row_sum[keep].unsqueeze(-1)  # [A', n]
        a_keep = a_sel[keep]

        # bf16 logits, fp32 log-softmax. Self exclusion is an additive mask in the matmul (an
        # in-place fill would clone the full [L, A, n] logits in backward).
        z = F.normalize(feats[:, vidx], dim=-1)  # [L, n, D]
        L, A = feats.shape[0], a_keep.numel()
        mask = torch.zeros(1, A, n, dtype=torch.bfloat16, device=feats.device)
        mask[0, torch.arange(A, device=feats.device), a_keep] = -1e9
        logits = torch.baddbmm(
            mask,
            z[:, a_keep].to(torch.bfloat16),
            z.transpose(1, 2).to(torch.bfloat16),
            alpha=1.0 / self.conf.tau,
        )  # [L, A, n]
        logsm = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        return -torch.einsum("an,lan->", target, logsm) / (L * A)

    def forward(self, layer_tokens, context):
        """Per-scene GeoCE.

        Args:
            layer_tokens: List of L token maps [B, V, T, D].
            context: Context views with image, camera, pose (c2w), depth and depth_mask.

        Returns:
            [B] loss, mean over the layers. Scenes without usable geometry get a
            graph-connected zero.
        """
        B, V, T, _ = layer_tokens[0].shape
        H, W = context["depth"].shape[-2:]
        pH = int(round((T * H / W) ** 0.5))
        pW = T // pH
        assert pH * pW == T, f"T={T} does not match the {H}x{W} patch grid"

        losses = []
        for b in range(B):
            camera, pose = context["camera"][b], context["pose"][b]
            depth, depth_mask = context["depth"][b], context["depth_mask"][b]
            pts, valid = patch_points(
                camera, pose, depth, depth_mask, pH, pW, self.conf.min_valid_frac
            )
            sigma = patch_footprint(pts, valid, V, pH, pW)
            vidx = valid.nonzero(as_tuple=True)[0]
            loss = None
            if sigma is not None and vidx.numel() >= 8:
                sig_tok = photometric_sigma(
                    context["image"][b],
                    valid,
                    pH,
                    pW,
                    sigma,
                    self.conf.sigma_min_scale,
                    self.conf.sigma_max_scale,
                )
                feats = torch.stack([tok[b].reshape(V * T, -1) for tok in layer_tokens]).float()
                a_sel = torch.randperm(vidx.numel(), device=feats.device)
                a_sel = a_sel[: int(self.conf.anchors)]
                pk, mk = key_pixel_points(
                    camera, pose, depth, depth_mask, pH, pW, vidx, self.conf.key_pixels
                )
                loss = self._scene_loss(feats, pts, vidx, a_sel, sig_tok, pk, mk)
            if loss is None:
                loss = layer_tokens[0][b].sum() * 0.0  # zero, graph-connected
            losses.append(loss)
        return torch.stack(losses)


class ConeLoss(nn.Module):
    """Cone loss: a token's Gaussians should project near its source patches.

    Each Gaussian is projected into the source view of every member token of its cluster. Its
    distance to the nearest member patch center (in patch units) is penalized by a hinged
    Cauchy: zero inside rho, bounded pull far away. Behind-camera projections get the sentinel
    distance.
    """

    default_conf = {
        "weight": 0.01,
        "rho": 2.0,  # free radius, patch units
        "c": 2.0,  # Cauchy scale, patch units
        "sentinel": 20.0,  # distance of a behind-camera projection, patch units
    }

    def __init__(self, conf=None):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf or {})

    def forward(self, means, assign, pose, camera, patch_grid, patch_px):
        """Per-scene cone loss.

        Args:
            means: Gaussian means grouped per token [B, K, G, 3].
            assign: Cluster id of every context token [B, M], M = S * pH * pW.
            pose: c2w Pose of the S context views [B, S].
            camera: Camera of the context views [B, S].
            patch_grid: (pH, pW) of the context images.
            patch_px: Patch size in pixels.

        Returns:
            [B] loss.
        """
        B, K, G, _ = means.shape
        M = assign.shape[1]
        pH, pW = patch_grid
        T = pH * pW

        idx = torch.arange(M, device=means.device)
        views = idx // T  # source view of each token
        centers = torch.stack(
            [(idx % T % pW + 0.5) * patch_px, (idx % T // pW + 0.5) * patch_px], dim=-1
        )  # [M, 2] patch centers in pixels (x, y)

        # Each token's cluster Gaussians, projected into the token's source view.
        w2c = pose.inv()[:, views].float()
        cam = camera[:, views].float()
        members = means.gather(1, assign[..., None, None].expand(B, M, G, 3))  # [B, M, G, 3]
        p2d, in_front = cam.project(w2c.transform(members))
        p2d, _ = cam.distort(p2d)
        p2d = cam.denormalize(p2d)  # [B, M, G, 2] pixels

        d = (p2d - centers[None, :, None]).norm(dim=-1) / patch_px  # [B, M, G] patch units
        d = torch.where(in_front, d, d.new_full((), self.conf.sentinel))

        # Nearest member cone per Gaussian: the free region is the union of member patches.
        d_min = d.new_full((B, K, G), self.conf.sentinel).scatter_reduce_(
            1, assign[..., None].expand(B, M, G), d, reduce="amin", include_self=True
        )
        excess = (d_min - self.conf.rho).clamp_min(0)
        c = self.conf.c
        return ((c**2 / 2) * torch.log1p((excess / c) ** 2)).mean(dim=(1, 2))


class PoseLoss(nn.Module):
    """Pose loss on the raw camera-head encoding [t, quat_xyzw, fov_h, fov_w] (c2w).

    VGGT's camera loss with one refinement stage: L1 per component against the GT encoded the
    same way, with VGGT's clamp on the translation L1. The GT quaternion is sign-aligned to the
    prediction (q and -q are the same rotation). Supervising the raw quaternion keeps
    matrix_to_quaternion, whose gradient is NaN near axis-aligned rotations, out of the graph.
    """

    default_conf = {
        "weight": 0.05,
        "t_weight": 1.0,
        "rot_weight": 1.0,
        "fov_weight": 0.5,
        "t_clamp": 100.0,  # outlier clamp on the per-element translation L1
    }

    def __init__(self, conf=None):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf or {})

    def forward(self, pose_enc, pred_pose, pred_camera, gt_pose, gt_camera):
        """Pose loss of one batch.

        Args:
            pose_enc: Raw c2w head encoding [B, S, 9].
            pred_pose: Decoded c2w Pose [B, S] (metrics only).
            pred_camera: Decoded Camera [B, S] (metrics only).
            gt_pose: GT c2w Pose [B, S].
            gt_camera: GT Camera [B, S].

        Returns:
            (losses, metrics), dicts of [B] tensors.
        """
        pred_t = pose_enc[..., :3].float()
        pred_q = F.normalize(pose_enc[..., 3:7].float(), dim=-1)
        pred_fov = pose_enc[..., 7:9].float()
        with torch.no_grad():
            tgt_t = gt_pose.t.float()
            tgt_q = matrix_to_quaternion(gt_pose.R.float())[..., [1, 2, 3, 0]]  # wxyz -> xyzw
            tgt_q = tgt_q * torch.where((pred_q * tgt_q).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
            tgt_fov = torch.stack([gt_camera.vfov, gt_camera.hfov], dim=-1).float()

        t_loss = (pred_t - tgt_t).abs().clamp(max=self.conf.t_clamp).mean(dim=(1, 2))
        rot_loss = (pred_q - tgt_q).abs().mean(dim=(1, 2))
        fov_loss = (pred_fov - tgt_fov).abs().mean(dim=(1, 2))
        losses = {
            "pose": (
                self.conf.t_weight * t_loss
                + self.conf.rot_weight * rot_loss
                + self.conf.fov_weight * fov_loss
            ),
            "pose_t": t_loss,
            "pose_rot": rot_loss,
            "pose_fov": fov_loss,
        }
        return losses, pose_errors(pred_pose, pred_camera, gt_pose, gt_camera)


@torch.no_grad()
def pose_errors(pred_pose, pred_camera, gt_pose, gt_camera):
    """Mean per-view errors [B] in w2c: translation, rotation angle (deg), fov (deg)."""
    pred_w2c, gt_w2c = pred_pose.inv(), gt_pose.inv()
    pred_t, tgt_t = pred_w2c.t.float(), gt_w2c.t.float()
    pred_R, tgt_R = pred_w2c.R.float(), gt_w2c.R.float()
    pred_fov = torch.stack([pred_camera.vfov, pred_camera.hfov], dim=-1).float()
    tgt_fov = torch.stack([gt_camera.vfov, gt_camera.hfov], dim=-1).float()
    trace = (pred_R.transpose(-1, -2) @ tgt_R).diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return {
        "pose/t_err": (pred_t - tgt_t).norm(dim=-1).mean(dim=1),
        "pose/rot_err_deg": torch.rad2deg(torch.acos(cos)).mean(dim=1),
        "pose/fov_err_deg": torch.rad2deg((pred_fov - tgt_fov).abs()).mean(dim=(1, 2)),
    }
