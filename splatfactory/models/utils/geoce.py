"""Patch geometry behind the GeoCE targets (models/losses.py GeoCELoss).

Every patch of a context view is unprojected to a 3D anchor, its median-depth pixel. The
kernel bandwidth is the scene's patch footprint (median 3D distance between adjacent patches),
shrunk per token by its photometric complexity. Key patches are represented by a small pixel
grid instead of one point, so a patch that straddles a depth edge gets fractional weight.

All functions work on ONE scene: V views, an H x W image and a pH x pW patch grid.

Author: Alexander Veicht
"""

import math

import torch
import torch.nn.functional as F


def _per_patch(t, pH, pW):
    """[V, H, W, C] -> [V, pH * pW, ph * pw, C]: the pixels of each patch, row-major."""
    V, H, W, C = t.shape
    ph, pw = H // pH, W // pW
    t = t.reshape(V, pH, ph, pW, pw, C).permute(0, 1, 3, 2, 4, 5)
    return t.reshape(V, pH * pW, ph * pw, C)


def patch_points(camera, pose, depth, depth_mask, pH, pW, min_valid_frac):
    """World point and validity of every patch.

    A patch is represented by its median-depth pixel: a real surface point on the dominant
    surface (the mean of a patch that straddles a depth edge lies in mid-air between them).

    Args:
        camera: Camera of the V views.
        pose: c2w Pose of the V views.
        depth: GT depth [V, H, W].
        depth_mask: Valid-depth mask [V, H, W].
        pH: Patch rows.
        pW: Patch columns.
        min_valid_frac: A patch is valid if at least this fraction of its pixels has depth.

    Returns:
        (points [V * pH * pW, 3] in world space, valid [V * pH * pW]).
    """
    p3d = camera.unproject_depth(depth * depth_mask)  # [V, H, W, 3] camera space
    frac = F.adaptive_avg_pool2d(depth_mask.float().unsqueeze(1), (pH, pW))
    valid = (frac.flatten(1) >= min_valid_frac).reshape(-1)

    d = _per_patch(depth.unsqueeze(-1), pH, pW).squeeze(-1)  # [V, T, ph * pw]
    d = torch.where(_per_patch(depth_mask.unsqueeze(-1), pH, pW).squeeze(-1), d, torch.nan)
    med = d.nanmedian(dim=-1, keepdim=True).values
    pick = (d - med).abs().nan_to_num(torch.inf).argmin(dim=-1)  # [V, T]
    pick = pick[..., None, None].expand(-1, -1, 1, 3)
    points = _per_patch(p3d, pH, pW).gather(2, pick)[:, :, 0]  # [V, T, 3]
    return pose.transform(points).reshape(-1, 3), valid


def patch_footprint(points, valid, V, pH, pW):
    """Median 3D distance between spatially adjacent valid patches, or None if there are none."""
    p = points.view(V, pH, pW, 3)
    m = valid.view(V, pH, pW)
    dx = (p[:, :, 1:] - p[:, :, :-1]).norm(dim=-1)[m[:, :, 1:] & m[:, :, :-1]]
    dy = (p[:, 1:] - p[:, :-1]).norm(dim=-1)[m[:, 1:] & m[:, :-1]]
    dists = torch.cat([dx, dy])
    return dists.median().clamp_min(1e-4) if dists.numel() else None


def photometric_sigma(image, valid, pH, pW, sigma, min_scale, max_scale):
    """Per-token kernel bandwidth [V * pH * pW] from photometric complexity.

    Complexity = per-patch mean of the squared grayscale forward differences. Valid tokens are
    ranked per scene and their sigma is log-interpolated from sigma * max_scale (flattest)
    down to sigma * min_scale (most complex). Invalid tokens keep the largest sigma, they are
    never anchors or keys.

    Args:
        image: Context images [V, 3, H, W] in [0, 1].
        valid: Patch validity [V * pH * pW].
        pH: Patch rows.
        pW: Patch columns.
        sigma: The scene's patch footprint.
        min_scale: Sigma factor at the highest complexity.
        max_scale: Sigma factor at the lowest complexity.
    """
    w = image.new_tensor([0.299, 0.587, 0.114])
    gray = (image * w[:, None, None]).sum(1)  # [V, H, W]
    energy = torch.zeros_like(gray)
    energy[:, :, :-1] += (gray[:, :, 1:] - gray[:, :, :-1]).square()
    energy[:, :-1] += (gray[:, 1:] - gray[:, :-1]).square()
    complexity = F.adaptive_avg_pool2d(energy.unsqueeze(1), (pH, pW)).reshape(-1)

    lo = math.log(sigma * min_scale)
    hi = math.log(sigma * max_scale)
    sig = torch.full_like(complexity, math.exp(hi))
    vidx = valid.nonzero(as_tuple=True)[0]
    if vidx.numel() > 1:
        rank = torch.argsort(torch.argsort(complexity[vidx])).float() / (vidx.numel() - 1)
        sig[vidx] = torch.exp(hi + rank * (lo - hi))
    return sig


def key_pixel_points(camera, pose, depth, depth_mask, pH, pW, vidx, num_pixels):
    """World points of an evenly spaced pixel grid inside every valid key patch.

    Args:
        camera: Camera of the V views.
        pose: c2w Pose of the V views.
        depth: GT depth [V, H, W].
        depth_mask: Valid-depth mask [V, H, W].
        pH: Patch rows.
        pW: Patch columns.
        vidx: Indices of the valid patches [n].
        num_pixels: Pixels per patch, on a square grid (16 -> 4 x 4).

    Returns:
        (points [n, m, 3], mask [n, m]) with m = round(sqrt(num_pixels)) ** 2.
    """
    V, H, W = depth.shape
    pp = H // pH
    side = max(1, int(round(float(num_pixels) ** 0.5)))
    offset = pp // (2 * side)
    lin = torch.linspace(offset, pp - 1 - offset, side, device=depth.device).round().long()
    ids = (lin[:, None] * pp + lin[None, :]).reshape(-1)

    p3d = camera.unproject_depth(depth * depth_mask).reshape(V, -1, 3)
    p3d = pose.transform(p3d).reshape(V, H, W, 3)
    points = _per_patch(p3d, pH, pW).reshape(V * pH * pW, pp * pp, 3)
    mask = _per_patch(depth_mask.unsqueeze(-1), pH, pW).reshape(V * pH * pW, pp * pp)
    return points[vidx][:, ids].float(), mask[vidx][:, ids]
