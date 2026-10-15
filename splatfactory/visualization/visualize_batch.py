"""Training and eval figures for one batch of predictions and ground truth."""

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.utils.scale import mean_cam_transform
from splatfactory.models.metrics import calculate_lpips, calculate_psnr
from splatfactory.utils.mappings import batch_to_device
from splatfactory.visualization import viz2d

PLOT_DPI = 50


def _short_scene_id(scene_id: str) -> str:
    """Shorten scene IDs with long hashes (e.g. dl3dv-06da79...-> dl3dv-06da79..)."""
    parts = scene_id.split("-", 1)
    if len(parts) == 2 and len(parts[1]) > 8:
        return f"{parts[0]}-{parts[1][:8]}"
    return scene_id


def tensor_to_img(t):
    """[3, H, W] tensor -> [H, W, 3] CPU image in [0, 1]."""
    return t.cpu().detach().permute(1, 2, 0).clip(0, 1)


def _scene_info_text(data, idx: int = 0) -> str:
    """Scene name + view counts, overlaid on the first ground-truth view."""
    scene = _short_scene_id(data["scene"][idx])
    n_ctx = len(data["context"]["image"][idx])
    n_tgt = len(data["target"]["image"][idx])
    return f"{scene}\n{n_ctx} ctx -> {n_tgt} tgt"


def compute_view_scores(pred_rgb: torch.Tensor, gt_image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute PSNR and LPIPS between predicted and ground truth views."""
    return {
        "psnr": calculate_psnr(pred_rgb, gt_image),
        "lpips": calculate_lpips(pred_rgb, gt_image),
    }


def format_score_text(scores: dict[str, torch.Tensor], view_idx: int) -> str:
    """Format PSNR/LPIPS scores for a single view as annotation text."""
    return (
        f"PSNR: {scores['psnr'][view_idx].item():.2f}\n"
        f"LPIPS: {scores['lpips'][view_idx].item():.3f}"
    )


def make_context_and_targets_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
    dpi: int = PLOT_DPI,
    show_context: bool = False,
) -> dict[str, Any]:
    """One row per view, one column per quantity. Optional columns are skipped when missing."""
    if "target_rgb" not in pred:
        return {}
    show_context = show_context and "context_rgb" in pred

    idx = 0
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    splits = (["context"] if show_context else []) + ["target"]

    n_context = len(data["context"]["image"][idx])
    has_overlap = all(f"overlap_{split}" in pred for split in splits)
    has_cluster = all(f"cluster_rgb_{split}" in pred for split in splits)
    has_token_pca = all(f"token_pca_{split}" in pred for split in splits)
    has_gt_depth = all("depth" in data[split] for split in splits)

    def _masked(t):
        t = t.clone().float()
        t[t < 0] = float("nan")  # invalid depth -> transparent
        return t

    gt_imgs, renderings, clusters, token_pcas, diffs = [], [], [], [], []
    gt_depths, depths, overlaps = [], [], []
    id_texts, score_texts = [], []
    for split in splits:
        gt = data[split]["image"][idx]
        rgb = pred[f"{split}_rgb"][idx]
        scores = compute_view_scores(rgb, gt)
        for i in range(len(gt)):
            gt_imgs.append(tensor_to_img(gt[i]))
            renderings.append(tensor_to_img(rgb[i]))
            diffs.append(tensor_to_img((rgb[i] - gt[i]).abs() * 3))  # diff amplified 3x
            depths.append(pred[f"{split}_depth"][idx][i])
            if has_cluster:
                clusters.append(tensor_to_img(pred[f"cluster_rgb_{split}"][idx][i]))
            if has_token_pca:
                token_pcas.append(tensor_to_img(pred[f"token_pca_{split}"][idx][i]))
            if has_gt_depth:
                gt_depths.append(_masked(data[split]["depth"][idx][i]))
            if has_overlap:
                overlaps.append(_masked(pred[f"overlap_{split}"][i]))
            img_id = int(data[split]["index"][idx][i].item())
            id_texts.append(f"{split.capitalize()} ID: {img_id}")
            score_texts.append(format_score_text(scores, i))
    id_texts[0] = _scene_info_text(data, idx) + "\n" + id_texts[0]

    # Shared scale for the two depth columns (GT invalids are nan already).
    depth_pool = torch.stack(depths).flatten()
    if has_gt_depth:
        g = torch.stack(gt_depths).flatten()
        depth_pool = torch.cat([depth_pool, g[g.isfinite()]])
    max_depth = depth_pool.quantile(0.95).item()
    depth_kw = dict(colorbar=True, vmax=max_depth, alpha=1.0)
    overlap_kw = dict(cmap="turbo", vmax=n_context, colorbar=True, alpha=1.0)

    # (header, per-view base images, per-view heatmaps or None, heatmap kwargs)
    columns = [
        ("GT", gt_imgs, None, None),
        ("Rendering", renderings, None, None),
        ("Diff x3", diffs, None, None),
    ]
    if has_overlap:
        columns.append(("Ctx Overlap", gt_imgs, overlaps, overlap_kw))
    if has_gt_depth:
        columns.append(("GT Depth", gt_imgs, gt_depths, depth_kw))
    columns.append(("Depth", gt_imgs, depths, depth_kw))
    if has_cluster:
        columns.append(("Clusters", clusters, None, None))
    if has_token_pca:
        columns.append(("Token PCA", token_pcas, None, None))

    n_views = len(gt_imgs)
    grid = [[col[1][i] for col in columns] for i in range(n_views)]
    titles = [[col[0] for col in columns]] + [[""] * len(columns) for _ in range(n_views - 1)]

    fig, ax = viz2d.plot_image_grid(grid, titles=titles, dpi=dpi)
    ax = np.array(ax).reshape(n_views, len(columns))

    for i in range(n_views):
        viz2d.add_text(0, id_texts[i], axes=ax[i])
        viz2d.add_text(1, score_texts[i], axes=ax[i])
    for c, (_, _, maps, kw) in enumerate(columns):
        if maps is None:
            continue
        for i in range(n_views):
            viz2d.plot_heatmaps([maps[i]], axes=[ax[i, c]], **kw)

    return {"context_and_target": fig}


def make_attention_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
    dpi: int = PLOT_DPI,
) -> dict[str, Any]:
    """Create a figure for visualizing the attention maps."""
    if "attention" not in pred or len(pred["attention"]) == 0:
        return {}

    n_pairs = min(n_pairs, len(data["context"]["image"]))
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    attention_maps = pred["attention"]
    attention_maps = torch.stack(attention_maps)
    attention_maps = rearrange(attention_maps, "L B N T -> B L T N")[:n_pairs]  # (B, L, T, N)

    title_row = [f"CA - Layer {i}" for i in range(attention_maps.shape[1])]
    titles = [title_row for _ in range(n_pairs)]

    fig, axes = viz2d.plot_image_grid(attention_maps, titles=titles)

    for i in range(n_pairs):
        maps = attention_maps[i]
        vmax = np.quantile(maps.cpu().numpy(), 0.995)
        viz2d.plot_heatmaps(
            maps,
            vmax=vmax,
            cmap="viridis",
            colorbar=True,
            alpha=1.0,
            axes=axes[i],
        )

        for idx, attn in enumerate(maps):
            attn = attn.T
            attn_entropy = -(attn * (attn + 1e-8).log()).sum(-1).mean().item()
            max_entropy = math.log(attn.shape[-1])
            max_attn = attn.max(-1)[0].mean().item()
            viz2d.add_text(
                idx,
                f"Entropy: {attn_entropy:.2f}/{max_entropy:.2f}"
                + f"\nRatio: {attn_entropy/max_entropy:.2%}"
                + f"\nMax: {max_attn:.4f} / {1/attn.shape[-1]:.4f}",
                fs=12,
                axes=axes[i],
            )

    return {"attention_maps": fig}


def make_gaussian_stats_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], dpi: int = PLOT_DPI
) -> dict[str, Any]:
    """Opacity and scale distribution histograms for predicted Gaussians."""
    if "gaussians" not in pred or "activated_gaussians" not in pred:
        return {}

    gs = pred["gaussians"][0]
    opacities = gs.opacities.detach().cpu().float()
    scales = gs.scales.detach().cpu().float()
    max_scale = scales.max(dim=-1).values

    dead = ~pred["activated_gaussians"][0].bool().detach().cpu()
    dead_frac = dead.float().mean().item()
    n_total = len(opacities)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- 1. Opacity histogram ---
    ax = axes[0]
    bins = np.linspace(0, 1, 101)
    ax.hist(opacities[~dead].numpy(), bins=bins, alpha=0.7, color="#2176AE", label="alive")
    ax.hist(opacities[dead].numpy(), bins=bins, alpha=0.7, color="#E8553A", label="dead")
    ax.set_xlabel("opacity")
    ax.set_ylabel("count")
    ax.set_title(f"Opacity  (dead={dead_frac:.1%}, n={n_total})")
    ax.legend(fontsize=8)

    # --- 2. Scale histogram (log x) ---
    ax = axes[1]
    s_all = max_scale.clamp(min=1e-7)
    lo = np.log10(max(s_all.min().item(), 1e-7))
    hi = np.log10(s_all.quantile(0.999).item())
    bins = np.logspace(lo, hi, 80)
    ax.hist(s_all[~dead].numpy(), bins=bins, alpha=0.7, color="#2176AE", label="alive")
    ax.hist(s_all[dead].numpy(), bins=bins, alpha=0.7, color="#E8553A", label="dead")
    ax.set_xscale("log")
    ax.set_xlabel("max scale")
    ax.set_ylabel("count")
    ax.set_title("Max Scale")
    ax.legend(fontsize=8)

    # --- 3. Opacity vs max-scale scatter (log x) ---
    ax = axes[2]
    n_plot = min(10000, n_total)
    idx = np.random.choice(n_total, n_plot, replace=False)
    opa_plot = opacities[idx].numpy()
    scale_plot = max_scale[idx].clamp(min=1e-7).numpy()
    dead_plot = dead[idx].numpy()

    ax.scatter(
        scale_plot[~dead_plot],
        opa_plot[~dead_plot],
        s=1,
        alpha=0.15,
        color="#2176AE",
        label="alive",
        rasterized=True,
    )
    ax.scatter(
        scale_plot[dead_plot],
        opa_plot[dead_plot],
        s=1,
        alpha=0.15,
        color="#E8553A",
        label="dead",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel("max scale")
    ax.set_ylabel("opacity")
    ax.set_title("Opacity vs Scale")
    ax.legend(fontsize=8, markerscale=5)

    fig.tight_layout()
    return {"gaussian_stats": fig}


def _draw_frusta_2d(ax, render_pose, K, poses, cameras, depth, color, label):
    """Project per-view camera frusta into a rendered panel and draw them as lines.

    Args:
        ax: panel axes.
        render_pose: [1] c2w pose of the panel's render camera.
        K: [3, 3] pixel intrinsics of the render camera.
        poses: [V] c2w poses to draw.
        cameras: [V] cameras to draw, each with its own fov.
        depth: frustum depth in scene units.
        color: line color.
        label: legend label.
    """
    w2c = render_pose.inv()
    Rw, tw = w2c.R.reshape(3, 3), w2c.t.reshape(3)
    R_v = poses.R.reshape(-1, 3, 3)
    t_v = poses.t.reshape(-1, 3)
    hf, vf = cameras.hfov.reshape(-1), cameras.vfov.reshape(-1)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for v in range(t_v.shape[0]):
        x = math.tan(hf[v].item() / 2) * depth
        y = math.tan(vf[v].item() / 2) * depth
        local = torch.tensor(
            [[0, 0, 0], [-x, -y, depth], [x, -y, depth], [x, y, depth], [-x, y, depth]],
            dtype=t_v.dtype,
            device=t_v.device,
        )
        world = local @ R_v[v].T + t_v[v]
        cam = world @ Rw.T + tw
        z = cam[:, 2].clamp(min=1e-6)
        uv = (cam[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
        uv, valid = uv.cpu().numpy(), (cam[:, 2] > 1e-4).cpu().numpy()
        for a, b in edges:
            if valid[a] and valid[b]:
                ax.plot(*np.stack([uv[a], uv[b]]).T, color=color, lw=0.9, alpha=0.9)
        if valid[0]:
            ax.plot(uv[0, 0], uv[0, 1], ".", color=color, ms=3)
    ax.plot([], [], color=color, lw=1, label=label)  # one legend entry per pose set


def _pulled_back_pose(R_base, t_base, pitch, pull):
    """Pitch the view down by `pitch` and pull the camera back by `pull` (elevated overview)."""
    fwd, down = R_base[:, 2], R_base[:, 1]
    fwd = torch.nn.functional.normalize(fwd * math.cos(pitch) + down * math.sin(pitch), dim=0)
    down = torch.nn.functional.normalize(down - (down @ fwd) * fwd, dim=0)
    right = torch.linalg.cross(down, fwd)
    R = torch.stack([right, down, fwd], dim=-1)
    return Pose.from_Rt(R[None], (t_base - fwd * pull)[None])


def make_gaussian_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], dpi: int = PLOT_DPI
) -> dict[str, Any]:
    """Gaussians from elevated mean-cam and identity views, with the context frusta.

    GT frusta are green, the predicted ones (when the model has a pose head) red.
    """
    if "gaussians" not in pred:
        return {}

    gaussians: Gaussians = pred["gaussians"][0]
    device = gaussians.means.device
    ctx_pose: Pose = data["context"]["pose"][0].to(device)
    ctx_cam: Camera = data["context"]["camera"][0].to(device)

    center = gaussians.means.median(dim=0).values
    radius = ((gaussians.means - center).norm(dim=-1).quantile(0.9) * 1.2).item()
    pull, pitch = 1.1 * radius, math.radians(20.0)

    H, W = 512, 512
    fov = torch.tensor([math.radians(65.0)], device=device)
    camera = Camera.from_fov(fov, fov, w=W, h=H).to(device)
    K = camera.K.reshape(3, 3)

    avg_c2w = torch.linalg.inv(mean_cam_transform(ctx_pose.Rt))
    mean_pose = _pulled_back_pose(avg_c2w[:3, :3], avg_c2w[:3, 3], pitch, pull)
    eye = torch.eye(3, device=device)
    id_pose = _pulled_back_pose(eye, torch.zeros(3, device=device), pitch, pull)

    poses = torch.cat([mean_pose, id_pose], dim=0)
    render = gaussians.render_view(
        camera.repeat(2), poses, backgrounds=torch.ones(2, 3, device=device)
    )
    imgs = [tensor_to_img(render["rendering"][i]) for i in range(2)]

    fig, axs = viz2d.plot_images(imgs, titles=["mean-cam", "identity"], dpi=dpi)

    pose_sets = [(ctx_pose, ctx_cam, "green", "GT")]
    if "pose_pred" in pred:
        pose_sets.append(
            (pred["pose_pred"][0].to(device), pred["camera_pred"][0].to(device), "red", "pred")
        )
    frustum_depth = 0.13 * radius
    for ax, rp in zip(axs, (mean_pose, id_pose)):
        for p, c, color, label in pose_sets:
            _draw_frusta_2d(ax, rp, K, p, c, frustum_depth, color, label)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    axs[0].legend(fontsize=7, loc="upper right")

    if "pose_pred" in pred:
        pp, pc = pose_sets[1][0], pose_sets[1][1]
        cos = ((pp.R.transpose(-1, -2) @ ctx_pose.R).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        rot_deg = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
        # center error and the scale-free pairwise translation-direction angle
        t_err = (pp.t - ctx_pose.t).norm(dim=-1)
        i, j = torch.triu_indices(len(t_err), len(t_err), offset=1)
        d_pred = torch.nn.functional.normalize(pp.t[i] - pp.t[j], dim=-1)
        d_gt = torch.nn.functional.normalize(ctx_pose.t[i] - ctx_pose.t[j], dim=-1)
        tdir_deg = torch.rad2deg(torch.acos((d_pred * d_gt).sum(-1).abs().clamp(max=1.0)))
        fov_deg = torch.rad2deg(
            torch.stack([pc.vfov - ctx_cam.vfov, pc.hfov - ctx_cam.hfov], -1).abs()
        )
        viz2d.add_text(
            0,
            f"R {rot_deg.median():.1f} deg\n"
            f"t {t_err.median():.3f}\n"
            f"t-dir {tdir_deg.median():.1f} deg\n"
            f"fov {fov_deg.median():.1f} deg",
            axes=axs,
        )
    return {"gaussians": fig}


def visualize_batch(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 1,
    dpi: int = 50,
    show_context: bool = False,
) -> dict[str, Any]:
    """Figures for one batch of predictions and ground truth.

    Args:
        pred: model predictions.
        data: the batch.
        n_pairs: number of batch elements to plot.
        dpi: figure resolution.
        show_context: also plot the context views (default: targets only).
    """
    n_pairs = max(1, n_pairs)

    if "image" in data:
        n_pairs = min(n_pairs, len(data["image"]))

    figures = {}
    figures |= make_context_and_targets_figure(
        pred, data, n_pairs=n_pairs, dpi=dpi, show_context=show_context
    )
    figures |= make_attention_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    figures |= make_gaussian_figure(pred, data, dpi=dpi)
    figures |= make_gaussian_stats_figure(pred, data, dpi=dpi)

    return figures
