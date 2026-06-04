"""Visualization of predicted and ground truth for a single batch."""

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.utils import PCA
from splatfactory.models.metrics import calculate_lpips, calculate_psnr
from splatfactory.utils.conversions import reconstruct_feature_img
from splatfactory.utils.mappings import batch_to_device
from splatfactory.visualization import viz2d, viz3d

PLOT_DPI = 50


def _short_scene_id(scene_id: str) -> str:
    """Shorten scene IDs with long hashes (e.g. dl3dv-06da79...-> dl3dv-06da79..)."""
    parts = scene_id.split("-", 1)
    if len(parts) == 2 and len(parts[1]) > 8:
        return f"{parts[0]}-{parts[1][:8]}"
    return scene_id


def tensor_to_img(t):
    return t.cpu().detach().permute(1, 2, 0).clip(0, 1)


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


def make_query_mask_figure(
    title: str,
    scores: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    iou: float,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Create a figure for visualizing the query mask and ground truth mask."""
    img = data["image"][0].permute(1, 2, 0).cpu()
    titles = [title, "Heatmap", "Pred Mask", "GT Mask"]
    fig, ax = viz2d.plot_images([img] * 4, titles=titles, dpi=PLOT_DPI)
    viz2d.plot_heatmaps(
        [scores, pred_mask, gt_mask], vmax=1.0, colorbar=True, alpha=0.9, axes=ax[1:]
    )
    viz2d.add_text(-2, f"IoU: {iou:.3f}", axes=ax)
    return {"query_mask": fig}


def make_rendering_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], n_pairs: int = 2
) -> dict[str, Any]:
    """Create a figure for visualizing the gt image and renderings."""
    if "rendering" not in pred:
        return {}

    n_pairs = min(n_pairs, len(data["image"]))
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    img_pairs, scores = [], []
    for i in range(n_pairs):
        img, rendering = data["image"][i], pred["rendering"][i]
        scores.append(
            {"psnr": calculate_psnr(img, rendering), "lpips": calculate_lpips(img, rendering)}
        )
        img_pairs.append([i.permute(1, 2, 0).clip(0, 1) for i in [img, rendering]])

    titles = ["Image", "Rendering"]
    fig, ax = viz2d.plot_image_grid(img_pairs, titles=[titles] * n_pairs, set_lim=True)

    for i in range(len(img_pairs)):
        img, rendering = data["image"][i], pred["rendering"][i]
        txt = f"PSNR: {scores[i]['psnr']:.2f}\nLPIPS: {scores[i]['lpips']:.4f}"
        viz2d.add_text(1, txt, axes=ax[i])

    return {"rendering": fig}


def make_context_and_targets_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
    dpi: int = PLOT_DPI,
) -> dict[str, Any]:
    if "context_rgb" not in pred or "target_rgb" not in pred:
        return {}

    idx = 0
    # TODO: support n_pairs > 1
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    n_context, n_target = (len(data["context"]["image"][idx]), len(data["target"]["image"][idx]))

    context_scores = compute_view_scores(pred["context_rgb"][idx], data["context"]["image"][idx])
    target_scores = compute_view_scores(pred["target_rgb"][idx], data["target"]["image"][idx])

    context_imgs = [tensor_to_img(img) for img in data["context"]["image"][idx]]
    target_imgs = [tensor_to_img(img) for img in data["target"]["image"][idx]]
    context_rgb = [tensor_to_img(img) for img in pred["context_rgb"][idx]]
    target_rgb = [tensor_to_img(img) for img in pred["target_rgb"][idx]]

    titles_1 = ["Context"] * n_context + ["Target"] * n_target
    titles_2 = ["Context Rendering"] * n_context + ["Target Rendering"] * n_target
    titles_3 = ["Context Depth"] * n_context + ["Target Depth"] * n_target
    titles_4 = ["Context Diff x3"] * n_context + ["Target Diff x3"] * n_target

    # Compute per-pixel RGB diff (amplified 3x)
    context_diff = [
        tensor_to_img((p - gt).abs() * 3)
        for p, gt in zip(pred["context_rgb"][idx], data["context"]["image"][idx])
    ]
    target_diff = [
        tensor_to_img((p - gt).abs() * 3)
        for p, gt in zip(pred["target_rgb"][idx], data["target"]["image"][idx])
    ]

    fig, ax = viz2d.plot_image_grid(
        [
            context_imgs + target_imgs,
            context_rgb + target_rgb,
            context_imgs + target_imgs,  # depth row (placeholders)
            context_diff + target_diff,
        ],
        titles=[titles_1, titles_2, titles_3, titles_4],
        dpi=dpi,
    )

    context_max = torch.quantile(pred["context_depth"], 0.95).item()
    target_max = torch.quantile(pred["target_depth"], 0.95).item()
    max_depth = max(context_max, target_max)

    for i in range(len(context_imgs)):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["context"]["index"][idx][i].item())
        viz2d.add_text(i, f"{scene}\nID: {img_id}", axes=ax[0])

        depth = pred["context_depth"][idx][i]
        viz2d.plot_heatmaps([depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[2, i]])

    for i in range(len(target_imgs)):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["target"]["index"][idx][i].item())
        viz2d.add_text(len(context_imgs) + i, f"{scene}\nID: {img_id}", axes=ax[0])

        depth = pred["target_depth"][idx][i]
        viz2d.plot_heatmaps(
            [depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[2, len(context_imgs) + i]]
        )

    for i in range(n_context):
        viz2d.add_text(i, format_score_text(context_scores, i), axes=ax[1])

    for i in range(n_target):
        viz2d.add_text(n_context + i, format_score_text(target_scores, i), axes=ax[1])

    return {"context_and_target": fig}


def make_cluster_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
    dpi: int = PLOT_DPI,
) -> dict[str, Any]:
    if "cluster_rgb_context" not in pred or "cluster_rgb_target" not in pred:
        return {}

    idx = 0
    # TODO: support n_pairs > 1
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    n_context, n_target = (len(data["context"]["image"][idx]), len(data["target"]["image"][idx]))

    context_scores = compute_view_scores(pred["context_rgb"][idx], data["context"]["image"][idx])
    target_scores = compute_view_scores(pred["target_rgb"][idx], data["target"]["image"][idx])

    context_imgs = [tensor_to_img(img) for img in data["context"]["image"][idx]]
    target_imgs = [tensor_to_img(img) for img in data["target"]["image"][idx]]
    context_rgb = [tensor_to_img(img) for img in pred["context_rgb"][idx]]
    target_rgb = [tensor_to_img(img) for img in pred["target_rgb"][idx]]
    context_cluster = [tensor_to_img(img) for img in pred["cluster_rgb_context"][idx]]
    target_cluster = [tensor_to_img(img) for img in pred["cluster_rgb_target"][idx]]

    titles_1 = ["Context"] * n_context + ["Target"] * n_target
    titles_2 = ["Context Rendering"] * n_context + ["Target Rendering"] * n_target
    titles_3 = ["Context Clusters"] * n_context + ["Target Clusters"] * n_target

    fig, ax = viz2d.plot_image_grid(
        [context_imgs + target_imgs, context_rgb + target_rgb, context_cluster + target_cluster],
        titles=[titles_1, titles_2, titles_3],
        dpi=dpi,
    )

    for i in range(n_context):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["context"]["index"][idx][i].item())
        viz2d.add_text(i, f"{scene}\nID: {img_id}", axes=ax[0])
        viz2d.add_text(i, format_score_text(context_scores, i), axes=ax[1])

    for i in range(n_target):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["target"]["index"][idx][i].item())
        viz2d.add_text(n_context + i, f"{scene}\nID: {img_id}", axes=ax[0])
        viz2d.add_text(n_context + i, format_score_text(target_scores, i), axes=ax[1])

    return {"cluster": fig}


def make_depth_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
    dpi: int = PLOT_DPI,
) -> dict[str, Any]:
    if "context_depth" not in pred or "depth" not in data["context"]:
        return {}

    idx = 0
    # TODO: support n_pairs > 1
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    context_imgs = [tensor_to_img(img) for img in data["context"]["image"][idx]]
    target_imgs = [tensor_to_img(img) for img in data["target"]["image"][idx]]

    n_context, n_target = len(context_imgs), len(target_imgs)
    titles_1 = ["Context"] * n_context + ["Target"] * n_target
    titles_2 = ["Context Rendering"] * n_context + ["Target Rendering"] * n_target

    fig, ax = viz2d.plot_image_grid(
        [context_imgs + target_imgs] * 2, titles=[titles_1, titles_2], dpi=dpi
    )

    context_max = torch.quantile(data["context"]["depth"], 0.95).item()
    target_max = torch.quantile(data["target"]["depth"], 0.95).item()
    max_depth = max(context_max, target_max)

    # Add GT depths to context images
    for i in range(len(context_imgs)):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["context"]["index"][idx][i].item())
        viz2d.add_text(i, f"{scene}\nID: {img_id}", axes=ax[0])

        depth = data["context"]["depth"][idx][i]
        viz2d.plot_heatmaps([depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[0, i]])

        depth = pred["context_depth"][idx][i]
        viz2d.plot_heatmaps([depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[1, i]])

    for i in range(len(target_imgs)):
        scene = _short_scene_id(data["scene"][idx])
        img_id = int(data["target"]["index"][idx][i].item())
        viz2d.add_text(len(context_imgs) + i, f"{scene}\nID: {img_id}", axes=ax[0])

        depth = data["target"]["depth"][idx][i]
        viz2d.plot_heatmaps(
            [depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[0, len(context_imgs) + i]]
        )

        depth = pred["target_depth"][idx][i]
        viz2d.plot_heatmaps(
            [depth], colorbar=True, vmax=max_depth, alpha=1.0, axes=[ax[1, len(context_imgs) + i]]
        )

    return {"depth": fig}


def make_feature_figure(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 2,
) -> dict[str, Any]:
    """Create a figure for visualizing the gt image and renderings."""
    if "rendering" not in pred:
        return {}

    n_pairs = min(n_pairs, len(data["image"]))
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    pca = None

    img_pairs, titles = [], []
    for i in range(n_pairs):
        img = data["image"][i]
        row = [img]
        titles = ["Image"]

        H, W = img.shape[-2:]
        levels = np.array(data["levels"]).squeeze()
        levels = levels[None] if levels.ndim == 0 else levels

        for level_id, level in enumerate(levels):
            gt_features = reconstruct_feature_img(
                features=data[f"{level}_features"],
                indices=data[f"{level}_indices"],
                target_shape=(H, W),
            )[0]
            features = pred["feature_map"][i, level_id]

            gt_features = rearrange(gt_features, "C H W -> (H W) C")
            features = rearrange(features, "C H W -> (H W) C")

            if pca is None:
                pca = PCA(n_components=3)
                pca.fit(torch.unique(gt_features, dim=0))

            if gt_features.shape[-1] == features.shape[-1]:
                gt_features = pca.normalize(pca.transform(gt_features))
                features = pca.normalize(pca.transform(features))
            else:
                pca = PCA(n_components=3)
                pca.fit(torch.unique(gt_features, dim=0))
                gt_features = pca.normalize(pca.transform(gt_features))

                pca.fit(torch.unique(features, dim=0))
                features = pca.normalize(pca.transform(features))

            gt_features = rearrange(gt_features, "(H W) C -> C H W", H=H, W=W)
            features = rearrange(features, "(H W) C -> C H W", H=H, W=W)

            titles += [f"GT Features {level}", f"Pred Features {level}"]
            row += [gt_features, features]

        img_pairs.append([i.permute(1, 2, 0).clip(0, 1) for i in row])

    fig, ax = viz2d.plot_image_grid(
        img_pairs, titles=[titles] * n_pairs, set_lim=True, dpi=PLOT_DPI
    )

    return {"features": fig}


def make_3d_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], n_pairs: int = 2
) -> dict[str, Any]:
    """Create a figure for visualizing the gt image and renderings."""
    if "gaussians" not in pred:
        return {}

    n_pairs = min(n_pairs, len(data["context"]["image"]))
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    fig = viz3d.init_figure()
    context = data["context"]
    viz3d.plot_cameras(fig, context["pose"][0], context["camera"][0], scale=3)

    gaussians: Gaussians = pred["gaussians"][0]
    viz3d.plot_points(fig, gaussians.means, color=gaussians.rgb)

    return {"3d_points": fig}


def make_prototype_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], n_pairs: int = 2
) -> dict[str, Any]:
    """Create a figure for visualizing the gt image and renderings."""
    if "gaussians" not in pred or "prototype_ids" not in pred:
        return {}

    n_pairs = min(n_pairs, len(data["context"]["image"]))
    pred = batch_to_device(pred, "cpu", detach=True, non_blocking=False)
    data = batch_to_device(data, "cpu", detach=True, non_blocking=False)

    fig = viz3d.init_figure()
    context = data["context"]
    viz3d.plot_cameras(fig, context["pose"][0], context["camera"][0], scale=3)

    gaussians: Gaussians = pred["gaussians"][0]
    cmap = plt.get_cmap("tab20")
    colors = []
    for pid in pred["prototype_ids"]:
        color = np.array(cmap(pid % 20)[:3])
        color = (color * 255).astype(np.uint8)
        colors.append(color)
    colors = np.array(colors)
    viz3d.plot_points(fig, gaussians.means, color=colors)

    return {"prototypes": fig}


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

    attention_maps = pred["attention"]  # (B, N, H, W)
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


def make_gaussian_figure(
    pred: dict[str, torch.Tensor], data: dict[str, torch.Tensor], dpi: int = PLOT_DPI
) -> dict[str, Any]:
    """Create a figure for visualizing Gaussians from multiple orbit views."""
    if "gaussians" not in pred:
        return {}

    gaussians: Gaussians = pred["gaussians"][0]
    center = gaussians.means.median(dim=0).values
    radius = (gaussians.means - center).norm(dim=-1).quantile(0.9) * 1.2
    # radius = 0.8

    # Create camera with 90 degree FOV
    H, W = 512, 512
    fov = torch.tensor([math.pi / 2], device=data["context"]["image"].device)
    camera = Camera.from_fov(fov, fov, H, W)

    # Define orbit views: (azimuth, elevation, title)
    azimuths = [0, 20, 90, 0]
    elevations = [0, 20, 0, 90]
    titles = ["Front", "Oblique", "Side", "Top"]

    poses = Pose.orbit(center, radius, azimuths, elevations)
    cameras = camera.repeat(len(azimuths))

    render = gaussians.render_view(cameras, poses)
    renderings = [tensor_to_img(render["rendering"][i]) for i in range(len(azimuths))]

    fig, _ = viz2d.plot_images(renderings, titles=titles, dpi=dpi)
    return {"gaussians": fig}


def visualize_batch(
    pred: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    n_pairs: int = 1,
    dpi: int = 50,
) -> dict[str, Any]:
    """Create visualization figures for a batch of predictions and ground truth."""
    n_pairs = max(1, n_pairs)

    if "image" in data:
        n_pairs = min(n_pairs, len(data["image"]))

    figures = {}
    # TODO: Add functions for figures
    # figures |= make_rendering_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    figures |= make_context_and_targets_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    figures |= make_depth_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    # figures |= make_3d_figure(pred, data, n_pairs=n_pairs)
    # figures |= make_prototype_figure(pred, data, n_pairs=n_pairs)
    figures |= make_cluster_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    figures |= make_attention_figure(pred, data, n_pairs=n_pairs, dpi=dpi)
    figures |= make_gaussian_figure(pred, data, dpi=dpi)
    figures |= make_gaussian_stats_figure(pred, data, dpi=dpi)

    # {f.tight_layout() for f in figures.values()}
    return figures
