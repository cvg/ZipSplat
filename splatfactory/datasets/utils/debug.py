"""Shared debug/test runner and visualization for dataset classes.

Usage: python -m splatfactory.datasets.dataset_re10k [--flags]

Author: Alexander Veicht
"""

import argparse
import time

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from splatfactory.utils import mappings

# --- Debug runner -------------------------------------------------------------


def debug_dataset(dataset_cls: type) -> None:
    """Load and iterate a dataset, optionally visualize or benchmark throughput."""
    parser = argparse.ArgumentParser(description=f"Debug {dataset_cls.__name__}")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument(
        "--shard-dir", type=str, default=None, help="Override both train/test shard dirs"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to load")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup batches to skip")
    parser.add_argument("--perf-test", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--visualize", action="store_true", help="Show 3D + 2D visualizations")
    args = parser.parse_args()

    conf = {
        "image_num_range": [args.views, args.views],
        "num_workers": args.num_workers,
        "train_batch_size": args.batch_size,
        "val_batch_size": args.batch_size,
        "test_batch_size": args.batch_size,
        "prefetch_factor": None if args.num_workers == 0 else 2,
    }
    if args.dataset_dir:
        conf["dataset_dir"] = args.dataset_dir
    if args.shard_dir:
        conf["train_shard_dir"] = args.shard_dir
        conf["test_shard_dir"] = args.shard_dir

    ds = dataset_cls(conf, split=args.split)
    loader = ds.get_loader(num_workers=args.num_workers)

    print(f"Dataset: {dataset_cls.__name__}, split={args.split}, shards={len(ds.shard_paths)}")

    if args.perf_test:
        _run_perf_test(ds, loader, args)
        return

    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        mappings.print_summary(batch)

        if args.visualize:
            visualize_data_batch(batch)

        if i + 1 >= args.num_batches:
            break


# --- Performance test ---------------------------------------------------------


def _run_perf_test(ds, loader, args) -> None:
    """Measure throughput and per-stage breakdown."""
    batch_times = []
    total_images = 0
    total_scenes = 0

    print(
        f"Running perf test: {args.num_batches} batches, "
        f"{args.warmup} warmup, {args.num_workers} workers"
    )

    for i, batch in enumerate(tqdm(loader, desc="Perf", total=args.num_batches + args.warmup)):
        if i < args.warmup:
            t_start = time.time()
            continue

        elapsed = time.time() - t_start
        batch_times.append(elapsed)

        bs = batch["context"]["image"].shape[0]
        n_ctx = batch["context"]["image"].shape[1]
        n_tgt = batch["target"]["image"].shape[1]
        total_images += bs * (n_ctx + n_tgt)
        total_scenes += bs
        t_start = time.time()

        if i >= args.num_batches + args.warmup - 1:
            break

    if not batch_times:
        print("No batches after warmup.")
        return

    bt = np.array(batch_times)
    wall = bt.sum()
    n = len(bt)

    print(f"\n{'=' * 60}")
    print(f"Throughput ({n} batches, {args.warmup} warmup):")
    print(f"  {n / wall:.1f} batches/sec | {total_images / wall:.1f} images/sec")
    print(f"  Batch time: {bt.mean() * 1000:.1f} +/- {bt.std() * 1000:.1f} ms")
    print(f"  Total: {total_scenes} scenes, {total_images} images in {wall:.1f}s")

    # Per-stage breakdown (only available with num_workers=0)
    if hasattr(ds, "step_timer") and ds.step_timer.num_steps() > 0:
        print(f"\n{'=' * 60}")
        print("Per-sample breakdown:")
        ds.step_timer.log_stats()
    if hasattr(ds, "view_timer") and ds.view_timer.num_steps() > 0:
        print(f"\n{'=' * 60}")
        print("Per-view breakdown:")
        ds.view_timer.log_stats()
    if args.num_workers > 0:
        print("\nNote: per-stage breakdown only available with --num-workers 0")


# --- Visualization ------------------------------------------------------------


def _t2img(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().detach().permute(1, 2, 0).clip(0, 1)


def visualize_data_batch(batch: dict, port: int = 8097) -> None:
    """Visualize a raw data batch with 3D point cloud and image/depth grid."""
    from splatfactory.visualization import viz2d, viz3d

    mappings.print_summary(batch)

    # --- 3D point cloud ---
    fig3d = viz3d.init_figure()
    poses = batch["context"]["pose"][0]
    cameras = batch["context"]["camera"][0]
    depths = batch["context"]["depth"][0]
    context_imgs = batch["context"]["image"][0]

    viz3d.plot_cameras(fig3d, poses, cameras, color="rgb(255, 0, 0)")

    for cam_idx in range(len(cameras)):
        depth = depths[cam_idx]
        if (depth < 0).all():
            continue
        p3d = cameras[cam_idx].normalized_image_coordinates()
        depth_flat = rearrange(depth, "H W -> (H W) 1")
        p3d = p3d * depth_flat
        p3d = poses[cam_idx].transform(p3d)

        colors = rearrange(context_imgs[cam_idx], "C H W -> (H W) C")
        valid = depth_flat.squeeze(-1) > 0
        viz3d.plot_points(fig3d, p3d[valid], color=colors[valid])

    target_poses = batch["target"]["pose"][0]
    target_cameras = batch["target"]["camera"][0]
    viz3d.plot_cameras(fig3d, target_poses, target_cameras, color="rgb(0, 255, 0)")
    viz3d.show_figure(fig3d, port=port)

    # --- Image/depth grid ---
    rows, titles = [], []
    b = 0

    for group_name in ["context", "target"]:
        if group_name not in batch:
            continue
        group = batch[group_name]
        prefix = "ctx" if group_name == "context" else "tgt"
        n_views = len(group["image"][b])
        idx_arr = group.get("index", [None] * n_views)
        idx_arr = idx_arr[b] if idx_arr is not None else None

        # RGB row
        rgb_row, rgb_titles = [], []
        for i in range(n_views):
            rgb_row.append(_t2img(group["image"][b][i]))
            idx_str = f" id:{int(idx_arr[i])}" if idx_arr is not None else ""
            rgb_titles.append(f"{prefix} {i}{idx_str}")
        rows.append(rgb_row)
        titles.append(rgb_titles)

        # Depth row
        if "depth" in group:
            depth_row, depth_titles = [], []
            for i in range(n_views):
                depth_row.append(_t2img(group["image"][b][i]))  # placeholder for heatmap overlay
                idx_str = f" id:{int(idx_arr[i])}" if idx_arr is not None else ""
                depth_titles.append(f"{prefix} depth {i}{idx_str}")
            rows.append(depth_row)
            titles.append(depth_titles)

    if not rows:
        return

    # Pad rows to equal length
    max_cols = max(len(r) for r in rows)
    h, w = rows[0][0].shape[:2]
    blank = torch.zeros(h, w, 3)
    for r, t in zip(rows, titles):
        while len(r) < max_cols:
            r.append(blank)
            t.append("")

    fig2d, ax = viz2d.plot_image_grid(rows, titles=titles, dpi=80)

    # Overlay depth heatmaps
    row_idx = 0
    for group_name in ["context", "target"]:
        if group_name not in batch:
            continue
        group = batch[group_name]
        n_views = len(group["image"][b])
        row_idx += 1  # skip RGB row

        if "depth" in group:
            depth_data = group["depth"][b]
            valid_mask = depth_data > 0
            vmax = (
                torch.quantile(depth_data[valid_mask].float(), 0.98).item()
                if valid_mask.any()
                else 1.0
            )
            for i in range(n_views):
                d = depth_data[i].clone().float()
                d[d < 0] = float("nan")
                viz2d.plot_heatmaps([d], vmax=vmax, colorbar=True, alpha=1.0, axes=[ax[row_idx, i]])
            row_idx += 1

    fig2d.tight_layout()
    fig2d.show()
