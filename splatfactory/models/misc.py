# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from splatfactory.models import BaseModel


class PeakMemTracker:
    """Track peak GPU memory per named region. Use as context manager."""

    def __init__(self):
        self.regions = {}
        self._active = False

    def track(self, name):
        self._name = name
        self._active = True
        return self

    def __enter__(self):
        if self._active:
            torch.cuda.synchronize()
            self._baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        if self._active:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            baseline = self._baseline
            self.regions[self._name] = {
                "peak": peak,
                "baseline": baseline,
                "delta": peak - baseline,
            }
            self._active = False

    def summary(self):
        print("\n=== Peak Memory per Region ===")
        print(f"{'Region':<22} {'Baseline':>10} {'Peak':>10} {'Delta':>10}")
        print("-" * 56)
        for name, m in self.regions.items():
            b = m["baseline"] / (1024**3)
            p = m["peak"] / (1024**3)
            d = m["delta"] / (1024**3)
            print(f"{name:<22} {b:>7.2f} GB {p:>7.2f} GB {d:>7.2f} GB")


def overfit_model(
    model: BaseModel,
    sample: Dict[str, Any],
    n_iters: int = 300,
    lr: float = 1e-3,
    loss_key: str = "total",
    weight_decay: float = 0,
    log_dir: str = None,
    log_every: int = 100,
) -> BaseModel:
    """Overfit a model on a single sample for debugging purposes."""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.1
    )

    pbar = tqdm(range(n_iters), desc="Overfitting", ncols=100)
    for i in pbar:
        # forward pass and loss computation
        optimizer.zero_grad()
        pred = model(sample)
        loss, metrics = model.loss(pred, sample)
        total_loss = loss[loss_key]
        total_loss.backward()

        # update weights
        optimizer.step()
        lr_scheduler.step()

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        metrics = {k: v.mean().item() for k, v in metrics.items()}
        pbar.set_postfix(
            {"loss": total_loss.item(), "alloc (GB)": allocated, "res (GB)": reserved} | metrics
        )

    model.eval()
    return model


def profile_model(
    model: nn.Module,
    sample: Dict[str, torch.Tensor],
    num_steps: int = 10,
    dtype: torch.dtype = torch.bfloat16,
):
    """Profile a model with torch profiler."""
    device_type = next(model.parameters()).device.type
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # Warmup
    for _ in tqdm(range(10), desc="Warmup", ncols=100):
        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=dtype):
            pred = model(sample)
            loss, _ = model.loss(pred, sample)
        scaler.scale(loss["total"].mean()).backward()
        scaler.step(optimizer)
        scaler.update()

    # Peak memory profiling pass (separate from torch profiler)
    tracker = PeakMemTracker()
    model._peak_mem_tracker = tracker
    optimizer.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with tracker.track("forward"), torch.autocast(device_type=device_type, dtype=dtype):
        pred = model(sample)

    with tracker.track("loss"), torch.autocast(device_type=device_type, dtype=dtype):
        loss_dict, _ = model.loss(pred, sample)

    with tracker.track("backward"):
        scaler.scale(loss_dict["total"].mean()).backward()

    with tracker.track("optimizer_step"):
        scaler.step(optimizer)
        scaler.update()

    tracker.summary()
    del model._peak_mem_tracker

    # Torch profiler pass
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in tqdm(range(num_steps), desc="Profiling", ncols=100):
            optimizer.zero_grad()
            with record_function("forward"), torch.autocast(device_type=device_type, dtype=dtype):
                pred = model(sample)

            with record_function("loss"), torch.autocast(device_type=device_type, dtype=dtype):
                loss, metrics = model.loss(pred, sample)

            total_loss = loss["total"].mean()
            with record_function("backward"):
                scaler.scale(total_loss).backward()

            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()

            prof.step()

    # NOTE: keep commented-out tables for debugging full profiler output
    # print("\n=== CUDA Time ===")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
    # print("\n=== Memory Usage (all ops) ===")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=60))

    named_regions = [
        "forward",
        "backbone",
        "color_embed",
        "fuse_tokens",
        "build_queries",
        "kmeans_clustering",
        "fps_init",
        "kmeans_iter_0",
        "query_refinement",
        "fuse_attention",
        "color_fusion",
        "gaussian_head",
        "loss",
        "backward",
        "optimizer_step",
    ]
    events_by_key = defaultdict(list)
    for e in prof.key_averages():
        events_by_key[e.key].append(e)

    print("\n=== Named Region Summary ===")
    print(f"{'Region':<22} {'CUDA time':>12} {'CUDA Mem':>12} {'Self CUDA Mem':>14}")
    print("-" * 64)
    for name in named_regions:
        if name not in events_by_key:
            continue
        entries = events_by_key[name]
        # Pick entry with largest absolute memory (the CPU-side record_function event)
        e = max(entries, key=lambda x: abs(x.device_memory_usage + x.cpu_memory_usage))
        dev_time = e.device_time_total / e.count / 1000  # ms avg per call
        dev_mem = e.device_memory_usage / e.count / (1024**3)
        self_dev_mem = e.self_device_memory_usage / e.count / (1024**3)
        print(f"{name:<22} {dev_time:>9.1f} ms {dev_mem:>9.2f} GB {self_dev_mem:>11.2f} GB")


def format_module_params(module, depth: int = 0) -> str:
    """Format trainable/total parameters for a module and its submodules as a string."""

    def _fmt(n):
        return f"{n / 1e6:.2f} M"

    def _recurse(mod, max_depth, cur_depth, is_last, prefix, name):
        lines = []

        total = sum(p.numel() for p in mod.parameters())
        trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)

        if cur_depth == 0:
            connector = ""
            label = mod.__class__.__name__
        else:
            connector = prefix + ("└─" if is_last else "├─")
            label = f"{name}: {mod.__class__.__name__}" if name else mod.__class__.__name__

        pct = 100 * trainable / total if total > 0 else 0
        lines.append(f"{connector}{label}: {_fmt(trainable)}/{_fmt(total)} ({pct:.1f}% trainable)")

        if cur_depth < max_depth:
            children = list(mod.named_children())
            for i, (child_name, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                new_prefix = "  " if cur_depth == 0 else prefix + ("  " if is_last else "│ ")
                lines.extend(
                    _recurse(child, max_depth, cur_depth + 1, is_last_child, new_prefix, child_name)
                )

        return lines

    return "\n".join(_recurse(module, depth, 0, True, "", ""))


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(
        embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid
