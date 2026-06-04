from typing import Any, Dict, List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_video_frames(video_path: str) -> torch.Tensor:
    """Loads video frames from a video file."""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(frame)
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    return torch.from_numpy(frames).permute(0, 3, 1, 2) / 255.0  # (T, 3, H, W)


def visualize_masks_on_frames(
    frames: torch.Tensor | np.ndarray,
    masks_per_frame: Dict[int, List[Dict[str, Any]]],
    cmap="tab20",
    alpha: float = 0.5,
):
    """Overlays instance masks on frames for visualization.

    Args:
        frames: List of frames as numpy arrays of shape (H, W, 3).
        masks_per_frame: Dictionary mapping frame indices to lists of masks. Each dict contains:
            - "mask": binary mask of shape (H, W)
            - "id": integer ID of the mask
        alpha: Opacity of the mask overlay.
        cmap: Colormap to use for overlaying masks.

    Returns:
        List of frames with masks overlaid.
    """
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().permute(0, 2, 3, 1).numpy()

    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)

    assert len(frames) == len(
        masks_per_frame
    ), f"Number of frames and masks must match ({len(frames)=}, {len(masks_per_frame)=})"

    cmap = plt.get_cmap(cmap)
    for f_idx, frame in enumerate(frames):
        colors = np.zeros_like(frame, dtype=np.float32)
        counts = np.zeros(frame.shape[:2], dtype=np.float32)

        for mask_dict in masks_per_frame[f_idx]:
            mask = mask_dict["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            mask = mask.astype(bool)
            color = np.array(cmap(mask_dict["id"] % 20)[:3])
            color = (color * 255).astype(np.uint8)
            colors += mask[..., None] * color[None, None]
            counts += mask.astype(np.float32)

        counts = np.clip(counts, a_min=1, a_max=None)
        colors = colors / counts[..., None]

        frames[f_idx] = (1 - alpha) * frame.astype(np.float32) + alpha * colors

    return frames


def add_frame_numbers(
    frames: torch.Tensor | np.ndarray,
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
):
    """Adds frame numbers to each frame in the sequence.

    Args:
        frames: List of frames as numpy arrays of shape (H, W, 3).
        font_scale: Scale of the font for the frame numbers.
        color: Color of the text.
        thickness: Thickness of the text.

    Returns:
        List of frames with frame numbers added.
    """
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)

    for i in range(frames.shape[0]):
        plt_frame = frames[i].copy()
        cv2.putText(
            plt_frame,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        frames[i] = plt_frame

    return frames


def frames_to_video(
    frames: torch.Tensor, output_path: str, fps: int = 30, frame_numbers: bool = False
):
    """Saves a sequence of frames as a video."""
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)

    frames = add_frame_numbers(frames) if frame_numbers else frames
    with imageio.get_writer(
        output_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
