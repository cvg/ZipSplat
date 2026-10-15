"""Image operations: format conversion, spatial transforms, I/O, and preprocessing.

Single home for all image manipulation. Functions accept both numpy arrays
and torch tensors where it makes sense, dispatching internally.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image as PILImage
from torch import nn

# --- Format conversion --------------------------------------------------------


def to_tensor(image):
    """Convert image to CHW float32 tensor.

    Accepts HWC uint8 numpy, HW numpy (grayscale), or passthrough if already a tensor.
    """
    if isinstance(image, torch.Tensor):
        return image
    if image.ndim == 3:
        image = image.transpose(2, 0, 1)  # HWC -> CHW
    elif image.ndim == 2:
        image = image[None]  # HW -> 1HW
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
    return torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)


def to_numpy(image):
    """Convert image to HWC uint8 numpy.

    Accepts CHW float32 tensor, 1HW tensor (grayscale), or passthrough if already numpy.
    """
    if isinstance(image, np.ndarray):
        return image
    x = image.detach().cpu()
    if x.ndim == 3:
        x = x.permute(1, 2, 0)  # CHW -> HWC
    elif x.ndim == 2:
        pass  # HW stays HW
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
    return (x * 255.0).clamp(0, 255).to(torch.uint8).numpy()


def chw_from_hwc(x):
    """...HWC -> ...CHW. Works on both numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -1, -3)
    return x.transpose(-2, -1).transpose(-3, -2)


def hwc_from_chw(x):
    """...CHW -> ...HWC. Works on both numpy arrays and torch tensors."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -3, -1)
    return x.transpose(-3, -2).transpose(-2, -1)


# --- Spatial operations -------------------------------------------------------
# Each returns (result, transform_3x3) where transform is a numpy float64 3x3 matrix.


def _get_hw(image):
    """Get (h, w) from numpy HWC/HW or torch ...CHW/...HW."""
    if isinstance(image, np.ndarray):
        return image.shape[0], image.shape[1]
    return image.shape[-2], image.shape[-1]


def _round_even(x):
    """Round to nearest even integer. Ensures symmetric center crops for camera cx/cy."""
    return 2 * round(x / 2)


def resize_context(data: dict, target_size: int) -> dict:
    """Resize context images (and update intrinsics) so the short edge is `target_size`.

    Used by baselines to send a downscaled image to their encoder while leaving
    `data["target"]` untouched so rendering uses the original eval-res cameras.
    """
    context = data["context"]
    images = context["image"]
    B, V, C, H, W = images.shape
    if min(H, W) == target_size:
        return data

    flat = images.flatten(0, 1)
    pairs = [resize_to_cover(im, target_size, target_size) for im in flat]
    out = torch.stack([p[0] for p in pairs])
    H_new, W_new = out.shape[-2:]
    images_new = out.view(B, V, C, H_new, W_new)

    new_context = {**context, "image": images_new}
    if "camera" in context:
        s = torch.tensor([W_new / W, H_new / H], device=images.device, dtype=images.dtype)
        new_context["camera"] = context["camera"].scale(s)
    return {**data, "context": new_context}


def resize_to_cover(image, target_h, target_w, method: str = "torch_bilinear_aa"):
    """Scale uniformly so both sides reach the target, keeping the aspect ratio.

    Args:
        image: HWC uint8 numpy array or CHW float tensor.
        target_h: Minimum output height.
        target_w: Minimum output width.
        method: "torch_bilinear_aa" (default), "pil_lanczos" or "cv2_inter_area".

    Returns:
        The resized image and its 3x3 scale matrix.
    """
    h, w = _get_hw(image)
    scale = max(target_h / h, target_w / w)
    h_new, w_new = _round_even(h * scale), _round_even(w * scale)

    if h_new == h and w_new == w:
        return image, np.eye(3)

    is_tensor = isinstance(image, torch.Tensor)

    if method == "pil_lanczos":
        np_img = to_numpy(image) if is_tensor else image
        resized = np.array(PILImage.fromarray(np_img).resize((w_new, h_new), PILImage.LANCZOS))
        if is_tensor:
            resized = to_tensor(resized).to(device=image.device, dtype=image.dtype)
    elif method == "torch_bilinear_aa":
        img_tens = to_tensor(image) if not is_tensor else image
        resized_tens = F.interpolate(
            img_tens[None], size=(h_new, w_new), mode="bilinear", antialias=True
        )[0]
        resized = resized_tens if is_tensor else to_numpy(resized_tens)
    elif method == "cv2_inter_area":
        np_img = to_numpy(image) if is_tensor else image
        resized = cv2.resize(np_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        if is_tensor:
            resized = to_tensor(resized).to(device=image.device, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown resize method: {method!r}")

    transform = np.diag([w_new / w, h_new / h, 1.0])
    return resized, transform


def crop_to_principal_point(image, cx, cy):
    """Largest crop with the input's aspect ratio that centres the principal point.

    Even sides keep the principal point exactly at the centre.

    Args:
        image: HWC numpy array or CHW tensor.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.

    Returns:
        The cropped image and its 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    ar = w / h

    half_w = min(cx, w - cx)
    half_h = min(cy, h - cy)

    # Constrain to original AR
    if half_w / half_h > ar:
        half_w = half_h * ar
    else:
        half_h = half_w / ar

    w_new = 2 * int(half_w)
    h_new = 2 * int(half_h)

    x0 = int(cx - w_new / 2)
    y0 = int(cy - h_new / 2)

    if w_new == w and h_new == h:
        return image, np.eye(3)

    if isinstance(image, np.ndarray):
        cropped = image[y0 : y0 + h_new, x0 : x0 + w_new]
    else:
        cropped = image[..., y0 : y0 + h_new, x0 : x0 + w_new]

    transform = np.eye(3)
    transform[0, 2] = -x0
    transform[1, 2] = -y0
    return cropped, transform


def crop_to_ar(image, aspect_ratio):
    """Centre-crop to the aspect ratio W / H.

    Args:
        image: HWC numpy array or CHW tensor.
        aspect_ratio: Target W / H, above 1 for landscape.

    Returns:
        The cropped image and its 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    target_w = _round_even(h * aspect_ratio)

    if target_w <= w:
        # Crop width
        cx = (w - target_w) // 2
        cy = 0
        h_new, w_new = h, target_w
    else:
        # Crop height
        target_h = _round_even(w / aspect_ratio)
        cy = (h - target_h) // 2
        cx = 0
        h_new, w_new = target_h, w

    if cx == 0 and cy == 0 and h_new == h and w_new == w:
        return image, np.eye(3)

    if isinstance(image, np.ndarray):
        cropped = image[cy : cy + h_new, cx : cx + w_new]
    else:
        cropped = image[..., cy : cy + h_new, cx : cx + w_new]

    transform = np.eye(3)
    transform[0, 2] = -cx
    transform[1, 2] = -cy
    return cropped, transform


def _center_crop(a, s):
    """Symmetric centre crop (crop, x0, y0) of an HW[C] array to at least `s` of its size.

    The margins are equal on both sides, since Camera.compose_image_transform derives the new
    size as size - 2 * offset.
    """
    h, w = a.shape[:2]
    y0, x0 = int((h - h * s) // 2), int((w - w * s) // 2)
    return a[y0 : h - y0, x0 : w - x0], x0, y0


def crop_to_divisible(image, divisor):
    """Centre-crop both sides down to a multiple of `divisor`, without resizing.

    Args:
        image: HWC numpy array or CHW tensor.
        divisor: Integer divisor.

    Returns:
        The cropped image and its 3x3 translation matrix.
    """
    h, w = _get_hw(image)
    h_new = h // divisor * divisor
    w_new = w // divisor * divisor

    if h_new == h and w_new == w:
        return image, np.eye(3)

    cy = (h - h_new) // 2
    cx = (w - w_new) // 2

    if isinstance(image, np.ndarray):
        cropped = image[cy : cy + h_new, cx : cx + w_new]
    else:
        cropped = image[..., cy : cy + h_new, cx : cx + w_new]

    transform = np.eye(3)
    transform[0, 2] = -cx
    transform[1, 2] = -cy
    return cropped, transform


# --- Depth --------------------------------------------------------------------


def resize_depth(depth, target_hw, invalid_value=-1.0):
    """Resize a depth map (H, W) with INTER_AREA, invalid wherever an invalid pixel contributes.

    Args:
        depth: Depth (H, W) as float32 numpy array, negative values are invalid.
        target_hw: Output size (H, W).
        invalid_value: Value of the invalid output pixels.
    """
    h_target, w_target = target_hw
    d = depth.astype(np.float32).copy()
    d[d < 0] = np.nan
    resized = cv2.resize(d, (w_target, h_target), interpolation=cv2.INTER_AREA)
    resized[np.isnan(resized)] = invalid_value
    return resized


# --- I/O ----------------------------------------------------------------------


def read_image(path, grayscale=False, as_tensor=False):
    """Image from disk as HWC uint8 numpy array, or CHW float32 tensor with `as_tensor`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]  # BGR -> RGB
    if as_tensor:
        return to_tensor(image)
    return image


# --- Normalization ------------------------------------------------------------


class ImageNetNormalizer(nn.Module):
    """Normalize images with ImageNet mean and std."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x):
        return ((x.transpose(-3, -1) - self.mean) / self.std).transpose(-3, -1)


# --- Composer -----------------------------------------------------------------


class ImagePreprocessor:
    """resize_to_cover, crop_to_ar, crop_to_divisible, optional augmentation, to_tensor.

    Works on uint8 numpy arrays and converts to a tensor at the end. Depth and masks follow the
    same spatial steps with nearest interpolation, and the steps compose into one 3x3
    transform for the intrinsics. `conf.resize` is the long edge of the output, the aspect
    ratio is passed per call.
    """

    default_conf = {
        "resize": None,  # long edge of output, None for no resize
        "edge_divisible_by": None,  # crop to nearest multiple
        "resize_method": "torch_bilinear_aa",  # or "pil_lanczos", "cv2_inter_area"
    }

    def __init__(self, conf, augmentation=None):
        """
        Args:
            conf: Preprocessing config, OmegaConf or dict.
            augmentation: Optional HWC uint8 to HWC uint8 callable.
        """
        default_conf = OmegaConf.create(self.default_conf)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        self.augmentation = augmentation

    def __call__(self, image, depth=None, mask=None, aspect_ratio=1.0, zoom=1.0):
        """Preprocess an image (and optional depth map / binary mask).

        Args:
            image: HWC uint8 numpy array, or a CHW float tensor converted first.
            depth: Optional depth (H, W), negative values are invalid.
            mask: Optional binary mask (H, W), nonzero is True.
            aspect_ratio: Target W / H, 1.0 for square.
            zoom: Centre-crop factor below 1 zooms in, clamped so nothing is upscaled. 1.0
                is off.

        Returns:
            dict with:
                "image": (3, H, W) float32 tensor
                "transform": (3, 3) float64 ndarray
                "depth": (H, W) float32 tensor (if depth provided)
                "depth_mask": (H, W) bool tensor (if depth provided)
                "mask": (H, W) bool tensor (if mask provided)
        """
        if isinstance(image, torch.Tensor):
            image = to_numpy(image)

        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = depth.astype(np.float32).copy()
            depth[depth < 0] = np.nan

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            mask = (np.asarray(mask) > 0).astype(np.uint8)

        transform = np.eye(3)

        # 0. Zoom: center crop by s = max(zoom, smallest crop that still covers the target)
        if zoom < 1.0 and self.conf.resize is not None:
            h, w = image.shape[:2]
            target_h, target_w = self._compute_target(aspect_ratio)
            s = max(zoom, target_h / h, target_w / w)
            if s < 1.0:
                image, x0, y0 = _center_crop(image, s)
                if depth is not None:
                    depth = _center_crop(depth, s)[0]
                if mask is not None:
                    mask = _center_crop(mask, s)[0]
                t = np.eye(3)
                t[0, 2], t[1, 2] = -x0, -y0
                transform = t @ transform

        # 1. Resize to cover target dimensions
        if self.conf.resize is not None:
            target_h, target_w = self._compute_target(aspect_ratio)
            image, t = resize_to_cover(image, target_h, target_w, method=self.conf.resize_method)
            transform = t @ transform
            if depth is not None:
                h, w = image.shape[:2]
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            if mask is not None:
                h, w = image.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Crop to target AR
        image, t = crop_to_ar(image, aspect_ratio)
        transform = t @ transform
        if depth is not None:
            h, w = image.shape[:2]
            dh, dw = depth.shape
            if dh != h or dw != w:
                cy, cx = int(-t[1, 2]), int(-t[0, 2])
                depth = depth[cy : cy + h, cx : cx + w]
        if mask is not None:
            h, w = image.shape[:2]
            mh, mw = mask.shape
            if mh != h or mw != w:
                cy, cx = int(-t[1, 2]), int(-t[0, 2])
                mask = mask[cy : cy + h, cx : cx + w]

        # 3. Crop to divisible
        if self.conf.edge_divisible_by is not None:
            image, t = crop_to_divisible(image, self.conf.edge_divisible_by)
            transform = t @ transform
            if depth is not None:
                h, w = image.shape[:2]
                dh, dw = depth.shape
                if dh != h or dw != w:
                    cy, cx = int(-t[1, 2]), int(-t[0, 2])
                    depth = depth[cy : cy + h, cx : cx + w]
            if mask is not None:
                h, w = image.shape[:2]
                mh, mw = mask.shape
                if mh != h or mw != w:
                    cy, cx = int(-t[1, 2]), int(-t[0, 2])
                    mask = mask[cy : cy + h, cx : cx + w]

        # 4. Color augmentation (on uint8, after spatial)
        if self.augmentation is not None:
            image = self.augmentation(image)

        # 5. To tensor
        result = {
            "image": to_tensor(image),
            "transform": transform,
        }

        if depth is not None:
            depth[np.isnan(depth)] = -1.0
            depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).float()
            result["depth"] = depth_tensor
            result["depth_mask"] = depth_tensor > 0

        if mask is not None:
            result["mask"] = torch.from_numpy(np.ascontiguousarray(mask)) > 0

        return result

    def _compute_target(self, aspect_ratio):
        """Compute (target_h, target_w) from resize and aspect_ratio."""
        if aspect_ratio >= 1.0:  # landscape: W >= H
            target_w = self.conf.resize
            target_h = round(self.conf.resize / aspect_ratio)
        else:  # portrait: H > W
            target_h = self.conf.resize
            target_w = round(self.conf.resize * aspect_ratio)
        return target_h, target_w
