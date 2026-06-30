"""Input preprocessing helpers for EviField-ERA optical-flow inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def load_rgb_image(path: str | Path) -> Tensor:
    """Load one RGB image as [3, H, W] float tensor in raw 0..255 scale."""
    image = Image.open(path).convert("RGB")
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float()


def normalize_optical_flow_image(image: Tensor) -> Tensor:
    """Match training preprocessing: 0..255 or 0..1 image -> [-1, 1]."""
    image = image.float()
    if image.numel() == 0:
        return image
    if float(image.min().item()) < 0.0:
        return image
    if float(image.max().item()) > 1.0:
        image = image / 255.0
    return image * 2.0 - 1.0


def prepare_optical_flow_pair(
    image1: Tensor,
    image2: Tensor,
    *,
    resize_to: Optional[Tuple[int, int]] = None,
    center_crop: Optional[Tuple[int, int]] = None,
) -> tuple[list[Tensor], dict[str, object]]:
    """Prepare a pair of optical-flow input images.

    Args:
        image1/image2: [3, H, W], [H, W, 3], or batched [1, 3, H, W].
        resize_to: optional (height, width). If used, output flow is in resized
            image coordinates and should be rescaled before overlaying on the
            original image.

    Returns:
        images: list of two tensors, each [1, 3, H, W].
        meta: original/final size and scale factors.
    """
    image1 = _ensure_bchw(image1)
    image2 = _ensure_bchw(image2)
    if image1.shape != image2.shape:
        raise ValueError(f"image1 and image2 must have the same shape, got {tuple(image1.shape)} vs {tuple(image2.shape)}")
    original_h, original_w = int(image1.shape[-2]), int(image1.shape[-1])
    if resize_to is not None:
        image1 = F.interpolate(image1, size=resize_to, mode="bilinear", align_corners=False)
        image2 = F.interpolate(image2, size=resize_to, mode="bilinear", align_corners=False)
    crop_meta = None
    if center_crop is not None:
        image1, crop_meta = _center_crop_bchw(image1, center_crop)
        image2, _ = _center_crop_bchw(image2, center_crop)
    image1 = normalize_optical_flow_image(image1)
    image2 = normalize_optical_flow_image(image2)
    final_h, final_w = int(image1.shape[-2]), int(image1.shape[-1])
    return [image1, image2], {
        "original_size": [original_h, original_w],
        "input_size": [final_h, final_w],
        "image_scale_x": final_w / max(original_w, 1),
        "image_scale_y": final_h / max(original_h, 1),
        "resize_to": list(resize_to) if resize_to is not None else None,
        "center_crop": crop_meta,
    }


def load_ltem_image(path: str | Path) -> Tensor:
    """Load one LTEM grayscale image as [1, H, W] float tensor in raw 0..255 scale."""
    image = Image.open(path).convert("L")
    array = np.array(image, copy=True)
    return torch.from_numpy(array).unsqueeze(0).contiguous().float()


def normalize_ltem_image(image: Tensor) -> Tensor:
    """Match LTEM preprocessing: grayscale 0..255 or 0..1 -> 0..1."""
    image = image.float()
    if image.numel() == 0:
        return image
    return image / 255.0 if float(image.max().item()) > 1.0 else image


def prepare_ltem_triplet(
    under: Tensor,
    infocus: Tensor,
    over: Tensor,
    *,
    resize_to: Optional[Tuple[int, int]] = (256, 256),
    center_crop: Optional[Tuple[int, int]] = (224, 224),
) -> tuple[list[Tensor], dict[str, object]]:
    """Prepare LTEM U/I/O images using the formal validation transform."""
    images = [_ensure_bchw(image) for image in (under, infocus, over)]
    # LTEM images are one-channel; _ensure_bchw repeats to RGB only for optical
    # helpers, so restore one channel if needed.
    images = [image[:, :1] for image in images]
    original_h, original_w = int(images[0].shape[-2]), int(images[0].shape[-1])
    if any(image.shape[-2:] != images[0].shape[-2:] for image in images):
        raise ValueError("All LTEM images must share spatial dimensions.")
    if resize_to is not None:
        images = [F.interpolate(image, size=resize_to, mode="bilinear", align_corners=False) for image in images]
    crop_meta = None
    if center_crop is not None:
        cropped = []
        for image in images:
            image, crop_meta = _center_crop_bchw(image, center_crop)
            cropped.append(image)
        images = cropped
    images = [normalize_ltem_image(image) for image in images]
    final_h, final_w = int(images[0].shape[-2]), int(images[0].shape[-1])
    return images, {
        "input_order": ["U", "I", "O"],
        "original_size": [original_h, original_w],
        "input_size": [final_h, final_w],
        "resize_to": list(resize_to) if resize_to is not None else None,
        "center_crop": crop_meta,
        "normalization": "0..255 or 0..1 -> 0..1",
    }


def resize_flow_to_original(flow: Tensor, original_size: Tuple[int, int]) -> Tensor:
    """Resize [B, 2, H, W] optical flow back to original coordinates."""
    target_h, target_w = original_size
    source_h, source_w = int(flow.shape[-2]), int(flow.shape[-1])
    if (source_h, source_w) == (target_h, target_w):
        return flow
    resized = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
    resized[:, 0:1] *= target_w / max(source_w, 1)
    resized[:, 1:2] *= target_h / max(source_h, 1)
    return resized


def _ensure_bchw(image: Tensor) -> Tensor:
    image = torch.as_tensor(image)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.unsqueeze(0)
    elif image.ndim == 3 and image.shape[-1] in {1, 3}:
        image = image.permute(2, 0, 1).unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image [3,H,W], [H,W,3], or [B,3,H,W], got {tuple(image.shape)}")
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.shape[1] != 3:
        raise ValueError(f"Expected 3 channels after adaptation, got {image.shape[1]}")
    return image.contiguous().float()


def _center_crop_bchw(image: Tensor, crop_size: Tuple[int, int]) -> tuple[Tensor, dict[str, int]]:
    crop_h = min(int(crop_size[0]), int(image.shape[-2]))
    crop_w = min(int(crop_size[1]), int(image.shape[-1]))
    top = max((int(image.shape[-2]) - crop_h) // 2, 0)
    left = max((int(image.shape[-1]) - crop_w) // 2, 0)
    return image[..., top : top + crop_h, left : left + crop_w], {
        "top": top,
        "left": left,
        "height": crop_h,
        "width": crop_w,
    }
