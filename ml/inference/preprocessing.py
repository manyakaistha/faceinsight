"""Preprocessing and image I/O utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags

from ..models.config import IMAGENET_MEAN, IMAGENET_STD


def _pil_exif_rotate(img: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        else:
            orientation = None
        if orientation is None:
            return img
        exif = img._getexif()
        if not exif:
            return img
        tag = exif.get(orientation)
        if tag == 3:
            img = img.rotate(180, expand=True)
        elif tag == 6:
            img = img.rotate(270, expand=True)
        elif tag == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def load_image_pil(data: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(data, Image.Image):
        img = data.convert("RGB")
    elif isinstance(data, (str, Path)):
        img = Image.open(str(data)).convert("RGB")
    elif isinstance(data, bytes):
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image data type {type(data)!r}")
    img = _pil_exif_rotate(img)
    return img


def pil_to_tensor_normalized(img: Image.Image, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    from torchvision import transforms

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return tfm(img)


def crop_face_from_pil(img: Image.Image, x1: int, y1: int, x2: int, y2: int, expand_factor: float = 1.1) -> Image.Image:
    w, h = img.size
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)
    nw = fw * expand_factor
    nh = fh * expand_factor
    nx1 = int(cx - nw / 2.0)
    ny1 = int(cy - nh / 2.0)
    nx2 = int(cx + nw / 2.0)
    ny2 = int(cy + nh / 2.0)
    nx1 = max(0, min(w, nx1))
    ny1 = max(0, min(h, ny1))
    nx2 = max(0, min(w, nx2))
    ny2 = max(0, min(h, ny2))
    return img.crop((nx1, ny1, nx2, ny2))


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    from torchvision import transforms

    return transforms.Resize(size)(transforms.CenterCrop(size)(img)) if False else transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])(img)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    if bgr.ndim == 2:
        return Image.fromarray(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
