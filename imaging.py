"""Tiny shared seam: accept a frame as a path, a PIL image, or an ndarray.

The face gate and the clothing tagger were first written against image *paths*
(`Image.open(...)`). Live capture hands us in-memory frames instead, so both
stages now route their input through `to_rgb()`. Paths still work unchanged (the
sample-image CLI and the warmup pass are untouched), but a webcam frame — a PIL
image, or the HWC uint8 ndarray that imageio yields (see camera.py) — now flows
through the exact same code with no copy on disk.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_rgb(src) -> Image.Image:
    """Coerce a path / PIL.Image / HWC uint8 ndarray to an RGB PIL.Image."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, np.ndarray):
        return Image.fromarray(src).convert("RGB")
    return Image.open(src).convert("RGB")
