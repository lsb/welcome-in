"""SSD anchor generation for the MediaPipe BlazeFace "back" (256x256) detector.

This is a NumPy port of MediaPipe's SsdAnchorsCalculator, configured with the
options used by the face_detection back-camera model. It produces 896 anchors,
matching the two output heads of the Qualcomm ONNX export:

    stride 16 : 16 x 16 grid x 2 anchors = 512   (-> box_coords_1 [1, 512, 16])
    stride 32 :  8 x  8 grid x 6 anchors = 384   (-> box_coords_2 [1, 384, 16])

Each anchor is (x_center, y_center, w, h) in normalized [0, 1] coordinates.
With fixed_anchor_size=True the width/height are always 1.0 (as the model
expects), so only the centers actually vary.
"""

from __future__ import annotations

import math

import numpy as np

# MediaPipe face_detection "back" model anchor options.
INPUT_SIZE = 256
MIN_SCALE = 0.15625
MAX_SCALE = 0.75
STRIDES = (16, 32, 32, 32)
ASPECT_RATIOS = (1.0,)
ANCHOR_OFFSET_X = 0.5
ANCHOR_OFFSET_Y = 0.5
INTERPOLATED_SCALE_ASPECT_RATIO = 1.0
FIXED_ANCHOR_SIZE = True
REDUCE_BOXES_IN_LOWEST_LAYER = False


def _scale_at(layer_id: int, num_layers: int) -> float:
    if num_layers == 1:
        return (MIN_SCALE + MAX_SCALE) * 0.5
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * layer_id / (num_layers - 1.0)


def generate_anchors() -> np.ndarray:
    """Return an (896, 4) float32 array of (x_center, y_center, w, h) anchors."""
    strides = STRIDES
    num_layers = len(strides)
    anchors: list[list[float]] = []

    layer_id = 0
    while layer_id < num_layers:
        anchor_heights: list[float] = []
        anchor_widths: list[float] = []
        aspect_ratios: list[float] = []
        scales: list[float] = []

        # Merge all consecutive layers that share the same stride.
        last = layer_id
        while last < num_layers and strides[last] == strides[layer_id]:
            scale = _scale_at(last, num_layers)
            if last == 0 and REDUCE_BOXES_IN_LOWEST_LAYER:
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                for ar in ASPECT_RATIOS:
                    aspect_ratios.append(ar)
                    scales.append(scale)
                if INTERPOLATED_SCALE_ASPECT_RATIO > 0.0:
                    scale_next = 1.0 if last == num_layers - 1 else _scale_at(last + 1, num_layers)
                    scales.append(math.sqrt(scale * scale_next))
                    aspect_ratios.append(INTERPOLATED_SCALE_ASPECT_RATIO)
            last += 1

        for ar in aspect_ratios:
            r = math.sqrt(ar)
            anchor_heights.append(1.0 / r)  # value only used when not fixed size
            anchor_widths.append(1.0 * r)

        stride = strides[layer_id]
        feature_map_height = math.ceil(INPUT_SIZE / stride)
        feature_map_width = math.ceil(INPUT_SIZE / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for a in range(len(anchor_heights)):
                    x_center = (x + ANCHOR_OFFSET_X) / feature_map_width
                    y_center = (y + ANCHOR_OFFSET_Y) / feature_map_height
                    if FIXED_ANCHOR_SIZE:
                        w = h = 1.0
                    else:
                        w = anchor_widths[a]
                        h = anchor_heights[a]
                    anchors.append([x_center, y_center, w, h])

        layer_id = last

    return np.asarray(anchors, dtype=np.float32)


if __name__ == "__main__":
    a = generate_anchors()
    print("anchors:", a.shape)  # expect (896, 4)
