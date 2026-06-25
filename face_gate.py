"""Stage 1 — the visual wake word.

Runs Qualcomm's MediaPipe BlazeFace ONNX detector (CPU, via onnxruntime) and
decides whether someone is *present and facing the camera*. The 6 BlazeFace
keypoints (eyes, nose, mouth, two ear tragions) give us a cheap, robust
frontal-pose test without a second model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from anchors import INPUT_SIZE, generate_anchors
from imaging import to_rgb
from models import ensure_face_model

# BlazeFace keypoint indices (within each decoded row, after the 4 bbox values).
KP_RIGHT_EYE, KP_LEFT_EYE, KP_NOSE, KP_MOUTH, KP_RIGHT_EAR, KP_LEFT_EAR = range(6)

# Tunable thresholds (revisit with --debug against the sample images).
DET_THRESH = 0.5      # min sigmoid score to count as a face
NMS_IOU = 0.3
YAW_MAX = 0.35        # |nose offset from eye-midpoint| / inter-ocular distance
EAR_LO, EAR_HI = 0.6, 1.67  # nose->left-ear vs nose->right-ear distance ratio
ROLL_MAX_DEG = 30.0   # head tilt


@dataclass
class FaceResult:
    person_present: bool
    facing_camera: bool
    facing_conf: float = 0.0
    score: float = 0.0
    box: tuple[float, float, float, float] | None = None  # original-image normalized xyxy
    keypoints: np.ndarray | None = None                   # (6, 2) original-image normalized
    description: str = ""
    metrics: dict = field(default_factory=dict)           # debug: yaw / ear_ratio / roll


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))  # ±30 saturates sigmoid, float32-safe


def _letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """Resize keeping aspect ratio, pad to a square. Returns (canvas, scale, pad_x, pad_y)."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _iou_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_thresh]
    return keep


class FaceGate:
    def __init__(self, onnx_path: Path | None = None):
        import onnxruntime as ort

        path = onnx_path or ensure_face_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # let ORT pick (uses all cores on the Pi)
        self.session = ort.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        out = [o.name for o in self.session.get_outputs()]
        self.coords_names = sorted(n for n in out if "coord" in n.lower())  # _1 (512) then _2 (384)
        self.scores_names = sorted(n for n in out if "score" in n.lower())
        self.anchors = generate_anchors()  # (896, 4)

    # -- inference -------------------------------------------------------
    def _detect(self, canvas: Image.Image):
        x = np.asarray(canvas, dtype=np.float32) / 255.0   # HWC, [0,1]
        x = np.transpose(x, (2, 0, 1))[None]               # 1,3,H,W
        outputs = dict(zip(
            [o.name for o in self.session.get_outputs()],
            self.session.run(None, {self.input_name: x}),
        ))
        raw = np.concatenate([outputs[n][0] for n in self.coords_names], axis=0)   # (896,16)
        logits = np.concatenate([outputs[n][0] for n in self.scores_names], axis=0)  # (896,1)
        scores = _sigmoid(logits).reshape(-1)
        return raw, scores

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """Decode raw anchor offsets -> (896, 16): xyxy box + 6 keypoints, canvas-normalized."""
        ax, ay, aw, ah = (self.anchors[:, i] for i in range(4))
        s = float(INPUT_SIZE)
        cx = raw[:, 0] / s * aw + ax
        cy = raw[:, 1] / s * ah + ay
        w = raw[:, 2] / s * aw
        h = raw[:, 3] / s * ah
        out = np.empty_like(raw)
        out[:, 0] = cx - w / 2
        out[:, 1] = cy - h / 2
        out[:, 2] = cx + w / 2
        out[:, 3] = cy + h / 2
        for k in range(6):
            out[:, 4 + 2 * k] = raw[:, 4 + 2 * k] / s * aw + ax
            out[:, 5 + 2 * k] = raw[:, 5 + 2 * k] / s * ah + ay
        return out

    # -- public API ------------------------------------------------------
    def analyze(self, image) -> FaceResult:
        # `image` is a path, a PIL.Image, or an HWC uint8 ndarray (a live frame).
        img = to_rgb(image)
        ow, oh = img.size
        canvas, scale, pad_x, pad_y = _letterbox(img)
        raw, scores = self._detect(canvas)

        mask = scores >= DET_THRESH
        if not mask.any():
            return FaceResult(person_present=False, facing_camera=False)

        decoded = self._decode(raw)[mask]
        sc = scores[mask]
        keep = _iou_nms(decoded[:, :4], sc, NMS_IOU)
        best = keep[int(np.argmax(sc[keep]))]
        det, score = decoded[best], float(sc[best])

        # Facing test in canvas space (uniform scale+translation -> invariant).
        kp_canvas = det[4:].reshape(6, 2)
        facing, conf, metrics = self._facing(kp_canvas)

        # Map box + keypoints back to original-image normalized coords for context.
        def to_orig(pts: np.ndarray) -> np.ndarray:
            px = (pts[..., 0] * INPUT_SIZE - pad_x) / scale / ow
            py = (pts[..., 1] * INPUT_SIZE - pad_y) / scale / oh
            return np.stack([px, py], axis=-1)

        box_pts = to_orig(det[:4].reshape(2, 2)).reshape(-1)
        kp_orig = to_orig(kp_canvas)
        box = (float(box_pts[0]), float(box_pts[1]), float(box_pts[2]), float(box_pts[3]))

        return FaceResult(
            person_present=True,
            facing_camera=facing,
            facing_conf=conf,
            score=score,
            box=box,
            keypoints=kp_orig,
            description=self._describe(box),
            metrics=metrics,
        )

    def _facing(self, kp: np.ndarray):
        re, le, nose = kp[KP_RIGHT_EYE], kp[KP_LEFT_EYE], kp[KP_NOSE]
        rear, lear = kp[KP_RIGHT_EAR], kp[KP_LEFT_EAR]
        iod = float(np.hypot(*(le - re)))
        if iod < 1e-6:
            return False, 0.0, {"reason": "degenerate eyes"}

        eye_mid = (re + le) / 2.0
        yaw = float((nose[0] - eye_mid[0]) / iod)

        d_l = float(np.hypot(*(nose - lear)))
        d_r = float(np.hypot(*(nose - rear)))
        ear_ratio = d_l / (d_r + 1e-6)

        roll = math.degrees(math.atan2(le[1] - re[1], le[0] - re[0]))
        roll = ((roll + 90.0) % 180.0) - 90.0  # fold to [-90, 90]

        ok = (abs(yaw) < YAW_MAX) and (EAR_LO < ear_ratio < EAR_HI) and (abs(roll) < ROLL_MAX_DEG)

        s_yaw = max(0.0, 1.0 - abs(yaw) / YAW_MAX)
        s_ear = max(0.0, 1.0 - abs(math.log(max(ear_ratio, 1e-6))) / abs(math.log(EAR_HI)))
        s_roll = max(0.0, 1.0 - abs(roll) / ROLL_MAX_DEG)
        conf = (s_yaw * s_ear * s_roll) ** (1.0 / 3.0)

        return ok, conf, {"yaw": round(yaw, 3), "ear_ratio": round(ear_ratio, 3), "roll": round(roll, 1)}

    @staticmethod
    def _describe(box: tuple[float, float, float, float]) -> str:
        cx = (box[0] + box[2]) / 2.0
        bh = abs(box[3] - box[1])
        if cx < 0.38:
            pos = "off to your left"
        elif cx > 0.62:
            pos = "off to your right"
        else:
            pos = "right in front of you"
        if bh > 0.45:
            prox = "very close"
        elif bh > 0.25:
            prox = "close by"
        elif bh > 0.12:
            prox = "a few steps away"
        else:
            prox = "across the room"
        return f"{pos}, {prox}"
