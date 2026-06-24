"""Stage 1.5 — what the visitor is wearing.

llama.cpp's multimodal (mtmd) path is currently broken on CPU for Qwen3.5 — it
mangles the image (a solid red square reads as "black"/"blue"), so the VLM can't
be trusted to describe clothing. Instead we get clothing tags with zero-shot
CLIP, which is cheap and robust: crop to the visitor's torso (using the BlazeFace
box), embed the crop with marqo-fashionCLIP's image encoder (ONNX, CPU, same
onnxruntime as the face detector — no torch), and compare against a fixed bank of
candidate phrases (colors, garments, style, hat) by cosine similarity. The
winning tags become a short description handed to the greeter.

The text encoder is only used once at startup, to embed the fixed candidate
phrases; per visitor we run just the small image encoder + a few dot products.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from models import ensure_clip

# CLIP image normalisation (from the model's preprocessor_config.json).
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# Candidate phrases per attribute group: {clean label: CLIP prompt}.
_COLORS = ["red", "orange", "yellow", "green", "blue", "navy", "purple",
           "pink", "brown", "beige", "gray", "black", "white"]
_GROUPS: dict[str, dict[str, str]] = {
    "color": {c: f"a person wearing {c} clothing" for c in _COLORS},
    "garment": {
        "hoodie": "a person wearing a hoodie",
        "t-shirt": "a person wearing a t-shirt",
        "button-up shirt": "a person wearing a button-up collared shirt",
        "blazer": "a person wearing a blazer or suit jacket",
        "sweater": "a person wearing a sweater",
        "dress": "a person wearing a dress",
        "denim jacket": "a person wearing a denim jacket",
        "polo shirt": "a person wearing a polo shirt",
        "jacket": "a person wearing a jacket or coat",
    },
    "style": {
        "casual": "casual everyday attire",
        "formal": "formal elegant attire",
        "athletic": "athletic sporty attire",
    },
    "hat": {
        "hat": "a person wearing a hat",
        "nohat": "a person not wearing any hat, bareheaded",
    },
}

# Below this image↔garment cosine similarity we don't trust the read and return
# no description (e.g. no clear person in the crop).
_MIN_CONF = 0.15
# Only mention a hat if it beats "no hat" by at least this margin (a raised hood
# scores hat-ish, so we want a clear win — a real hat clears ~0.04).
_HAT_MARGIN = 0.03


class ClipTagger:
    def __init__(self):
        import onnxruntime as ort

        paths = ensure_clip()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # let ORT use all cores (matches the Pi)
        self._vis = ort.InferenceSession(
            str(paths["vision"]), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._groups = self._embed_candidates(paths)  # name -> (labels, embeds[N,512])

    def _embed_candidates(self, paths) -> dict[str, tuple[list[str], np.ndarray]]:
        """Embed every candidate phrase once. The text encoder + tokenizer are
        dropped afterwards (only the image encoder is needed per visitor)."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(paths["tokenizer"]))
        tok.enable_padding(length=77, pad_id=0)
        tok.enable_truncation(max_length=77)
        txt = ort.InferenceSession(str(paths["text"]), providers=["CPUExecutionProvider"])

        groups = {}
        for name, mapping in _GROUPS.items():
            labels = list(mapping)
            ids = np.array([e.ids for e in tok.encode_batch(list(mapping.values()))],
                           dtype=np.int64)
            emb = txt.run(None, {"input_ids": ids})[0]
            groups[name] = (labels, _l2(emb))
        return groups

    # -- image side ------------------------------------------------------
    def _embed_image(self, image_path, box) -> np.ndarray:
        im = Image.open(image_path).convert("RGB")
        if box is not None:
            im = _torso_crop(im, box)
        arr = (np.asarray(_resize_center(im), np.float32) / 255.0 - _MEAN) / _STD
        pixel_values = np.transpose(arr, (2, 0, 1))[None]
        emb = self._vis.run(None, {"pixel_values": pixel_values})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def describe(self, image_path, box=None) -> str | None:
        """Return a short clothing description like 'a red hoodie, in a casual
        style' (or '… and a hat …'), or None if nothing is read confidently."""
        ie = self._embed_image(image_path, box)
        color, _ = self._top("color", ie)
        garment, gconf = self._top("garment", ie)
        if gconf < _MIN_CONF:
            return None
        style, _ = self._top("style", ie)

        desc = f"{_article(color)} {color} {garment}"
        labels, hat_emb = self._groups["hat"]
        hat_sims = hat_emb @ ie
        if hat_sims[labels.index("hat")] - hat_sims[labels.index("nohat")] >= _HAT_MARGIN:
            desc += " and a hat"
        return f"{desc}, in {_article(style)} {style} style"

    def _top(self, group: str, image_emb: np.ndarray) -> tuple[str, float]:
        labels, emb = self._groups[group]
        sims = emb @ image_emb
        i = int(sims.argmax())
        return labels[i], float(sims[i])


# -- helpers -------------------------------------------------------------
def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _torso_crop(im: Image.Image, box) -> Image.Image:
    """Expand the (normalised) face box down to the torso and up a little for a
    hat, so the clothing fills the frame."""
    w, h = im.size
    x1, y1, x2, y2 = box
    fw, fh, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
    cx1 = max(0.0, cx - 1.8 * fw)
    cx2 = min(1.0, cx + 1.8 * fw)
    cy1 = max(0.0, y1 - 2.0 * fh)
    cy2 = min(1.0, y2 + 5.0 * fh)
    return im.crop((int(cx1 * w), int(cy1 * h), int(cx2 * w), int(cy2 * h)))


def _resize_center(im: Image.Image, size: int = 224) -> Image.Image:
    """Resize the shortest edge to `size` (bicubic) and center-crop a square."""
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    nw, nh = im.size
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))
