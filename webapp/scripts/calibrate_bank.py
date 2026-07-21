"""Calibrate the MobileCLIP phrase bank against the sample PNGs.

Principle: change as little as possible from the desktop bank (clip_tags.py);
add machinery only where the data shows the desktop design fails on MobileCLIP.

Variants, most minimal first:
  A. desktop bank verbatim (single prompts, absolute MIN_CONF gate,
     hat-vs-nohat margin) — the null hypothesis;
  B. A + a "hood" distractor in the hat group — motivated by 5.png (hood up,
     hat margin +0.022) outscoring 6.png (real feathered hat, +0.006): no
     threshold separates them, the desktop design fails here;
  C. B + prompt-template ensembling (mean of a few standard "a photo of ..."
     variants per label) — only if A/B leave no usable MIN_CONF gap between
     the far-painting no-read (2.png) and the weakest real read.

Desired outcomes (ground truth by eye + desktop parity):
  2.png  far painting, mostly wall     -> NO read (fashionCLIP: no read)
  4.png  close painting of a hoodie   -> read (fashionCLIP reads it too)
  5.png  red hoodie, HOOD UP          -> read, red hoodie, hat=False
  6.png  black blazer + feathered hat -> read, black blazer, hat=True
  7.png  beige/gray polo              -> read, polo/button-up, hat=False
  8.png  navy blazer                  -> read, navy/blue blazer, hat=False

Run from repo root: .venv/bin/python webapp/scripts/calibrate_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clip_tags import _GROUPS, _torso_crop  # noqa: E402

MOBILECLIP_DIR = next(
    (ROOT / "models/hf/models--Xenova--mobileclip_s0/snapshots").glob("*"))

HOOD_PROMPT = "a person with the hood of their hoodie pulled up over their head"
TEMPLATES = ["{}", "a photo of {}", "a picture of {}"]


def variants():
    base = {g: {k: [v] for k, v in m.items()} for g, m in _GROUPS.items()}
    yield "A: desktop verbatim", base
    b = {g: {k: list(v) for k, v in m.items()} for g, m in base.items()}
    b["hat"]["hood"] = [HOOD_PROMPT]
    yield "B: + hood distractor", b
    c = {g: {k: [t.format(p) for p in v for t in TEMPLATES]
             for k, v in m.items()} for g, m in b.items()}
    yield "C: B + template ensemble", c


def main() -> None:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MOBILECLIP_DIR / "tokenizer.json"))
    tok.enable_padding(length=77, pad_id=0)
    tok.enable_truncation(max_length=77)
    txt = ort.InferenceSession(str(MOBILECLIP_DIR / "onnx/text_model.onnx"),
                               providers=["CPUExecutionProvider"])
    txt_inputs = [i.name for i in txt.get_inputs()]

    def embed_texts(prompts: list[str]) -> np.ndarray:
        enc = tok.encode_batch(prompts)
        feeds = {"input_ids": np.array([e.ids for e in enc], np.int64)}
        if "attention_mask" in txt_inputs:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], np.int64)
        emb = txt.run(None, feeds)[0]
        return emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)

    vis = ort.InferenceSession(str(MOBILECLIP_DIR / "onnx/vision_model_fp16.onnx"),
                               providers=["CPUExecutionProvider"])
    vname = vis.get_inputs()[0].name
    vhalf = vis.get_inputs()[0].type == "tensor(float16)"

    def embed_image(im: Image.Image, box) -> np.ndarray:
        if box is not None:
            im = _torso_crop(im, box)
        w, h = im.size
        s = 256 / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        nw, nh = im.size
        left, top = (nw - 256) // 2, (nh - 256) // 2
        im = im.crop((left, top, left + 256, top + 256))
        arr = np.asarray(im, np.float32) / 255.0
        feed = np.transpose(arr, (2, 0, 1))[None]
        emb = vis.run(None, {vname: feed.astype(np.float16) if vhalf else feed})[0][0]
        return (lambda e: e / (np.linalg.norm(e) + 1e-9))(emb.astype(np.float32))

    ref = json.load(open(ROOT / "webapp/scripts/reference_outputs.json"))
    images = {}
    for name in ("2.png", "4.png", "5.png", "6.png", "7.png", "8.png"):
        images[name] = embed_image(Image.open(ROOT / name).convert("RGB"),
                                   ref["faces"][name]["box"])

    for vname_, groups in variants():
        print(f"\n=== {vname_} ===")
        bank = {}
        for g, mapping in groups.items():
            labels = list(mapping)
            embs = []
            for label in labels:
                e = embed_texts(mapping[label]).mean(axis=0)
                embs.append(e / (np.linalg.norm(e) + 1e-9))
            bank[g] = (labels, np.stack(embs))
        for name, ie in images.items():
            row = {}
            for g in ("color", "garment", "style"):
                labels, emb = bank[g]
                sims = emb @ ie
                i = int(sims.argmax())
                row[g] = (labels[i], float(sims[i]))
            labels, hemb = bank["hat"]
            hs = dict(zip(labels, hemb @ ie))
            nohat_best = max(v for k, v in hs.items() if k != "hat")
            print(f"{name}: garment={row['garment'][0]}({row['garment'][1]:.3f})  "
                  f"color={row['color'][0]}({row['color'][1]:.3f})  "
                  f"style={row['style'][0]}  "
                  f"hat_margin={hs['hat'] - nohat_best:+.4f}")


if __name__ == "__main__":
    main()
