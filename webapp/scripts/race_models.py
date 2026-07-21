"""Race candidate CLIP image encoders on the sample PNGs.

Each candidate is scored with the same bank (rich prompts + hood distractor,
desktop group structure, see calibrate_bank.py for why), embedded by ITS OWN
text encoder. Reports per-image tags, the gate gap (2.png's garment conf — the
far painting that must NOT read — vs the weakest real read), and hat margins
(6.png real hat must clear +0.03; 5.png raised hood must not).

Run from repo root: .venv/bin/python webapp/scripts/race_models.py [candidate ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clip_tags import _COLORS, _torso_crop  # noqa: E402

HF = ROOT / "models/hf"

CANDIDATES = {
    "s0-fp16": ("models--Xenova--mobileclip_s0", "onnx/vision_model_fp16.onnx",
                "onnx/text_model.onnx", 256),
    "s1-fp16": ("models--Xenova--mobileclip_s1", "onnx/vision_model_fp16.onnx",
                "onnx/text_model.onnx", 256),
    "s2-int8": ("models--Xenova--mobileclip_s2", "onnx/vision_model_int8.onnx",
                "onnx/text_model.onnx", 256),
    "s2-fp16": ("models--Xenova--mobileclip_s2", "onnx/vision_model_fp16.onnx",
                "onnx/text_model.onnx", 256),
    # MobileCLIP2-S0 (OpenCLIP-style export: visual/text at the repo root,
    # text input is token ids under a different name; same preprocessing).
    "mc2-s0": ("models--RuteNL--MobileCLIP2-S0-OpenCLIP-ONNX", "visual.onnx",
               "text.onnx", 256),
}

TEMPLATES = ["{}", "a photo of {}"]

GROUPS: dict[str, dict[str, list[str]]] = {
    "color": {c: [t.format(f"a person wearing {c} clothing") for t in TEMPLATES]
              + [f"a person dressed in {c}"] for c in _COLORS},
    "garment": {
        "hoodie": ["a person wearing a hoodie", "a photo of a person in a hooded sweatshirt"],
        "t-shirt": ["a person wearing a t-shirt", "a photo of a person in a tee shirt"],
        "button-up shirt": ["a person wearing a button-up collared shirt",
                            "a photo of a person in a collared dress shirt"],
        "blazer": ["a person wearing a blazer or suit jacket",
                   "a photo of a person in a tailored blazer"],
        "sweater": ["a person wearing a sweater", "a photo of a person in a knit sweater"],
        "dress": ["a person wearing a dress", "a photo of a person in a dress"],
        "denim jacket": ["a person wearing a denim jacket",
                         "a photo of a person in a jean jacket"],
        "polo shirt": ["a person wearing a polo shirt",
                       "a photo of a person in a short-sleeved polo shirt"],
        "jacket": ["a person wearing a jacket or coat",
                   "a photo of a person in a jacket"],
    },
    "style": {
        "casual": ["casual everyday attire", "a photo of a casually dressed person"],
        "formal": ["formal elegant attire", "a photo of a formally dressed person"],
        "athletic": ["athletic sporty attire", "a photo of a person in sporty athletic wear"],
    },
    "hat": {
        "hat": ["a person wearing a hat on their head", "a photo of a person in a hat"],
        "hood": ["a person with the hood of their hoodie pulled up over their head",
                 "a photo of a person wearing a raised hood"],
        "nohat": ["a person with nothing on their head, bareheaded",
                  "a photo of a bareheaded person"],
    },
}


def build_bank(snap_dir: Path, text_file: str):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snap_dir / "tokenizer.json"))
    tok.enable_padding(length=77, pad_id=0)
    tok.enable_truncation(max_length=77)
    so = ort.SessionOptions()
    # The fp16 text exports trip an ORT layer-norm fusion bug at load; the text
    # encoder runs once offline, so optimizations are not worth working around.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    txt = ort.InferenceSession(str(snap_dir / text_file), sess_options=so,
                               providers=["CPUExecutionProvider"])
    names = [i.name for i in txt.get_inputs()]

    def embed(prompts):
        enc = tok.encode_batch(prompts)
        ids = np.array([e.ids for e in enc], np.int64)
        feeds = {names[0]: ids}   # "input_ids" (HF-style) or "text" (OpenCLIP-style)
        if "attention_mask" in names:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], np.int64)
        emb = txt.run(None, feeds)[0].astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)

    bank = {}
    for g, mapping in GROUPS.items():
        labels = list(mapping)
        rows = []
        for label in labels:
            e = embed(mapping[label]).mean(axis=0)
            rows.append(e / (np.linalg.norm(e) + 1e-9))
        bank[g] = (labels, np.stack(rows))
    return bank


def run(name: str) -> None:
    import onnxruntime as ort

    repo_dir, vis_file, txt_file, size = CANDIDATES[name]
    snap = next((HF / repo_dir / "snapshots").glob("*"))
    bank = build_bank(snap, txt_file)
    vis = ort.InferenceSession(str(snap / vis_file),
                               providers=["CPUExecutionProvider"])
    vname = vis.get_inputs()[0].name
    vhalf = vis.get_inputs()[0].type == "tensor(float16)"
    vsize_mb = (snap / vis_file).stat().st_size / 1e6

    def embed_image(im, box):
        if box is not None:
            im = _torso_crop(im, box)
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        nw, nh = im.size
        left, top = (nw - size) // 2, (nh - size) // 2
        im = im.crop((left, top, left + size, top + size))
        arr = np.asarray(im, np.float32) / 255.0
        feed = np.transpose(arr, (2, 0, 1))[None]
        emb = vis.run(None, {vname: feed.astype(np.float16) if vhalf else feed})[0][0]
        emb = emb.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    ref = json.load(open(ROOT / "webapp/scripts/reference_outputs.json"))
    print(f"\n=== {name}  ({vis_file}, {vsize_mb:.1f} MB) ===")
    reads, t0 = {}, time.perf_counter()
    for img in ("2.png", "4.png", "5.png", "6.png", "7.png", "8.png"):
        ie = embed_image(Image.open(ROOT / img).convert("RGB"),
                         ref["faces"][img]["box"])
        row = {}
        for g in ("color", "garment", "style"):
            labels, emb = bank[g]
            sims = emb @ ie
            i = int(sims.argmax())
            row[g] = (labels[i], float(sims[i]))
        labels, hemb = bank["hat"]
        hs = dict(zip(labels, hemb @ ie))
        row["hat_margin"] = hs["hat"] - max(hs["hood"], hs["nohat"])
        reads[img] = row
        print(f"{img}: {row['color'][0]:6s}({row['color'][1]:.3f}) "
              f"{row['garment'][0]:15s}({row['garment'][1]:.3f}) "
              f"{row['style'][0]:8s} hat_margin={row['hat_margin']:+.4f}")
    dt = (time.perf_counter() - t0) / 6 * 1000
    reject = reads["2.png"]["garment"][1]
    weakest = min(reads[i]["garment"][1] for i in ("4.png", "5.png", "6.png", "7.png", "8.png"))
    print(f"gate: reject 2.png at {reject:.3f} vs weakest real {weakest:.3f} "
          f"-> gap {weakest - reject:+.3f}")
    print(f"hat:  6.png {reads['6.png']['hat_margin']:+.4f} (needs >= +0.03)  "
          f"5.png hood {reads['5.png']['hat_margin']:+.4f} (needs < 0.03)")
    print(f"avg embed time {dt:.0f} ms/image (python ORT CPU)")


if __name__ == "__main__":
    for n in (sys.argv[1:] or CANDIDATES):
        run(n)
