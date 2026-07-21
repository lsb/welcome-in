"""Build the shipping CLIP text bank (webapp/public/models/clip_bank.json).

The web app ships only the MobileCLIP2-S0 *image* encoder (fp16, 23.9 MB); the
text encoder runs once here, offline, to embed the candidate phrase bank —
mirroring the desktop design (clip_tags.py embeds its bank once at startup),
just moved to build time.

MobileCLIP2-S0 won the model race (race_models.py): it ties MobileCLIP-S0 on
desktop-parity tags over the sample PNGs but with a ~55% wider no-read gate
gap and +3.7pt general zero-shot accuracy at the same size; bigger models
(S1/S2) did worse on the actual samples. The fp16 conversion is lossless
(cosine 0.99999 vs fp32, identical tags/margins on every sample).

The bank keeps the desktop's group structure and its two-knob thresholds (an
absolute garment-confidence gate + a hat margin). Two calibrated changes, both
forced by measurements on the sample PNGs (see calibrate_bank.py /
race_models.py for the experiments):
  * a "hood" distractor label in the hat group — MobileCLIP scores a raised
    hood (5.png) far above a real hat (6.png), so hat must beat
    max(hood, nohat); with the distractor the desktop's 0.03 margin works;
  * two prompt phrasings per label, averaged — with single prompts the no-read
    gap collapses to 0.006 (unusable); with these the gate gap is 0.028 wide
    around MIN_CONF 0.185 (reject the far painting at 0.171, weakest real
    read 0.199).

Also appends the expected web-tagger outputs for the sample PNGs to
reference_outputs.json so the browser selftest has a target to diff against.

Run from repo root: .venv/bin/python webapp/scripts/make_clip_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clip_tags import _torso_crop  # noqa: E402
from race_models import GROUPS, build_bank  # noqa: E402

MOBILECLIP_DIR = next(
    (ROOT / "models/hf/models--RuteNL--MobileCLIP2-S0-OpenCLIP-ONNX/snapshots").glob("*"))
OUT_DIR = ROOT / "webapp/public/models"
VISION_ONNX = OUT_DIR / "mobileclip2_s0_visual_fp16.onnx"

# Thresholds on the MobileCLIP2-S0 similarity scale; same shape as the
# desktop's (_MIN_CONF, _HAT_MARGIN) pair in clip_tags.py.
MIN_CONF = 0.185
HAT_MARGIN = 0.03
IMAGE_SIZE = 256


def web_read(bank, ie: np.ndarray):
    """The web tagger's read, mirrored in TS (clipTagger.ts): returns the tag
    dict or None, exactly as the browser will decide it."""
    def top(g):
        labels, emb = bank[g]
        sims = emb @ ie
        i = int(sims.argmax())
        return labels[i], float(sims[i])

    garment, gconf = top("garment")
    if gconf < MIN_CONF:
        return None
    color, _ = top("color")
    style, _ = top("style")
    labels, hemb = bank["hat"]
    hs = dict(zip(labels, hemb @ ie))
    has_hat = (hs["hat"] - max(hs["hood"], hs["nohat"])) >= HAT_MARGIN
    return {"color": color, "garment": garment, "style": style,
            "hat": bool(has_hat), "gconf": round(gconf, 4)}


def main() -> None:
    import onnxruntime as ort

    bank = build_bank(MOBILECLIP_DIR, "text.onnx")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX (MobileCLIP2-S0)",
        "image_size": IMAGE_SIZE,
        "thresholds": {"min_conf": MIN_CONF, "hat_margin": HAT_MARGIN},
        "groups": {
            g: {"labels": labels,
                "embeddings": [[round(float(v), 6) for v in row] for row in emb]}
            for g, (labels, emb) in bank.items()
        },
        "prompts": GROUPS,
    }
    bank_path = OUT_DIR / "clip_bank.json"
    bank_path.write_text(json.dumps(out))
    print(f"wrote {bank_path} ({bank_path.stat().st_size // 1024} KB)")

    # Expected web tags for the sample PNGs -> reference_outputs.json, so the
    # browser selftest can diff its own reads against this exact pipeline.
    ref_path = ROOT / "webapp/scripts/reference_outputs.json"
    ref = json.load(open(ref_path))

    vis = ort.InferenceSession(str(VISION_ONNX), providers=["CPUExecutionProvider"])
    vname = vis.get_inputs()[0].name
    vhalf = vis.get_inputs()[0].type == "tensor(float16)"

    def embed_image(im: Image.Image, box) -> np.ndarray:
        if box is not None:
            im = _torso_crop(im, box)
        w, h = im.size
        s = IMAGE_SIZE / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        nw, nh = im.size
        left, top = (nw - IMAGE_SIZE) // 2, (nh - IMAGE_SIZE) // 2
        im = im.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        arr = np.asarray(im, np.float32) / 255.0
        feed = np.transpose(arr, (2, 0, 1))[None]
        emb = vis.run(None, {vname: feed.astype(np.float16) if vhalf else feed})[0][0]
        emb = emb.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    expected = {}
    for name, face in ref["faces"].items():
        if face["box"] is None:
            expected[name] = None
            continue
        ie = embed_image(Image.open(ROOT / name).convert("RGB"), face["box"])
        expected[name] = web_read(bank, ie)
        print(f"{name}: {expected[name]}")
    ref["web_expected"] = expected
    ref_path.write_text(json.dumps(ref, indent=2))
    print(f"updated {ref_path}")


if __name__ == "__main__":
    main()
