"""Parity harness: desktop pipeline (reference) vs the web MobileCLIP pipeline.

Runs on the sample PNGs in the repo root:

  * Face gate — the same BlazeFace ONNX ships to the browser, so the desktop
    FaceGate output *is* the web reference (present/facing counts, primary box).
  * Clothing tags — the desktop reads fashionCLIP; the web app ships a
    MobileCLIP-S0 image encoder with the text bank pre-embedded offline. This
    harness runs both over the identical torso crops and reports agreement per
    attribute group for each candidate quantization (fp32 / fp16 / int8), plus
    the raw similarity ranges needed to calibrate MIN_CONF / HAT_MARGIN on the
    MobileCLIP similarity scale.

Writes webapp/scripts/reference_outputs.json for the browser selftest to
compare against.

Run from the repo root:  .venv/bin/python webapp/scripts/parity_harness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clip_tags import _GROUPS, _torso_crop, ClipTagger  # noqa: E402
from face_gate import FaceGate  # noqa: E402

MOBILECLIP_DIR = next(
    (ROOT / "models/hf/models--Xenova--mobileclip_s0/snapshots").glob("*"))

SAMPLES = sorted(ROOT.glob("[0-9].png"))


class MobileClipTagger:
    """The web pipeline, bit-for-bit: MobileCLIP-S0 preprocessing is resize
    shortest-edge 256 (bilinear), center-crop 256, scale to [0,1] — no mean/std
    normalisation (do_normalize: false in the export's preprocessor config)."""

    def __init__(self, vision_onnx: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._vis = ort.InferenceSession(str(vision_onnx),
                                         providers=["CPUExecutionProvider"])
        txt = ort.InferenceSession(str(MOBILECLIP_DIR / "onnx/text_model.onnx"),
                                   providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(str(MOBILECLIP_DIR / "tokenizer.json"))
        tok.enable_padding(length=77, pad_id=0)
        tok.enable_truncation(max_length=77)
        self._txt_inputs = [i.name for i in txt.get_inputs()]

        self.groups = {}
        for name, mapping in _GROUPS.items():
            labels = list(mapping)
            enc = tok.encode_batch(list(mapping.values()))
            feeds = {"input_ids": np.array([e.ids for e in enc], np.int64)}
            if "attention_mask" in self._txt_inputs:
                feeds["attention_mask"] = np.array(
                    [e.attention_mask for e in enc], np.int64)
            emb = txt.run(None, feeds)[0]
            self.groups[name] = (labels, _l2(emb))

    def embed_image(self, im: Image.Image, box) -> np.ndarray:
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
        name = self._vis.get_inputs()[0].name
        if self._vis.get_inputs()[0].type == "tensor(float16)":
            feed = feed.astype(np.float16)
        emb = self._vis.run(None, {name: feed})[0][0].astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def read(self, im: Image.Image, box):
        ie = self.embed_image(im, box)
        out = {}
        for g in ("color", "garment", "style"):
            labels, emb = self.groups[g]
            sims = emb @ ie
            i = int(sims.argmax())
            out[g] = (labels[i], float(sims[i]))
        labels, hat_emb = self.groups["hat"]
        hs = hat_emb @ ie
        out["hat_margin"] = float(hs[labels.index("hat")] - hs[labels.index("nohat")])
        return out


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main() -> None:
    gate = FaceGate()
    ref_tagger = ClipTagger()

    faces = {}
    for p in SAMPLES:
        r = gate.analyze(p)
        faces[p.name] = {
            "present": r.present_count, "facing": r.facing_count,
            "person_present": r.person_present, "facing_camera": r.facing_camera,
            "proximity": round(r.proximity, 4),
            "box": [round(v, 4) for v in r.box] if r.box else None,
            "metrics": r.metrics,
        }
        print(f"{p.name}: present={r.present_count} facing={r.facing_count} "
              f"prox={r.proximity:.3f} metrics={r.metrics}")

    print("\n--- reference tags (fashionCLIP, the desktop read) ---")
    ref = {}
    for p in SAMPLES:
        box = faces[p.name]["box"]
        im = Image.open(p).convert("RGB")
        read = ref_tagger._read(im, box)
        ie = ref_tagger._embed_image(im, box)
        gl, ge = ref_tagger._groups["garment"]
        gconf = float((ge @ ie).max())
        if read is None:
            ref[p.name] = {"read": None, "gconf": round(gconf, 4)}
            print(f"{p.name}: no confident read (gconf={gconf:.3f})")
        else:
            color, garment, style, hat = read
            ref[p.name] = {"color": color, "garment": garment, "style": style,
                           "hat": bool(hat), "gconf": round(gconf, 4)}
            print(f"{p.name}: {color} {garment} / {style} / hat={hat} (gconf={gconf:.3f})")

    results = {"faces": faces, "fashionclip": ref, "mobileclip": {}}
    for variant in ("vision_model.onnx", "vision_model_fp16.onnx", "vision_model_int8.onnx"):
        print(f"\n--- MobileCLIP {variant} ---")
        tagger = MobileClipTagger(MOBILECLIP_DIR / "onnx" / variant)
        out = {}
        for p in SAMPLES:
            box = faces[p.name]["box"]
            im = Image.open(p).convert("RGB")
            r = tagger.read(im, box)
            out[p.name] = {k: (v if not isinstance(v, tuple) else
                               {"label": v[0], "conf": round(v[1], 4)})
                           for k, v in r.items()}
            rr = ref[p.name]
            agree = "".join(
                "=" if rr.get(g) == r[g][0] else "X"
                for g in ("color", "garment", "style")) if "color" in rr else "?"
            print(f"{p.name}: color={r['color'][0]}({r['color'][1]:.3f}) "
                  f"garment={r['garment'][0]}({r['garment'][1]:.3f}) "
                  f"style={r['style'][0]}({r['style'][1]:.3f}) "
                  f"hat_margin={r['hat_margin']:+.4f}  vs-ref[{agree}]")
        results["mobileclip"][variant] = out

    out_path = ROOT / "webapp/scripts/reference_outputs.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
