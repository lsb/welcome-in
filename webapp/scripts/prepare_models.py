"""Stage the web app's models into webapp/public/models/ (~24 MB total).

  * face_detector.onnx (0.6 MB) — the desktop's Qualcomm MediaPipe BlazeFace
    export with its external-data sidecar embedded into a single file
    (onnxruntime-web wants one fetchable file);
  * mobileclip2_s0_visual_fp16.onnx (23.9 MB) — MobileCLIP2-S0's image tower,
    fp16-converted from RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX's fp32 export
    (lossless on the sample tags: cosine 0.99999, identical labels/margins),
    then e5m6-squashed (squash_fp16.py: mantissa rounded to 6 bits in the
    fp16 container) so gzip halves the low bytes — 21.1 -> 16.3 MB on the
    wire at pigz -11, still tag-identical (cos >= 0.991 vs unsquashed);
  * clip_bank.json (150 KB) — the pre-embedded text phrase bank, built by
    make_clip_bank.py (run separately; it needs only the plain venv).

Needs onnx + onnxconverter-common (not project deps), so run it as:

  uv run --with onnx,onnxconverter-common python webapp/scripts/prepare_models.py

The desktop face model must already be present (main.py --setup downloads it);
the MobileCLIP2 export is fetched into models/hf on first run.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from squash_fp16 import squash_fp16_bits  # noqa: E402

OUT_DIR = ROOT / "webapp/public/models"
FACE_SRC = ROOT / "models/mediapipe_face-onnx-float/face_detector.onnx"
MC2_REPO = "RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX"
SQUASH_MANTISSA_BITS = 6  # e5m6: the measured accuracy-free wire-size win


def stage_face() -> None:
    import onnx

    if not FACE_SRC.exists():
        sys.exit(f"missing {FACE_SRC} — run `uv run python main.py --setup` first")
    m = onnx.load(str(FACE_SRC))  # loads the .data sidecar
    out = OUT_DIR / "face_detector.onnx"
    onnx.save(m, str(out))  # saves with weights embedded
    print(f"face: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def stage_clip() -> None:
    import onnx
    from huggingface_hub import hf_hub_download
    from onnxconverter_common import float16

    hf_hub_download(repo_id=MC2_REPO, filename="visual.onnx",
                    cache_dir=str(ROOT / "models/hf"))
    data = hf_hub_download(repo_id=MC2_REPO, filename="visual.onnx.data",
                           cache_dir=str(ROOT / "models/hf"))
    # onnx refuses external-data symlinks (the HF cache layout), so load from
    # a dereferenced copy.
    with tempfile.TemporaryDirectory() as td:
        for src in (Path(data).parent / "visual.onnx", Path(data)):
            shutil.copy(src.resolve(), Path(td) / src.name)
        m = onnx.load(str(Path(td) / "visual.onnx"))
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    import numpy as np

    for init in m16.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16 and init.raw_data:
            u = np.frombuffer(init.raw_data, "<u2")
            init.raw_data = squash_fp16_bits(u, SQUASH_MANTISSA_BITS).astype("<u2").tobytes()
    out = OUT_DIR / "mobileclip2_s0_visual_fp16.onnx"
    onnx.save(m16, str(out))
    print(f"clip: {out.name} ({out.stat().st_size / 1e6:.1f} MB, "
          f"e5m{SQUASH_MANTISSA_BITS}-squashed fp16)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_face()
    stage_clip()
    print("now run: .venv/bin/python webapp/scripts/make_clip_bank.py")


if __name__ == "__main__":
    main()
