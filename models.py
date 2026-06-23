"""Model resolution / download for the welcome-in pipeline.

Two model families, both cached under ./models so a Raspberry Pi deploy can be
warmed once and then run offline:

  * Face detector  - Qualcomm's MediaPipe BlazeFace ONNX export (face_detector.onnx
                     plus its external-data sidecar face_detector.data).
  * Greeter        - Qwen3.5 GGUF (0.8B / 2B) from unsloth, fetched via huggingface_hub.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
HF_CACHE = MODELS_DIR / "hf"

# Qualcomm release manifest -> ONNX zip download URL.
FACE_RELEASE_JSON = (
    "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/raw/main/release_assets.json"
)
# Pinned fallback (v0.56.0) in case the manifest layout changes.
FACE_ONNX_ZIP_FALLBACK = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/"
    "models/mediapipe_face/releases/v0.56.0/mediapipe_face-onnx-float.zip"
)
FACE_DIR = MODELS_DIR / "mediapipe_face-onnx-float"
FACE_DETECTOR_ONNX = FACE_DIR / "face_detector.onnx"

# Qwen3.5 GGUF (Q4_K_M) — good speed/quality balance for CPU.
GGUF_REPOS = {
    "0.8B": ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    "2B": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
}


def _face_zip_url() -> str:
    try:
        with urllib.request.urlopen(FACE_RELEASE_JSON, timeout=20) as r:
            manifest = json.load(r)
        return manifest["precisions"]["float"]["universal_assets"]["onnx"]["download_url"]
    except Exception:
        return FACE_ONNX_ZIP_FALLBACK


def ensure_face_model() -> Path:
    """Download + extract the BlazeFace ONNX export if absent; return the .onnx path."""
    if FACE_DETECTOR_ONNX.exists():
        return FACE_DETECTOR_ONNX
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = _face_zip_url()
    print(f"[models] downloading face detector ONNX … ({url.rsplit('/', 1)[-1]})")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(MODELS_DIR)  # contains mediapipe_face-onnx-float/...
    if not FACE_DETECTOR_ONNX.exists():
        raise FileNotFoundError(f"expected {FACE_DETECTOR_ONNX} after extraction")
    print(f"[models] face detector ready: {FACE_DETECTOR_ONNX.relative_to(ROOT)}")
    return FACE_DETECTOR_ONNX


def ensure_gguf(size: str) -> Path:
    """Download (cached) the selected Qwen3.5 GGUF; return its local path."""
    from huggingface_hub import hf_hub_download

    if size not in GGUF_REPOS:
        raise ValueError(f"unknown model size {size!r}; choose from {list(GGUF_REPOS)}")
    repo_id, filename = GGUF_REPOS[size]
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[models] resolving {repo_id}/{filename} …")
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(HF_CACHE))
    return Path(path)


def download_all() -> None:
    ensure_face_model()
    for size in GGUF_REPOS:
        ensure_gguf(size)
    print("[models] all models present.")
