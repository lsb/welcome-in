# welcome-in

An art installation: a gallery doorway that **greets a visitor only when a person is
present *and* facing the camera**. Two FOSS stages, **CPU-only**, made to develop on a
Mac and deploy on a Raspberry Pi (aarch64), on Python 3.14.

1. **Visual wake word** — Qualcomm's MediaPipe **BlazeFace** face detector
   (the ONNX export from [`qualcomm/MediaPipe-Face-Detection`](https://huggingface.co/qualcomm/MediaPipe-Face-Detection)),
   run via **ONNX Runtime**. The 6 face keypoints (eyes, nose, mouth, ear tragions) are
   used to decide *facing the camera*.
2. **Greeting** — **Qwen3.5** (`0.8B` / `2B`, GGUF from
   [`unsloth`](https://huggingface.co/unsloth)) via **llama-cpp-python**, generating a
   short warm greeting only when stage 1 fires.

The three sample images map to the three cases:

| image    | situation                          | result      |
|----------|------------------------------------|-------------|
| `1.png`  | person present, **facing away**    | stay quiet  |
| `2.png`  | person **facing the camera**       | **greet**   |
| `3.png`  | **no person**                      | stay quiet  |

## Why ONNX Runtime (not the `mediapipe` package)?

`mediapipe` publishes no linux/aarch64 wheel, so it cannot `pip install` on a 64-bit
Raspberry Pi. We run the *same* BlazeFace model through `onnxruntime`, which has clean
`cp314` + `aarch64` + macOS-arm64 wheels. `opencv-python` also lacks cp314 wheels, so
image I/O uses **Pillow**.

## Setup

```bash
# CPU-only llama build (matches the Pi; skips Metal on macOS)
CMAKE_ARGS="-DGGML_METAL=OFF" uv sync

# Download models once: face ONNX (~2.5 MB) + both Qwen3.5 GGUFs
uv run python main.py --setup
```

`llama-cpp-python` builds from source (it ships only an sdist); you need a C/C++
toolchain + cmake (Xcode CLT on macOS, `build-essential cmake` on the Pi).

## Run

```bash
uv run python main.py                 # 1.png 2.png 3.png with Qwen3.5-0.8B
uv run python main.py --model 2B      # larger model
uv run python main.py --debug         # show detection scores / pose metrics
uv run python main.py path/to/img.png # any image(s)
```

## Layout

| file          | role |
|---------------|------|
| `main.py`     | CLI + pipeline glue; `deliver()` is the output seam (swap for TTS later) |
| `face_gate.py`| ONNX detect → decode → NMS → facing-camera decision (`FaceResult`) |
| `anchors.py`  | BlazeFace back-256 SSD anchors (896) |
| `greeter.py`  | Qwen3.5 greeting generation (lazy-loaded) |
| `models.py`   | model download/caching into `models/` (gitignored) |

## Tuning

Facing-the-camera thresholds live at the top of `face_gate.py`
(`DET_THRESH`, `YAW_MAX`, `EAR_LO/HI`, `ROLL_MAX_DEG`). Use `--debug` to see the
per-image `yaw / ear_ratio / roll` metrics and adjust.

## Next steps (not yet wired)

- Live camera frames (picamera2 / a USB cam) feeding `FaceGate.analyze` in a loop.
- Speak greetings aloud via `deliver()` (`say` on macOS, `espeak-ng`/`piper` on the Pi).
- Optional `face_landmark_detector.onnx` (468-pt mesh) for precise head-pose gating.
