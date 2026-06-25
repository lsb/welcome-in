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

The acrostic constraint is the strongest lever on the greeting's tone, and it's
tunable from the environment (no code edits) — a short secret lets the host say
its piece and stop (crisp, plain), a long one forces it to pad past its content:

```bash
# terse, closer to plain speech: spell OPEN over four 32-48 char lines
WELCOME_ACROSTIC=OPEN WELCOME_MIN_LINE=32 WELCOME_MAX_LINE=48 \
  uv run python main.py 2.png

# turn the acrostic off entirely: raw, unconstrained tiny-LLM output
WELCOME_ACROSTIC=off uv run python main.py 2.png
```

| var | default | meaning |
|-----|---------|---------|
| `WELCOME_ACROSTIC` | `SLOPLEEB` | secret to spell; ASCII letters only, `\|` splits paragraphs (`HELLO\|FRIEND`). `off` (or `none`/`no`/`raw`) drops the grammar mask entirely for raw output |
| `WELCOME_MIN_LINE` | `10` | min characters per line |
| `WELCOME_MAX_LINE` | `100` | max characters per line |

The startup banner echoes the active acrostic and line bounds so you can confirm
the override took.

## Kiosk — live camera + printed card

`kiosk.py` runs the whole thing as a standing installation: a USB webcam feeds
`FaceGate.analyze` a few times a second on its **own capture/detect thread**, and
when a visitor **stops and holds the camera's gaze**, the greeting **streams onto
a full-screen display as the model writes it** (tens of seconds on the Pi — the
wait is the show), then the finished question card is **printed** for them to take.

Wake-word behaviour (all keyed on *facing the camera*, never on mere presence, so
a flowing exhibit where someone is always in frame still re-arms):

- **Fire on a near frontal face** — greets as soon as a near visitor faces the
  camera (`--dwell 0` default). Set `--dwell` above zero to require a *sustained*
  gaze (e.g. `1.2`) so glancing passers-by don't trip it.
- **Plural for groups** — every facing face is counted; two or more and the hello
  is addressed to the group ("hi everyone"), complimenting the **closest** person's
  outfit (the one we read with CLIP).
- **Abort on departure** — the detect thread keeps watching *during* the long
  generation (both llama.cpp and onnxruntime release the GIL), so if nobody faces
  the camera for `--abort-after` seconds (1.5 default) the half-written greeting is
  dropped, nothing prints, and the door re-arms.
- **Cooldown after a greeting** — once a greeting ends (finished *or* aborted) the
  door stays quiet for `--cooldown` seconds (3 default) before re-arming. The
  start/stop gate (near + frontal) is stricter than the "someone's here" presence
  cue (any detected face), so without this an abort would snap straight back to
  "someone's here" for a visitor merely angled away — and with `--dwell 0` it would
  instantly re-greet. The cooldown is the hysteresis that prevents the flap.
- **Proximity floor** — a face must be at least `--near-prox` box-height (0.10) to
  count as a near visitor; at a doorway this keeps far wall art and passers-by out
  of the trigger with no painting-specific logic.

A discreet near-black debug readout in the lower-right exposes every variable that
drives the gate — face counts, the closest face's pose/proximity, the dwell
countdown to fire and the gone countdown to abort/re-arm, stage timings, and the
active acrostic/model — invisible across the room, legible up close for tuning.

```bash
uv sync                                     # picks up imageio (USB capture)
uv run python kiosk.py                       # full-screen, default camera + printer
uv run python kiosk.py --windowed --cam 1    # windowed, second camera
uv run python kiosk.py --dwell 1.2 --cooldown 6       # require a held gaze, longer quiet gap
WELCOME_PRINTER=off uv run python kiosk.py   # run with no printer attached
```

| piece | how |
|-------|-----|
| **camera** | USB on both Mac and Pi, via `imageio`'s ffmpeg device reader — no `opencv-python` (no cp314 wheel). `camera.py` |
| **display** | a Tk full-screen skin that streams the text live; everything routes through a sink seam, so it can be reskinned (web / pygame) without touching the loop. Needs Tk: `apt install python3-tk` on the Pi |
| **printer** | the finished card prints once, complete, as a one-page **PostScript** document through CUPS (`lp`): the hello set in serif italic (wrapped at 80 cols), the question card in fixed-width Courier with each line's first letter **bold** to surface the acrostic, **landscape** so ~100-char lines don't wrap. `WELCOME_PRINTER=<queue>` picks a printer; `off` disables it; `WELCOME_PRINT_CPI` tunes the body density. `printer.py` |

The acrostic / tone env vars (above) apply to the kiosk too.

## Still not wired

- Speak greetings aloud (`say` on macOS, `espeak-ng`/`piper` on the Pi).
- Optional `face_landmark_detector.onnx` (468-pt mesh) for precise head-pose gating.
