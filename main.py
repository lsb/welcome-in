"""welcome-in — greet a visitor only when they are present and facing the camera.

Pipeline:
  1. BlazeFace (ONNX, CPU) detects a face + keypoints and tests frontal pose.
  2. If facing the camera, Qwen3.5 (GGUF, CPU) generates a warm greeting.

On startup we run the whole pipeline once to warm up (load models + JIT the
inference paths), then run it again "for real" with timing reported — so the
numbers reflect steady-state latency on the deployment hardware (the Pi).

Usage:
  uv run python main.py                  # warmup + real pass over 1.png 2.png 3.png
  uv run python main.py 2.png            # long, effusive acrostic welcome
  uv run python main.py --model 2B       # use the larger model
  uv run python main.py --debug img.png  # show detection metrics
  uv run python main.py --no-warmup      # skip the warmup pass
  uv run python main.py --setup          # prefetch all models, then exit

The acrostic style decodes a welcome whose lines secretly spell TIAT / LEEB
(60-80 chars each) under a single GBNF grammar mask — the grammar-masking idea
from github.com/lsb/sidechat. See acrostic.py.
"""

from __future__ import annotations

import argparse
import sys
import time

DEFAULT_IMAGES = ["1.png", "2.png", "3.png"]


def deliver(text: str) -> None:
    """Single seam for greeting output. Console for now; swap for TTS on the Pi."""
    lines = text.splitlines() or [""]
    print(f"  \U0001f44b  {lines[0]}")
    for line in lines[1:]:
        print(f"      {line}")


def pipeline_pass(gate, greeter, images, debug, announce, greet=True) -> float:
    """Run the gate (+ greeter) over every image; print per-stage timing.

    Returns the wall-clock seconds for the whole pass. When announce is False
    (warmup) the greeting text is generated but not printed. When greet is False
    only the detector runs — used to keep the acrostic warmup pass cheap (a full
    effusive greeting takes tens of seconds; the model is already warm from load).
    """
    t_total = time.perf_counter()
    for path in images:
        print(f"\n--- {path} ---")
        t0 = time.perf_counter()
        try:
            res = gate.analyze(path)
        except FileNotFoundError:
            print(f"  (no such image: {path})")
            continue
        det_ms = (time.perf_counter() - t0) * 1000.0

        if debug:
            print(f"  [debug] present={res.person_present} facing={res.facing_camera} "
                  f"score={res.score:.2f} conf={res.facing_conf:.2f} "
                  f"metrics={res.metrics} desc={res.description!r}")

        if res.person_present and res.facing_camera and greet:
            t1 = time.perf_counter()
            text = greeter.greet(res)
            greet_ms = (time.perf_counter() - t1) * 1000.0
            print(f"  detect {det_ms:5.0f} ms · greet {greet_ms:6.0f} ms")
            if announce:
                deliver(text)
        elif res.person_present and res.facing_camera:
            print(f"  detect {det_ms:5.0f} ms · (facing — greeting skipped this pass)")
        elif res.person_present:
            print(f"  detect {det_ms:5.0f} ms · (someone here, not facing — staying quiet)")
        else:
            print(f"  detect {det_ms:5.0f} ms · (no one here — staying quiet)")

    return time.perf_counter() - t_total


def run(images: list[str], model: str, debug: bool, warmup: bool) -> None:
    from face_gate import FaceGate
    from greeter import Greeter

    # --- load models (timed) ---
    print("=== startup ===")
    t = time.perf_counter()
    gate = FaceGate()
    load_face = time.perf_counter() - t
    print(f"  face detector loaded in {load_face:6.2f} s")

    t = time.perf_counter()
    greeter = Greeter(size=model).load()
    load_llm = time.perf_counter() - t
    print(f"  Qwen3.5-{model} loaded in {load_llm:6.2f} s")
    print(f"  acrostic: {' '.join(greeter.paragraphs)}")

    # --- warmup pass ---
    if warmup:
        print("\n=== warmup pass ===")
        # The acrostic greeting is expensive; the model is already warm from load,
        # so warm only the detector here and save the generation for the real pass.
        warm = pipeline_pass(gate, greeter, images, debug, announce=False, greet=False)
        print(f"\n[warmup] full pass in {warm:.2f} s")

    # --- real pass ---
    print("\n=== welcome in (real) ===")
    real = pipeline_pass(gate, greeter, images, debug, announce=True)
    print(f"\n[timing] full pass in {real:.2f} s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="welcome-in visual-wake-word greeter")
    p.add_argument("images", nargs="*", default=DEFAULT_IMAGES,
                   help="image paths (default: 1.png 2.png 3.png)")
    p.add_argument("--model", choices=["0.8B", "2B"], default="0.8B",
                   help="Qwen3.5 size (default: 0.8B)")
    p.add_argument("--debug", action="store_true", help="print detection metrics")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="skip the startup warmup pass")
    p.add_argument("--setup", action="store_true",
                   help="download all models (face ONNX + both GGUFs) and exit")
    args = p.parse_args(argv)

    if args.setup:
        from models import download_all

        download_all()
        return 0

    images = args.images if args.images else DEFAULT_IMAGES
    run(images, args.model, args.debug, args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
