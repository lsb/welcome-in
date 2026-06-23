"""welcome-in — greet a visitor only when they are present and facing the camera.

Pipeline:
  1. BlazeFace (ONNX, CPU) detects a face + keypoints and tests frontal pose.
  2. If facing the camera, Qwen3.5 (GGUF, CPU) generates a warm greeting.

Usage:
  uv run python main.py                  # run over 1.png 2.png 3.png with Qwen3.5-0.8B
  uv run python main.py --model 2B       # use the larger model
  uv run python main.py --debug img.png  # show detection metrics
  uv run python main.py --setup          # prefetch all models, then exit
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_IMAGES = ["1.png", "2.png", "3.png"]


def deliver(text: str) -> None:
    """Single seam for greeting output. Console for now; swap for TTS on the Pi."""
    print(f"  \U0001f44b  {text}")


def run(images: list[str], model: str, debug: bool) -> None:
    from face_gate import FaceGate

    gate = FaceGate()
    greeter = None  # lazy: only built once someone faces the camera

    for path in images:
        print(f"\n=== {path} ===")
        try:
            res = gate.analyze(path)
        except FileNotFoundError:
            print(f"  (no such image: {path})")
            continue

        if debug:
            print(f"  present={res.person_present} facing={res.facing_camera} "
                  f"score={res.score:.2f} conf={res.facing_conf:.2f} "
                  f"metrics={res.metrics} desc={res.description!r}")

        if res.person_present and res.facing_camera:
            if greeter is None:
                from greeter import Greeter

                greeter = Greeter(size=model)
            deliver(greeter.greet(res))
        elif res.person_present:
            print("  (someone is here, but not facing the doorway — staying quiet)")
        else:
            print("  (no one here — staying quiet)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="welcome-in visual-wake-word greeter")
    p.add_argument("images", nargs="*", default=DEFAULT_IMAGES,
                   help="image paths (default: 1.png 2.png 3.png)")
    p.add_argument("--model", choices=["0.8B", "2B"], default="0.8B",
                   help="Qwen3.5 size (default: 0.8B)")
    p.add_argument("--debug", action="store_true", help="print detection metrics")
    p.add_argument("--setup", action="store_true",
                   help="download all models (face ONNX + both GGUFs) and exit")
    args = p.parse_args(argv)

    if args.setup:
        from models import download_all

        download_all()
        return 0

    images = args.images if args.images else DEFAULT_IMAGES
    run(images, args.model, args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
