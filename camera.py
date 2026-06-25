"""USB webcam capture, no OpenCV.

`opencv-python` has no cp314 wheel (see README), so we grab frames through
imageio's ffmpeg device reader, which speaks v4l2 on Linux (the Pi) and
avfoundation on macOS (dev) behind one portable ``"<videoN>"`` spec — the same
USB-webcam code path on both targets. Frames come back as HWC uint8 RGB ndarrays;
`to_rgb()` turns them into the PIL images the gate and tagger already understand.

imageio (and its ffmpeg binary) is imported lazily inside ``__init__`` so the
module imports fine on a machine that hasn't run ``uv sync`` yet.
"""

from __future__ import annotations

from PIL import Image

from imaging import to_rgb


class Camera:
    """A USB webcam as a blocking frame source. ``frame()`` returns the next
    frame; the reader paces itself at the camera's frame rate."""

    def __init__(self, index: int = 0, size: tuple[int, int] | None = None,
                 fps: int | None = None):
        import imageio.v2 as imageio

        kwargs: dict = {}
        if size is not None:
            kwargs["size"] = size
        if fps is not None:
            kwargs["fps"] = fps
        # "<video0>" is imageio's portable webcam spec (v4l2 / avfoundation / dshow).
        self._reader = imageio.get_reader(f"<video{index}>", **kwargs)

    def frame(self) -> Image.Image:
        """Grab the next frame as an RGB PIL image (blocks until one is ready)."""
        return to_rgb(self._reader.get_next_data())

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
