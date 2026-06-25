"""The installation: a live doorway greeter with a printed card.

Wires the existing pipeline into a standing kiosk. A USB webcam (camera.py) feeds
``FaceGate.analyze`` a few times a second; when a visitor faces the camera, the
greeting **streams onto a full-screen display as the model writes it** (tens of
seconds on the Pi — the wait is the show, see greeter.py), and the finished
question card is **printed** for them to take (printer.py, CUPS landscape).

Structure — one generation stream, two sinks, behind a clean seam:

  capture+detect thread          Tk main thread
  ----------------------         --------------------------------
  camera.frame()                 drains a queue every ~33 ms and is the *only*
  gate.analyze()       --push-->  thing that touches widgets (Tk isn't thread-
  on fire: greeter.greet(on_delta)  safe). Streamed deltas append live; on
    -> screen deltas + final     "settle" the screen snaps to the finished text;
    -> printer.print_card()      the printer fires once, complete.

The screen is the live-text sink. Reskinning it (web / pygame) means swapping
``_handle`` / ``_build_ui`` — the capture/detect/greet loop and the sinks don't
change. State machine: IDLE -> (visitor faces) GREETING -> SHOWING -> (visitor
leaves) IDLE. A presence latch means one greeting per visitor: the card holds
until they've been gone `clear_after` seconds, then the door re-arms.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

from camera import Camera
from printer import Printer

_BG = "#0a0a0a"


class Kiosk:
    def __init__(self, model: str = "0.8B", cam_index: int = 0, fullscreen: bool = True,
                 detect_interval: float = 0.3, min_show: float = 8.0,
                 clear_after: float = 4.0):
        self.model = model
        self.cam_index = cam_index
        self.fullscreen = fullscreen
        self.detect_interval = detect_interval   # seconds between detections
        self.min_show = min_show                 # hold a card at least this long
        self.clear_after = clear_after           # clear once gone this long

        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._closing = False
        self._thread: threading.Thread | None = None

        # Heavy components are built on the worker thread (see _run), so the UI
        # stays responsive while models load.
        self.gate = self.tagger = self.greeter = self.camera = None
        self.printer = Printer.from_env()

        # Live accumulators for the two streamed parts (touched on the UI thread).
        self._hello = ""
        self._card = ""

    # -- lifecycle -------------------------------------------------------
    def run(self) -> None:
        self._build_ui()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.root.after(33, self._drain)
        self.root.mainloop()

    def _post(self, msg: tuple) -> None:
        self._q.put(msg)

    # -- worker: load, then capture/detect/greet loop --------------------
    def _run(self) -> None:
        try:
            self._post(("status", "warming up the gallery…"))
            from clip_tags import ClipTagger
            from face_gate import FaceGate
            from greeter import Greeter

            self.gate = FaceGate()
            self.tagger = ClipTagger()
            self.greeter = Greeter(size=self.model).load()
            self.camera = Camera(index=self.cam_index)
            self._post(("status", "waiting"))
        except Exception as e:  # surface startup failures on screen, don't crash silently
            self._post(("error", f"could not start: {e}"))
            return
        self._loop()

    def _loop(self) -> None:
        state = "IDLE"
        last_present = 0.0
        show_since = 0.0
        while not self._stop.is_set():
            try:
                frame = self.camera.frame()
            except Exception:
                if not self._stop.is_set():
                    self._post(("error", "camera read failed"))
                break
            res = self.gate.analyze(frame)
            now = time.monotonic()
            present = res.person_present
            facing = present and res.facing_camera
            if present:
                last_present = now

            if state == "IDLE":
                self._post(("status", "someone's here" if present else "waiting"))
                if facing:
                    state = "GREETING"
                    self._greet(frame, res)        # blocks here, streaming to the UI
                    show_since = time.monotonic()
                    state = "SHOWING"
            elif state == "SHOWING":
                if (now - show_since) > self.min_show and (now - last_present) > self.clear_after:
                    self._post(("clear",))
                    state = "IDLE"

            time.sleep(self.detect_interval)

    def _greet(self, frame, res) -> None:
        """Run the expensive path for one visitor: read clothing, stream both
        greeting parts to the screen, then print the finished card."""
        self._post(("begin",))
        try:
            clothing = self.tagger.describe(frame, res.box)
        except Exception:
            clothing = None
        self._post(("clothing", clothing))

        greeting = self.greeter.greet(
            res, clothing, on_delta=lambda part, d: self._post(("delta", part, d)))

        # Snap the screen to the finished, settled text (the streamed text was raw;
        # the printer and this 'settle' both use greeter's polished strings).
        self._post(("settle", greeting.hello, greeting.topic, greeting.questions))
        self.printer.print_card(greeting)

    # -- UI (main thread only) -------------------------------------------
    def _build_ui(self) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        # Tcl picks its "system encoding" from the locale at Tk startup; on some
        # bundled-Tk builds (uv's Python on the Pi) it lands on iso8859-1 even
        # under a UTF-8 locale, so _tkinter's UTF-8 bytes get decoded as Latin-1
        # and non-ASCII labels mojibake ("…" -> "â€¦"). Pin it to UTF-8 so the
        # screen renders em-dashes/ellipses correctly.
        try:
            self.root.tk.call("encoding", "system", "utf-8")
        except tk.TclError:
            pass
        self.root.title("welcome-in")
        self.root.configure(bg=_BG)
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry("1100x800")
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        wrap = max(400, self.root.winfo_screenwidth() - 240)
        self.status_var = tk.StringVar(value="warming up the gallery…")
        self.hello_var = tk.StringVar(value="")
        self.topic_var = tk.StringVar(value="")
        self.card_var = tk.StringVar(value="")

        tk.Label(self.root, textvariable=self.status_var, bg=_BG, fg="#555",
                 font=("Helvetica", 16)).pack(pady=(48, 0))
        tk.Label(self.root, textvariable=self.hello_var, bg=_BG, fg="#f4f4f4",
                 font=("Helvetica", 40), wraplength=wrap, justify="center").pack(
                     pady=(40, 0), padx=120)
        tk.Label(self.root, textvariable=self.topic_var, bg=_BG, fg="#7a7a7a",
                 font=("Helvetica", 18, "italic"), wraplength=wrap,
                 justify="center").pack(pady=(48, 0), padx=120)
        tk.Label(self.root, textvariable=self.card_var, bg=_BG, fg="#e6e6e6",
                 font=("Helvetica", 22), wraplength=wrap, justify="center").pack(
                     pady=(16, 0), padx=120)

    def _drain(self) -> None:
        try:
            while True:
                self._handle(self._q.get_nowait())
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(33, self._drain)

    def _handle(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "status":
            self.status_var.set(msg[1])
        elif kind == "error":
            self.status_var.set(msg[1])
        elif kind == "begin":
            self._hello = self._card = ""
            self.hello_var.set("")
            self.topic_var.set("")
            self.card_var.set("")
            self.status_var.set("greeting you…")
        elif kind == "clothing":
            if msg[1]:
                self.status_var.set(f"noticing your {msg[1]}…")
        elif kind == "delta":
            if msg[1] == "hello":
                self._hello += msg[2]
                self.hello_var.set(self._hello)
            else:
                self._card += msg[2]
                self.card_var.set(self._card)
        elif kind == "settle":
            _, hello, topic, questions = msg
            self.hello_var.set(hello)
            self.topic_var.set(f"— {topic} —")
            self.card_var.set(questions)
            self.status_var.set("your card is printing — take it as you come in"
                                if self.printer.enabled
                                else "linger on these as long as you like")
        elif kind == "clear":
            self._hello = self._card = ""
            self.hello_var.set("")
            self.topic_var.set("")
            self.card_var.set("")
            self.status_var.set("waiting")

    def _on_close(self) -> None:
        self._closing = True
        self._stop.set()
        if self.camera is not None:        # unblock a pending camera.frame()
            self.camera.close()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="welcome-in live kiosk")
    p.add_argument("--model", choices=["0.8B", "2B"], default="0.8B",
                   help="Qwen3.5 size (default: 0.8B)")
    p.add_argument("--cam", type=int, default=0, help="USB camera index (default: 0)")
    p.add_argument("--windowed", action="store_true",
                   help="run in a window instead of full-screen")
    args = p.parse_args(argv)
    Kiosk(model=args.model, cam_index=args.cam, fullscreen=not args.windowed).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
