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
from dataclasses import dataclass
from pathlib import Path

from acrostic import GenerationAborted
from camera import Camera
from printer import Printer

_BG = "#0a0a0a"

# What the screen says when nobody's been gated yet — the inviting default that
# greets the empty gallery instead of a small grey "waiting". The headline rides
# the big hello label; the line under it rides the italic subtitle.
_IDLE_HELLO = "WELCOME IN"
_IDLE_SUBTITLE = "I've been looking for you!"


@dataclass(frozen=True)
class Snapshot:
    """One reading from the capture/detect thread, published as a single atomic
    reference (Python assignment is atomic under the GIL) that the control loop and
    the abort hook read without locking.

    ``is_facing`` is whether at least one *near* face (box height >= near_prox) is
    turned to the camera; ``facing_for`` is how long that has held continuously
    (the dwell gate that separates a stopper from a passer-by); ``gone_for`` is how
    long since a near face last faced — it drives both the mid-greeting abort and
    the SHOWING -> IDLE re-arm. Both timers are kept by the detect thread, which is
    the only thing sampling the camera continuously."""
    t: float = 0.0
    res: object = None           # the FaceResult (handed to the greet that follows)
    frame: object = None         # the RGB frame (for the closest-person outfit read)
    present: int = 0             # faces detected over threshold
    facing: int = 0              # of those, how many face the camera
    is_facing: bool = False      # a near face is facing (drives dwell/abort/re-arm)
    proximity: float = 0.0       # closest facing face's box height (0..1)
    facing_for: float = 0.0      # seconds the near-facing streak has held (dwell)
    gone_for: float = 1e9        # seconds since a near face last faced
    det_ms: float = 0.0


class Kiosk:
    def __init__(self, model: str = "0.8B", cam_index: int = 0, fullscreen: bool = True,
                 detect_interval: float = 0.1, dwell: float = 0.0, min_show: float = 8.0,
                 abort_after: float = 1.5, clear_after: float = 4.0, cooldown: float = 3.0,
                 near_prox: float = 0.10, warmup: bool = True,
                 warmup_image: str = "6.png"):
        self.model = model
        self.cam_index = cam_index
        self.fullscreen = fullscreen
        self.detect_interval = detect_interval   # seconds between detections
        self.dwell = dwell                       # must keep facing this long to fire
        self.min_show = min_show                 # hold a card at least this long
        self.abort_after = abort_after           # stop a greeting once gone this long
        self.clear_after = clear_after           # re-arm once gone this long
        self.cooldown = cooldown                 # stay quiet this long after a greeting ends
        self.near_prox = near_prox               # min face box height to count as near
        self.warmup = warmup                     # full e2e pass on startup
        self.warmup_image = warmup_image

        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._painted = threading.Event()   # UI-thread handshake (see _flush_ui)
        self._closing = False
        self._thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        # The capture/detect thread publishes the latest reading here; the control
        # loop and the abort hook read it lock-free (atomic reference assignment).
        self._snap = Snapshot()

        # Heavy components are built on the worker thread (see _run), so the UI
        # stays responsive while models load.
        self.gate = self.tagger = self.greeter = self.camera = None
        self.printer = Printer.from_env()

        # Live accumulators for the two streamed parts (touched on the UI thread).
        self._hello = ""
        self._card = ""
        # Latest values behind the discreet debug readout (UI thread only).
        self._dbg: dict = {}

    # -- lifecycle -------------------------------------------------------
    def run(self) -> None:
        self._build_ui()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.root.after(33, self._drain)
        self.root.mainloop()

    def _post(self, msg: tuple) -> None:
        self._q.put(msg)

    def _flush_ui(self, timeout: float = 1.0) -> None:
        """Block the worker until the UI thread has painted everything queued so
        far. Called right before the long LLM generation so the clothing line is
        on screen first — and since the worker waits here while the cores are
        still idle, that paint isn't starved by the generation that follows."""
        self._painted.clear()
        self._post(("flush",))
        self._painted.wait(timeout)

    # -- worker: load, then run capture/detect + control loops -----------
    def _run(self) -> None:
        try:
            self._post(("status", "warming up the gallery..."))
            from clip_tags import ClipTagger
            from face_gate import FaceGate
            from greeter import Greeter

            self.gate = FaceGate()
            self.tagger = ClipTagger()
            self.greeter = Greeter(size=self.model).load()
            self.camera = Camera(index=self.cam_index)
            self._post(("config", " ".join(self.greeter.paragraphs) or "off",
                        f"{self.greeter.min_line}-{self.greeter.max_line}", self.model))
            if self.warmup:
                self._warmup()
            self._post(("clear",))   # drop into the standing WELCOME IN screen
        except Exception as e:  # surface startup failures on screen, don't crash silently
            self._post(("error", f"could not start: {e}"))
            return
        # Capture/detect runs continuously on its own thread so the camera keeps
        # watching *during* the long generation — that is what lets a greeting
        # abort the instant its audience leaves. The control loop below only reads
        # the snapshots it publishes; it never touches the camera itself.
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._detect_thread.start()
        self._control_loop()

    def _warmup(self) -> None:
        """Run the whole pipeline once on a sample image before going live, so the
        first real visitor doesn't pay first-call overhead — the lazy chat
        formatter, llama.cpp's decode warmup, and the CLIP / face-gate paths. The
        model is already loaded; this exercises the actual decode end to end, and
        seeds the admin timing readout with its numbers."""
        img = Path(self.warmup_image)
        if not img.is_absolute():
            img = Path(__file__).resolve().parent / img   # robust to a service cwd
        self._post(("status", f"warming up the gallery (running {img.name})..."))
        try:
            t_det = time.perf_counter()
            res = self.gate.analyze(str(img))
            det_ms = (time.perf_counter() - t_det) * 1000.0
            t_clip = time.perf_counter()
            clothing = self.tagger.describe(str(img), res.box)
            clip_ms = (time.perf_counter() - t_clip) * 1000.0
            t_greet = time.perf_counter()
            self.greeter.greet(res, clothing)
            greet_ms = (time.perf_counter() - t_greet) * 1000.0
            print(f"[kiosk] warmup {img.name}: detect {det_ms:.0f}ms  "
                  f"clip {clip_ms:.0f}ms  greet {greet_ms:.0f}ms")
            self._post(("timing", det_ms, clip_ms, greet_ms))
        except FileNotFoundError:
            print(f"[kiosk] warmup image {img} not found; skipping warmup")
        except Exception as e:
            print(f"[kiosk] warmup failed: {e}")

    def _detect_loop(self) -> None:
        """Capture + detect, forever, on its own thread. Publishes a Snapshot per
        frame (including the dwell/abort timers it alone can keep) and feeds the
        live debug readout. Runs *through* the greeting too — both llama.cpp and
        onnxruntime release the GIL during compute, so this thread keeps sampling
        in the gaps and the abort hook stays live."""
        facing_since = 0.0
        last_facing = 0.0
        # A few frames of grace so brief detection flicker (a blink, proximity
        # oscillating at the floor) doesn't shatter the dwell streak or trip the
        # abort — this is the "tolerate a few consecutive misses" hysteresis.
        grace = max(0.3, 3 * self.detect_interval)
        while not self._stop.is_set():
            try:
                frame = self.camera.frame()
            except Exception:
                if not self._stop.is_set():
                    self._post(("error", "camera read failed"))
                break
            t_det = time.perf_counter()
            res = self.gate.analyze(frame)
            det_ms = (time.perf_counter() - t_det) * 1000.0
            now = time.monotonic()
            # "Engaged" = a face that is both facing AND near enough (box height).
            # The proximity floor is the stopper/passer-by gate; at a doorway it
            # also keeps far wall art out of the trigger without any special-casing.
            near_facing = res.facing_camera and res.proximity >= self.near_prox
            if near_facing:
                if (now - last_facing) > grace:   # a genuine gap -> fresh streak
                    facing_since = now
                last_facing = now
            gone_for = (now - last_facing) if last_facing else 1e9
            engaged = gone_for <= grace            # facing now, or within the grace
            self._snap = Snapshot(
                t=now, res=res, frame=frame,
                present=res.present_count, facing=res.facing_count,
                is_facing=engaged, proximity=res.proximity,
                facing_for=(now - facing_since) if engaged else 0.0,
                gone_for=gone_for, det_ms=det_ms)
            self._post(("vars", self._snap))
            time.sleep(self.detect_interval)

    def _control_loop(self) -> None:
        """The state machine, driven entirely off published snapshots (it never
        reads the camera). IDLE -> (a near visitor faces for `dwell`) GREETING ->
        SHOWING -> COOLDOWN -> IDLE. Keyed on *facing*, never on mere presence: in a
        flowing exhibit someone is always in frame, so a presence latch would never
        re-arm.

        COOLDOWN is a refractory period after every greeting ends. It exists because
        the gate's start/stop signal (near + frontal) is stricter than the presence
        signal behind "someone's here" (any detected face): when a greeting aborts
        — meaning nobody *faced* for abort_after — a visitor merely angled away is
        still *present*, so without a cooldown the screen snaps straight back to
        "someone's here", and with dwell=0 it would instantly re-greet, flapping."""
        state, show_since, cool_since, last_status = "IDLE", 0.0, 0.0, None
        self._post(("state", state))
        while not self._stop.is_set():
            snap = self._snap
            now = time.monotonic()
            if state == "IDLE":
                # Nobody here → the standing WELCOME IN screen invites them in;
                # a person present (but not yet gated) gets the quiet "someone's
                # here" acknowledgement. Only post on the transition.
                status = "present" if snap.present else "idle"
                if status != last_status:
                    self._post(("status", "someone's here") if snap.present
                               else ("clear",))
                    last_status = status
                # Fire once a near face is frontal (and, if dwell>0, has held it).
                if snap.is_facing and snap.facing_for >= self.dwell:
                    state = "GREETING"
                    self._post(("state", state))
                    try:
                        self._greet(snap)            # blocks, streaming + abortable
                        show_since = time.monotonic()
                        state = "SHOWING"
                    except GenerationAborted:
                        self._post(("aborted",))     # audience left mid-greeting
                        cool_since = time.monotonic()
                        state = "COOLDOWN"
                    self._post(("state", state))
            elif state == "SHOWING":
                # Finish-then-re-arm: hold the printed card at least min_show, then
                # clear once the greeted audience has moved on (gone_for) OR a fresh
                # audience is already waiting (is_facing), so the next group is served.
                if (now - show_since) > self.min_show and (
                        snap.is_facing or snap.gone_for > self.clear_after):
                    self._post(("clear",))
                    cool_since = time.monotonic()
                    state = "COOLDOWN"
                    self._post(("state", state))
            elif state == "COOLDOWN":
                # Stay quiet (no re-announce, no re-trigger) until the refractory
                # period elapses, then re-arm clean.
                if (now - cool_since) >= self.cooldown:
                    state = "IDLE"
                    last_status = None
                    self._post(("state", state))
            time.sleep(self.detect_interval)

    def _greet(self, snap: Snapshot) -> None:
        """Run the expensive path for one (group of) visitor(s): read the closest
        person's clothing, stream both greeting parts to the screen, then print the
        finished card. Aborts (raising GenerationAborted) if the audience leaves —
        `gone_for` keeps climbing on the detect thread while we generate here."""
        self._post(("begin",))
        res, frame = snap.res, snap.frame
        t_clip = time.perf_counter()
        try:
            clothing = self.tagger.describe(frame, res.box)
        except Exception:
            clothing = None
        clip_ms = (time.perf_counter() - t_clip) * 1000.0
        self._post(("clothing", clothing))
        self._flush_ui()   # paint the appearance line before the LLM generation starts

        t_greet = time.perf_counter()
        greeting = self.greeter.greet(
            res, clothing,
            on_delta=lambda part, d: self._post(("delta", part, d)),
            abort=lambda: self._stop.is_set() or self._snap.gone_for >= self.abort_after)
        greet_ms = (time.perf_counter() - t_greet) * 1000.0
        print(f"[kiosk] detect {snap.det_ms:.0f}ms  clip {clip_ms:.0f}ms  greet {greet_ms:.0f}ms")
        self._post(("timing", snap.det_ms, clip_ms, greet_ms))

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
            # "-fullscreen" is a *request* to the window manager. Set immediately
            # after Tk() — before the window is mapped — the WM can drop it, and
            # since this branch sets no geometry the window then shrinks to fit its
            # (nearly empty) labels: the "tiny window" failure. Make it
            # deterministic — size to the whole screen as a floor, then re-assert
            # fullscreen once the window is actually on screen.
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
            self.root.attributes("-fullscreen", True)
            self.root.after(200, lambda: self.root.attributes("-fullscreen", True))
        else:
            self.root.geometry("1100x800")
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        wrap = max(400, self.root.winfo_screenwidth() - 240)
        self.status_var = tk.StringVar(value="warming up the gallery...")
        self.hello_var = tk.StringVar(value="")
        self.topic_var = tk.StringVar(value="")
        self.card_var = tk.StringVar(value="")
        self.debug_var = tk.StringVar(value="")

        tk.Label(self.root, textvariable=self.status_var, bg=_BG, fg="#555",
                 font=("Helvetica", 16)).pack(pady=(48, 0))
        # The hello label carries two very different headlines: a personal greeting
        # while we engage someone, and the large standing "WELCOME IN" while we wait.
        # We hold onto it so the idle screen can swell the font and shrink it back.
        self._hello_font = ("Helvetica", 40)
        self._idle_font = ("Helvetica", 96, "bold")
        self.hello_label = tk.Label(self.root, textvariable=self.hello_var, bg=_BG,
                 fg="#f4f4f4", font=self._hello_font, wraplength=wrap,
                 justify="center")
        self.hello_label.pack(pady=(40, 0), padx=120)
        tk.Label(self.root, textvariable=self.topic_var, bg=_BG, fg="#7a7a7a",
                 font=("Helvetica", 18, "italic"), wraplength=wrap,
                 justify="center").pack(pady=(48, 0), padx=120)
        # The question card reads like the printed card: left-justified monospace,
        # so each acrostic line starts at the same column and the hidden first-letter
        # acrostic lines up vertically down the left edge.
        tk.Label(self.root, textvariable=self.card_var, bg=_BG, fg="#e6e6e6",
                 font=("Courier", 20), wraplength=wrap, justify="left",
                 anchor="w").pack(pady=(16, 0), padx=120, anchor="w", fill="x")

        # Discreet admin debug readout pinned to the lower-right in near-black grey:
        # invisible across the room, legible up close if you know to look. Every
        # variable that drives the gate lives here — counts, the closest face's
        # pose/proximity, the dwell and abort/clear countdowns, stage timings, and
        # the active acrostic/model — so the install can be tuned in place.
        tk.Label(self.root, textvariable=self.debug_var, bg=_BG, fg="#2a2a2a",
                 font=("Courier", 11), justify="left").place(
                     relx=1.0, rely=1.0, x=-10, y=-6, anchor="se")

        # Quiet standing byline along the bottom edge, centered and dim so it reads
        # as a signature rather than competing with the greeting above it.
        tk.Label(self.root, text="by Lee Butterman", bg=_BG, fg="#555",
                 font=("Helvetica", 14, "italic")).place(
                     relx=0.5, rely=1.0, y=-12, anchor="s")

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
        elif kind == "vars":
            snap = msg[1]
            res = snap.res
            self._dbg.update(
                present=snap.present, facing=snap.facing, proximity=snap.proximity,
                facing_for=snap.facing_for, gone_for=snap.gone_for, det_ms=snap.det_ms,
                desc=(res.description if res and res.person_present else "-"),
                conf=(res.facing_conf if res else 0.0),
                metrics=(res.metrics if res else None))
            self._render_debug()
        elif kind == "state":
            self._dbg["fsm"] = msg[1]
            self._render_debug()
        elif kind == "config":
            _, acrostic, lines, model = msg
            self._dbg.update(acrostic=acrostic, lines=lines, model=model)
            self._render_debug()
        elif kind == "timing":
            _, _det, clip, greet = msg   # det comes live from "vars"; keep the latest
            self._dbg.update(clip_ms=clip, greet_s=greet / 1000.0)
            self._render_debug()
        elif kind == "flush":
            self.root.update_idletasks()   # force pending label repaints to land now
            self._painted.set()
        elif kind == "begin":
            self._hello = self._card = ""
            self.hello_label.config(font=self._hello_font)
            self.hello_var.set("")
            self.topic_var.set("")
            self.card_var.set("")
            self.status_var.set("greeting you...")
        elif kind == "clothing":
            if msg[1]:
                self.status_var.set(f"noticing your {msg[1]}...")
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
            self.topic_var.set(f"- {topic} -")
            self.card_var.set(questions)
            self.status_var.set("your card is printing - take it as you come in"
                                if self.printer.enabled
                                else "linger on these as long as you like")
        elif kind == "clear":
            self._idle()
        elif kind == "aborted":
            # The audience walked off mid-greeting: drop the half-written card,
            # print nothing, and quietly re-arm for the next visitor.
            self._idle()

    def _idle(self) -> None:
        """Return the screen to its inviting standing state: a large WELCOME IN
        over its subtitle, with no half-written card left behind. This is what the
        empty gallery shows while we wait for someone to gate."""
        self._hello = self._card = ""
        self.hello_label.config(font=self._idle_font)
        self.hello_var.set(_IDLE_HELLO)
        self.topic_var.set(_IDLE_SUBTITLE)
        self.card_var.set("")
        self.status_var.set("")

    def _render_debug(self) -> None:
        """Paint the discreet lower-right readout from the latest variables. Every
        line maps to a lever in the loop: face counts, the primary face's
        pose/proximity, the dwell countdown to fire, the gone countdown to
        abort/re-arm, stage timings, and the active acrostic/model."""
        d = self._dbg
        m = d.get("metrics") or {}
        pose = (f"yaw {m.get('yaw', 0.0):+.2f}  ear {m.get('ear_ratio', 0.0):.2f}  "
                f"roll {m.get('roll', 0.0):+.0f}") if m else "-"
        gone = d.get("gone_for", 0.0)
        gone = 999.0 if gone > 999 else gone
        self.debug_var.set("\n".join([
            f"state {d.get('fsm', '?'):8s}  faces {d.get('facing', 0)}/{d.get('present', 0)} facing/present",
            f"near  {d.get('desc', '-')}   conf {d.get('conf', 0.0):.2f}   "
            f"prox {d.get('proximity', 0.0):.2f} (>{self.near_prox:.2f})",
            f"pose  {pose}",
            f"dwell {d.get('facing_for', 0.0):4.1f}s (>{self.dwell:.1f} fires)   "
            f"gone {gone:4.1f}s (>{self.abort_after:.1f} abort  >{self.clear_after:.1f} clear)",
            f"time  det {d.get('det_ms', 0.0):3.0f}ms  clip {d.get('clip_ms', 0.0):4.0f}ms  "
            f"greet {d.get('greet_s', 0.0):.1f}s",
            f"acr   {d.get('acrostic', '?')} {d.get('lines', '')}   model {d.get('model', '?')}",
        ]))

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
    p.add_argument("--dwell", type=float, default=0.0,
                   help="seconds a near visitor must keep facing before greeting (default 0)")
    p.add_argument("--abort-after", type=float, default=1.5,
                   help="abort a greeting once nobody has faced for this long (default 1.5)")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="stay quiet this long after a greeting ends, before re-arming (default 3)")
    p.add_argument("--near-prox", type=float, default=0.10,
                   help="min face box height (0..1) to count as a near visitor (default 0.10)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="skip the startup end-to-end warmup pass")
    p.add_argument("--warmup-image", default="6.png",
                   help="image for the startup warmup pass (default: 6.png)")
    args = p.parse_args(argv)
    Kiosk(model=args.model, cam_index=args.cam, fullscreen=not args.windowed,
          dwell=args.dwell, abort_after=args.abort_after, cooldown=args.cooldown,
          near_prox=args.near_prox, warmup=args.warmup, warmup_image=args.warmup_image).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
