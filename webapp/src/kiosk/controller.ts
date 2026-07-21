// The installation loop — port of kiosk.py onto browser timers. A webcam feeds
// FaceGate.analyze a few times a second; when a near visitor faces the camera,
// the templated hello appears at once and a pre-generated card is pulled from
// the pool and typed onto the screen a word at a time.
//
// Structure mirrors the desktop: a detect tick publishes Snapshots (with the
// dwell/gone timers), a control tick runs the state machine
// IDLE -> GREETING -> SHOWING -> COOLDOWN -> IDLE keyed on *facing* (never
// mere presence), and the UI receives the same message vocabulary the Tk skin
// consumed ("status" / "vars" / "state" / "begin" / "clothing" / "delta" /
// "settle" / "clear" / ...), so the React layer is just a renderer.
import type { FaceResult } from "../pipeline/faceGate";
import { FaceGate } from "../pipeline/faceGate";
import { ClipTagger } from "../pipeline/clipTagger";
import { CardPool } from "./pool";
import { FALLBACK_CARD, FALLBACK_TOPIC, HELLO_TEMPLATE_COUNT, renderHello } from "./greeter";
import poolData from "../generated/pool.json";

export interface Snapshot {
  t: number;
  res: FaceResult | null;
  present: number;
  facing: number;
  isFacing: boolean; // a near face is facing (drives dwell/abort/re-arm)
  proximity: number;
  facingFor: number; // seconds the near-facing streak has held (dwell)
  goneFor: number; // seconds since a near face last faced
  detMs: number;
}

export type KioskMessage =
  | { kind: "status"; text: string }
  | { kind: "error"; text: string }
  | { kind: "config"; acrostic: string; lines: string; model: string }
  | { kind: "vars"; snap: Snapshot }
  | { kind: "state"; state: string }
  | { kind: "timing"; clipMs: number; greetMs: number }
  | { kind: "begin" }
  | { kind: "clothing"; compliment: string | null }
  | { kind: "delta"; part: "hello" | "card"; piece: string }
  | { kind: "settle"; hello: string; topic: string; questions: string }
  | { kind: "clear" }
  | { kind: "idleCard"; reset: boolean } // fresh idle card starts typing
  | { kind: "idleFade" }; // idle card fades out (CSS transition)

export interface KioskOptions {
  detectInterval: number; // seconds between detections
  dwell: number; // must keep facing this long to fire
  minShow: number; // hold a card at least this long
  clearAfter: number; // re-arm once gone this long
  cooldown: number; // stay quiet this long after a greeting ends
  nearProx: number; // min face box height to count as near
  wordMs: number; // screen reveal cadence (per word)
  idleHoldS: number; // hold a finished idle card this long before fading
  idleFadeS: number; // idle fade-out duration (CSS transition time)
  idleEnabled: boolean;
  /** Camera source override (e.g. the ?demo=N canvas stream); defaults to
   * getUserMedia. */
  getStream?: () => Promise<MediaStream>;
}

export const KIOSK_DEFAULTS: KioskOptions = {
  detectInterval: 0.1, dwell: 0.0, minShow: 8.0, clearAfter: 4.0,
  cooldown: 3.0, nearProx: 0.10, wordMs: 30, idleHoldS: 5.0, idleFadeS: 0.85,
  idleEnabled: true,
};

const now = () => performance.now() / 1000;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class KioskController {
  readonly opts: KioskOptions;
  private video: HTMLVideoElement;
  private listeners = new Set<(m: KioskMessage) => void>();

  private gate: FaceGate | null = null;
  private tagger: ClipTagger | null = null;
  private pool = new CardPool(poolData.cards);
  private idlePool = new CardPool(poolData.idle);

  // Latest reading, published by the detect tick, read by everyone else.
  private snap: Snapshot = {
    t: 0, res: null, present: 0, facing: 0, isFacing: false,
    proximity: 0, facingFor: 0, goneFor: 1e9, detMs: 0,
  };

  // The frame the greeting reads: the live canvas is redrawn every detect
  // tick; on fire it is copied to the snap canvas so the outfit read matches
  // the gating moment even while detection keeps running.
  private frameCanvas = document.createElement("canvas");
  private snapCanvas = document.createElement("canvas");

  private stopped = false;
  private timers: number[] = [];
  private helloIndex = 0; // round-robin cursor over the hello templates
  private idleEpoch = 0; // bump to invalidate pending idle steps
  private idleRunning = false;

  constructor(video: HTMLVideoElement, opts: Partial<KioskOptions> = {}) {
    this.video = video;
    this.opts = { ...KIOSK_DEFAULTS, ...opts };
  }

  subscribe(fn: (m: KioskMessage) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private post(m: KioskMessage): void {
    for (const fn of this.listeners) fn(m);
  }

  async start(): Promise<void> {
    this.post({ kind: "status", text: "Preheating..." });
    try {
      [this.gate, this.tagger] = await Promise.all([FaceGate.create(), ClipTagger.create()]);
      const stream = await (this.opts.getStream?.() ??
        navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        }));
      this.video.srcObject = stream;
      await this.video.play();
    } catch (e) {
      this.post({ kind: "error", text: `could not start: ${e instanceof Error ? e.message : e}` });
      return;
    }
    this.post({
      kind: "config", acrostic: "SLOP+TIAT+LEEB",
      lines: `${this.pool.size} pooled`, model: "MobileCLIP2-S0",
    });
    this.clearToIdle();
    this.detectLoop();
    this.controlLoop();
  }

  stop(): void {
    this.stopped = true;
    this.idleEpoch += 1;
    for (const t of this.timers) clearTimeout(t);
    const stream = this.video.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
  }

  // -- capture + detect ------------------------------------------------
  private async detectLoop(): Promise<void> {
    let facingSince = 0;
    let lastFacing = 0;
    // A few frames of grace so brief detection flicker doesn't shatter the
    // dwell streak or trip the abort.
    const grace = Math.max(0.3, 3 * this.opts.detectInterval);
    while (!this.stopped) {
      const t0 = performance.now();
      const vw = this.video.videoWidth, vh = this.video.videoHeight;
      if (vw > 0 && this.gate) {
        if (this.frameCanvas.width !== vw) {
          this.frameCanvas.width = vw;
          this.frameCanvas.height = vh;
        }
        this.frameCanvas.getContext("2d")!.drawImage(this.video, 0, 0, vw, vh);
        let res: FaceResult;
        try {
          res = await this.gate.analyze(this.frameCanvas, vw, vh);
        } catch (e) {
          this.post({ kind: "error", text: `detect failed: ${e instanceof Error ? e.message : e}` });
          break;
        }
        const detMs = performance.now() - t0;
        const t = now();
        // "Engaged" = a face that is both facing AND near enough.
        const nearFacing = res.facingCamera && res.proximity >= this.opts.nearProx;
        if (nearFacing) {
          if (t - lastFacing > grace) facingSince = t; // a genuine gap -> fresh streak
          lastFacing = t;
        }
        const goneFor = lastFacing ? t - lastFacing : 1e9;
        const engaged = goneFor <= grace;
        this.snap = {
          t, res, present: res.presentCount, facing: res.facingCount,
          isFacing: engaged, proximity: res.proximity,
          facingFor: engaged ? t - facingSince : 0, goneFor, detMs,
        };
        this.post({ kind: "vars", snap: this.snap });
      }
      const elapsed = performance.now() - t0;
      await sleep(Math.max(0, this.opts.detectInterval * 1000 - elapsed));
    }
  }

  // -- state machine ---------------------------------------------------
  private async controlLoop(): Promise<void> {
    let state = "IDLE";
    let showSince = 0, coolSince = 0;
    let lastStatus: string | null = null;
    this.post({ kind: "state", state });
    while (!this.stopped) {
      const snap = this.snap;
      const t = now();
      if (state === "IDLE") {
        const status = snap.present ? "present" : "idle";
        if (status !== lastStatus) {
          if (snap.present) this.post({ kind: "status", text: "Noticing..." });
          else this.clearToIdle();
          lastStatus = status;
        }
        if (snap.isFacing && snap.facingFor >= this.opts.dwell) {
          state = "GREETING";
          this.post({ kind: "state", state });
          if (await this.greet(snap)) {
            showSince = now();
            state = "SHOWING";
          } else {
            this.clearToIdle(); // visitor left during the show
            coolSince = now();
            state = "COOLDOWN";
          }
          this.post({ kind: "state", state });
        }
      } else if (state === "SHOWING") {
        // Finish-then-re-arm: hold at least minShow, then clear once the
        // greeted audience moved on OR a fresh audience is already waiting.
        if (t - showSince > this.opts.minShow &&
            (snap.isFacing || snap.goneFor > this.opts.clearAfter)) {
          this.clearToIdle();
          coolSince = now();
          state = "COOLDOWN";
          this.post({ kind: "state", state });
        }
      } else if (state === "COOLDOWN") {
        if (t - coolSince >= this.opts.cooldown) {
          state = "IDLE";
          lastStatus = null;
          this.post({ kind: "state", state });
        }
      }
      await sleep(this.opts.detectInterval * 1000);
    }
  }

  /** Return to the standing WELCOME IN screen with the ambient rotation. */
  private clearToIdle(): void {
    this.post({ kind: "clear" });
    this.idleStart();
  }

  // -- one greeting ----------------------------------------------------
  /** Serve one (group of) visitor(s). The whole show always plays out; returns
   * whether the visitor was still facing at the end (the desktop's "printed"). */
  private async greet(snap: Snapshot): Promise<boolean> {
    this.idleStop(); // the greeting takes over the card area
    this.post({ kind: "begin" });
    // Freeze the gating frame for the outfit read.
    this.snapCanvas.width = this.frameCanvas.width;
    this.snapCanvas.height = this.frameCanvas.height;
    this.snapCanvas.getContext("2d")!.drawImage(this.frameCanvas, 0, 0);

    const t0 = performance.now();
    let compliment: string | null = null;
    try {
      compliment = await this.tagger!.compliment(
        this.snapCanvas, this.snapCanvas.width, this.snapCanvas.height,
        snap.res?.box ?? null);
    } catch {
      compliment = null;
    }
    const clipMs = performance.now() - t0;
    this.post({ kind: "clothing", compliment });

    const groupSize = Math.max(1, snap.res?.facingCount ?? 1);
    const hello = renderHello(compliment, groupSize, this.helloIndex);
    this.helloIndex = (this.helloIndex + 1) % HELLO_TEMPLATE_COUNT;
    const card = this.pool.takeLeastViewed();
    const topic = card?.topic ?? FALLBACK_TOPIC;
    const questions = card?.text ?? FALLBACK_CARD;

    this.post({ kind: "delta", part: "hello", piece: hello }); // hello shows at once
    const tShow = performance.now();
    await this.typeCard(questions); // full word-by-word reveal
    this.post({ kind: "timing", clipMs, greetMs: performance.now() - tShow });

    if (!this.snap.isFacing) return false; // they walked off during the show
    this.post({ kind: "settle", hello, topic, questions });
    return true;
  }

  /** Reveal a pooled card one word every wordMs. The split keeps each word's
   * trailing whitespace so blank-line stanza breaks survive and the acrostic
   * stays column-aligned as it appears. */
  private async typeCard(text: string): Promise<void> {
    for (const piece of text.match(/\S+\s*/g) ?? []) {
      if (this.stopped) return;
      this.post({ kind: "delta", part: "card", piece });
      await sleep(this.opts.wordMs);
    }
  }

  // -- ambient idle rotation -------------------------------------------
  /** Begin the rotating idle cards under the WELCOME headline. Idempotent. */
  idleStart(): void {
    if (!this.opts.idleEnabled || this.idleRunning) return;
    this.idleRunning = true;
    this.idleEpoch += 1;
    void this.idleShowNext(this.idleEpoch);
  }

  /** Stop the rotation; bumping the epoch invalidates pending steps. */
  idleStop(): void {
    this.idleRunning = false;
    this.idleEpoch += 1;
  }

  private async idleShowNext(epoch: number): Promise<void> {
    if (epoch !== this.idleEpoch || this.stopped) return;
    const card = this.idlePool.takeLeastViewed();
    if (card === null) {
      this.post({ kind: "idleCard", reset: true }); // leave the area clean
      this.idleRunning = false; // static WELCOME IN simply stays up
      return;
    }
    this.post({ kind: "idleCard", reset: true });
    for (const piece of card.text.match(/\S+\s*/g) ?? []) {
      if (epoch !== this.idleEpoch || this.stopped) return;
      this.post({ kind: "delta", part: "card", piece });
      await sleep(this.opts.wordMs);
    }
    await sleep(this.opts.idleHoldS * 1000);
    if (epoch !== this.idleEpoch || this.stopped) return;
    this.post({ kind: "idleFade" });
    await sleep(this.opts.idleFadeS * 1000);
    void this.idleShowNext(epoch);
  }
}
