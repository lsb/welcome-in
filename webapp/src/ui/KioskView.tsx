// The screen — port of the Tk skin in kiosk.py (_build_ui / _handle / _idle):
// a near-black full-screen page with the status line, the hello (swelling to
// the WELCOME IN headline while idle), the topic, the typed card, a discreet
// near-black debug readout in the lower-right, and the byline.
import { useEffect, useMemo, useReducer, useRef } from "react";
import type { KioskMessage, KioskOptions, Snapshot } from "../kiosk/controller";
import { KioskController } from "../kiosk/controller";

const BG = "#0a0a0a";
const CARD_FG = "#e6e6e6";
const IDLE_HELLO = "WELCOME IN";
const IDLE_SUBTITLE = "I've been looking for you!";

interface Dbg {
  fsm?: string;
  snap?: Snapshot;
  clipMs?: number;
  greetS?: number;
  acrostic?: string;
  lines?: string;
  model?: string;
}

interface ViewState {
  status: string;
  hello: string;
  topic: string;
  card: string;
  idleHeadline: boolean; // hello label is the big WELCOME IN
  cardFading: boolean; // idle card fading out (CSS color transition)
  dbg: Dbg;
}

const INITIAL: ViewState = {
  status: "Preheating...", hello: "", topic: "", card: "",
  idleHeadline: false, cardFading: false, dbg: {},
};

function reduce(s: ViewState, m: KioskMessage): ViewState {
  switch (m.kind) {
    case "status":
    case "error":
      return { ...s, status: m.text };
    case "config":
      return { ...s, dbg: { ...s.dbg, acrostic: m.acrostic, lines: m.lines, model: m.model } };
    case "vars":
      return { ...s, dbg: { ...s.dbg, snap: m.snap } };
    case "state":
      return { ...s, dbg: { ...s.dbg, fsm: m.state } };
    case "timing":
      return { ...s, dbg: { ...s.dbg, clipMs: m.clipMs, greetS: m.greetMs / 1000 } };
    case "begin":
      return { ...s, hello: "", topic: "", card: "", idleHeadline: false,
               cardFading: false, status: "Greeting..." };
    case "clothing":
      return m.compliment ? { ...s, status: `Noticing your ${m.compliment}...` } : s;
    case "delta":
      return m.part === "hello"
        ? { ...s, hello: s.hello + m.piece }
        : { ...s, card: s.card + m.piece };
    case "settle":
      return { ...s, hello: m.hello, topic: `- ${m.topic} -`, card: m.questions,
               status: "Lingering..." };
    case "clear":
      return { ...s, hello: IDLE_HELLO, topic: IDLE_SUBTITLE, idleHeadline: true,
               status: "Awaiting your gaze..." };
    case "idleCard":
      return { ...s, card: "", cardFading: false };
    case "idleFade":
      return { ...s, cardFading: true };
  }
}

/** Every line maps to a lever in the loop — port of _render_debug. */
function renderDebug(d: Dbg, opts: KioskOptions): string {
  const snap = d.snap;
  const res = snap?.res ?? null;
  const m = res?.metrics ?? {};
  const pose = m.yaw !== undefined
    ? `yaw ${fmt(m.yaw!, 2, true)}  ear ${m.ear_ratio!.toFixed(2)}  roll ${fmt(m.roll!, 0, true)}`
    : "-";
  const gone = Math.min(snap?.goneFor ?? 0, 999);
  return [
    `state ${(d.fsm ?? "?").padEnd(8)}  faces ${snap?.facing ?? 0}/${snap?.present ?? 0} facing/present`,
    `near  ${res?.personPresent ? res.description : "-"}   conf ${(res?.facingConf ?? 0).toFixed(2)}   ` +
      `prox ${(snap?.proximity ?? 0).toFixed(2)} (>${opts.nearProx.toFixed(2)})`,
    `pose  ${pose}`,
    `dwell ${(snap?.facingFor ?? 0).toFixed(1).padStart(4)}s (>${opts.dwell.toFixed(1)} fires)   ` +
      `gone ${gone.toFixed(1).padStart(4)}s (>${opts.clearAfter.toFixed(1)} clear)`,
    `time  det ${(snap?.detMs ?? 0).toFixed(0).padStart(3)}ms  clip ${(d.clipMs ?? 0).toFixed(0).padStart(4)}ms  ` +
      `greet ${(d.greetS ?? 0).toFixed(1)}s`,
    `acr   ${d.acrostic ?? "?"} ${d.lines ?? ""}   model ${d.model ?? "?"}`,
  ].join("\n");
}

function fmt(v: number, digits: number, sign: boolean): string {
  const s = v.toFixed(digits);
  return sign && v >= 0 ? `+${s}` : s;
}

export function KioskView({ options }: { options?: Partial<KioskOptions> }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, dispatch] = useReducer(reduce, INITIAL);
  const controllerRef = useRef<KioskController | null>(null);
  const opts = useMemo(() => options ?? {}, [options]);

  useEffect(() => {
    const controller = new KioskController(videoRef.current!, opts);
    controllerRef.current = controller;
    const unsub = controller.subscribe(dispatch);
    void controller.start();
    return () => {
      unsub();
      controller.stop();
    };
  }, [opts]);

  const fullOpts = controllerRef.current?.opts;

  return (
    <div style={{
      position: "fixed", inset: 0, background: BG, overflow: "hidden",
      fontFamily: "Helvetica, Arial, sans-serif", textAlign: "center",
      cursor: "none",
    }}>
      <video ref={videoRef} playsInline muted style={{ display: "none" }} />

      <div style={{ paddingTop: 48, color: "#555", fontSize: 16 }}>{state.status}</div>

      <div style={{
        padding: "24px 120px 0", color: "#f4f4f4",
        fontSize: state.idleHeadline ? "min(144px, 13vw)" : 40,
        fontWeight: state.idleHeadline ? "bold" : "normal",
        overflowWrap: "break-word",
      }}>
        {state.hello}
      </div>

      <div style={{
        padding: "12px 120px 0", color: "#7a7a7a", fontSize: 18, fontStyle: "italic",
      }}>
        {state.topic}
      </div>

      <div style={{
        padding: "16px 120px 0", color: state.cardFading ? BG : CARD_FG,
        fontSize: 20, fontFamily: "Courier, monospace", textAlign: "left",
        whiteSpace: "pre-wrap",
        transition: state.cardFading ? "color 0.8s linear" : "none",
      }}>
        {state.card}
      </div>

      <div style={{
        position: "absolute", right: 16, bottom: 12, color: "#2a2a2a",
        fontSize: 11, fontFamily: "Courier, monospace", textAlign: "left",
        whiteSpace: "pre",
      }}>
        {fullOpts ? renderDebug(state.dbg, fullOpts) : ""}
      </div>

      <div style={{
        position: "absolute", left: 0, right: 0, bottom: 12, color: "#555",
        fontSize: 14, fontStyle: "italic",
      }}>
        by Lee Butterman
      </div>
    </div>
  );
}
