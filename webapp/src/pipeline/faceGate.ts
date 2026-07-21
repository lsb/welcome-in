// Stage 1 — the visual wake word. Direct port of face_gate.py: the same
// Qualcomm MediaPipe BlazeFace ONNX (public/models/face_detector.onnx, external
// data embedded) run through onnxruntime-web, decode -> NMS -> facing test on
// the 6 keypoints (eyes, nose, mouth, ear tragions).
import { generateAnchors, INPUT_SIZE } from "./anchors";
import { createSession, ort } from "./ort";

// Keypoint indices (within each decoded row, after the 4 bbox values).
const KP_RIGHT_EYE = 0, KP_LEFT_EYE = 1, KP_NOSE = 2;
const KP_RIGHT_EAR = 4, KP_LEFT_EAR = 5;

// Tunable thresholds — mirror face_gate.py.
export const DET_THRESH = 0.5;
const NMS_IOU = 0.3;
const YAW_MAX = 0.35;
const EAR_LO = 0.6, EAR_HI = 1.67;
const ROLL_MAX_DEG = 30.0;

export interface FaceMetrics {
  yaw?: number;
  ear_ratio?: number;
  roll?: number;
  reason?: string;
}

export interface FaceResult {
  personPresent: boolean;
  facingCamera: boolean;
  facingConf: number;
  score: number;
  box: [number, number, number, number] | null; // original-image normalized xyxy
  keypoints: Float32Array | null; // (6*2) original-image normalized
  description: string;
  metrics: FaceMetrics;
  presentCount: number;
  facingCount: number;
  proximity: number; // primary face's box height (0..1)
}

const EMPTY: FaceResult = {
  personPresent: false, facingCamera: false, facingConf: 0, score: 0,
  box: null, keypoints: null, description: "", metrics: {},
  presentCount: 0, facingCount: 0, proximity: 0,
};

function sigmoid(x: number): number {
  const c = Math.max(-30, Math.min(30, x));
  return 1 / (1 + Math.exp(-c));
}

interface Detected {
  facing: boolean;
  conf: number;
  metrics: FaceMetrics;
  score: number;
  box: [number, number, number, number];
  kp: Float32Array;
  bh: number;
}

export class FaceGate {
  private session: ort.InferenceSession;
  private coordsNames: string[];
  private scoresNames: string[];
  private anchors: Float32Array;
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
    const out = session.outputNames;
    this.coordsNames = out.filter((n) => n.toLowerCase().includes("coord")).sort();
    this.scoresNames = out.filter((n) => n.toLowerCase().includes("score")).sort();
    this.anchors = generateAnchors();
    this.canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    // "medium" tracks PIL's antialiased bilinear letterbox closest on the
    // sample PNGs (Chrome's default "low" flips marginal far-face detections
    // at the 0.5 score threshold; see the selftest).
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = "medium";
  }

  static async create(modelUrl = "models/face_detector.onnx"): Promise<FaceGate> {
    return new FaceGate(await createSession(modelUrl));
  }

  /** Letterbox `source` into the 256 canvas (black pad) and return CHW floats. */
  private letterbox(source: CanvasImageSource, sw: number, sh: number) {
    const scale = INPUT_SIZE / Math.max(sw, sh);
    const nw = Math.round(sw * scale), nh = Math.round(sh * scale);
    const padX = Math.floor((INPUT_SIZE - nw) / 2), padY = Math.floor((INPUT_SIZE - nh) / 2);
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    this.ctx.drawImage(source, 0, 0, sw, sh, padX, padY, nw, nh);
    const { data } = this.ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const n = INPUT_SIZE * INPUT_SIZE;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      chw[i] = data[i * 4] / 255;
      chw[n + i] = data[i * 4 + 1] / 255;
      chw[2 * n + i] = data[i * 4 + 2] / 255;
    }
    return { chw, scale, padX, padY };
  }

  /** Decode raw anchor offsets -> (896,16): xyxy box + 6 keypoints, canvas-normalized. */
  private decode(raw: Float32Array): Float32Array {
    const out = new Float32Array(raw.length);
    const s = INPUT_SIZE;
    for (let i = 0; i < 896; i++) {
      const ax = this.anchors[i * 4], ay = this.anchors[i * 4 + 1];
      const aw = this.anchors[i * 4 + 2], ah = this.anchors[i * 4 + 3];
      const o = i * 16;
      const cx = (raw[o] / s) * aw + ax;
      const cy = (raw[o + 1] / s) * ah + ay;
      const w = (raw[o + 2] / s) * aw;
      const h = (raw[o + 3] / s) * ah;
      out[o] = cx - w / 2;
      out[o + 1] = cy - h / 2;
      out[o + 2] = cx + w / 2;
      out[o + 3] = cy + h / 2;
      for (let k = 0; k < 6; k++) {
        out[o + 4 + 2 * k] = (raw[o + 4 + 2 * k] / s) * aw + ax;
        out[o + 5 + 2 * k] = (raw[o + 5 + 2 * k] / s) * ah + ay;
      }
    }
    return out;
  }

  async analyze(source: CanvasImageSource, sw: number, sh: number): Promise<FaceResult> {
    const { chw, scale, padX, padY } = this.letterbox(source, sw, sh);
    const feeds: Record<string, ort.Tensor> = {
      [this.session.inputNames[0]]: new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    };
    const outputs = await this.session.run(feeds);

    // Concatenate the two heads: coords (512+384, 16) and scores (512+384).
    const raw = new Float32Array(896 * 16);
    let off = 0;
    for (const n of this.coordsNames) {
      const d = outputs[n].data as Float32Array;
      raw.set(d, off);
      off += d.length;
    }
    const scores = new Float32Array(896);
    off = 0;
    for (const n of this.scoresNames) {
      const d = outputs[n].data as Float32Array;
      for (let i = 0; i < d.length; i++) scores[off + i] = sigmoid(d[i]);
      off += d.length;
    }

    const idx: number[] = [];
    for (let i = 0; i < 896; i++) if (scores[i] >= DET_THRESH) idx.push(i);
    if (idx.length === 0) return { ...EMPTY };

    const decoded = this.decode(raw);
    const keep = nms(decoded, scores, idx, NMS_IOU);

    // Map canvas-normalized points back to original-image normalized coords.
    const toOrigX = (x: number) => (x * INPUT_SIZE - padX) / scale / sw;
    const toOrigY = (y: number) => (y * INPUT_SIZE - padY) / scale / sh;

    // Evaluate every kept face — the kiosk greets groups and needs the room.
    const faces: Detected[] = keep.map((i) => {
      const o = i * 16;
      const kpCanvas = decoded.subarray(o + 4, o + 16); // facing test in canvas space
      const [facing, conf, metrics] = this.facing(kpCanvas);
      const box: [number, number, number, number] = [
        toOrigX(decoded[o]), toOrigY(decoded[o + 1]),
        toOrigX(decoded[o + 2]), toOrigY(decoded[o + 3]),
      ];
      const kp = new Float32Array(12);
      for (let k = 0; k < 6; k++) {
        kp[2 * k] = toOrigX(kpCanvas[2 * k]);
        kp[2 * k + 1] = toOrigY(kpCanvas[2 * k + 1]);
      }
      return { facing, conf, metrics, score: scores[i], box, kp, bh: Math.abs(box[3] - box[1]) };
    });

    const facingFaces = faces.filter((f) => f.facing);
    // Primary = closest facing person if anyone faces, else closest present.
    const pool = facingFaces.length > 0 ? facingFaces : faces;
    const primary = pool.reduce((a, b) => (b.bh > a.bh ? b : a));

    return {
      personPresent: true,
      facingCamera: facingFaces.length > 0,
      facingConf: primary.conf,
      score: primary.score,
      box: primary.box,
      keypoints: primary.kp,
      description: describe(primary.box),
      metrics: primary.metrics,
      presentCount: faces.length,
      facingCount: facingFaces.length,
      proximity: primary.bh,
    };
  }

  private facing(kp: Float32Array): [boolean, number, FaceMetrics] {
    const rex = kp[2 * KP_RIGHT_EYE], rey = kp[2 * KP_RIGHT_EYE + 1];
    const lex = kp[2 * KP_LEFT_EYE], ley = kp[2 * KP_LEFT_EYE + 1];
    const nx = kp[2 * KP_NOSE], ny = kp[2 * KP_NOSE + 1];
    const rearx = kp[2 * KP_RIGHT_EAR], reary = kp[2 * KP_RIGHT_EAR + 1];
    const learx = kp[2 * KP_LEFT_EAR], leary = kp[2 * KP_LEFT_EAR + 1];

    const iod = Math.hypot(lex - rex, ley - rey);
    if (iod < 1e-6) return [false, 0, { reason: "degenerate eyes" }];

    const emx = (rex + lex) / 2;
    const yaw = (nx - emx) / iod;

    const dL = Math.hypot(nx - learx, ny - leary);
    const dR = Math.hypot(nx - rearx, ny - reary);
    const earRatio = dL / (dR + 1e-6);

    let roll = (Math.atan2(ley - rey, lex - rex) * 180) / Math.PI;
    roll = ((((roll + 90) % 180) + 180) % 180) - 90; // fold to [-90, 90]

    const ok =
      Math.abs(yaw) < YAW_MAX && EAR_LO < earRatio && earRatio < EAR_HI &&
      Math.abs(roll) < ROLL_MAX_DEG;

    const sYaw = Math.max(0, 1 - Math.abs(yaw) / YAW_MAX);
    const sEar = Math.max(0, 1 - Math.abs(Math.log(Math.max(earRatio, 1e-6))) / Math.abs(Math.log(EAR_HI)));
    const sRoll = Math.max(0, 1 - Math.abs(roll) / ROLL_MAX_DEG);
    const conf = Math.cbrt(sYaw * sEar * sRoll);

    return [ok, conf, {
      yaw: Math.round(yaw * 1000) / 1000,
      ear_ratio: Math.round(earRatio * 1000) / 1000,
      roll: Math.round(roll * 10) / 10,
    }];
  }
}

function nms(decoded: Float32Array, scores: Float32Array, idx: number[], iouThresh: number): number[] {
  const order = [...idx].sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];
  const area = (i: number) => {
    const o = i * 16;
    return Math.max(0, decoded[o + 2] - decoded[o]) * Math.max(0, decoded[o + 3] - decoded[o + 1]);
  };
  const alive = new Set(order);
  for (const i of order) {
    if (!alive.has(i)) continue;
    keep.push(i);
    alive.delete(i);
    const oi = i * 16;
    for (const j of [...alive]) {
      const oj = j * 16;
      const xx1 = Math.max(decoded[oi], decoded[oj]);
      const yy1 = Math.max(decoded[oi + 1], decoded[oj + 1]);
      const xx2 = Math.min(decoded[oi + 2], decoded[oj + 2]);
      const yy2 = Math.min(decoded[oi + 3], decoded[oj + 3]);
      const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
      const union = area(i) + area(j) - inter;
      const iou = union > 0 ? inter / union : 0;
      if (iou > iouThresh) alive.delete(j);
    }
  }
  return keep;
}

/** Position/proximity phrase for the debug readout — port of FaceGate._describe. */
export function describe(box: [number, number, number, number]): string {
  const cx = (box[0] + box[2]) / 2;
  const bh = Math.abs(box[3] - box[1]);
  const pos = cx < 0.38 ? "off to your left" : cx > 0.62 ? "off to your right" : "right in front of you";
  const prox = bh > 0.45 ? "very close" : bh > 0.25 ? "close by" : bh > 0.12 ? "a few steps away" : "across the room";
  return `${pos}, ${prox}`;
}
