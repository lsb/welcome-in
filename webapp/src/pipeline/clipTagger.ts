// Stage 1.5 — what the visitor is wearing. Port of clip_tags.py: crop to the
// visitor's torso (from the BlazeFace box), embed with the MobileCLIP2-S0
// image encoder (fp16 ONNX), compare against the pre-embedded phrase bank
// (clip_bank.json — the text encoder ran once offline, at build time) by
// cosine similarity.
//
// MobileCLIP2 preprocessing: resize shortest edge to 256 (bilinear), center
// crop 256, scale to [0,1] — no mean/std normalisation.
import { createSession, ort } from "./ort";

export interface ClipRead {
  color: string;
  garment: string;
  style: string;
  hat: boolean;
  gconf: number;
}

interface BankGroup {
  labels: string[];
  embeddings: number[][];
}

export interface ClipBank {
  model: string;
  image_size: number;
  thresholds: { min_conf: number; hat_margin: number };
  groups: Record<string, BankGroup>;
}

/** Convert a float32 value to its IEEE 754 half-precision bit pattern. */
function f32ToF16(val: number): number {
  f32buf[0] = val;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const frac = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (frac ? 0x200 : 0); // Inf/NaN
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> Inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow -> 0
    const f = (frac | 0x800000) >>> (1 - e + 13);
    return sign | f;
  }
  return sign | (e << 10) | (frac >>> 13);
}
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

export class ClipTagger {
  private session: ort.InferenceSession;
  private bank: ClipBank;
  private size: number;
  private fp16Input: boolean;
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;

  private constructor(session: ort.InferenceSession, bank: ClipBank) {
    this.session = session;
    this.bank = bank;
    this.size = bank.image_size;
    // The shipped visual encoder is fp16 end-to-end; introspect when the
    // runtime exposes metadata, else assume fp16.
    const meta = (session as unknown as { inputMetadata?: { type?: string }[] }).inputMetadata;
    this.fp16Input = meta?.[0]?.type ? meta[0].type === "float16" : true;
    this.canvas = new OffscreenCanvas(this.size, this.size);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  }

  static async create(
    modelUrl = "models/mobileclip2_s0_visual_fp16.onnx",
    bankUrl = "models/clip_bank.json",
  ): Promise<ClipTagger> {
    const [session, bank] = await Promise.all([
      createSession(modelUrl),
      fetch(bankUrl).then((r) => r.json() as Promise<ClipBank>),
    ]);
    return new ClipTagger(session, bank);
  }

  /** Embed the torso crop of `box` (original-image normalized xyxy; null for
   * the whole frame) and L2-normalize. */
  async embedImage(
    source: CanvasImageSource, sw: number, sh: number,
    box: [number, number, number, number] | null,
  ): Promise<Float32Array> {
    // Torso crop in source pixels — port of clip_tags._torso_crop (expand the
    // face box down to the torso, up a little for a hat). int() truncates.
    let sx = 0, sy = 0, cw = sw, ch = sh;
    if (box !== null) {
      const [x1, y1, x2, y2] = box;
      const fw = x2 - x1, fh = y2 - y1, cx = (x1 + x2) / 2;
      const cx1 = Math.max(0, cx - 1.8 * fw), cx2 = Math.min(1, cx + 1.8 * fw);
      const cy1 = Math.max(0, y1 - 2.0 * fh), cy2 = Math.min(1, y2 + 5.0 * fh);
      sx = Math.trunc(cx1 * sw);
      sy = Math.trunc(cy1 * sh);
      cw = Math.trunc(cx2 * sw) - sx;
      ch = Math.trunc(cy2 * sh) - sy;
    }
    // Resize shortest edge to `size` and center-crop a square, in one draw.
    const s = this.size / Math.min(cw, ch);
    const rw = Math.round(cw * s), rh = Math.round(ch * s);
    const left = Math.floor((rw - this.size) / 2), top = Math.floor((rh - this.size) / 2);
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.size, this.size);
    this.ctx.drawImage(source, sx, sy, cw, ch, -left, -top, rw, rh);

    const { data } = this.ctx.getImageData(0, 0, this.size, this.size);
    const n = this.size * this.size;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      chw[i] = data[i * 4] / 255;
      chw[n + i] = data[i * 4 + 1] / 255;
      chw[2 * n + i] = data[i * 4 + 2] / 255;
    }

    let tensor: ort.Tensor;
    if (this.fp16Input) {
      const half = new Uint16Array(chw.length);
      for (let i = 0; i < chw.length; i++) half[i] = f32ToF16(chw[i]);
      tensor = new ort.Tensor("float16", half, [1, 3, this.size, this.size]);
    } else {
      tensor = new ort.Tensor("float32", chw, [1, 3, this.size, this.size]);
    }
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const raw = out[this.session.outputNames[0]].data;
    // fp16 outputs surface as a native Float16Array where the JS engine has
    // one (values, not bit patterns), else as a Uint16Array of half bits.
    const F16 = (globalThis as Record<string, unknown>).Float16Array as
      (new (...a: unknown[]) => ArrayLike<number>) | undefined;
    const emb: ArrayLike<number> =
      raw instanceof Float32Array ? raw
        : F16 && raw instanceof F16 ? (raw as ArrayLike<number>)
        : f16BitsToF32(raw as Uint16Array);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) norm += Number(emb[i]) * Number(emb[i]);
    norm = Math.sqrt(norm) + 1e-9;
    const res = new Float32Array(emb.length);
    for (let i = 0; i < emb.length; i++) res[i] = Number(emb[i]) / norm;
    return res;
  }

  private top(group: string, ie: Float32Array): [string, number] {
    const { labels, embeddings } = this.bank.groups[group];
    let best = -Infinity, bi = 0;
    for (let r = 0; r < embeddings.length; r++) {
      let s = 0;
      const row = embeddings[r];
      for (let i = 0; i < row.length; i++) s += row[i] * ie[i];
      if (s > best) { best = s; bi = r; }
    }
    return [labels[bi], best];
  }

  /** The shared read — top tag per group, gated by the garment confidence
   * (mirrors make_clip_bank.web_read, which wrote the expected outputs). */
  readEmbedding(ie: Float32Array): ClipRead | null {
    const { min_conf, hat_margin } = this.bank.thresholds;
    const [garment, gconf] = this.top("garment", ie);
    if (gconf < min_conf) return null;
    const [color] = this.top("color", ie);
    const [style] = this.top("style", ie);
    const { labels, embeddings } = this.bank.groups.hat;
    const sims = new Map(labels.map((l, r) => {
      let s = 0;
      for (let i = 0; i < embeddings[r].length; i++) s += embeddings[r][i] * ie[i];
      return [l, s] as const;
    }));
    const hat = (sims.get("hat")! - Math.max(sims.get("hood")!, sims.get("nohat")!)) >= hat_margin;
    return { color, garment, style, hat, gconf };
  }

  async read(
    source: CanvasImageSource, sw: number, sh: number,
    box: [number, number, number, number] | null,
  ): Promise<ClipRead | null> {
    return this.readEmbedding(await this.embedImage(source, sw, sh, box));
  }

  /** 'a red hoodie, in a casual style' (or '… and a hat …'), or null. */
  async describe(source: CanvasImageSource, sw: number, sh: number,
                 box: [number, number, number, number] | null): Promise<string | null> {
    const r = await this.read(source, sw, sh, box);
    if (r === null) return null;
    let desc = `${article(r.color)} ${r.color} ${r.garment}`;
    if (r.hat) desc += " and a hat";
    return `${desc}, in ${article(r.style)} ${r.style} style`;
  }

  /** Bare noun phrase for the 'love the {…}' hello template, or null. */
  async compliment(source: CanvasImageSource, sw: number, sh: number,
                   box: [number, number, number, number] | null): Promise<string | null> {
    const r = await this.read(source, sw, sh, box);
    return r === null ? null : `${r.color} ${r.garment}`;
  }
}

function f16BitsToF32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h >>> 10) & 0x1f;
    const frac = h & 0x3ff;
    let bits32: number;
    if (exp === 0) {
      if (frac === 0) bits32 = sign;
      else {
        // subnormal half -> normal float
        let e = -1, f = frac;
        do { e++; f <<= 1; } while ((f & 0x400) === 0);
        bits32 = sign | ((127 - 15 - e) << 23) | ((f & 0x3ff) << 13);
      }
    } else if (exp === 0x1f) {
      bits32 = sign | 0x7f800000 | (frac << 13);
    } else {
      bits32 = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    u32out[0] = bits32;
    out[i] = f32out[0];
  }
  return out;
}
const f32out = new Float32Array(1);
const u32out = new Uint32Array(f32out.buffer);

function article(word: string): string {
  return "aeiou".includes(word[0]?.toLowerCase() ?? "") ? "an" : "a";
}
