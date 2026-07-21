# welcome-in web

The kiosk as a web app: the same doorway greeter — face-gated wake word,
zero-shot outfit compliment, pooled acrostic question cards — running entirely
in the browser. **All local**: models, wasm runtime, and cards are served from
this origin; no CDN, no remote model downloads at runtime. Model footprint is
**~24 MB** (vs the desktop's multi-GB stack), within the <50 MB budget.

| stage | desktop | web |
|-------|---------|-----|
| face gate | Qualcomm BlazeFace ONNX via onnxruntime, `face_gate.py` | the **same ONNX** (0.6 MB, external data embedded) via onnxruntime-web, `src/pipeline/faceGate.ts` |
| outfit read | marqo-fashionCLIP int8 (~85 MB vision) | **MobileCLIP2-S0** image encoder, fp16 (23.9 MB), `src/pipeline/clipTagger.ts` |
| CLIP text side | embedded at startup on-device | embedded **offline at build time** into `clip_bank.json` (150 KB) — the browser never runs a text encoder |
| question cards | pooled `.txt` cards from the 27B producer | the **same pooled cards**, baked into the bundle by `scripts/prepare_assets.mjs` |
| hello | five templates, round-robin | same templates, `src/kiosk/greeter.ts` |
| screen | Tk full-screen skin | React skin, same layout/copy/debug readout, `src/ui/KioskView.tsx` |
| printer | CUPS PostScript | not wired (the show ends at "Lingering...") |

## Run

```bash
npm install
npm run dev        # http://localhost:5173 — grant camera access
npm run build && npm run preview   # production build
```

The models in `public/models/` are committed (24 MB), so this just works. To
regenerate them (new model choice, edited phrase bank):

```bash
# needs the desktop face model once: `uv run python main.py --setup` from the repo root
uv run --with onnx,onnxconverter-common python scripts/prepare_models.py
../.venv/bin/python scripts/make_clip_bank.py
```

Routes:
- `/` — the kiosk (needs a webcam)
- `/?demo=5` — no webcam: sample PNG 5 played through a canvas stream (any of 1–8)
- `/?selftest` — runs the browser pipeline over the 8 sample PNGs and diffs
  against the Python reference (`scripts/reference_outputs.json`); shows
  PASS/FAIL per image and exposes `window.__SELFTEST__` for automation

## Why these models

Measured on the repo's sample PNGs (`scripts/race_models.py`,
`scripts/calibrate_bank.py` — run them to reproduce):

- **BlazeFace**: byte-identical to the desktop detector, so gate parity is by
  construction. Canvas letterboxing uses `imageSmoothingQuality: "medium"`,
  which tracks PIL's bilinear closest (Chrome's default flips marginal
  far-face detections at the 0.5 score threshold).
- **MobileCLIP2-S0 fp16**: ties MobileCLIP-S0 on desktop-parity tags (11/15
  labels + 2/2 hats vs fashionCLIP) with a ~55% wider no-read margin, at the
  same 23 MB. Bigger models did worse on the actual images: S1-fp16 (43 MB,
  higher ImageNet score) misread the samples and halved the no-read gap;
  int8 quantization collapses these conv-hybrid encoders entirely (S0/S2
  int8 read everything as "gray polo shirt"); MatMulNBits q4 doesn't compress
  conv-heavy towers (S0 q4 is *bigger* than fp16 — 4-bit pays off only for
  pure-ViT towers like MobileCLIP-B/fashionCLIP, none of which fit <50 MB).
  The fp16 conversion is lossless on the samples (cosine 0.99999 vs fp32).
- **Text bank** (`public/models/clip_bank.json`): the desktop's phrase groups
  with two calibrated changes the measurements forced — a **hood** distractor
  in the hat group (MobileCLIP scores a raised hood above a real hat; with it,
  the desktop's 0.03 hat margin transfers) and **two prompt phrasings per
  label, averaged** (single prompts left a 0.006 no-read gap; these widen it
  to 0.028 around `min_conf` 0.185). Regenerate with `make_clip_bank.py`
  after editing the groups in `race_models.py`.

Known sample-level deltas vs the desktop read (both defensible by eye):
4.png (a painted portrait) reads "black jacket" vs fashionCLIP's "beige
jacket"; 7.png reads "gray button-up shirt" vs "beige polo shirt". 5.png's
second, hood-shadowed face sits at ear-ratio 1.58 vs the 1.67 facing bound and
may count as facing on one side only — same greet either way, "Hi" vs "Hi
everyone".

## Deploy (static files)

`npm run build` puts the whole app in `dist/` — plain static files (hashed
JS + the wasm runtime in `assets/`, models under `models/`). Serve `dist/`,
not the source tree (the source needs Vite's dev server to transpile). The
build is path-relative (`base: "./"` + relative fetches), so it works mounted
anywhere — domain root, `/kiosk/`, or browsing to `webapp/dist/` under a
server rooted at the repo. Any static server works. For nginx:

```nginx
server {
    listen 443 ssl;                      # getUserMedia REQUIRES https
    root /var/www/welcome-in/dist;       # (localhost is the only exemption)

    # wasm threads want cross-origin isolation; without these headers the
    # app still runs, single-threaded.
    add_header Cross-Origin-Opener-Policy same-origin always;
    add_header Cross-Origin-Embedder-Policy require-corp always;

    # Serve pre-compressed model/wasm files (see below) instead of
    # compressing 24 MB on the fly.
    gzip_static on;          # serves foo.onnx.gz as Content-Encoding: gzip
    # brotli_static on;      # if ngx_brotli is built in

    location /assets/ {      # content-hashed -> cache forever
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
}
```

Pre-compress the big files once (`.gz` sits next to the original):

```bash
pigz -11 -k dist/models/*.onnx dist/assets/*.wasm
```

The wire cost is ~20 MB: vision model 16.3 MB + wasm runtime 3.5 MB + face
model 0.55 MB + bank 0.04 MB. The shipped vision model is already
**e5m6-squashed** (prepare_models.py rounds the fp16 mantissas to 6 bits so
gzip halves the low bytes — 21.1 -> 16.3 MB, measured tag-identical); more
aggressive squashes degrade — see the collapse curve in `squash_fp16.py`.

The camera needs a **secure context**: `https://` anywhere except
`localhost`. For a LAN kiosk either terminate TLS in nginx (self-signed +
`chrome --ignore-certificate-errors` on the kiosk box, or a real cert) or run
the browser on the serving box against `http://localhost`.

## Notes

- The kiosk state machine, timers, and message vocabulary are a direct port of
  `kiosk.py` (`src/kiosk/controller.ts`); the React layer only renders
  messages, mirroring the Tk `_handle` seam.
- onnxruntime-web's runtime is imported via Vite `?url` assets
  (`src/pipeline/ort.ts`) — it cannot live in `public/` because ORT loads its
  `.mjs` with a dynamic `import()`, which Vite refuses for public-dir files.
- fp16 ONNX outputs surface as a native `Float16Array` (values, not bit
  patterns) in current Chrome + onnxruntime-web; inputs accept either form.
- COOP/COEP headers (vite.config.ts) make the page cross-origin-isolated so
  the wasm runtime can use threads; without them it still runs, single-threaded.
