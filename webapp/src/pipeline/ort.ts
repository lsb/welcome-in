// onnxruntime-web setup. We use the cpu/wasm-only entry (no webgpu bundle)
// and hand Vite the runtime files as ordinary same-origin assets via ?url —
// everything is served locally, no CDN. (The runtime .mjs cannot live in
// public/: ORT loads it with a dynamic import(), which Vite refuses for
// public-directory files.) Threads need cross-origin isolation (COOP/COEP
// headers, set in vite.config.ts); without it we run single-threaded.
import * as ort from "onnxruntime-web/wasm";
// Relative paths: the package's exports map doesn't expose ./dist/*.
import ortWasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";

ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
// ?threads=N overrides for benchmarking (thread count is fixed at wasm init,
// so changing it means reloading the page).
const threadsOverride = Number(new URLSearchParams(location.search).get("threads")) || 0;
ort.env.wasm.numThreads = threadsOverride > 0
  ? threadsOverride
  : self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;

export async function createSession(url: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
}

export { ort };
