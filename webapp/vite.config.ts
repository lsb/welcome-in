import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// COOP/COEP make the page cross-origin-isolated so onnxruntime-web can use
// SharedArrayBuffer (multi-threaded wasm). Without them it still runs, single
// threaded. Everything is served from this origin — no CDN, no remote models.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  // Relative base so dist/ works mounted at any path (domain root or a
  // subdirectory) — runtime fetches in src/ are likewise relative.
  base: "./",
  plugins: [react()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
});
