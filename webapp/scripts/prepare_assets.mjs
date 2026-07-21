// Stage the app's local assets before dev/build. Everything the page loads is
// served from this origin — no CDN, no remote models:
//   * the pre-generated card pools (../pool/*.txt, written by the desktop
//     producer) -> src/generated/pool.json
//   * the sample PNGs + the Python harness's reference outputs -> public/samples/
//     (the ?selftest route diffs the browser pipeline against them)
// (The onnxruntime-web wasm runtime is bundled by Vite itself via ?url imports
// in src/pipeline/ort.ts — no copying needed.)
import { copyFileSync, globSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webapp = join(here, "..");
const repo = join(webapp, "..");

// -- card pools ---------------------------------------------------------
// Mirrors pool.py's on-disk format: "# key: value" header lines, blank line,
// then the card body (stanzas separated by blank lines).
function parseCard(raw) {
  const lines = raw.split("\n");
  const header = {};
  let i = 0;
  for (; i < lines.length && lines[i].startsWith("#"); i++) {
    const m = lines[i].slice(1).match(/^\s*([^:]+):\s*(.*)$/);
    if (m) header[m[1].trim()] = m[2].trim();
  }
  while (i < lines.length && lines[i].trim() === "") i++;
  return { header, body: lines.slice(i).join("\n").trim() };
}

function loadPool(dir) {
  const cards = [];
  for (const f of globSync(join(dir, "*.txt")).sort()) {
    const { header, body } = parseCard(readFileSync(f, "utf-8"));
    if (!body || !header.topic || header.provisional === "1") continue;
    cards.push({ topic: header.topic, text: body, model: header.model ?? "?" });
  }
  return cards;
}

const pool = {
  cards: loadPool(join(repo, "pool/SLOP+TIAT+LEEB")),
  idle: loadPool(join(repo, "pool/SLOP")),
};
mkdirSync(join(webapp, "src/generated"), { recursive: true });
writeFileSync(join(webapp, "src/generated/pool.json"), JSON.stringify(pool, null, 1));
console.log(`pool: ${pool.cards.length} cards, ${pool.idle.length} idle cards -> src/generated/pool.json`);

// -- selftest fixtures --------------------------------------------------
const samplesDir = join(webapp, "public/samples");
mkdirSync(samplesDir, { recursive: true });
let n = 0;
for (const f of globSync(join(repo, "[0-9].png"))) { copyFileSync(f, join(samplesDir, basename(f))); n++; }
copyFileSync(join(webapp, "scripts/reference_outputs.json"),
             join(samplesDir, "reference_outputs.json"));
console.log(`samples: ${n} PNGs + reference_outputs.json -> public/samples/`);
