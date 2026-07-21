import { KioskView } from "./ui/KioskView";
import { SelfTest } from "./ui/SelfTest";

/** ?demo=5 — stand in for the webcam with a sample PNG played through a
 * canvas stream, so the whole kiosk flow runs without a camera. */
function demoStream(n: string): () => Promise<MediaStream> {
  return async () => {
    const img = new Image();
    img.src = `samples/${n}.png`;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d")!;
    const draw = () => ctx.drawImage(img, 0, 0);
    draw();
    setInterval(draw, 100); // keep frames flowing on the static scene
    return canvas.captureStream(10);
  };
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("selftest")) return <SelfTest />;
  const demo = params.get("demo");
  return <KioskView options={demo ? { getStream: demoStream(demo) } : undefined} />;
}
