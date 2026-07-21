// SSD anchor generation for the MediaPipe BlazeFace "back" (256x256) detector.
// Direct port of anchors.py (itself a NumPy port of MediaPipe's
// SsdAnchorsCalculator). Produces 896 anchors of (x_center, y_center, w, h) in
// normalized [0,1]; with fixed anchor size the w/h are always 1.0.

export const INPUT_SIZE = 256;
const MIN_SCALE = 0.15625;
const MAX_SCALE = 0.75;
const STRIDES = [16, 32, 32, 32];
const ASPECT_RATIOS = [1.0];
const ANCHOR_OFFSET_X = 0.5;
const ANCHOR_OFFSET_Y = 0.5;
const INTERPOLATED_SCALE_ASPECT_RATIO = 1.0;
const FIXED_ANCHOR_SIZE = true;

function scaleAt(layerId: number, numLayers: number): number {
  if (numLayers === 1) return (MIN_SCALE + MAX_SCALE) * 0.5;
  return MIN_SCALE + ((MAX_SCALE - MIN_SCALE) * layerId) / (numLayers - 1.0);
}

/** Returns a (896*4) float32 array of (x_center, y_center, w, h) anchors. */
export function generateAnchors(): Float32Array {
  const numLayers = STRIDES.length;
  const anchors: number[] = [];

  let layerId = 0;
  while (layerId < numLayers) {
    const anchorHeights: number[] = [];
    const anchorWidths: number[] = [];
    const aspectRatios: number[] = [];
    const scales: number[] = [];

    // Merge all consecutive layers that share the same stride.
    let last = layerId;
    while (last < numLayers && STRIDES[last] === STRIDES[layerId]) {
      const scale = scaleAt(last, numLayers);
      for (const ar of ASPECT_RATIOS) {
        aspectRatios.push(ar);
        scales.push(scale);
      }
      if (INTERPOLATED_SCALE_ASPECT_RATIO > 0.0) {
        const scaleNext = last === numLayers - 1 ? 1.0 : scaleAt(last + 1, numLayers);
        scales.push(Math.sqrt(scale * scaleNext));
        aspectRatios.push(INTERPOLATED_SCALE_ASPECT_RATIO);
      }
      last += 1;
    }

    for (const ar of aspectRatios) {
      const r = Math.sqrt(ar);
      anchorHeights.push(1.0 / r);
      anchorWidths.push(1.0 * r);
    }

    const stride = STRIDES[layerId];
    const featureMapSize = Math.ceil(INPUT_SIZE / stride);
    for (let y = 0; y < featureMapSize; y++) {
      for (let x = 0; x < featureMapSize; x++) {
        for (let a = 0; a < anchorHeights.length; a++) {
          const xc = (x + ANCHOR_OFFSET_X) / featureMapSize;
          const yc = (y + ANCHOR_OFFSET_Y) / featureMapSize;
          const w = FIXED_ANCHOR_SIZE ? 1.0 : anchorWidths[a];
          const h = FIXED_ANCHOR_SIZE ? 1.0 : anchorHeights[a];
          anchors.push(xc, yc, w, h);
        }
      }
    }

    layerId = last;
  }

  return new Float32Array(anchors); // expect 896*4
}
