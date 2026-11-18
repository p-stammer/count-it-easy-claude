/**
 * Full recognition: predict -> decode, optionally refining the board
 * alignment by trying small shifts/scales and keeping the most likely result.
 */
import { DEFAULT_GEOMETRY, transformed, type Geometry } from './geometry';
import { decode, type Decoded, type Probs } from './decode';
import { predict } from './model';

export interface Recognition extends Decoded {
  geometry: Geometry;
  probs: Probs;
  refined: boolean;
  ms: number;
}

/** Confidence below which a tile is shown as "please check". */
export const UNSURE = 0.85;

const needsRefine = (d: Decoded) => d.confidence.some((c) => c < UNSURE) || d.corrected.length > 0;

export async function recognize(work: HTMLCanvasElement, geometry: Geometry = DEFAULT_GEOMETRY,
  opts: { refine?: boolean } = {}): Promise<Recognition> {
  const t0 = performance.now();
  let probs = await predict(work, geometry);
  let best: Recognition = { ...decode(probs), geometry, probs, refined: false, ms: 0 };
  if (opts.refine !== false && needsRefine(best)) {
    // coarse-to-fine search around the user's alignment
    for (const step of [0.02, 0.01]) {
      const center = best.geometry;
      const candidates = [
        transformed(center, step, 0), transformed(center, -step, 0),
        transformed(center, 0, step), transformed(center, 0, -step),
        transformed(center, 0, 0, 1 + step * 2), transformed(center, 0, 0, 1 - step * 2),
      ];
      for (const g of candidates) {
        probs = await predict(work, g);
        const d = decode(probs);
        if (d.logLikelihood > best.logLikelihood + 0.5) best = { ...d, geometry: g, probs, refined: true, ms: 0 };
      }
      if (!needsRefine(best)) break;
    }
  }
  best.ms = performance.now() - t0;
  return best;
}
