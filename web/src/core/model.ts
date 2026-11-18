/** Digit classifier inference with onnxruntime-web (WASM, fully on-device). */
import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import { digitBoxes, N_TILES, type Geometry } from './geometry';
import { frameToWork, newCanvas } from './frame';
import type { Probs } from './decode';

export const INPUT = 128;
let session: Promise<ort.InferenceSession> | null = null;

export function loadModel(): Promise<ort.InferenceSession> {
  if (!session) {
    ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    session = ort.InferenceSession.create(`${import.meta.env.BASE_URL}models/digits.onnx`, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    session.catch(() => { session = null; });
  }
  return session;
}

const cropCanvas = typeof document !== 'undefined' ? newCanvas(INPUT) : null;

/** Cut the 57 digit crops from the work image into an NCHW float tensor. */
export function cropsTensor(work: HTMLCanvasElement, g: Geometry): Float32Array {
  const ctx = cropCanvas!.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const S = work.width;
  const data = new Float32Array(N_TILES * 3 * 3 * INPUT * INPUT);
  const plane = INPUT * INPUT;
  let n = 0;
  for (const row of digitBoxes(g)) {
    for (const [x0, y0, x1, y1] of row) {
      const sx = frameToWork(x0, S), sy = frameToWork(y0, S);
      ctx.drawImage(work, sx, sy, frameToWork(x1, S) - sx, frameToWork(y1, S) - sy, 0, 0, INPUT, INPUT);
      const px = ctx.getImageData(0, 0, INPUT, INPUT).data;
      const base = n * 3 * plane;
      for (let i = 0; i < plane; i++) {
        data[base + i] = px[4 * i] / 255;
        data[base + plane + i] = px[4 * i + 1] / 255;
        data[base + 2 * plane + i] = px[4 * i + 2] / 255;
      }
      n++;
    }
  }
  return data;
}

/** logits (57 x 9) -> probs[tile][direction][class] using each direction's 3 logits. */
export function logitsToProbs(logits: Float32Array): Probs {
  const out: Probs = [];
  for (let t = 0; t < N_TILES; t++) {
    const row: number[][] = [];
    for (let d = 0; d < 3; d++) {
      const o = (t * 3 + d) * 9 + 3 * d;
      const z = [logits[o], logits[o + 1], logits[o + 2]];
      const m = Math.max(...z);
      const e = z.map((v) => Math.exp(v - m));
      const s = e[0] + e[1] + e[2];
      row.push(e.map((v) => v / s));
    }
    out.push(row);
  }
  return out;
}

export async function predict(work: HTMLCanvasElement, g: Geometry): Promise<Probs> {
  const sess = await loadModel();
  const input = new ort.Tensor('float32', cropsTensor(work, g), [N_TILES * 3, 3, INPUT, INPUT]);
  const res = await sess.run({ [sess.inputNames[0]]: input });
  return logitsToProbs(res[sess.outputNames[0]].data as Float32Array);
}
