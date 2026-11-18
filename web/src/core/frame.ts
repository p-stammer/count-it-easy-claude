/**
 * The "work image": a square canvas covering frame coordinates
 * [-PAD, 1 + PAD]. Frame coordinates are the canonical board frame shown by
 * the camera overlay. The padding guarantees that crop boxes of a shifted or
 * enlarged board never fall outside the image.
 */
export const PAD = 0.15;
export const WORK_SIZE = 1200;

export const frameToWork = (f: number, size = WORK_SIZE) => ((f + PAD) / (1 + 2 * PAD)) * size;
export const workToFrame = (px: number, size = WORK_SIZE) => (px / size) * (1 + 2 * PAD) - PAD;

type Source = CanvasImageSource & { width?: number; height?: number };

/**
 * Draw the source rectangle (sx, sy, sw, sh) of an image/video onto the whole
 * destination canvas, correctly handling parts outside the source (filled
 * grey) – plain drawImage would silently rescale in that case.
 */
export function drawClipped(ctx: CanvasRenderingContext2D, src: Source, srcW: number, srcH: number,
  sx: number, sy: number, sw: number, sh: number, dw: number, dh: number) {
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, dw, dh);
  const x0 = Math.max(0, sx), y0 = Math.max(0, sy);
  const x1 = Math.min(srcW, sx + sw), y1 = Math.min(srcH, sy + sh);
  if (x1 <= x0 || y1 <= y0) return;
  const kx = dw / sw, ky = dh / sh;
  ctx.drawImage(src, x0, y0, x1 - x0, y1 - y0, (x0 - sx) * kx, (y0 - sy) * ky, (x1 - x0) * kx, (y1 - y0) * ky);
}

export function newCanvas(w: number, h = w): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

/**
 * Grab the work image from a live <video> shown with object-fit: cover in a
 * square container of `containerPx` CSS pixels whose frame (the overlay)
 * is inset by `insetPx` on every side.
 */
export function captureFromVideo(video: HTMLVideoElement, containerPx: number, insetPx: number): HTMLCanvasElement {
  const vw = video.videoWidth, vh = video.videoHeight;
  const k = Math.max(containerPx / vw, containerPx / vh); // cover scale (CSS px per video px)
  const ox = (containerPx - vw * k) / 2, oy = (containerPx - vh * k) / 2;
  const frameCss = containerPx - 2 * insetPx;
  const toVideo = (f: number, o: number) => (insetPx + f * frameCss - o) / k;
  const sx = toVideo(-PAD, ox), sy = toVideo(-PAD, oy);
  const side = ((1 + 2 * PAD) * frameCss) / k;
  const out = newCanvas(WORK_SIZE);
  const ctx = out.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  drawClipped(ctx, video, vw, vh, sx, sy, side, side, WORK_SIZE, WORK_SIZE);
  return out;
}

/**
 * Work image from a photo: the central square of the photo is treated as the
 * frame (the board should fill it, like the training photos).
 */
export function captureFromImage(img: HTMLImageElement | ImageBitmap): HTMLCanvasElement {
  const w = img.width, h = img.height, s = Math.min(w, h);
  const fx = (w - s) / 2, fy = (h - s) / 2;
  const out = newCanvas(WORK_SIZE);
  const ctx = out.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  drawClipped(ctx, img, w, h, fx - PAD * s, fy - PAD * s, (1 + 2 * PAD) * s, (1 + 2 * PAD) * s, WORK_SIZE, WORK_SIZE);
  return out;
}
