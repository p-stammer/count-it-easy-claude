/**
 * Board geometry – mirrors ml/src/cie/geometry.py. The parameters live in
 * geometry.json, which is exported from Python so both sides stay identical.
 */
import data from './geometry.json';

export interface Geometry {
  cx: number;
  cy: number;
  radius: number;
  rotation: number;
  digit_radius: number;
  box_w: number;
  box_h: number;
}

export type Point = [number, number];
export type Box = [number, number, number, number];

export const DEFAULT_GEOMETRY: Geometry = data.geometry;
export const TILES = data.tiles as [number, number][];
export const VALUES = data.values as [number, number, number][];
export const LINES = data.lines as number[][][];
export const DIGIT_ANGLES = data.digitAngles as number[];
export const N_TILES = TILES.length;

export function transformed(g: Geometry, dx = 0, dy = 0, scale = 1, rotation = 0): Geometry {
  return { ...g, cx: g.cx + dx, cy: g.cy + dy, radius: g.radius * scale, rotation: g.rotation + rotation };
}

function toFrame(g: Geometry, [x, y]: Point): Point {
  const a = (g.rotation * Math.PI) / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return [(c * x - s * y) * g.radius + g.cx, (s * x + c * y) * g.radius + g.cy];
}

function tileBase([q, r]: [number, number]): Point {
  return [1.5 * q, Math.sqrt(3) * (r + q / 2)];
}

export function tileCenters(g: Geometry): Point[] {
  return TILES.map((t) => toFrame(g, tileBase(t)));
}

export function tilePolygons(g: Geometry, shrink = 1): Point[][] {
  return TILES.map((t) => {
    const [bx, by] = tileBase(t);
    return Array.from({ length: 6 }, (_, k) => {
      const a = (k * Math.PI) / 3;
      return toFrame(g, [bx + shrink * Math.cos(a), by + shrink * Math.sin(a)]);
    });
  });
}

/** [tile][direction] centre of the printed digit, in frame coordinates. */
export function digitCenters(g: Geometry): Point[][] {
  return TILES.map((t) => {
    const [bx, by] = tileBase(t);
    return DIGIT_ANGLES.map((deg) => {
      const a = (deg * Math.PI) / 180;
      return toFrame(g, [bx + g.digit_radius * Math.cos(a), by + g.digit_radius * Math.sin(a)]);
    });
  });
}

/** [tile][direction] crop box (x0, y0, x1, y1) in frame coordinates. */
export function digitBoxes(g: Geometry): Box[][] {
  const hw = (g.box_w * g.radius) / 2, hh = (g.box_h * g.radius) / 2;
  return digitCenters(g).map((row) => row.map(([x, y]) => [x - hw, y - hh, x + hw, y + hh] as Box));
}

/** Unit vector along the lines of a direction (frame coordinates, y down). */
export function lineDirection(g: Geometry, d: number): Point {
  const base: Point = d === 0 ? [0, 1] : d === 1 ? [1.5, -Math.sqrt(3) / 2] : [1.5, Math.sqrt(3) / 2];
  const a = (g.rotation * Math.PI) / 180;
  const v: Point = [Math.cos(a) * base[0] - Math.sin(a) * base[1], Math.sin(a) * base[0] + Math.cos(a) * base[1]];
  const n = Math.hypot(v[0], v[1]);
  return [v[0] / n, v[1] / n];
}
