import { LINES, N_TILES, VALUES } from './geometry';

/** Printed values of a tile: [1|5|9, 2|6|7, 3|4|8]. */
export type Tile = [number, number, number];

export const ALL_TILES: Tile[] = VALUES[0].flatMap((a) => VALUES[1].flatMap((b) => VALUES[2].map((c) => [a, b, c] as Tile)));

export const tileKey = (t: Tile) => t.join('-');
export function parseTile(s: string): Tile {
  const t = s.split('-').map(Number) as Tile;
  if (t.length !== 3 || t.some((v, d) => !VALUES[d].includes(v))) throw new Error(`invalid tile ${s}`);
  return t;
}

export interface LineScore {
  direction: number;
  tiles: number[];
  value: number | null; // uniform value or null
  points: number;
}

export function lineScores(board: (Tile | null)[]): LineScore[] {
  if (board.length !== N_TILES) throw new Error(`expected ${N_TILES} tiles`);
  const out: LineScore[] = [];
  LINES.forEach((lines, d) =>
    lines.forEach((tiles) => {
      const vals = new Set(tiles.map((i) => board[i]?.[d] ?? null));
      const value = vals.size === 1 ? [...vals][0] : null;
      out.push({ direction: d, tiles, value, points: value === null ? 0 : value * tiles.length });
    }),
  );
  return out;
}

export const score = (board: (Tile | null)[]) => lineScores(board).reduce((s, l) => s + l.points, 0);
