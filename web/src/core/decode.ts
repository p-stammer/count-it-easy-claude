/**
 * Probabilities -> consistent board. Mirrors ml/src/cie/decode.py: each of the
 * 27 physical tiles exists once, so we choose the most likely assignment of
 * distinct tiles to the 19 positions.
 */
import { N_TILES, VALUES } from './geometry';
import { hungarian } from './hungarian';
import { ALL_TILES, score, type Tile } from './scoring';

/** probs[tile][direction][class] */
export type Probs = number[][][];

export interface Decoded {
  board: Tile[];
  naive: Tile[];
  confidence: number[]; // probability of the chosen tile per position
  corrected: number[];
  score: number;
  logLikelihood: number;
}

const IDX = ALL_TILES.map((t) => t.map((v, d) => VALUES[d].indexOf(v)));

export function decode(probs: Probs, eps = 1e-6): Decoded {
  if (probs.length !== N_TILES) throw new Error('bad probs');
  const lp = probs.map((t) => t.map((d) => d.map((p) => Math.log(Math.min(1, Math.max(eps, p))))));
  const ll = lp.map((t) => IDX.map(([a, b, c]) => t[0][a] + t[1][b] + t[2][c]));
  const cols = hungarian(ll.map((row) => row.map((x) => -x)));
  const board = cols.map((c) => ALL_TILES[c]);
  const naive = probs.map((t) => t.map((d, di) => VALUES[di][d.indexOf(Math.max(...d))]) as Tile);
  const confidence = cols.map((c, i) => Math.exp(ll[i][c]));
  const corrected = board.flatMap((t, i) => (t.join() === naive[i].join() ? [] : [i]));
  return { board, naive, confidence, corrected, score: score(board), logLikelihood: cols.reduce((s, c, i) => s + ll[i][c], 0) };
}
