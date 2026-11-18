import { describe, expect, it } from 'vitest';
import ref from './__fixtures__/python_reference.json';
import { DEFAULT_GEOMETRY, digitBoxes, LINES, transformed } from './geometry';
import { decode } from './decode';
import { hungarian } from './hungarian';
import { ALL_TILES, parseTile, score, tileKey } from './scoring';

const close = (a: number[][][], b: number[][][]) =>
  a.forEach((row, i) => row.forEach((box, d) => box.forEach((v, k) => expect(v).toBeCloseTo(b[i][d][k], 5))));

describe('geometry (parity with Python)', () => {
  it('default digit boxes', () => close(digitBoxes(DEFAULT_GEOMETRY), ref.digitBoxes));
  it('transformed digit boxes', () => close(digitBoxes(transformed(DEFAULT_GEOMETRY, 0.01, -0.02, 1.05, 2)), ref.shiftedBoxes));
  it('15 lines of length 3-4-5-4-3', () => {
    expect(LINES.flat().map((l) => l.length).sort()).toEqual([3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5]);
  });
});

describe('scoring', () => {
  it('27 unique tiles', () => expect(new Set(ALL_TILES.map(tileKey)).size).toBe(27));
  it.each(ref.boards.map((b) => [b.id, b] as const))('board %s', (_, b) => {
    expect(score(b.tiles.map(parseTile))).toBe(b.score);
  });
  it('original notebook reference scores', () => {
    const s = Object.fromEntries(ref.boards.map((b) => [b.id, b.score]));
    expect([s.t1, s.t2, s.t3, s.t4, s.t5]).toEqual([170, 181, 167, 245, 307]);
  });
  it('rejects invalid tiles', () => expect(() => parseTile('2-1-3')).toThrow());
});

describe('decode (parity with Python)', () => {
  it.each(ref.decodeCases.map((c, i) => [i, c] as const))('case %i', (_, c) => {
    const d = decode(c.probs);
    expect(d.board.map(tileKey)).toEqual(c.board);
    expect(d.score).toBe(c.score);
    expect(new Set(d.board.map(tileKey)).size).toBe(19);
  });
  it('hungarian finds optimum on a small matrix', () => {
    expect(hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]])).toEqual([1, 0, 2]);
  });
});
