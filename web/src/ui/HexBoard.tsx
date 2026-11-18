import { DEFAULT_GEOMETRY, digitCenters, lineDirection, tileCenters, tilePolygons, type Geometry } from '../core/geometry';
import { lineScores, type Tile } from '../core/scoring';
import { STRIPE } from './colors';

interface Props {
  board: (Tile | null)[];
  unsure?: boolean[];
  duplicates?: boolean[];
  selected?: number | null;
  onTile?: (i: number) => void;
  showLines?: boolean;
  geometry?: Geometry;
}

/** Renders the board like the real game: dark hexes with coloured stripes. */
export function HexBoard({ board, unsure = [], duplicates = [], selected = null, onTile, showLines = true, geometry = DEFAULT_GEOMETRY }: Props) {
  const polys = tilePolygons(geometry, 0.97);
  const centers = tileCenters(geometry);
  const digits = digitCenters(geometry);
  const r = geometry.radius;
  const scoring = showLines ? lineScores(board).filter((l) => l.points > 0) : [];
  const pts = (p: [number, number][]) => p.map(([x, y]) => `${x},${y}`).join(' ');
  return (
    <svg class="hexboard" viewBox="0 0.02 1 0.96" role="img" aria-label="Spielbrett">
      <defs>
        {polys.map((p, i) => <clipPath id={`clip${i}`}><polygon points={pts(p)} /></clipPath>)}
      </defs>
      {polys.map((p, i) => {
        const t = board[i];
        const [cx, cy] = centers[i];
        const cls = ['tile', unsure[i] && 'unsure', duplicates[i] && 'dup', selected === i && 'selected'].filter(Boolean).join(' ');
        return (
          <g class={cls} onClick={() => onTile?.(i)} style={{ cursor: onTile ? 'pointer' : 'default' }}>
            <polygon points={pts(p)} class="tile-bg" />
            {t && (
              <g clip-path={`url(#clip${i})`}>
                {[1, 2, 0].map((d) => {
                  const [ux, uy] = lineDirection(geometry, d);
                  const L = 1.2 * r;
                  return <line x1={cx - ux * L} y1={cy - uy * L} x2={cx + ux * L} y2={cy + uy * L}
                    stroke={STRIPE[t[d]]} stroke-width={r * 0.34} />;
                })}
              </g>
            )}
            <polygon points={pts(p)} class="tile-border" />
            {t ? digits[i].map(([x, y], d) => (
              <text x={x} y={y} class="digit" font-size={r * 0.5}>{t[d]}</text>
            )) : <text x={cx} y={cy} class="digit missing" font-size={r * 0.6}>?</text>}
          </g>
        );
      })}
      {scoring.map((l) => {
        const a = centers[l.tiles[0]], b = centers[l.tiles[l.tiles.length - 1]];
        return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} class="score-line" stroke={STRIPE[l.value!]} stroke-width={r * 0.12} />;
      })}
    </svg>
  );
}

/** Small read-only tile glyph (used in the correction sheet). */
export function TileGlyph({ tile, size = 56 }: { tile: Tile; size?: number }) {
  const g: Geometry = { ...DEFAULT_GEOMETRY, cx: 0.5, cy: 0.5, radius: 0.48 };
  const poly = Array.from({ length: 6 }, (_, k) => [0.5 + 0.48 * Math.cos((k * Math.PI) / 3), 0.5 + 0.48 * Math.sin((k * Math.PI) / 3)]);
  const pts = poly.map(([x, y]) => `${x},${y}`).join(' ');
  const offs = [-90, 150, 30].map((a) => [0.5 + 0.29 * Math.cos((a * Math.PI) / 180), 0.5 + 0.29 * Math.sin((a * Math.PI) / 180)]);
  return (
    <svg width={size} height={size} viewBox="0 0 1 1" class="glyph">
      <clipPath id="gclip"><polygon points={pts} /></clipPath>
      <polygon points={pts} class="tile-bg" />
      <g clip-path="url(#gclip)">
        {[1, 2, 0].map((d) => {
          const [ux, uy] = lineDirection(g, d);
          return <line x1={0.5 - ux} y1={0.5 - uy} x2={0.5 + ux} y2={0.5 + uy} stroke={STRIPE[tile[d]]} stroke-width={0.16} />;
        })}
      </g>
      {offs.map(([x, y], d) => <text x={x} y={y} class="digit" font-size={0.24}>{tile[d]}</text>)}
    </svg>
  );
}
