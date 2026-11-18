export function Sparkline({ values, color = '#fd9b1c', width = 120, height = 32 }: { values: number[]; color?: string; width?: number; height?: number }) {
  if (values.length < 2) return <svg width={width} height={height} />;
  const v = values.slice(-30);
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((y, i) => `${(i / (v.length - 1)) * (width - 4) + 2},${height - 3 - ((y - lo) / span) * (height - 6)}`).join(' ');
  return (
    <svg width={width} height={height} aria-hidden="true">
      <polyline points={pts} fill="none" stroke={color} stroke-width="2" stroke-linejoin="round" />
    </svg>
  );
}
