const COLORS = ['#88B358', '#BA2525', '#FFEE00', '#00a3d4', '#ffddfb', '#82ffff', '#dddddd', '#fd9b1c', '#ff00e6', '#88B358', '#00a3d4', '#BA2525'];

export function Logo({ small = false }: { small?: boolean }) {
  let i = 0;
  return (
    <h1 class={small ? 'logo small' : 'logo'} aria-label="Count it Easy!">
      {'Count it Easy!'.split('').map((ch) => (ch === ' ' ? ' ' : <span style={{ color: COLORS[i++ % COLORS.length] }}>{ch}</span>))}
    </h1>
  );
}
