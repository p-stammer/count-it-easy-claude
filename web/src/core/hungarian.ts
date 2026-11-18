/**
 * Minimum-cost assignment (Hungarian / Kuhn-Munkres, O(n^2 m)).
 * cost: n rows x m columns with n <= m. Returns column index for each row.
 */
export function hungarian(cost: number[][]): number[] {
  const n = cost.length, m = cost[0].length;
  if (n > m) throw new Error('need rows <= columns');
  const INF = Number.POSITIVE_INFINITY;
  const u = new Float64Array(n + 1), v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1), way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const ans = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j] > 0) ans[p[j] - 1] = j - 1;
  return ans;
}
