import { useEffect, useState } from 'preact/hooks';
import { deleteGame, listGames, listPlayers, statsFor, type Game, type Player } from '../db';
import { parseTile } from '../core/scoring';
import { Header } from '../ui/Header';
import { HexBoard } from '../ui/HexBoard';
import { Sparkline } from '../ui/Sparkline';

export function Stats() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<Game[]>([]);
  const [open, setOpen] = useState<Game | null>(null);
  const reload = () => { listPlayers(true).then(setPlayers); listGames().then(setGames); };
  useEffect(reload, []);
  const byId = Object.fromEntries(players.map((p) => [p.id, p]));
  const rows = players.map((p) => ({ p, s: statsFor(games, p.id) })).filter((r) => r.s.games > 0).sort((a, b) => b.s.avg - a.s.avg);
  return (
    <main class="screen">
      <Header title="Statistik" />
      {rows.length === 0 && <p class="muted center">Noch keine gespeicherten Spiele.</p>}
      {rows.length > 0 && (
        <div class="card table-wrap">
          <table class="stats">
            <thead><tr><th>Spieler</th><th>Spiele</th><th>Ø</th><th>Best</th><th>Verlauf</th></tr></thead>
            <tbody>
              {rows.map(({ p, s }) => (
                <tr>
                  <td><span class="dot" style={{ background: p.color }} /> {p.name}{p.archived ? ' (entfernt)' : ''}</td>
                  <td>{s.games}</td><td>{Math.round(s.avg)}</td><td><b>{s.best}</b></td>
                  <td><Sparkline values={s.history} color={p.color} width={80} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {games.length > 0 && (
        <section class="card">
          <h2>Alle Spiele</h2>
          <ul class="list">
            {games.map((g) => (
              <li onClick={() => setOpen(g)} style={{ cursor: 'pointer' }}>
                <span class="dot" style={{ background: byId[g.playerId]?.color ?? '#888' }} />
                <span class="grow">{byId[g.playerId]?.name ?? '?'}{g.round ? <small class="muted"> · Runde {g.round}</small> : null}</span>
                <span class="muted">{new Date(g.createdAt).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
                <b class="score-num">{g.score}</b>
              </li>
            ))}
          </ul>
        </section>
      )}
      {open && (
        <div class="sheet-backdrop" onClick={() => setOpen(null)}>
          <div class="sheet" onClick={(e) => e.stopPropagation()}>
            <b>{byId[open.playerId]?.name} · {open.score} Punkte</b>
            <HexBoard board={open.tiles.map(parseTile)} />
            <div class="row2">
              <button class="btn danger" onClick={async () => { await deleteGame(open.id); setOpen(null); reload(); }}>Spiel löschen</button>
              <button class="btn" onClick={() => setOpen(null)}>Schließen</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
