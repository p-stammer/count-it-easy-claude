import { useEffect, useState } from 'preact/hooks';
import { listGames, listPlayers, listSessions, statsFor, type Game, type Player, type Session } from '../db';
import { go } from '../router';
import { startCount } from '../store';
import { Logo } from '../ui/Logo';

export function Home() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<Game[]>([]);
  const [open, setOpen] = useState<Session | null>(null);
  useEffect(() => {
    listPlayers().then(setPlayers);
    listGames().then(setGames);
    listSessions().then((s) => setOpen(s.find((x) => !x.finishedAt) ?? null));
  }, []);
  const names = Object.fromEntries(players.map((p) => [p.id, p]));
  return (
    <main class="screen home">
      <Logo />
      <div class="stack">
        <button class="btn big primary" onClick={() => { startCount({}); go('camera'); }}>Brett zählen</button>
        {open
          ? <button class="btn big" style={{ borderColor: '#88B358' }} onClick={() => go(`session/${open.id}`)}>Spielrunde fortsetzen</button>
          : <button class="btn big" style={{ borderColor: '#88B358' }} onClick={() => go('session/new')}>Neue Spielrunde</button>}
        {open && <button class="link" style={{ alignSelf: 'center' }} onClick={() => go('session/new')}>oder neue Spielrunde starten</button>}
        <div class="row2">
          <button class="btn" style={{ borderColor: '#fd9b1c' }} onClick={() => go('players')}>Spieler</button>
          <button class="btn" style={{ borderColor: '#ff00e6' }} onClick={() => go('stats')}>Statistik</button>
        </div>
      </div>
      {games.length > 0 && (
        <section class="card">
          <h2>Zuletzt gezählt</h2>
          <ul class="list">
            {games.slice(0, 5).map((g) => (
              <li>
                <span class="dot" style={{ background: names[g.playerId]?.color ?? '#888' }} />
                <span class="grow">{names[g.playerId]?.name ?? 'Unbekannt'}</span>
                <span class="muted">{new Date(g.createdAt).toLocaleDateString()}</span>
                <b class="score-num">{g.score}</b>
              </li>
            ))}
          </ul>
        </section>
      )}
      {players.length > 0 && games.length > 0 && (
        <section class="card">
          <h2>Bestenliste</h2>
          <ul class="list">
            {players.map((p) => ({ p, s: statsFor(games, p.id) })).filter((x) => x.s.games).sort((a, b) => b.s.best - a.s.best).slice(0, 5)
              .map(({ p, s }) => (
                <li><span class="dot" style={{ background: p.color }} /><span class="grow">{p.name}</span>
                  <span class="muted">Ø {Math.round(s.avg)}</span><b class="score-num">{s.best}</b></li>
              ))}
          </ul>
        </section>
      )}
      <footer class="home-footer">
        <button class="link" onClick={() => go('help')}>So funktioniert's</button>
        <button class="link" onClick={() => go('settings')}>Einstellungen</button>
      </footer>
    </main>
  );
}
