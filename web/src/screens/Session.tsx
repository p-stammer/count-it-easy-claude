import { useEffect, useState } from 'preact/hooks';
import { createSession, deleteSession, gamesOfSession, getSession, listPlayers, updateSession, type Game, type Player, type Session } from '../db';
import { go } from '../router';
import { startCount } from '../store';
import { Header } from '../ui/Header';

export function SessionSetup() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [rounds, setRounds] = useState(3);
  useEffect(() => { listPlayers().then((p) => { setPlayers(p); setChosen(new Set(p.map((x) => x.id))); }); }, []);
  const toggle = (id: string) => { const n = new Set(chosen); n.has(id) ? n.delete(id) : n.add(id); setChosen(n); };
  return (
    <main class="screen">
      <Header title="Neue Spielrunde" />
      {players.length === 0 ? (
        <div class="card center">
          <p>Lege zuerst Spieler an.</p>
          <button class="btn primary" onClick={() => go('players')}>Spieler anlegen</button>
        </div>
      ) : (
        <>
          <section class="card">
            <h2>Wer spielt mit?</h2>
            <div class="player-chips">
              {players.map((p) => (
                <button class={`pchip ${chosen.has(p.id) ? 'active' : ''}`} style={{ borderColor: p.color }} onClick={() => toggle(p.id)}>{p.name}</button>
              ))}
            </div>
          </section>
          <section class="card">
            <h2>Anzahl Runden</h2>
            <div class="seg">
              {[1, 2, 3, 4, 5].map((n) => <button class={rounds === n ? 'active' : ''} onClick={() => setRounds(n)}>{n}</button>)}
            </div>
          </section>
          <button class="btn big primary" disabled={chosen.size === 0}
            onClick={async () => { const s = await createSession(players.filter((p) => chosen.has(p.id)).map((p) => p.id), rounds); go(`session/${s.id}`, true); }}>
            Los geht's
          </button>
        </>
      )}
    </main>
  );
}

export function SessionView({ id }: { id: string }) {
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<Game[]>([]);
  const [confirm, setConfirm] = useState(false);
  const reload = () => { getSession(id).then((s) => setSession(s ?? null)); listPlayers(true).then(setPlayers); gamesOfSession(id).then(setGames); };
  useEffect(reload, [id]);
  const allDone = !!session && session.playerIds.every((pid) =>
    Array.from({ length: session.rounds }, (_, i) => i + 1).every((r) => games.some((g) => g.playerId === pid && g.round === r)));
  useEffect(() => {
    if (session && allDone && !session.finishedAt) updateSession({ ...session, finishedAt: Date.now() }).then(reload);
  }, [allDone, session?.id, session?.finishedAt]);
  if (session === undefined) return null;
  if (session === null) return <main class="screen"><Header title="Spielrunde" /><p class="center muted">Nicht gefunden.</p></main>;

  const ps = session.playerIds.map((pid) => players.find((p) => p.id === pid)).filter(Boolean) as Player[];
  const cell = (pid: string, r: number) => games.filter((g) => g.playerId === pid && g.round === r).sort((a, b) => b.createdAt - a.createdAt)[0];
  const rounds = Array.from({ length: session.rounds }, (_, i) => i + 1);
  // current round = first round in which somebody is still missing
  const currentRound = rounds.find((r) => ps.some((p) => !cell(p.id, r))) ?? null;
  const totals = ps.map((p) => rounds.reduce((s, r) => s + (cell(p.id, r)?.score ?? 0), 0));
  const best = Math.max(...totals, 0);
  const done = currentRound === null;

  return (
    <main class="screen">
      <Header title={done ? 'Endstand' : `Runde ${currentRound} von ${session.rounds}`} />
      <div class="card table-wrap">
        <table class="stats session">
          <thead><tr><th /> {ps.map((p) => <th style={{ color: p.color }}>{p.name}</th>)}</tr></thead>
          <tbody>
            {rounds.map((r) => (
              <tr class={r === currentRound ? 'current' : ''}>
                <td>R{r}</td>
                {ps.map((p) => {
                  const g = cell(p.id, r);
                  const top = g && g.score === Math.max(...ps.map((q) => cell(q.id, r)?.score ?? -1));
                  return <td class={top ? 'top' : ''}>{g ? g.score : r === currentRound ? '·' : ''}</td>;
                })}
              </tr>
            ))}
            <tr class="total"><td>Σ</td>{totals.map((t) => <td class={t === best && t > 0 ? 'top' : ''}>{t}</td>)}</tr>
          </tbody>
        </table>
      </div>
      {done && best > 0 && <p class="winner">🏆 {ps.filter((_, i) => totals[i] === best).map((p) => p.name).join(' & ')} gewinnt mit {best} Punkten!</p>}
      {!done && (
        <section class="stack">
          {ps.map((p) => {
            const g = cell(p.id, currentRound!);
            return (
              <button class="btn big" style={{ borderColor: p.color, opacity: g ? 0.5 : 1 }}
                onClick={() => { startCount({ playerId: p.id, sessionId: session.id, round: currentRound! }); go('camera'); }}>
                {g ? `${p.name}: ${g.score} – neu zählen` : `${p.name} zählen`}
              </button>
            );
          })}
        </section>
      )}
      <div class="row2">
        <button class="btn" onClick={() => go('home')}>Zur Startseite</button>
        <button class="btn danger" onClick={() => setConfirm(true)}>Runde löschen</button>
      </div>
      {confirm && (
        <div class="sheet-backdrop" onClick={() => setConfirm(false)}>
          <div class="sheet" onClick={(e) => e.stopPropagation()}>
            <b>Spielrunde inkl. aller Ergebnisse löschen?</b>
            <div class="row2">
              <button class="btn" onClick={() => setConfirm(false)}>Abbrechen</button>
              <button class="btn danger" onClick={async () => { await deleteSession(session.id); go('home', true); }}>Löschen</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
