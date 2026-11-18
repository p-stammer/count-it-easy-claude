import { useEffect, useState } from 'preact/hooks';
import { addPlayer, listGames, listPlayers, removePlayer, renamePlayer, statsFor, type Game, type Player } from '../db';
import { Header } from '../ui/Header';

export function Players() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<Game[]>([]);
  const [name, setName] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<Player | null>(null);
  const reload = () => { listPlayers().then(setPlayers); listGames().then(setGames); };
  useEffect(reload, []);

  async function add(e: Event) {
    e.preventDefault();
    try { await addPlayer(name); setName(''); setErr(null); reload(); } catch (x) { setErr((x as Error).message); }
  }

  return (
    <main class="screen">
      <Header title="Spieler" />
      <form class="add-row" onSubmit={add}>
        <input class="input" placeholder="Name" value={name} maxLength={30} onInput={(e) => setName((e.target as HTMLInputElement).value)} />
        <button class="btn primary" type="submit">Hinzufügen</button>
      </form>
      {err && <p class="error">{err}</p>}
      {players.length === 0 && <p class="muted center">Noch keine Spieler. Du kannst Spieler auch direkt beim Speichern eines Ergebnisses anlegen.</p>}
      <ul class="list card">
        {players.map((p) => {
          const s = statsFor(games, p.id);
          return (
            <li>
              <span class="dot" style={{ background: p.color }} />
              {editing === p.id
                ? <input class="input grow" value={p.name} autoFocus maxLength={30}
                  onBlur={async (e) => { await renamePlayer(p.id, (e.target as HTMLInputElement).value); setEditing(null); reload(); }}
                  onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />
                : <span class="grow" onClick={() => setEditing(p.id)}>{p.name}<small class="muted"> · {s.games} Spiele</small></span>}
              <button class="icon-btn" aria-label="Umbenennen" onClick={() => setEditing(p.id)}>✎</button>
              <button class="icon-btn" aria-label="Löschen" onClick={() => setConfirm(p)}>🗑</button>
            </li>
          );
        })}
      </ul>
      {confirm && (
        <div class="sheet-backdrop" onClick={() => setConfirm(null)}>
          <div class="sheet" onClick={(e) => e.stopPropagation()}>
            <b>{confirm.name} entfernen?</b>
            <p class="muted">Bereits gespeicherte Ergebnisse bleiben in der Statistik erhalten.</p>
            <div class="row2">
              <button class="btn" onClick={() => setConfirm(null)}>Abbrechen</button>
              <button class="btn danger" onClick={async () => { await removePlayer(confirm.id); setConfirm(null); reload(); }}>Entfernen</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
