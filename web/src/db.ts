/**
 * Local persistence (IndexedDB). Everything stays on the device; a JSON
 * export/import exists as backup because browsers may evict site data.
 */
import { openDB, type DBSchema, type IDBPDatabase } from 'idb';

export interface Player { id: string; name: string; color: string; createdAt: number; archived?: boolean }
export interface Game {
  id: string;
  playerId: string;
  score: number;
  tiles: string[];        // 19 tiles "a-b-c", canonical order
  createdAt: number;
  sessionId?: string;
  round?: number;
  edited: boolean;        // user corrected at least one tile
}
export interface Session {
  id: string;
  createdAt: number;
  playerIds: string[];
  rounds: number;
  finishedAt?: number;
}
export interface Sample {   // optional: captured photo + final labels, for improving the model
  id: string;
  createdAt: number;
  image: Blob;            // work image (JPEG)
  geometry: unknown;
  tiles: string[];
  predicted: string[];
}

interface CieDB extends DBSchema {
  players: { key: string; value: Player };
  games: { key: string; value: Game; indexes: { byPlayer: string; bySession: string } };
  sessions: { key: string; value: Session };
  samples: { key: string; value: Sample };
  settings: { key: string; value: unknown };
}

let dbp: Promise<IDBPDatabase<CieDB>> | null = null;
export function db() {
  dbp ??= openDB<CieDB>('count-it-easy', 1, {
    upgrade(d) {
      d.createObjectStore('players', { keyPath: 'id' });
      const g = d.createObjectStore('games', { keyPath: 'id' });
      g.createIndex('byPlayer', 'playerId');
      g.createIndex('bySession', 'sessionId');
      d.createObjectStore('sessions', { keyPath: 'id' });
      d.createObjectStore('samples', { keyPath: 'id' });
      d.createObjectStore('settings');
    },
  });
  return dbp;
}

export const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);

export const PLAYER_COLORS = ['#88B358', '#BA2525', '#FFEE00', '#00a3d4', '#ff9ef0', '#82ffff', '#fd9b1c', '#ff00e6', '#dddddd', '#9c7cff'];

export async function listPlayers(includeArchived = false) {
  const all = await (await db()).getAll('players');
  return all.filter((p) => includeArchived || !p.archived).sort((a, b) => a.createdAt - b.createdAt);
}

export async function addPlayer(name: string): Promise<Player> {
  const clean = name.trim().slice(0, 30);
  if (!clean) throw new Error('Bitte einen Namen eingeben.');
  const players = await listPlayers(true);
  if (players.some((p) => !p.archived && p.name.toLowerCase() === clean.toLowerCase())) throw new Error('Diesen Namen gibt es schon.');
  const used = new Set(players.filter((p) => !p.archived).map((p) => p.color));
  const color = PLAYER_COLORS.find((c) => !used.has(c)) ?? PLAYER_COLORS[players.length % PLAYER_COLORS.length];
  const p: Player = { id: uid(), name: clean, color, createdAt: Date.now() };
  await (await db()).put('players', p);
  return p;
}

export async function renamePlayer(id: string, name: string) {
  const d = await db();
  const p = await d.get('players', id);
  const clean = name.trim().slice(0, 30);
  if (p && clean) await d.put('players', { ...p, name: clean });
}

/** Players with games are archived (history stays intact), others deleted. */
export async function removePlayer(id: string) {
  const d = await db();
  const games = await d.getAllFromIndex('games', 'byPlayer', id);
  if (games.length) {
    const p = await d.get('players', id);
    if (p) await d.put('players', { ...p, archived: true });
  } else await d.delete('players', id);
}

export async function saveGame(g: Omit<Game, 'id' | 'createdAt'>): Promise<Game> {
  const game: Game = { ...g, id: uid(), createdAt: Date.now() };
  await (await db()).put('games', game);
  return game;
}

export async function updateGame(g: Game) { await (await db()).put('games', g); }
export async function deleteGame(id: string) { await (await db()).delete('games', id); }
export async function listGames() { return (await (await db()).getAll('games')).sort((a, b) => b.createdAt - a.createdAt); }
export async function gamesOfSession(id: string) { return (await db()).getAllFromIndex('games', 'bySession', id); }

export async function createSession(playerIds: string[], rounds: number): Promise<Session> {
  const s: Session = { id: uid(), createdAt: Date.now(), playerIds, rounds };
  await (await db()).put('sessions', s);
  return s;
}
export async function getSession(id: string) { return (await db()).get('sessions', id); }
export async function updateSession(s: Session) { await (await db()).put('sessions', s); }
export async function listSessions() { return (await (await db()).getAll('sessions')).sort((a, b) => b.createdAt - a.createdAt); }
export async function deleteSession(id: string) {
  const d = await db();
  for (const g of await gamesOfSession(id)) await d.delete('games', g.id);
  await d.delete('sessions', id);
}

export async function getSetting<T>(key: string, fallback: T): Promise<T> {
  const v = await (await db()).get('settings', key);
  return (v as T) ?? fallback;
}
export async function setSetting(key: string, value: unknown) { await (await db()).put('settings', value, key); }

export async function addSample(s: Omit<Sample, 'id' | 'createdAt'>) {
  await (await db()).put('samples', { ...s, id: uid(), createdAt: Date.now() });
}
export async function countSamples() { return (await db()).count('samples'); }
export async function listSamples() { return (await db()).getAll('samples'); }
export async function clearSamples() { await (await db()).clear('samples'); }

export interface Backup { app: 'count-it-easy'; version: 1; exportedAt: number; players: Player[]; games: Game[]; sessions: Session[] }

export async function exportBackup(): Promise<Backup> {
  const d = await db();
  return { app: 'count-it-easy', version: 1, exportedAt: Date.now(), players: await d.getAll('players'),
    games: await d.getAll('games'), sessions: await d.getAll('sessions') };
}

export async function importBackup(b: Backup) {
  if (b?.app !== 'count-it-easy' || b.version !== 1) throw new Error('Keine gültige Count-it-Easy-Sicherung.');
  const d = await db();
  const tx = d.transaction(['players', 'games', 'sessions'], 'readwrite');
  for (const p of b.players) await tx.objectStore('players').put(p);
  for (const g of b.games) await tx.objectStore('games').put(g);
  for (const s of b.sessions) await tx.objectStore('sessions').put(s);
  await tx.done;
}

export interface PlayerStats { games: number; best: number; avg: number; last: number | null; history: number[] }
export function statsFor(games: Game[], playerId: string): PlayerStats {
  const g = games.filter((x) => x.playerId === playerId).sort((a, b) => a.createdAt - b.createdAt);
  const s = g.map((x) => x.score);
  return { games: s.length, best: s.length ? Math.max(...s) : 0, avg: s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0,
    last: s.length ? s[s.length - 1] : null, history: s };
}
