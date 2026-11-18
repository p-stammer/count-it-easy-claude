import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { DEFAULT_GEOMETRY, tilePolygons, transformed, VALUES, type Geometry } from '../core/geometry';
import { PAD } from '../core/frame';
import { recognize, UNSURE } from '../core/recognize';
import { lineScores, score, tileKey, type Tile } from '../core/scoring';
import { addPlayer, addSample, deleteGame, gamesOfSession, getSetting, listPlayers, saveGame, type Player } from '../db';
import { back, backThenGo, go } from '../router';
import { current } from '../store';
import { Header } from '../ui/Header';
import { HexBoard, TileGlyph } from '../ui/HexBoard';
import { STRIPE } from '../ui/colors';

function useCountUp(target: number) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf = 0;
    const t0 = performance.now(), from = 0, dur = Math.min(1200, 300 + target * 4);
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / dur);
      setV(Math.round(from + (target - from) * (1 - (1 - k) ** 3)));
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return v;
}

export function Result() {
  const rec = current.recognition;
  const work = current.work;
  useEffect(() => { if (!rec || !work) go('home', true); }, []);
  if (!rec || !work) return null;
  return <ResultView key={rec.ms} />;
}

function ResultView() {
  const [rec, setRec] = useState(current.recognition!);
  const work = current.work!;
  const [board, setBoard] = useState<Tile[]>(rec.board);
  const [edited, setEdited] = useState<Set<number>>(new Set());
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<'board' | 'photo'>('board');
  const [players, setPlayers] = useState<Player[]>([]);
  const [playerId, setPlayerId] = useState<string | undefined>(current.context.playerId);
  const [newName, setNewName] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { listPlayers().then(setPlayers); }, []);

  const total = score(board);
  const shown = useCountUp(total);
  const counts = new Map<string, number>();
  board.forEach((t) => counts.set(tileKey(t), (counts.get(tileKey(t)) ?? 0) + 1));
  const duplicates = board.map((t) => (counts.get(tileKey(t)) ?? 0) > 1);
  const unsure = rec.confidence.map((c, i) => c < UNSURE && !edited.has(i));
  const nUnsure = unsure.filter(Boolean).length;
  const lines = lineScores(board).filter((l) => l.points > 0);

  function setDigit(i: number, d: number, v: number) {
    const next = board.slice();
    next[i] = next[i].slice() as Tile;
    next[i][d] = v;
    setBoard(next);
    setEdited(new Set(edited).add(i));
  }

  async function recount(g: Geometry) {
    setBusy(true);
    try {
      const r = await recognize(work, g, { refine: false });
      current.recognition = r;
      setRec(r); setBoard(r.board); setEdited(new Set()); setTab('board');
    } finally { setBusy(false); }
  }

  async function save() {
    setMsg(null);
    let pid = playerId;
    try {
      if (!pid && newName.trim()) pid = (await addPlayer(newName)).id;
    } catch (e) { setMsg((e as Error).message); return; }
    if (!pid) { setMsg('Bitte wähle, für wen gezählt wurde.'); return; }
    if (duplicates.some(Boolean)) { setMsg('Ein Plättchen kommt doppelt vor – bitte zuerst korrigieren.'); return; }
    const ctx = current.context;
    if (ctx.sessionId && ctx.round) {
      // a recount within a game round replaces the previous result of that player
      for (const g of await gamesOfSession(ctx.sessionId)) if (g.playerId === pid && g.round === ctx.round) await deleteGame(g.id);
    }
    await saveGame({ playerId: pid, score: total, tiles: board.map(tileKey), sessionId: ctx.sessionId, round: ctx.round, edited: edited.size > 0 });
    if (await getSetting('collectSamples', false)) {
      const blob = await new Promise<Blob | null>((res) => work.toBlob(res, 'image/jpeg', 0.9));
      if (blob) await addSample({ image: blob, geometry: rec.geometry, tiles: board.map(tileKey), predicted: rec.board.map(tileKey) });
    }
    current.recognition = null;
    backThenGo(2, ctx.sessionId ? `session/${ctx.sessionId}` : 'stats');
  }

  const fixedPlayer = current.context.playerId ? players.find((p) => p.id === current.context.playerId) : undefined;

  return (
    <main class="screen result">
      <Header title="Ergebnis" right={<button class="btn small" onClick={back}>Neu</button>} />
      <div class="score-big">{shown}<span>Punkte</span></div>
      {nUnsure > 0
        ? <p class="banner warn">{nUnsure === 1 ? '1 Feld ist' : `${nUnsure} Felder sind`} unsicher (gelb). Antippen zum Prüfen.</p>
        : rec.corrected.length > 0
          ? <p class="banner info">{rec.corrected.length} Ziffer(n) automatisch korrigiert (jedes Plättchen gibt es nur einmal).</p>
          : null}
      {duplicates.some(Boolean) && <p class="banner error">Doppeltes Plättchen (rot) – bitte korrigieren.</p>}

      <div class="tabs">
        <button class={tab === 'board' ? 'active' : ''} onClick={() => setTab('board')}>Brett</button>
        <button class={tab === 'photo' ? 'active' : ''} onClick={() => setTab('photo')}>Foto &amp; Ausrichtung</button>
      </div>
      {tab === 'board'
        ? <HexBoard board={board} unsure={unsure} duplicates={duplicates} selected={selected} onTile={setSelected} />
        : <PhotoAlign work={work} geometry={rec.geometry} busy={busy} onRecount={recount} />}

      {tab === 'board' && lines.length > 0 && (
        <div class="chips">
          {lines.map((l) => <span class="chip" style={{ borderColor: STRIPE[l.value!] }}>{l.value}×{l.tiles.length} = {l.points}</span>)}
        </div>
      )}

      <section class="card save">
        {fixedPlayer ? <p>Für <b style={{ color: fixedPlayer.color }}>{fixedPlayer.name}</b>{current.context.round ? ` · Runde ${current.context.round}` : ''}</p> : (
          <>
            <h2>Für wen?</h2>
            <div class="player-chips">
              {players.map((p) => (
                <button class={`pchip ${playerId === p.id ? 'active' : ''}`} style={{ borderColor: p.color }} onClick={() => { setPlayerId(p.id); setNewName(''); }}>{p.name}</button>
              ))}
              <input class="input" placeholder="+ Neuer Spieler" value={newName} maxLength={30}
                onInput={(e) => { setNewName((e.target as HTMLInputElement).value); setPlayerId(undefined); }} />
            </div>
          </>
        )}
        {msg && <p class="error">{msg}</p>}
        <button class="btn big primary" onClick={save}>Speichern</button>
      </section>

      {selected !== null && (
        <div class="sheet-backdrop" onClick={() => setSelected(null)}>
          <div class="sheet" onClick={(e) => e.stopPropagation()}>
            <div class="sheet-head">
              <TileGlyph tile={board[selected]} size={72} />
              <div>
                <b>Feld korrigieren</b>
                <p class="muted">Erkannt mit {Math.round(rec.confidence[selected] * 100)} % Sicherheit</p>
              </div>
            </div>
            {VALUES.map((vals, d) => (
              <div class="seg">
                {vals.map((v) => (
                  <button class={board[selected][d] === v ? 'active' : ''} style={{ '--c': STRIPE[v] } as Record<string, string>}
                    onClick={() => setDigit(selected, d, v)}>{v}</button>
                ))}
              </div>
            ))}
            <button class="btn big primary" onClick={() => setSelected(null)}>Fertig</button>
          </div>
        </div>
      )}
    </main>
  );
}

function PhotoAlign({ work, geometry, busy, onRecount }: { work: HTMLCanvasElement; geometry: Geometry; busy: boolean; onRecount: (g: Geometry) => void }) {
  const [g, setG] = useState(geometry);
  const url = useMemo(() => work.toDataURL('image/jpeg', 0.85), [work]);
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; g: Geometry } | null>(null);
  const polys = tilePolygons(g);
  const span = 1 + 2 * PAD;
  const scale = g.radius / DEFAULT_GEOMETRY.radius;
  const onDown = (e: PointerEvent) => { (e.target as Element).setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, g }; };
  const onMove = (e: PointerEvent) => {
    if (!drag.current || !ref.current) return;
    const k = span / ref.current.getBoundingClientRect().width;
    const d = drag.current;
    setG({ ...d.g, cx: d.g.cx + (e.clientX - d.x) * k, cy: d.g.cy + (e.clientY - d.y) * k });
  };
  return (
    <div class="photo-align">
      <div class="photo" ref={ref}>
        <img src={url} alt="Aufgenommenes Brett" />
        <svg viewBox={`${-PAD} ${-PAD} ${span} ${span}`} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={() => (drag.current = null)}>
          {polys.map((p) => <polygon points={p.map(([x, y]) => `${x},${y}`).join(' ')} />)}
        </svg>
      </div>
      <p class="muted small">Sechsecke verschieben, bis sie auf den Plättchen liegen.</p>
      <label class="slider">Größe
        <input type="range" min={0.8} max={1.2} step={0.005} value={scale}
          onInput={(e) => setG({ ...g, radius: DEFAULT_GEOMETRY.radius * +(e.target as HTMLInputElement).value })} />
      </label>
      <label class="slider">Drehung
        <input type="range" min={-8} max={8} step={0.25} value={g.rotation}
          onInput={(e) => setG({ ...g, rotation: +(e.target as HTMLInputElement).value })} />
      </label>
      <div class="row2">
        <button class="btn" onClick={() => setG(transformed(DEFAULT_GEOMETRY))}>Zurücksetzen</button>
        <button class="btn primary" disabled={busy} onClick={() => onRecount(g)}>{busy ? 'Zähle …' : 'Neu zählen'}</button>
      </div>
    </div>
  );
}
