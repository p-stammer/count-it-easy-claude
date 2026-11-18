import { useEffect, useRef, useState } from 'preact/hooks';
import { DEFAULT_GEOMETRY, tilePolygons } from '../core/geometry';
import { captureFromImage, captureFromVideo, newCanvas } from '../core/frame';
import { recognize } from '../core/recognize';
import { go } from '../router';
import { current } from '../store';
import { Header } from '../ui/Header';

/** Overlay inset inside the square viewfinder (fraction of its size). */
const INSET = 0.04;
const POLYS = tilePolygons(DEFAULT_GEOMETRY);

type CamState = 'starting' | 'live' | 'denied' | 'unavailable' | 'insecure';

export function Camera() {
  const video = useRef<HTMLVideoElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CamState>('starting');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<{ min: number; max: number; step: number; value: number } | null>(null);
  const [torch, setTorch] = useState<boolean | null>(null);
  const [glare, setGlare] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { setState('insecure'); return; }
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1920 } },
    }).then((s) => {
      if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
      stream.current = s;
      const v = video.current!;
      v.srcObject = s;
      v.play().catch(() => undefined);
      const track = s.getVideoTracks()[0];
      const caps = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { zoom?: { min: number; max: number; step: number }; torch?: boolean };
      const settings = track.getSettings() as MediaTrackSettings & { zoom?: number };
      if (caps.zoom && caps.zoom.max > caps.zoom.min) setZoom({ ...caps.zoom, value: settings.zoom ?? caps.zoom.min });
      if (caps.torch) setTorch(false);
      setState('live');
    }).catch((e: DOMException) => {
      setState(e?.name === 'NotAllowedError' || e?.name === 'SecurityError' ? 'denied' : 'unavailable');
    });
    return () => { cancelled = true; stream.current?.getTracks().forEach((t) => t.stop()); };
  }, []);

  // cheap live reflection detector: share of blown-out pixels on the board
  useEffect(() => {
    if (state !== 'live') return;
    const c = newCanvas(64);
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    const id = setInterval(() => {
      const v = video.current;
      if (!v || !v.videoWidth || busy) return;
      const s = Math.min(v.videoWidth, v.videoHeight);
      ctx.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, 64, 64);
      const d = ctx.getImageData(8, 8, 48, 48).data;
      let hot = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] > 240 && d[i + 1] > 240 && d[i + 2] > 240) hot++;
      setGlare(hot / (d.length / 4) > 0.01);
    }, 700);
    return () => clearInterval(id);
  }, [state, busy]);

  async function run(work: HTMLCanvasElement) {
    setBusy('Zähle …');
    setError(null);
    try {
      current.work = work;
      current.recognition = await recognize(work);
      go('result');
    } catch (e) {
      console.error(e);
      setError('Die Erkennung ist fehlgeschlagen. Bitte noch einmal versuchen.');
    } finally {
      setBusy(null);
    }
  }

  function capture() {
    const v = video.current, b = box.current;
    if (!v || !b || !v.videoWidth) return;
    const size = b.getBoundingClientRect().width;
    run(captureFromVideo(v, size, size * INSET));
  }

  async function pickFile(e: Event) {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    try {
      const bmp = await createImageBitmap(f, { imageOrientation: 'from-image' } as ImageBitmapOptions);
      run(captureFromImage(bmp));
    } catch {
      setError('Das Bild konnte nicht geladen werden.');
    }
    (e.target as HTMLInputElement).value = '';
  }

  async function setTrack(c: Record<string, unknown>) {
    const track = stream.current?.getVideoTracks()[0];
    try { await track?.applyConstraints({ advanced: [c] } as MediaTrackConstraints); } catch { /* unsupported */ }
  }

  const message = {
    starting: 'Kamera wird gestartet …',
    denied: 'Kein Kamerazugriff. Bitte die Kamera-Berechtigung für diese Seite in den Browser-Einstellungen erlauben – oder unten ein Foto wählen.',
    unavailable: 'Keine Kamera gefunden. Du kannst stattdessen ein Foto wählen.',
    insecure: 'Die Kamera funktioniert nur über HTTPS. Du kannst stattdessen ein Foto wählen.',
    live: '',
  }[state];

  return (
    <main class="screen camera">
      <Header title="Brett fotografieren" right={torch !== null && (
        <button class={`icon-btn ${torch ? 'on' : ''}`} aria-label="Licht" onClick={() => { setTorch(!torch); setTrack({ torch: !torch }); }}>💡</button>
      )} />
      <div class="viewfinder" ref={box}>
        <video ref={video} playsInline muted autoPlay />
        <svg class="overlay" viewBox="0 0 1 1" style={{ inset: `${INSET * 100}%` }}>
          {POLYS.map((p) => <polygon points={p.map(([x, y]) => `${x},${y}`).join(' ')} />)}
        </svg>
        {message && <div class="vf-message">{message}</div>}
        {busy && <div class="vf-busy"><div class="spinner" />{busy}</div>}
      </div>
      <p class={`hint ${glare ? 'warn' : ''}`}>
        {glare ? 'Spiegelung erkannt – Handy etwas kippen oder Lampe meiden.'
          : 'Sechsecke auf die Plättchen legen. Handy parallel zum Brett halten.'}
      </p>
      {zoom && (
        <label class="zoom">
          <span>Zoom</span>
          <input type="range" min={zoom.min} max={zoom.max} step={zoom.step || 0.1} value={zoom.value}
            onInput={(e) => { const v = +(e.target as HTMLInputElement).value; setZoom({ ...zoom, value: v }); setTrack({ zoom: v }); }} />
        </label>
      )}
      {error && <p class="error">{error}</p>}
      <div class="camera-actions">
        <label class="btn small">
          Foto wählen
          <input type="file" accept="image/*" hidden onChange={pickFile} />
        </label>
        <button class="shutter" onClick={capture} disabled={state !== 'live' || !!busy} aria-label="Zählen" />
        <span class="spacer" />
      </div>
    </main>
  );
}
