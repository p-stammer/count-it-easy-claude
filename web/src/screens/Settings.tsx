import { useEffect, useState } from 'preact/hooks';
import { clearSamples, countSamples, exportBackup, getSetting, importBackup, listSamples, setSetting } from '../db';
import { Header } from '../ui/Header';

function download(name: string, text: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const blobToDataUrl = (b: Blob) => new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(r.result as string); r.readAsDataURL(b); });

export function Settings() {
  const [collect, setCollect] = useState(false);
  const [samples, setSamples] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  const [persisted, setPersisted] = useState<boolean | null>(null);
  useEffect(() => {
    getSetting('collectSamples', false).then(setCollect);
    countSamples().then(setSamples);
    navigator.storage?.persisted?.().then(setPersisted);
  }, []);
  const stamp = new Date().toISOString().slice(0, 10);

  return (
    <main class="screen">
      <Header title="Einstellungen" />
      <section class="card">
        <h2>Datensicherung</h2>
        <p class="muted">Alle Daten liegen nur auf diesem Gerät{persisted === false ? ' – der Browser könnte sie bei Speichermangel löschen' : ''}. Exportiere ab und zu eine Sicherung.</p>
        <div class="row2">
          <button class="btn" onClick={async () => { download(`count-it-easy-${stamp}.json`, JSON.stringify(await exportBackup())); }}>Exportieren</button>
          <label class="btn">Importieren
            <input type="file" accept="application/json,.json" hidden onChange={async (e) => {
              const f = (e.target as HTMLInputElement).files?.[0];
              if (!f) return;
              try { await importBackup(JSON.parse(await f.text())); setMsg('Sicherung importiert.'); } catch (x) { setMsg((x as Error).message); }
            }} />
          </label>
        </div>
        {msg && <p class="muted">{msg}</p>}
      </section>
      <section class="card">
        <h2>Erkennung verbessern</h2>
        <label class="toggle">
          <input type="checkbox" checked={collect} onChange={async (e) => { const v = (e.target as HTMLInputElement).checked; setCollect(v); await setSetting('collectSamples', v); }} />
          <span>Fotos gespeicherter Spiele lokal aufbewahren</span>
        </label>
        <p class="muted">Die Fotos verlassen das Gerät nicht. Du kannst sie samt korrigierter Ergebnisse exportieren und damit das Modell nachtrainieren (siehe README).</p>
        <div class="row2">
          <button class="btn" disabled={!samples} onClick={async () => {
            const all = await listSamples();
            const out = await Promise.all(all.map(async (s) => ({ ...s, image: await blobToDataUrl(s.image) })));
            download(`cie-samples-${stamp}.json`, JSON.stringify({ app: 'count-it-easy', kind: 'samples', samples: out }));
          }}>{samples} Fotos exportieren</button>
          <button class="btn danger" disabled={!samples} onClick={async () => { await clearSamples(); setSamples(0); }}>Fotos löschen</button>
        </div>
      </section>
      <section class="card">
        <h2>Über</h2>
        <p class="muted">Count it Easy! {__APP_VERSION__} · Modell {__MODEL_VERSION__}. Die Erkennung läuft vollständig auf deinem Gerät. „Take it Easy!" ist ein Spiel von Peter Burley; diese App ist ein inoffizielles Hobbyprojekt.</p>
      </section>
    </main>
  );
}
