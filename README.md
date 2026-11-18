# Count it Easy!

Zählt die Punkte beim Brettspiel **Take it Easy!** automatisch: Handy über das Brett halten, auslösen, fertig.
Läuft als Web-App (PWA) auf Android und iPhone, ist installierbar und funktioniert offline. Die Bilderkennung
läuft vollständig auf dem Gerät, es werden keine Fotos hochgeladen.

```
web/     Web-App (Vite + Preact + TypeScript, onnxruntime-web)
ml/      ML-Pipeline (Python): Geometrie, Punkte, Dekodierung, Training, Evaluation, ONNX-Export
data/    39 Brettfotos mit Ground Truth (boards.json) + Zusatz-Crops
legacy/  Ursprüngliche Kivy-App, Notebook und Modelle (2022), nur zur Referenz
```

## So funktioniert die Erkennung

1. **Ein Rahmen für alles.** Die Lage der 19 Sechsecke wird durch eine einzige, parametrische Geometrie beschrieben
   ([`ml/src/cie/geometry.py`](ml/src/cie/geometry.py) → exportiert nach `web/src/core/geometry.json`).
   Kamera-Overlay, Trainings-Ausschnitte und Erkennung in der App benutzen dieselben Zahlen.
2. **Ziffern klassifizieren.** Für jedes Feld werden die drei Ziffern ausgeschnitten (57 Ausschnitte, 128×128).
   Ein kleines CNN (~18k Parameter, 78 KB) liefert pro Ausschnitt Wahrscheinlichkeiten für die drei möglichen Werte.
3. **Spielregeln nutzen.** Jedes der 27 Plättchen existiert genau einmal. Die App wählt die wahrscheinlichste
   Belegung *verschiedener* Plättchen (Hungarian-Algorithmus). Einzelne Lesefehler werden so automatisch korrigiert.
4. **Ausrichtung nachjustieren.** Bei unsicheren Feldern testet die App kleine Verschiebungen und Skalierungen
   und nimmt das wahrscheinlichste Ergebnis.
5. **Mensch im Loop.** Unsichere Felder sind markiert und lassen sich per Antippen korrigieren. Die Punkte
   werden live neu berechnet.

## Web-App

```bash
cd web
npm install
npm run dev        # http://localhost:5173/count-it-easy/
npm test           # Paritätstests gegen die Python-Referenz
npm run build      # Produktions-Build nach web/dist
```

Für Kamera-Tests am Handy braucht es HTTPS: entweder das deployte GitHub-Pages-Build nutzen oder
`npm run dev -- --host` mit einem Tunnel (z. B. `cloudflared tunnel --url http://localhost:5173`).

**Deployment:** Der Workflow `.github/workflows/deploy.yml` baut bei jedem Push auf `main` und veröffentlicht auf
GitHub Pages (Einstellungen → Pages → Source: *GitHub Actions*). Bei einem privaten Repo braucht GitHub Pages einen
bezahlten Plan; alternativ lässt sich `web/dist` auf Cloudflare Pages oder Netlify hosten (dann mit `BASE=/`
bauen).

**Funktionen:** Brett zählen, Felder korrigieren, Ausrichtung manuell anpassen, Foto aus der Galerie, beliebig viele
Spieler, Spielrunden mit frei wählbarer Rundenzahl (z. B. Best of 3), Statistik mit Verlauf, JSON-Sicherung und
-Wiederherstellung. Optional lassen sich Fotos zum Nachtrainieren sammeln.

## ML-Pipeline

```bash
cd ml
pip install -e '.[train,dev]'
pytest -q                                                    # Kernlogik + Konsistenz aller 39 Bretter
python scripts/train.py --out runs/v2 --epochs 45            # ~1 h auf 2 CPU-Kernen
python scripts/evaluate.py --model runs/... --split test     # Board-Level-Evaluation
python scripts/export_onnx.py runs/v2/best.pt --version 2.0  # -> web/public/models/
python scripts/export_web_assets.py                          # Geometrie + Test-Fixtures für die Web-App
```

**Daten:** `data/boards.json` enthält 39 Bretter (Labels aus dem ursprünglichen Notebook, drei Labelfehler korrigiert)
mit festem Split: **28 train / 5 val / 6 test**. Die Testbretter (t2, t6, t7, t8, t38, t39) sind dieselben, die das
alte Modell als Validierung nutzte. Damit ist der Vergleich alt gegen neu fair.

**Training:** Jede Epoche rendert jedes Trainingsbrett mehrfach neu: mit zufälliger Fehlausrichtung (Verschiebung,
Skalierung, Drehung) und mit Bildstörungen, vor allem **Spiegelungen** (gesättigte Glanzlichter, breite
Reflexionen), Schatten, Weißabgleich, Unschärfe, Rauschen und JPEG-Artefakten. Ein gemeinsames Netz mit 9 Ausgängen
ersetzt die drei getrennten Modelle.

**Evaluation** ([`ml/src/cie/evaluate.py`](ml/src/cie/evaluate.py)) misst auf ganzen Brettern unter 25 Bedingungen:
sauber, verschoben, skaliert, gedreht, Spiegelung, starke Spiegelung, dunkel und unscharf.
Die Ergebnisse stehen in [`ml/RESULTS.md`](ml/RESULTS.md).

## Modell mit eigenen Fotos verbessern

1. In der App unter *Einstellungen → Erkennung verbessern* das Aufbewahren von Fotos einschalten.
2. Normal spielen und Fehler korrigieren; gespeichert wird das korrigierte Ergebnis.
3. *Fotos exportieren* → `python ml/scripts/import_samples.py cie-samples-….json`
4. Neu trainieren, evaluieren, exportieren.
