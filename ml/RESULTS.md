# Ergebnisse

Gemessen auf den **6 zurückgehaltenen Testbrettern** (t2, t6, t7, t8, t38, t39), unter 25 Aufnahme-Bedingungen.
Beide Modelle laufen durch die komplette App-Pipeline: Ziffern-Erkennung, Plättchen-Eindeutigkeit und automatische
Feinausrichtung. Das alte Modell profitiert hier also schon von den neuen Verbesserungen. In der alten App war es
schwächer (ohne Eindeutigkeits-Regel: 81 % statt 88 % komplett richtige Bretter).

Reproduzieren:

```bash
python scripts/evaluate.py --legacy models/legacy --interp nearest --refine   # altes Modell (2022)
python scripts/evaluate.py --model ../web/public/models/digits.onnx --refine   # neues Modell v2.0
```

| Bedingung | alt: Ziffern | alt: Brett komplett | alt: Punkte korrekt | **v2: Ziffern** | **v2: Brett komplett** | **v2: Punkte korrekt** | v2: Fehler markiert |
|---|---|---|---|---|---|---|---|
| clean | 100.0 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| shift 0.020 | 100.0 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| shift 0.035 | 99.4 % | 96 % | 100 % | 99.9 % | 100 % | 100 % | 100 % |
| scale 0.94 | 100.0 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| scale 1.06 | 100.0 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| scale 0.9 | 99.4 % | 83 % | 100 % | 99.1 % | 83 % | 100 % | 100 % |
| scale 1.1 | 100.0 % | 100 % | 100 % | 99.7 % | 100 % | 100 % | 100 % |
| rotate 3.0 | 100.0 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| glare | 98.3 % | 67 % | 92 % | 100.0 % | 100 % | 100 % | 100 % |
| strong glare | 90.4 % | 17 % | 25 % | 98.9 % | 88 % | 88 % | 100 % |
| dark+blurry | 99.9 % | 100 % | 100 % | 100.0 % | 100 % | 100 % | 100 % |
| **Mittel** | 98.9 % | 88 % | 92 % | **99.8 %** | **97 %** | **99 %** | 100 % |

- **Brett komplett:** Alle 19 Plättchen wurden richtig erkannt.
- **Punkte korrekt:** Die Punktzahl stimmt, auch wenn ein Fehler keine Linie betrifft.
- **Fehler markiert:** Anteil der falsch erkannten Felder, die die App gelb als „unsicher" markiert. Schwelle 0,75,
  kalibriert auf den Validierungsbrettern. Dabei werden im Schnitt etwa 0,5 korrekte Felder pro Brett unnötig markiert.

## Was die Verbesserung bringt

| Maßnahme | Effekt |
|---|---|
| Plättchen-Eindeutigkeit (27 Plättchen, jedes nur einmal) | Altes Modell bei Spiegelung: 42 % → 67 % komplett richtige Bretter, ohne neues Training |
| Training mit Spiegelungs-, Ausrichtungs- und Kamera-Augmentierung | Spiegelung: 67 % → 100 %; starke Spiegelung: 17 % → 88 % |
| Konfidenz-Markierung | 100 % der verbleibenden Fehler werden zur Prüfung markiert (altes Modell: 82 %; bei Fehlausrichtung war es falsch und trotzdem sicher) |
| Ein Netz mit 9 Ausgängen statt drei Netzen | gleiche Größe (78 KB) und Rechenzeit pro Ziffer, ein Modell weniger auszuliefern |

## Einschränkungen (ehrlich)

- **Nur 6 Testbretter:** Die Prozentwerte sind entsprechend grob.
- **Spiegelungen sind synthetisch:** Gaußsche Glanzlichter imitieren echte Reflexionen nur. Die eigentliche Probe ist
  der Einsatz am echten Spieltisch. Dafür gibt es in der App die Funktion zum Sammeln korrigierter Fotos (siehe README).
- **Leichter Vorteil für das alte Modell:** Es hat die Testbretter früher als Validierung (Early Stopping) genutzt.
- **Brett deutlich kleiner als das Overlay (−10 %):** Das bleibt der schwächste Fall bei der Ziffern-Genauigkeit.
  Die Punktzahl stimmt trotzdem dank Eindeutigkeit und Feinausrichtung. Der Zoom-Regler in der App hilft zusätzlich.

## Training

`python scripts/train.py --out runs/v2 --epochs 45 --workers 1`: ca. 50 Minuten auf 2 CPU-Kernen, Seed 0.
Das Log liegt in `results/train_log_v2.json`. Verlauf auf den Validierungsbrettern (Mittel über alle Bedingungen):
Epoche 10: 99,0 % Ziffern / 83 % Bretter → Epoche 20: 99,7 % / 97 % → Epoche 45: 99,9 % / 99 %.
