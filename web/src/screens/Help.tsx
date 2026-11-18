import { Header } from '../ui/Header';

export function Help() {
  return (
    <main class="screen">
      <Header title="So funktioniert's" />
      <section class="card prose">
        <ol>
          <li><b>Brett fotografieren:</b> Handy parallel über das Brett halten, bis die Sechsecke auf den Plättchen liegen. Der Zoom-Regler hilft, wenn das Brett zu klein oder zu groß ist.</li>
          <li><b>Spiegelungen vermeiden:</b> Lampen spiegeln sich leicht auf den Plättchen. Erscheint der Hinweis „Spiegelung erkannt", das Handy leicht kippen oder einen Schritt zur Seite gehen.</li>
          <li><b>Prüfen:</b> Unsichere Felder sind gelb markiert. Antippen, um einzelne Zahlen zu korrigieren – die Punkte werden sofort neu berechnet.</li>
          <li><b>Ausrichtung:</b> Lag das Raster daneben, im Reiter „Foto &amp; Ausrichtung" die Sechsecke verschieben und „Neu zählen" tippen.</li>
          <li><b>Spielrunden:</b> Mit „Neue Spielrunde" zählt ihr mehrere Runden (z. B. Best of 3) und seht den Gesamtstand.</li>
        </ol>
        <p class="muted">Tipp: Über das Teilen-Menü des Browsers „Zum Home-Bildschirm" wählen – dann startet die App wie eine normale App und funktioniert auch offline.</p>
      </section>
    </main>
  );
}
