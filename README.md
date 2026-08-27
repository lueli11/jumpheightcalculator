# jumpheightcalculator

Misst Flugzeit, Absprunggeschwindigkeit und Sprunghöhe aus einem Handyvideo.

## Deployen

    docker compose up --build -d

Danach: Weboberflaeche auf Port **3050**, API auf Port **8000** (Doku unter
`/docs`). Beide sind verstellbar:

    WEB_PORT=3050 API_PORT=8000 docker compose up --build -d

Port 3000 ist bewusst nicht die Voreinstellung - auf einem Dokploy-Host läuft
dort Dokploy selbst. Wer `API_PORT` ändert, muss auch `NEXT_PUBLIC_API_BASE`
setzen, sonst sucht der Browser die API weiterhin auf 8000.
Vier Container: `web`, `api`, `worker`, `redis`. Videos und Datenbank liegen im
Volume `jumpheightcalculator-data`.

Der Browser spricht die API **direkt** auf Port 8000 an, nicht über die
Web-App. Läuft alles auf demselben Host, funktioniert das ohne Konfiguration.
Liegt die API woanders, setze `NEXT_PUBLIC_API_BASE` beim Bauen des
`web`-Containers.

Beide Ports müssen vom Handy aus erreichbar sein.

## Ohne Docker

    py -3.12 -m venv .venv
    .venv/Scripts/pip install -e ".[dev]"
    .venv/Scripts/python -m jumpheightcalculator analyze VIDEO.MOV --jump-type cmj --mass-kg 65

Weboberflaeche lokal, in drei Terminals:

    .venv/Scripts/python -m uvicorn jumpheightcalculator.api:app --port 8000
    .venv/Scripts/python scripts/local_worker.py
    cd web && npm run dev

Dann http://localhost:3050 . `local_worker.py` ersetzt den arq-Worker auf
Maschinen ohne Redis.

## Aufnehmen

Slo-Mo ab 120 fps (240 ist besser), kurze Belichtung, Handy quer auf ein Stativ
so hoch wie möglich, 3-4 m Abstand, senkrecht zur Sprungebene, ganzer Körper
im Bild, enge Kleidung, kontrastreicher Boden.

## Kameraposition

Abstand und Kamerahöhe sind **egal**. Das ist keine Behauptung, sondern
gerechnet: eine Lochkamera-Simulation über eine echte Wurfparabel, durch
denselben Fit und dieselbe Selbstkalibrierung (R5) wie ein echtes Video.

    Abstand 2 - 10 m          Abweichung 0.0000 %
    Kamerahöhe 0.05 - 2.5 m   Abweichung 0.0000 %
    Neigung 0 - 30 Grad       Abweichung max. +0.04 %

Der Grund: R5 leitet den Massstab aus der Krümmung derselben Kurve ab. Ein
konstanter Massstabsfehler kürzt sich dabei heraus - `h = g·v0²/(2a²)` bleibt
unverändert, wenn v0 und a mit demselben Faktor skalieren.

Auch die Genauigkeit leidet kaum. Mit 2 px Keypoint-Rauschen bei 120 fps:

    Abstand   Person im Bild   Streuung
      2 m        1225 px        0.06 cm
      6 m         408 px        0.17 cm
     14 m         175 px        0.39 cm

Der Fit mittelt über rund hundert Flugframes, deshalb faellt Rauschen wenig ins
Gewicht.

**Was nicht egal ist: eine Massstabsänderung *während* des Sprungs.** Wer auf die
Kamera zu springt, wird im Bild grösser, und dann gilt kein einziger Massstab
mehr für die ganze Flugbahn. Genau das prüft die App an Ober- und
Unterschenkel - Knochen wachsen nicht. Ab 12 % Änderung gibt es keinen Wert,
sondern den Grund. An einem abgelehnten Video waren es 33 %.

## Das Gewicht

Das Gewicht ist **optional**. Es wird für genau zwei Grössen gebraucht:

    J = m · v₀                          Absprungimpuls        [Ns]
    F̄ = m · v₀ / t_abdruck + m · g      mittlere Kraft        [N]

Höhe, Absprunggeschwindigkeit und Flugzeit kommen ohne aus.

**Wattzahlen gibt es bewusst nicht.** Peak Power braucht den Kraft-Zeit-Verlauf
einer Kraftmessplatte; aus Video ist nur der Mittelwert über die Abdruckphase
seriös. Regressionsformeln wie Sayers oder Lewis sind an bestimmten
Populationen kalibriert und bleiben draussen - eine Wattzahl mit ±20 %
Unsicherheit sieht präziser aus, als sie ist.

## Was gemessen wird

`E2` ist der Referenzwert: der Scheitel der gefitteten Hüfte-Parabel. `E1` aus
der Flugzeit überschätzt bauartbedingt, weil man gestreckt abspringt und
gebeugt landet - `Delta` sagt, um wie viel. `E3` fittet dieselbe Bewegung auf dem
Ganzkörper-Schwerpunkt; `|E2-E3|` ist das Gütemass. `E4` nutzt nur die
Fallphase.

Keine Wattzahlen: Peak Power braucht den Kraft-Zeit-Verlauf einer
Kraftmessplatte. Impuls und mittlere Kraft gibt es mit Körpermasse.

Bei schlechter Aufnahme gibt es keine Zahl, sondern den Grund.

## Tests

    .venv/Scripts/python -m pytest -q -m "not slow"   # schnell
    .venv/Scripts/python -m pytest -q                 # inkl. Videogenerierung
    cd web && npx playwright test                     # braucht laufende Dienste
