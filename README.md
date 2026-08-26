# jumpheightcalculator

Misst Flugzeit, Absprunggeschwindigkeit und Sprunghöhe aus einem Handyvideo.

## Deployen

    docker compose up --build -d

Danach: Web-Oberflaeche auf Port **3050**, API auf Port **8000** (Doku unter
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
