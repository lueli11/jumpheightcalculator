# jumpheightcalculator

Sprunghoehe, Flugzeit und Absprunggeschwindigkeit aus einem Handyvideo. Python
3.12 (FastAPI + arq-Worker + Redis + SQLite, MediaPipe Pose, ffmpeg), Weboberflaeche
in Next.js 15.

## Die sieben Regeln

Sie stehen als `R1`..`R7` in den Kommentaren und sind der Grund, weshalb der Code
so aussieht, wie er aussieht. Wer eine davon aufweicht, macht die Messung kaputt,
ohne dass ein Test zwingend rot wird - **erst pruefen, ob die Regel noch gilt.**

- **R1 - Zeit kommt aus dem Container.** Die Zeitachse sind die PTS aus
  `ffprobe`, nie `n/fps`. Handys zeichnen variabel auf. Keine Zeitstempel, keine
  Messung (`ingest.py`).
- **R2 - E1 muss ueberschaetzen.** Man springt gestreckt ab und landet gebeugt,
  also liegt die Flugzeitmethode ueber der Huefte. Tut sie es nicht, ist der
  Absprung falsch erkannt - das ist eine Verweigerung, kein Rundungsfehler.
- **R3 - Timing in voller Aufloesung.** Das Kontaktsignal entsteht in
  `pixels.py` auf den Originalpixeln; herunterskaliert wird nur fuer die Pose.
- **R4 - Randframes raus.** Der Frame am Uebergang ist weder Kontakt noch
  sauberer Flug und verzieht den Fit.
- **R5 - Selbstkalibrierung.** Der Massstab kommt aus der Kruemmung der
  Flugparabel (`a_px` muss 9.81 m/s^2 entsprechen), nicht vom Nutzer. Kein
  Massband, kein Referenzobjekt. **Darum sind Kameraabstand und -hoehe egal** -
  ein konstanter Massstabsfehler kuerzt sich in `h = g*v0^2/(2a^2)` heraus.
  Nicht egal ist eine Massstabsaenderung *waehrend* des Sprungs.
- **R6 - Impuls und Kraft nur mit Masse.** `J = m*v0`,
  `F = m*v0/t_abdruck + m*g`. **Keine Wattzahlen** - Peak Power braucht den
  Kraft-Zeit-Verlauf einer Kraftmessplatte, Regressionsformeln (Sayers, Lewis)
  sind an fremden Populationen kalibriert. Das ist eine Entscheidung, kein
  fehlendes Feature.
- **R7 - lieber kein Wert als ein falscher, und immer mit Grund.** Jede
  Verweigerung (`pipeline.Refusal`, `IngestError`, `FitError`) nennt die
  gemessene Zahl, die Grenze und was der Nutzer anders machen soll.

## Aufbau

```
ingest.py    ffprobe: PTS, Rotation, fps            -> IngestError
pixels.py    Kontaktsignal in voller Aufloesung     (R3)
pose.py      MediaPipe-Keypoints auf dem Proxy
fusion.py    Kontaktsignal + Koerperbewegung gegeneinander
physics.py   robuster Parabelfit, Selbstkalibrierung (R5)
pipeline.py  die vier Schaetzer, alle Verweigerungen, Warnungen
api.py       FastAPI; Analyse laeuft **nie** im Request
worker.py    arq; scripts/local_worker.py ersetzt ihn ohne Redis
```

`E2` (Huefte) ist der Referenzwert, `|E2-E3|` das Guetemass. `ball.py` ist die
Gegenprobe mit einem geworfenen Ball: dort gilt R2 *nicht*, die Bahn ist eine
exakte Parabel - laeuft die Ballmessung schief, liegt der Fehler in der Physik
und nicht in der Pose.

## Konventionen

- **Kommentare und Docstrings sind transliteriert** (`noetig`, `waehrend`,
  `faellt`) - nicht schoen, aber durchgaengig. **Alles, was der Nutzer sieht,
  hat echte Umlaute**: Verweigerungen, Warnungen, Weboberflaeche. Beim Bearbeiten
  einer Meldung darauf achten.
- Schweizer Schreibung: `Massstab`, `gross`, `heisst` - kein Eszett.
- Kommentare erklaeren das **Warum**, nie das Was. Viele halten einen Fehlversuch
  fest ("die erste Fassung testete X, das sagt nichts ueber Y") - solche
  Kommentare nicht wegkuerzen, sie sind der teuerste Teil der Datei.
- Keine Abstraktionsschicht mit genau einem Implementierer (siehe `db.py`).

## Entwickeln

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q -m "not slow"    # ~2-4 min
.venv/Scripts/python -m pytest -q                  # inkl. synthetischer Videos
cd web && npx playwright test                      # braucht laufende Dienste
```

Die Tests heissen `test_p0_gate.py` .. `test_p2_gate.py` und pruefen die Regeln
direkt (R5 ohne Massband, R1 mit gejittertem Video, R2 auf einem simulierten
Sprung). Ein neuer Schaetzer oder eine geaenderte Konstante gehoert dort hinein,
nicht in einen neuen Test daneben.

Lokal ohne Redis in drei Terminals:

```bash
.venv/Scripts/python -m uvicorn jumpheightcalculator.api:app --port 8000
.venv/Scripts/python scripts/local_worker.py
cd web && npm run dev
```

## Deployment (Dokploy)

Vier Container (`web`, `api`, `worker`, `redis`), Daten im Volume
`jumpheightcalculator-data`. **Push nach `main` deployt automatisch** - es gibt
keinen Knopf und keinen SSH-Zugang auf den Host.

Vier Dinge, die je einen kaputten Deploy gekostet haben:

- **Port 3000 gehoert auf einem Dokploy-Host Dokploy selbst.** Web laeuft auf
  **3050**, API auf 8000 (`WEB_PORT`/`API_PORT`).
- **Der Next-Container lauscht innen wie aussen auf 3050**, und
  `HOSTNAME=0.0.0.0` ist Pflicht. Traefik verbindet sich ueber das Docker-Netz
  auf die Container-IP; ohne beides gibt es 502, obwohl Container, Netz und
  Domain-Regel stimmen.
- **`NEXT_PUBLIC_API_BASE` wird zur Bauzeit eingebacken.** Der Browser spricht
  die API **direkt** auf 8000 an, nicht ueber die Web-App. Andere API-Adresse
  heisst: `web`-Container neu bauen.
- **MediaPipe braucht `libegl1` und `libgles2` auch headless.** Fehlen sie, baut
  das Image sauber durch und erst die erste echte Analyse stirbt - darum wird
  der Import schon im Dockerfile geprueft, genau wie die Existenz des
  Pose-Modells in site-packages (nicht im Quellbaum, das sagt nichts aus).

Keine Build-Artefakte einchecken. `build/` lag einmal im Repo und zog den
30-MB-`pose_landmarker_heavy.task` mit, den `.gitignore` direkt daneben
ausschliesst - 35 MiB je Klon. `build/`, `dist/`, `web/tsconfig.tsbuildinfo`
stehen jetzt in `.gitignore` **und** `.dockerignore`.
