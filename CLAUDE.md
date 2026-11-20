# jumpheightcalculator

Misst aus einem Handyvideo die **Flugzeit**, die **Absprunggeschwindigkeit** und
die **Sprunghöhe** eines Vertikalsprungs.

**Das ist ein Messgerät, keine CRUD-App.** Der Wert steckt in der
Erkennungsgenauigkeit, nicht im Web-Teil drumherum. Eine App, die plausibel
aussehende falsche Zahlen liefert, ist schlechter als keine — jemand richtet
sein Training danach aus.

Daraus folgt die wichtigste Arbeitsregel: **lieber keine Zahl als eine falsche.**
Wo die Datenlage nicht trägt, gibt es eine Diagnose und keinen Wert. Wer eine
Verweigerung „wegoptimiert", macht das Projekt kaputt.

Das ursprüngliche Konzept liegt als `konzept.html` im Repo (Arbeitstitel
„Absprung" / JumpLab). Es ist die fachliche Begründung für fast alles hier.

---

## Vor jeder Änderung lesen

Die Kommentare in `pipeline/limits.py`, `pipeline/measurement.py`, `physics.py`
und `pixels.py` halten fest, **was
schon gemessen und verworfen wurde** — mit Zahlen. Mehrere naheliegende
„Verbesserungen" wurden ausprobiert und waren nachweislich schlechter. Die
Begründungen stehen dort, wo man sie brauchen würde, nicht in einem Dokument,
das niemand liest.

Wenn eine Änderung eine dieser Begründungen ungültig macht: **nachmessen, nicht
argumentieren.** Es gibt einen synthetischen Videogenerator mit bekannter
Wahrheit, damit genau das billig ist.

---

## Schnellstart

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m jumpheightcalculator analyze VIDEO.MOV --jump-type cmj --mass-kg 65
```

**Python 3.12 ist Pflicht.** MediaPipe deklariert nur 3.9–3.12; auf 3.13/3.14
gibt es keine getesteten Wheels.

Weboberfläche lokal, drei Terminals:

```bash
.venv/Scripts/python -m uvicorn jumpheightcalculator.api:app --port 8000
.venv/Scripts/python scripts/local_worker.py
cd web && npm run dev
```

Dann `http://localhost:3050`. `local_worker.py` ersetzt den arq-Worker auf
Maschinen ohne Redis — er pollt die Datenbank und ruft dieselbe Funktion, die
arq aufrufen würde.

Tests:

```bash
.venv/Scripts/python -m pytest -q -m "not slow"   # ~2 min
.venv/Scripts/python -m pytest -q                 # inkl. Videogenerierung, ~5 min
cd web && npx playwright test                     # braucht laufende Dienste
```

---

## Modulkarte

| Datei | Stufe | Zuständigkeit |
|---|---|---|
| `__init__.py` | — | `G = 9.81`, `PIPELINE_VERSION` |
| `ingest.py` | S0 | Zeitstempel aus dem Container, Decode, Skalierung, Zuschnitt, Plausibilitätsgrenzen |
| `pose.py` | S1 | MediaPipe-Keypoints, Körperschwerpunkt, Körperskala, Bodenlinie, Savitzky-Golay |
| `pixels.py` | S2 | Kontaktsignale im Streifen unter dem Fussballen, Kamerabewegung |
| `fusion.py` | S3 | Normierung, Zwei-Zustands-HMM mit Viterbi, Sub-Frame-Schnittpunkte |
| `physics.py` | S4 | RANSAC-Parabel, Selbstkalibrierung, kubischer Residuentest |
| `pipeline/limits.py` | S4/S5 | alle Schwellen (Flugzeit, R², Perspektive, Kontaktsignal) mit ihrer Begruendung |
| `pipeline/track.py` | S4 | `TrackData` - NPZ mit Keypoints, Kontaktsignal, Zeitstempeln |
| `pipeline/result.py` | S5 | `Result` - Messwerte, Güte, Warnungen, Pipeline-Version |
| `pipeline/timebase.py` | S4/S5 | `Refusal`/`TrajectoryUnusable`, gestreckte Zeitachse, Massstabsgegenprobe |
| `pipeline/geometry.py` | S4 | grobes Flugfenster, Standphase, Skalen-Drift |
| `pipeline/extraction.py` | S4 | `extract` - Decode, Pose, Pixelzweig, Fusion (der teure Teil) |
| `pipeline/measurement.py` | S5 | `measure` - Fit, die vier Schätzer, Güte, alle Verweigerungen (der billige Teil) |
| `ball.py` | — | Kugel-Tracker für das synthetische Video und den Ballwurf-Test |
| `synth.py` | — | Testvideo-Generator mit bekannter Wahrheit |
| `api.py` | — | FastAPI |
| `worker.py` | — | arq-Worker, Vorschauvideo |
| `db.py` | — | SQLite-Schema |
| `web/` | — | Next.js: Upload und Review-Screen |

**`extract` und `measure` sind getrennt, und das ist wichtig.** `extract` ist der
teure Teil (Decode, MediaPipe, Pixelzweig, Fusion — Minuten) und schreibt einen
`TrackData`-NPZ. `measure` rechnet daraus in Millisekunden. Eine
±1-Frame-Korrektur im Review-Screen darf nicht bedeuten, dass MediaPipe erneut
über das ganze Video läuft.

---

## Die sieben Regeln

### R1 — Echte Frame-Zeitstempel, niemals die nominelle Framerate
`ingest.probe()` liest die Präsentationszeitstempel jedes Frames per `ffprobe`.
Gerechnet wird ausschliesslich mit `t[n]`, nie mit `n/fps`.

**Der Fehler skaliert quadratisch:** unter einer Zeitskalierung `t → k·t` gilt
`a_px → a_px/k²` und `v₀,px → v₀,px/k`, also `h → h·k²`. Ein 8-fach gestrecktes
Slo-Mo lieferte die 64-fache Höhe — lautlos.

Zwei Wächter dagegen:
1. Das mediane Frame-Intervall muss 20–1000 fps entsprechen.
2. Das gefittete Flugfenster muss in 0.15–1.10 s liegen.

**Negativprobe im Test:** derselbe Lauf mit angenommener nomineller Framerate
muss sichtbar danebenliegen (gemessen −30.6 %), sonst beweist der Test nicht,
dass die Zeitstempel überhaupt tragen. Gleichverteilter Jitter allein genügt als
Beweis **nicht** — er ist mittelwertfrei.

**Die eine erlaubte Ausnahme: `capture_fps`.** Ein verlangsamt exportiertes
Slo-Mo trägt die Abspielrate im Container; die echte Rate steht nirgends in der
Datei. `probe(path, capture_fps=…)` staucht die PTS um
`Container-Rate / Aufnahmerate`. Das ist **kein** Bruch von R1: gerechnet wird
weiter mit den echten, ungleich verteilten Zeitstempeln, nur auf der richtigen
Skala — ein konstanter Faktor lässt den Jitter unberührt (Test
`test_capture_fps_keeps_the_jitter`). Geprüft wird das Ergebnis, nicht die
Eingabe: `_check_timebase` läuft auf der **korrigierten** Achse.

Der Wert ist eine Tatsachenbehauptung des Nutzers, wie die Körpermasse — und
eine falsche Behauptung ist quadratisch falsch. Der Schutz wirkt **nur in eine
Richtung**: zu niedrig angegeben fängt die Flugzeitgrenze ab, zu hoch angegeben
liefert stillschweigend zu wenig (gemessen 10.00 statt 40.00 cm bei R² > 0.99).
Im Sprung-Pfad schliesst `_check_scale_plausible` das über die Körpergrösse, im
Ball-Pfad gibt es nichts. Darum steht neben jedem so gerechneten Ergebnis eine
Warnung.

### R2 — Die Höhe kommt aus der Hüfte, nicht aus der Flugzeit
`h = g·t²/8` setzt gleiche Schwerpunkthöhe bei Absprung und Landung voraus. Man
springt gestreckt ab und landet gebeugt; der Fehler ist systematisch. Der
Referenzwert ist der Scheitel der gefitteten Hüft-Parabel (**E2**).

### R3 — Der Knöchel liefert die Referenzhöhe, nie das Timing
Beim Abdruck streckt sich der Fuss: der Knöchel steigt bereits, während die Zehe
noch aufliegt. Das Timing kommt aus dem Pixelsignal am Auflagepunkt
(`pixels.contact_signals`), der Knöchel platziert nur den Streifen.

### R4 — Parabel fitten, nicht Frames zählen
`physics.fit_parabola` fittet per RANSAC über `[n0+1, n1−1]`; die Ereigniszeiten
sind Fliesskommazahlen. **Die Randframes bleiben draussen** — auf dem
synthetischen Video war ein einziger solcher Frame 0.14 % Höhenfehler
(Krümmung 9.797 statt 9.810).

Der RNG-Seed ist festgesetzt. Ohne ihn wäre dasselbe Video nicht bitgleich
reproduzierbar, und Gate P2.1 nicht erreichbar.

### R5 — Selbstkalibrierung über g
```
a_px = 2 · c2              [px/s²]
s    = 9.81 / a_px         [m/px]
h    = 9.81 · v0px² / (2 · a_px²)
```
Kein Massband, kein Referenzobjekt, keine Körpergrösse. **Das ist der Grund,
weshalb Kameraabstand und Kamerahöhe keine Rolle spielen** (siehe Messungen).

### R6 — Keine Wattzahlen
Peak Power braucht den Kraft-Zeit-Verlauf einer Kraftmessplatte. Impuls
(`J = m·v₀`) und mittlere Kraft (`F̄ = m·v₀/t_abdruck + m·g`) sind erlaubt und
brauchen Körpermasse. Sayers, Lewis und andere Regressionsformeln bleiben
draussen — sie sind an fremden Populationen kalibriert.

### R7 — Bei schlechter Güte den Grund nennen, keine Zahl erfinden
Alle Verweigerungen sind `pipeline.Refusal` mit einem Text, der sagt, **was der
Nutzer anders machen soll**. Vollständige Liste weiter unten.

---

## Die vier Höhenschätzer

| | Definition | misst |
|---|---|---|
| **E1** | `g·t_f²/8`, `t_f` aus dem Kontaktsignal | Vergleichswert zu anderen Apps; überschätzt bauartbedingt |
| **E2** | Scheitel der **Hüft**-Parabel über der Hüfthöhe bei t₀ | **Referenzwert der App** |
| **E3** | dasselbe auf einem eigenen Fit des **Ganzkörperschwerpunkts** (Winter 2009) | Uneinigkeit zu E2 → **hierauf gatet R7** |
| **E4** | `½·g·t_fall²`, Scheitel bis Landung | nur Fallphase; liegt systematisch bei E2 + Δ |

Dazu `Δ = (v_land² − v₀²)/(2g)` — die Absenkung des Schwerpunkts beim Landen.

### Warum E3 auf anderen Landmarks fittet

**Für jede Parabel gilt: Scheitelhöhe über dem Startpunkt ist identisch mit
`v(t₀)²/(2a)`.** Das ist eine Identität, keine zweite Herleitung. Ein Schätzer,
der aus **demselben Fit** abgeleitet wird, ist ein exaktes Duplikat von E2 und
kann per Konstruktion keinen Fit-Fehler aufdecken — numerisch bestätigt,
Differenz exakt null.

Nur weil E3 **andere Landmarks** fittet, ist der Abstand zu E2 überhaupt ein
Gütesignal.

**Die Spannweite über alle vier ist das falsche Fehlermass.** E1 und E4 weichen
von E2 um Δ ab, und Δ ist gemessene Physiologie, kein Messfehler. Wer auf die
Spannweite gatet, gatet auf die Landetechnik des Athleten.

**Verworfen:** die Schulter als E3. Sie geht beim Hochgreifen mit (2.6 %
Abweichung gemessen) — das wäre ein Fehlalarm bei jedem Reach-Jump. Der
Schwerpunkt weicht 0.6 % ab und hat das beste R².

**Ebenfalls verworfen:** der Schwerpunkt als *Ersatz* für die Hüfte bei E2.
Gemessen R² 0.99645 gegen 0.99585 — lohnt den Aufwand nicht.

---

## Zwei Messwege: Parabel, sonst Flugzeit

`Result.mode` sagt, welcher gelaufen ist.

- **`parabel`** — der Normalfall. Referenzwert ist **E2**, der Scheitel der
  gefitteten Hüft-Parabel.
- **`flugzeit`** — die Bahn war unbrauchbar. Geblieben ist **E1** aus
  `h = g·t²/8`; alle parabelabhängigen Felder sind `None`, `fallback_reason`
  sagt warum, und die erste Warnung erklärt es dem Nutzer.

**Der Unterschied, der darüber entscheidet:** in `h = g·t²/8` steht keine
einzige Pixelkoordinate. E1 braucht keinen Massstab, keine Pose, keine
Perspektive — nur die zwei Kontaktframes und eine richtige Zeitachse. Genau
deshalb überlebt es, was E2 umbringt: verrauschte Keypoints, schräge
Kameraachse, Massstabsänderung im Flug.

Umgekehrt gilt es nicht. Steht das **Timing** in Frage, fällt E1 mit — dann
bleibt es bei der Verweigerung. `TrajectoryUnusable` (Unterklasse von `Refusal`)
markiert die Fälle, die nur die Bahn betreffen; `measure` fängt sie ab, alles
andere geht durch.

**E1 überschätzt bauartbedingt** (R2), und um wie viel, sagt nur die Parabel —
die hier gerade weggefallen ist. Gemessen: saubere Referenz +3.0 cm,
Anlaufsprung mit tiefer Landung +24.3 cm. Der Wert kommt darum mit Ansage und
heisst im Review-Screen „nur Flugzeit", nicht „Sprunghöhe".

## Alle Verweigerungen (R7)

Die Reihenfolge in `measure()` ist bewusst gewählt: **Ursachen vor Symptomen.**

**Vier davon sind keine Verweigerung mehr, sondern der Rückfall auf die
Flugzeit** (1, 5, 6, 8 unten) — sie betreffen die Bahn, nicht das Timing.

1. **Massstabsänderung im Flug** (über 12 %, gleichmässig über alle vier
   Segmente) → Rückfall. Der Massstab ist hin, die Flugzeit nicht.
2. **Kontaktsignal und Körperbewegung widersprechen sich** (Verhältnis ausserhalb
   0.6–1.45) — typisch für einen Sprung mit Anlauf.
3. **Zu wenige Flugframes** (unter 10).
4. **Flugzeit unplausibel** (ausserhalb 0.15–1.10 s).
5. **R² der Hüft-Parabel unter 0.98** → Rückfall.
6. **Abstand E2 zu E3 über 4 cm** → Rückfall.
7. **E1 kleiner als E2** → Rückfall. Widerspruch in der Physik. Die Flugzeitmethode *muss*
   überschätzen; ist E2 grösser, ist die Hüfte angeblich höher gestiegen, als die
   Flugzeit erlaubt. In der Praxis heisst das: Absprung zu früh erkannt.
8. **Kubischer Anteil der Residuen signifikant** (t über 4) → Rückfall. —
   Objektivverzeichnung oder Bewegung zur Kamera.

Davor, schon in `extract`: **gestreckte Zeitachse** (`_refuse_if_stretched`) —
länger in der Luft als möglich, dabei eine saubere Parabel, und daraus eine
unmögliche Körpergrösse. Alle drei Bedingungen zusammen, sonst schlägt das
Weggehen nach der Landung an (auf der Referenz 2.58 s, aber R² 0.956).

Dazu Warnungen ohne Verweigerung: unter 120 fps, Kamerawackeln (Restversatz über
0.5 px), Person unter 140 px gross im Bild, ungleichmässige Beinlängenänderung im
Flug, und eine per `capture_fps` erklärte Zeitachse.

**Auch die Zeitachsen-Verweigerung ist keine Sackgasse.** Sie passiert schon in
`extract`, es gibt also keinen Track und nichts von Hand zu korrigieren. Der
Review-Screen zeigt dort stattdessen ein Feld für die Aufnahmerate;
`POST /api/analyses/{id}/capture-fps` setzt sie **am Video**, verwirft Track und
Vorschau (beide tragen die alte Achse) und reiht dieselbe Analyse neu ein. Ohne
das wäre der einzige Ausweg ein zweiter Upload derselben 20–140 MB.

**Eine Verweigerung ist keine Sackgasse.** Track und Vorschauvideo werden auch im
Fehlerfall in der Datenbank vermerkt; der Review-Screen zeigt dann Video,
Kontaktsignal und Korrekturknöpfe. Wer die Ereignisframes von Hand setzt, macht
aus `failed` wieder `ready`.

---

## Gemessene Befunde

### Kameraposition ist fast egal
Lochkamera-Simulation über eine echte Wurfparabel, durch denselben Fit und
dieselbe Selbstkalibrierung:

| | Abweichung |
|---|---|
| Abstand 2 – 10 m | 0.0000 % |
| Kamerahöhe 0.05 – 2.5 m | 0.0000 % |
| Neigung 0 – 30° | max. +0.04 % |

Grund: R5 leitet den Massstab aus der Krümmung derselben Kurve ab. Ein
konstanter Massstabsfehler kürzt sich in `h = g·v₀²/(2a²)` heraus.

Auch die Genauigkeit leidet kaum (2 px Keypoint-Rauschen, 120 fps):

| Abstand | Person im Bild | Streuung |
|---|---|---|
| 2 m | 1225 px | 0.06 cm |
| 6 m | 408 px | 0.17 cm |
| 14 m | 175 px | 0.39 cm |

### Was nicht geht
- **Bewegung zur Kamera.** Auf einem abgelehnten Video wurden die Oberschenkel
  im Flug 33 % länger im Bild. Knochen wachsen nicht — das ist Tiefenbewegung,
  und dann gilt kein einziger Massstab mehr für die Flugbahn. Ein sauberes Video
  streut 3 % ohne Trend.
- **Sprünge mit Anlauf.** Der Messstreifen liegt an der Standposition; beim
  Anlaufen ist die Person dort weg, bevor sie abspringt. Gemessen 137 gegen 76
  Flugframes, Flugzeit 1128 statt rund 600 ms.
- **30 fps.** Der Parabel-Fit senkt die Streuung, nicht die Bewegungsunschärfe.

### Empfindlichkeit
**E2 hängt mit rund 3.8 cm pro 10 ms am Absprungzeitpunkt.** Ein Frame bei
120 fps sind 8.3 ms. Das ist die harte Grenze der Genauigkeit und die
quantitative Begründung für R3 und R4.

### Referenzmessung
`IMG_5418.MOV` (120 fps, Kamera seitlich): E2 = 74.4 cm, v₀ 3.82 m/s, Flugzeit
779 ms, Δ +2.1 cm. Auf Server und Laptop **bitgleich**. Die gefittete Parabel
liegt im Mittel 1.81 px auf der Hüfte (bei 960 px Bildhöhe).

---

## Fallstricke

- **`ffmpeg` beim Decodieren braucht `-fps_mode passthrough`** (Ausgabeoption,
  hinter `-i`). Ohne sie wandelt ffmpeg selbst nach konstanter Framerate und
  dupliziert: aus 135 Frames wurden **224 422**. Das ist dieselbe stille
  Frame-zu-Zeit-Verschiebung, wegen der `cv2.VideoCapture` ausscheidet. Der
  Abgleich der Frameanzahl gegen die PTS-Liste fängt es ab.
- **Der ffmpeg-`concat`-Demuxer ignoriert `duration` bei Bild-Input** und
  kollabiert aufs 25-fps-Standardraster. `synth.py` nutzt deshalb **PyAV** und
  setzt die PTS je Frame direkt (Round-Trip auf 0.73 µs genau).
- **MediaPipe im IMAGE-Modus, nie VIDEO.** Der VIDEO-Modus glättet zeitlich und
  dämpft genau die Krümmung, die R4 fittet. Ausserdem nimmt `detect_for_video`
  ganzzahlige Millisekunden — bei 240 fps sind das 4.17 ms je Frame.
- **Bild-Landmarks, nicht World-Landmarks.** Gemessen wird in Pixeln, der
  Massstab kommt aus R5.
- **Savitzky-Golay vor dem Fit ist schädlich.** Die Höhe ändert sich um 0.03 cm
  (nichts), aber die Residuen-Autokorrelation steigt von 0.75 auf 0.90 und der
  Verzeichnungstest schlägt falsch an. Der Least-Squares-Fit mittelt das Rauschen
  ohnehin über hundert Frames.
- **Der kubische Test ist um Autokorrelation korrigiert** (`n_eff = n(1−ρ)/(1+ρ)`).
  Eine gewöhnliche t-Statistik setzt unabhängige Residuen voraus; bei einer
  Trajektorie sind sie das nie (gemessen ρ = 0.75), und ein naiver Test schlägt
  an, wo nichts ist.
- **R² ist blind für Zeitbasis-Fehler.** Die um 30 % falsche Rechnung hatte
  R² = 0.99997 — besser als die richtige. R7 darf nie auf R² allein gaten.
- **Arbeitsauflösung an eine Zielbreite binden (540 px), nie fester Faktor.** Ein
  über WhatsApp geschicktes Video kommt mit 464 px Breite an; ein Viertel davon
  sind 116 px, und darauf findet MediaPipe nichts Brauchbares. Hochskaliert wird
  nie.
- **`pyproject.toml` braucht `[tool.setuptools.package-data]`** für
  `models/*.task`, sonst landet das Pose-Modell nicht in site-packages. Der
  Docker-Guard prüft die **installierte** Stelle über den Import — die erste
  Fassung prüfte den Quellbaum und war damit wertlos.
- **MediaPipe braucht `libegl1` und `libgles2`** auch headless. Das Dockerfile
  importiert MediaPipe beim Bauen, damit eine fehlende Systembibliothek den Build
  kippt und nicht den ersten Job des Nutzers.
- **iOS zeigt ein `<video>` erst nach einmaligem Abspielen.** `currentTime` allein
  malt kein Bild, die Fläche bleibt schwarz. Nach dem Laden kurz anspielen und
  pausieren — erlaubt, weil stumm und `playsInline`.

- **Das Vorschauvideo behält die Container-Zeitstempel** (`-fps_mode passthrough`),
  `TrackData.t` dagegen ist bei gesetztem `capture_fps` gestaucht. Der
  Review-Screen muss deshalb mit `t / time_scale` suchen, sonst springt jeder
  Frame um genau diesen Faktor daneben — die Zahlen stimmen, das Bild darunter
  ist das falsche. Der leiseste Fehler der ganzen Kette.
- **`CREATE TABLE IF NOT EXISTS` zieht keine neuen Spalten nach.** Die Datenbank
  liegt im Volume und überlebt jedes Deploy; ohne `db._migrate` stirbt nicht der
  Start, sondern der erste Upload danach. Neue Spalten gehören in SCHEMA **und**
  in die Migrationsliste.

### Verworfene Ansätze (nicht nochmal probieren)
- **Den Messstreifen an den Fuss im Absprungmoment legen** statt an die
  Standposition. Sollte Sprünge mit Anlauf retten, half dort nicht (der Streifen
  ist auch vor dem Aufsetzen leer) und verschlechterte den Normalfall: E1 kippte
  unter E2, physikalisch unmöglich.
- **Die Schulter als E3** — siehe oben.
- **Den Ganzkörperschwerpunkt als E2** — misst 0.06 Prozentpunkte besser, lohnt
  den Aufwand nicht.

---

## API

```
POST /api/videos                 → video_id, erkannte fps, Auflösung, Warnung < 120 fps
                                   Formularfeld `capture_fps` optional (R1-Ausnahme)
POST /api/analyses               → job_id      {video_id, jump_type, mass_kg?}
GET  /api/analyses/{id}          → Status, alle Messwerte, Warteschlangenposition
GET  /api/analyses/{id}/track    → Keypoints, Kontaktsignal, Zeitstempel (fürs Overlay)
GET  /api/analyses/{id}/video    → herunterskaliertes Vorschauvideo
POST /api/analyses/{id}/correct  → korrigierte Ereignisframes, rechnet neu
POST /api/analyses/{id}/capture-fps → Aufnahmerate nachtragen, reiht neu ein
GET  /api/sessions               → Verlauf, optional als CSV
```

Der Browser spricht die API **direkt** an, nicht über den Next-Proxy: ein
134-MB-Upload durch die Rewrite-Schicht riss reproduzierbar mit `ECONNRESET` ab.
CORS ist offen — ein Nutzer im eigenen Netz, keine Anmeldung, keine Cookies.

`NEXT_PUBLIC_API_BASE` wird von Next **zur Bauzeit** eingebacken, ist also ein
Build-Argument und keine Laufzeitvariable.

---

## Datenmodell

`video` und `analysis` in SQLite; Videos, Track-NPZ und Vorschauvideos als
Dateien auf Disk, in der Datenbank nur Pfade.

**`pipeline_version` steht an jedem Ergebnis.** Ohne das sieht ein
Algorithmus-Update im Verlauf später aus wie ein Formhoch.

`db.connect` setzt `check_same_thread=False`: FastAPI erzeugt Sync-Dependencies
im Threadpool, der async-Endpunkt läuft danach im Event-Loop-Thread. Sicher, weil
jede Anfrage ihre eigene Verbindung bekommt und wieder schliesst.

---

## Worker

`max_jobs=1` — zwei parallele Analysen würden sich auf einer CPU nur gegenseitig
ausbremsen. Die Job-Übernahme ist **atomar** (`UPDATE ... WHERE status='queued'`);
ohne die Bedingung griffen zwei Worker dieselbe Zeile, und der langsamere
überschrieb das Ergebnis des schnelleren. Genau das ist beim Testen passiert:
dieselbe Datei kam einmal als 74.0 cm und einmal als Verweigerung heraus, obwohl
`extract` für sich reproduzierbar ist.

Der Worker erzeugt ein **Vorschauvideo** in Arbeitsauflösung. Im Browser lässt
sich ein 4K-Video nicht frame-genau steppen; das Proxy hat dieselbe Frameanzahl
und dieselben Zeitstempel und liegt in genau der Auflösung, in der auch die
Landmarks gelten — das Overlay braucht dadurch keine Umrechnung.

---

## Deployment (k3s, seit 20.11.2026)

- Live unter https://jumplab.jonasleuenberger.com, Namespace `jumplab` im Homelab-k3s.
  Manifest `deploy/k3s/jumplab.yaml`, Release `bash deploy/release.sh <tag>` (baut aus `main`
  auf der Build-VM, Images in zot, Ausrollen per Digest, trägt die Digests ins Manifest ein).
- Traefik schickt `/api` direkt an `api`, alles andere an `web`. `NEXT_PUBLIC_API_BASE` ist die
  öffentliche URL (gleiche Herkunft, kein CORS nötig).
- Cloudflare (orange) lässt höchstens 100 MB pro Anfrage durch. Videos über 50 MB lädt der Browser
  in Stücken (`POST /api/uploads/{id}`, dann `POST /api/videos` mit `upload_id`). Nicht
  transkodieren: die Zeitachse kommt aus den Container-Zeitstempeln (R1).
- Kein Deploy über GitHub Actions: das Repo ist öffentlich, ein Self-hosted Runner würde über
  Fork-PRs fremden Code auf der Build-VM ausführen.

## Deployment (Dokploy, bis 20.11.2026)

```bash
docker compose up --build -d
```

Vier Container: `web` (3050), `api` (8000), `worker`, `redis`.

- **Service-Typ „Docker Compose"**, nicht „Application". Compose-Services sind
  **normale Container, keine Swarm-Services** — der dnsrr-Cron des Hosts sieht
  sie nicht, und der Swarm-VIP-Bug betrifft sie nicht.
- **In der Domain-Konfiguration gehört der Container-Port.** Web und API lauschen
  absichtlich innen wie aussen auf derselben Zahl (3050 und 8000), damit es
  nichts zu verwechseln gibt.
- **`HOSTNAME=0.0.0.0` im Web-Container ist Pflicht.** Ohne das bindet Next'
  `server.js` an localhost; Traefik bekommt „connection refused" und antwortet
  mit 502, obwohl Netz, Router-Regel und Portangabe alle stimmen. Der
  veröffentlichte Host-Port verdeckt das, weil er einen anderen Weg nimmt.
- **Im Reverse Proxy** (NPM): `client_max_body_size 500M;` und
  `proxy_read_timeout 600s;`. Standard sind 1 MB — ein Sprungvideo bekäme 413.
- Zwei Domains: eine auf `web`, eine auf `api`, HTTPS in Dokploy aus (NPM
  terminiert TLS). `NEXT_PUBLIC_API_BASE` auf die API-Domain setzen.

Laufzeit: rund 3–4 Minuten für ein 4K-Video mit 120 fps, etwa 80 Sekunden für
einen 3-Sekunden-Ausschnitt.

---

## Tests und Gates

| Datei | deckt ab |
|---|---|
| `tests/test_p0_gate.py` | Kernphysik gegen synthetisches Video mit bekannter Wahrheit |
| `tests/test_p1_gate.py` | Wiederholbarkeit, Verweigerungen, Schätzer-Algebra, Perspektive |
| `tests/test_p2_gate.py` | API, Worker, Warteschlange, Korrektur nach Verweigerung |
| `web/e2e/review.spec.ts` | Upload → Ergebnis → ±1-Frame-Korrektur ändert die Zahl |

Die langsamen Tests (`-m slow`) erzeugen Videos und laufen über eine Minute.

**Gate-Ergebnisse:** P0 4/4 (Krümmung 9.81006 m/s², Höhe +0.022 %), P1 4/4
(σ = 0.0080 cm über 10 Rausch-Seeds, R² 0.9974, Abstand E2/E3 1.68 cm), P2
bestanden (API-Ergebnis bitgleich zur Pipeline), P3 bestanden (Parabel 1.81 px
auf der Hüfte), P4 bestanden (auf dem Server).

`tests/data/jump_clip.mp4` ist ein 3-Sekunden-Ausschnitt um den Sprung — dieselbe
Codestrecke, aber Minuten statt einer Viertelstunde je Testlauf. Nicht im Repo
(gitignored), erzeugbar mit:

```bash
ffmpeg -ss 6.3 -to 9.3 -i IMG_5418.MOV -fps_mode passthrough -c:v libx264 -crf 20 -an tests/data/jump_clip.mp4
```

---

## Aufnahme

Slo-Mo ab 120 fps (240 ist besser), kurze Belichtung, Handy quer auf ein Stativ,
ganzer Körper im Bild, enge Kleidung, kontrastreicher Boden.

Abstand und Kamerahöhe sind unkritisch — das ist nachgerechnet. **Entscheidend
ist: seitlich filmen und senkrecht springen, aus dem Stand.** Wer auf die Kamera
zu springt oder Anlauf nimmt, bekommt keine Messung.

---

## Offene Punkte

- **Ob die gemessenen 74.4 cm stimmen**, ist nicht gegen eine externe Referenz
  geprüft. Der Massstab ist anthropometrisch plausibel (1.36 m Schulter–Knöchel),
  aber das ist kein Beweis. Der Ballwurf-Test (`validate ball`) prüft die
  Pipeline gegen reine Physik und ist der beste verfügbare nächste Schritt.
- **Sprünge mit Anlauf** werden erkannt und abgelehnt, aber nicht gemessen. Dafür
  müsste der Messstreifen dem Absprungfuss folgen — ein Versuch dazu ist
  gescheitert, siehe oben.
- **Der direkte Host-Port `10.11.5.134:3050`** war zeitweise geschlossen, obwohl
  Compose ihn veröffentlicht. Über die Domain funktioniert alles.
- **`pose_landmarker_heavy.task`** liegt nicht im Repo (30 MB). Die
  Fehlermeldung in `pose.py` nennt die Download-URL.
