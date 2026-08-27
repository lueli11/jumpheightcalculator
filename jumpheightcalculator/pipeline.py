"""S4/S5 - Sub-Frame-Verfeinerung, Physik, Güte.

Die vier Höhenschaetzer und was sie unterscheidet:

  E1  g*t_f^2/8, t_f aus dem **Kontaktsignal**. Setzt gleiche Schwerpunkthöhe
      bei Absprung und Landung voraus - man springt aber gestreckt ab und landet
      gebeugt (R2). Überschätzt darum systematisch; dient als Vergleichswert.
  E2  Scheitel der Hüfte-Parabel über der Hüfthöhe zum Absprungzeitpunkt.
      **Referenzwert der App.**
  E3  Dasselbe auf einem eigenen Fit des Ganzkörper-Schwerpunkts (Winter 2009).
  E4  0.5*g*t_fall^2, nur die Fallphase (Scheitel bis Landung aus dem
      Kontaktsignal). Liegt systematisch bei E2 + Delta.

Wichtig, damit es niemand "vereinfacht": **E2 ist identisch mit v0^2/(2g)**.
Für jede Parabel gilt Scheitelhöhe-über-Startpunkt == v(t0)^2/(2a); das ist
dieselbe Algebra, keine zweite Herleitung. Ein Schätzer aus demselben Fit wäre
also ein exaktes Duplikat von E2 und koennte per Konstruktion keinen Fit-Fehler
aufdecken. Nur weil E3 auf **anderen Landmarks** fittet, ist |E2-E3| überhaupt
ein Gütesignal - und genau darauf gatet R7.

Der teure Teil (`extract`) und der billige (`measure`) sind getrennt: eine
Korrektur der Ereignisframes um +/-1 im Review-Screen darf nicht bedeuten, dass
MediaPipe erneut über das ganze Video läuft.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import G, PIPELINE_VERSION
from . import pixels as pixmod
from . import pose as posemod
from .fusion import (CONTACT, crossing_time, flight_window, fuse, normalise,
                     viterbi)
from .ingest import probe
from .physics import fit_parabola

MIN_FLIGHT_S, MAX_FLIGHT_S = 0.15, 1.10
MIN_FLIGHT_FRAMES = 10
MIN_R2 = 0.98

# Das Gütemass ist |E2-E3|, nicht die Spannweite über alle vier Schätzer.
# E1 und E4 weichen von E2 um Delta ab - die Landeasymmetrie. Die ist gemessene
# Physiologie, kein Messfehler, und gehoert darum nicht in ein Fehlermass.
# Nur E2 und E3 messen dieselbe Grösse auf verschiedenen Landmarks.
MAX_DISAGREE_M = 0.04
MAX_CUBIC_T = 4.0
MAX_SHIFT_RESID_PX = 0.5

# Zielbreite fuer den Pose-Zweig. Ein 4K-Video wird darauf heruntergerechnet,
# weil MediaPipe intern ohnehin kleiner arbeitet und die Rohpipe sonst der
# Flaschenhals ist. Ein **fester** Faktor waere falsch: ein ueber WhatsApp
# geschicktes Video kommt mit 464 px Breite an, ein Viertel davon sind 116 px,
# und darauf findet das Modell keine brauchbaren Keypoints mehr. Hochskaliert
# wird nie - erfundene Pixel sind keine Information.
TARGET_POSE_WIDTH = 540


def pose_scale(width: int) -> float:
    return min(1.0, TARGET_POSE_WIDTH / max(width, 1))


# Perspektive: um wie viel darf die Person im Bild waehrend des Flugs groesser
# oder kleiner werden? Gemessen an einem sauberen Video (Kamera seitlich, Sprung
# senkrecht zur Achse) bleiben die starren Segmente ueber den ganzen Flug ohne
# Trend, bei rund 3 % Streuung. Ein Video, auf dem die Person auf die Kamera zu
# springt, kam auf +33 % - ein Oberschenkelknochen waechst nicht, das ist reine
# Tiefenbewegung. Damit aendert sich der Massstab mitten im Sprung, die Kruemmung
# der Parabel wird bedeutungslos und die Selbstkalibrierung (R5) ungueltig.
MAX_SCALE_DRIFT = 0.12

# Kontaktsignal (Fuss) und Koerperbewegung (Knoechelhub) schaetzen das Flugfenster
# unabhaengig voneinander. Nach R3 ist die Knoechelversion systematisch zu lang -
# der Knoechel steigt beim Abdruck schon, waehrend die Zehe noch aufliegt - aber
# sie muessen in derselben Groessenordnung liegen. Auf einem sauberen Video sind
# es 93 gegen 95 Frames. Bei einem Sprung mit Anlauf mass das Kontaktsignal 137
# gegen 76: der Streifen war leer, weil die Person weggelaufen war, nicht weil
# sie flog. Wo die beiden so weit auseinanderliegen, ist das Timing nicht zu
# gebrauchen - und eine Hoehe daraus waere geraten.
CONTACT_VS_BODY_RANGE = (0.6, 1.45)

# Ab wann ist die Person zu klein im Bild? Gemessen (Lochkamera, 2 px
# Keypoint-Rauschen, 120 fps) bleibt die Streuung noch bei 175 px Koerpergroesse
# unter 0.4 cm - der Fit mittelt ueber rund hundert Frames. Weiter unten wird
# aber die Pose-Erkennung selbst unzuverlaessig, und das faellt nicht mehr in die
# Zustaendigkeit der Geometrie. Darum eine Warnung, keine Verweigerung.
MIN_BODY_PX = 140


class Refusal(RuntimeError):
    """R7 - lieber keinen Wert als einen falschen, und immer mit Grund."""


@dataclass(frozen=True)
class TrackData:
    """Alles, was `measure` braucht - ohne Video, ohne Pose-Modell.

    Wird als NPZ neben der Analyse abgelegt und speist sowohl das Overlay im
    Review-Screen als auch die Neuberechnung nach einer Korrektur.
    """
    t: np.ndarray            # (N,) Zeitstempel [s], aus dem Container (R1)
    xy: np.ndarray           # (N, 33, 2) alle Landmarks [px] - für das Skelett-Overlay
    width: int               # Arbeitsaufloesung, in der xy gilt
    height: int
    hip_y: np.ndarray        # (N,) Hüfthöhe [px], Bildkoordinaten
    com_y: np.ndarray        # (N,) Schwerpunkthöhe [px]
    ankle_y: np.ndarray      # (N,) für die Anzeige
    contact: np.ndarray      # (N,) Kontaktsignal c(n) in [0,1], 1 = Kontakt
    confidence: np.ndarray   # (N,) mittlere Keypoint-Konfidenz
    n0: int                  # Flugfenster aus dem HMM (Kontaktsignal, R3)
    n1: int
    coarse_n0: int           # Flugfenster aus dem Knoechelhub - Gegenprobe
    coarse_n1: int
    stance_start: int        # letzte echte Standphase vor dem Absprung
    stance_end: int
    fps: float
    shift_residual: float
    body_scale_px: float
    scale_drift: float       # Groessenaenderung starrer Segmente Stand -> Flug

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in asdict(self).items()})

    @staticmethod
    def load(path: str | Path) -> "TrackData":
        d = np.load(path)
        return TrackData(
            t=d["t"], xy=d["xy"], width=int(d["width"]), height=int(d["height"]),
            hip_y=d["hip_y"], com_y=d["com_y"], ankle_y=d["ankle_y"],
            contact=d["contact"], confidence=d["confidence"],
            n0=int(d["n0"]), n1=int(d["n1"]),
            coarse_n0=int(d["coarse_n0"]) if "coarse_n0" in d else int(d["n0"]),
            coarse_n1=int(d["coarse_n1"]) if "coarse_n1" in d else int(d["n1"]),
            stance_start=(int(d["stance_start"]) if "stance_start" in d
                          else max(0, int(d["stance_end"]) - 60)),
            stance_end=int(d["stance_end"]),
            fps=float(d["fps"]), shift_residual=float(d["shift_residual"]),
            body_scale_px=float(d["body_scale_px"]),
            scale_drift=float(d["scale_drift"]) if "scale_drift" in d else 0.0,
        )


@dataclass(frozen=True)
class Result:
    h_e1: float
    h_e2: float
    h_e3: float
    h_e4: float
    spread: float
    disagreement: float
    t_f: float
    v0: float
    v_land: float
    delta: float
    t_takeoff: float
    t_land: float
    takeoff_frame: float
    landing_frame: float
    contact_time: float | None     # nur beim Drop Jump definiert
    push_time: float               # tiefster Punkt der Gegenbewegung bis Absprung
    rsi: float | None
    impulse: float | None
    mean_force: float | None
    # Die Masse gehört ins Ergebnis, nicht nur in die Rechnung: ohne sie lässt
    # sich später nicht sagen, mit welchem Gewicht Impuls und Kraft entstanden.
    mass_kg: float | None
    scale: float
    a_px: float
    r2_hip: float
    r2_com: float
    rms_hip_px: float
    cubic_t: float
    n_flight_frames: int
    mean_confidence: float
    shift_residual_px: float
    fps_measured: float
    # Gefittete Parabel in Arbeitspixeln: y(t) = c0 + c1*(t-t_ref) + c2*(t-t_ref)^2.
    # Das Overlay zeichnet damit genau die Kurve, aus der die Zahl kommt - wer sie
    # sieht, kann der Zahl widersprechen.
    fit_c0: float
    fit_c1: float
    fit_c2: float
    fit_t_ref: float
    jump_type: str
    quality: float
    corrected: bool = False
    warnings: list[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _coarse_air(track: posemod.PoseTrack, t: np.ndarray) -> tuple[int, int]:
    """Grobes Flugfenster aus der Knöchelhöhe.

    Nur zur Platzierung des ROI und als Referenz für den leeren Boden -
    **nicht** als Timing (R3). Der Knöchel steigt beim Abdruck bereits, während
    die Zehe noch aufliegt, und meldet den Absprung dadurch deutlich zu frueh.
    """
    ank = track.pt(posemod.ANKLE)[:, 1]
    bs = track.body_scale()
    base = float(np.nanmedian(ank[:max(10, len(ank) // 5)]))
    lift = np.nan_to_num(base - ank, nan=-1e9)
    air = lift > 0.05 * bs

    # Genommen wird der Abschnitt mit dem **höchsten** Knöchelhub, nicht der
    # längste. Nach der Landung geht die Person weg, und dieses Weggehen ist
    # leicht länger als der Sprung selbst - auf einem Ausschnitt wurde so das
    # Fenster 252-360 statt 148-241 gewaehlt und der Fit kruemmte sich nach oben.
    # Hoch springt man nur einmal; herumlaufen kann man beliebig lange.
    best, best_lift = None, -np.inf
    i = 0
    while i < air.size:
        if air[i]:
            j = i
            while j < air.size and air[j]:
                j += 1
            dur = t[j - 1] - t[i]
            peak = float(lift[i:j].max())
            if MIN_FLIGHT_S <= dur <= MAX_FLIGHT_S and peak > best_lift:
                best, best_lift = (i, j), peak
            i = j
        else:
            i += 1
    if best is None:
        raise Refusal(
            "Kein Sprung gefunden: die Knöchel heben nie für 0.15-1.10 s deutlich "
            "vom Boden ab. Ist die Person im Bild und der ganze Sprung aufgenommen?"
        )
    return best


def _stance_window(track: posemod.PoseTrack, until: int) -> slice:
    """Die letzte Standphase vor dem Absprung: wo der Fuss zuletzt stillstand.

    Gesucht ist das **laengste** Fenster vor `until`, in dem die Fuss-x-Position
    innerhalb einer halben Fusslaenge bleibt. Bei Gleichstand gewinnt das
    spaetere - der Fuss soll dort liegen, wo er beim Abdruck liegt.

    Vorher wurde blind das Fenster [t0-1.0s, t0-0.5s] genommen. Das stimmt nur,
    wenn die Person die ganze Zeit stillsteht. Auf IMG_0366 laeuft sie ins Bild
    und stellt sich erst 0.25 s vor dem Sprung hin: der Streifen landete bei
    Fuss-x 38 statt 204 px - sechs Fusslaengen daneben. Er lief damit leer,
    sobald die Person daran vorbeiging, und der Absprung wurde 41 Frames
    (342 ms) zu frueh gemeldet.

    **Nicht** zu verwechseln mit dem verworfenen Ansatz "Streifen an den Fuss im
    Absprungmoment": gemittelt wird weiterhin ueber die ganze Standphase, nur
    ueber die richtige.
    """
    x = np.nanmean(track.xy[:, [29, 30, 31, 32], 0], axis=1)
    flen = float(np.nanmedian(np.linalg.norm(
        track.xy[:, [31, 32]] - track.xy[:, [29, 30]], axis=2)))
    if not np.isfinite(flen) or flen < 4:
        flen = track.body_scale() * 0.15
    tol = 0.5 * flen

    ende = int(np.clip(until, 0, x.size - 1))
    best, start = (0, 0), 0
    for e in range(ende + 1):
        while start < e and (np.nanmax(x[start:e + 1])
                             - np.nanmin(x[start:e + 1])) > tol:
            start += 1
        if e - start >= best[1] - best[0]:
            best = (start, e)

    # Hinten auf eine Sekunde begrenzt. Der Streifen ist der Median ueber das
    # Fenster, und wer lange dasteht, scharrt: auf IMG_5418 lief das Fenster
    # ueber 5.2 s, der Streifen wanderte dadurch und das Flugfenster fiel auf
    # 88 statt 93 Frames. Mit Deckel sind es 92 - ein Frame von der
    # dokumentierten Referenz. Auf IMG_0366 aendert der Deckel nichts, dort
    # steht die Person ohnehin nur 33 Frames.
    dt = float(np.median(np.diff(track.t))) if track.t.size > 1 else 1 / 120
    max_len = max(5, int(round(1.0 / max(dt, 1e-6))))
    return slice(max(best[0], best[1] + 1 - max_len), best[1] + 1)


def _scale_drift(track: posemod.PoseTrack, stance: slice, n0: int, n1: int) -> float:
    """Groessenaenderung **starrer** Segmente zwischen Stand und Flug.

    Genommen werden Oberschenkel und Unterschenkel beider Seiten: das sind
    Knochen, ihre Bildlaenge kann sich nur aendern, wenn sich der Abstand zur
    Kamera aendert (oder das Bein aus der Bildebene dreht). Die Koerperlaenge
    Schulter-Knoechel taugt dafuer nicht - die aendert sich schon dadurch, dass
    man die Beine anzieht.
    """
    pairs = ((23, 25), (24, 26), (25, 27), (26, 28))
    lens = np.stack([np.linalg.norm(track.xy[:, a] - track.xy[:, b], axis=1)
                     for a, b in pairs])
    # Gemessen wird gegen die **echte** Standphase. Mit dem alten festen Fenster
    # (0.5 s vor dem groben Absprung) lagen auf IMG_0366 Frames, in denen die
    # Person noch halb ausserhalb des Bildes ins Bild lief: die Gliedmassen waren
    # dort verstuemmelt, und die Pipeline meldete +34 % Massstabsaenderung, wo
    # keine war.
    with np.errstate(invalid="ignore"):
        stand = np.nanmedian(lens[:, stance])
        air = np.nanmedian(lens[:, n0:max(n0 + 1, n1)])
    if not (np.isfinite(stand) and np.isfinite(air)) or stand <= 0:
        return 0.0
    return float(air / stand - 1.0)


def extract(path: str | Path, *, scale: float | None = None,
            model: str = "full") -> TrackData:
    """Der teure Teil: Decode, Pose, Pixelzweig, Fusion. Rund 2-3 Minuten auf 4K.

    `scale` wird aus der Bildbreite bestimmt, wenn nichts angegeben ist.
    """
    info = probe(path)
    if scale is None:
        scale = pose_scale(info.width)
    track = posemod.track(info, model=model, scale=scale)
    if track.found.mean() < 0.8:
        raise Refusal(
            f"Person nur in {track.found.mean()*100:.0f} % der Frames erkannt. "
            f"Bitte den ganzen Körper im Bild halten und enge Kleidung tragen."
        )

    c_n0, c_n1 = _coarse_air(track, info.t)

    # Zweiter Streifen an der Landestelle: wer zum Korb springt, landet nicht
    # dort, wo er abgesprungen ist.
    fps_i = max(info.measured_fps, 1.0)
    land_stance = slice(min(c_n1 + int(0.15 * fps_i), info.n_frames - 2),
                        min(c_n1 + int(0.6 * fps_i), info.n_frames - 1))
    stance = _stance_window(track, c_n0)
    sig = pixmod.contact_signals(info, track, stance=stance,
                                 air_frames=np.arange(c_n0, c_n1),
                                 landing_stance=land_stance, scale=scale)

    # Ausserhalb der Umgebung des Sprungs ist der Streifen bedeutungslos: sobald
    # die Person weggeht, ist er dauerhaft leer und sieht damit aus wie Flug.
    # Gemessen las Viterbi so 3.6 s "Flug" - 0.8 s Sprung plus 2.8 s Weggehen.
    guard = int(0.4 * fps_i)
    lo, hi = max(0, c_n0 - guard), min(sig.t.size, c_n1 + guard)

    # Kontaktsignal: 1 = Kontakt.
    #
    # Nur Belegung und Spaltanteil, **nicht** die Differenzenergie zum Vorframe.
    # Die ist während des Abdrucks gross, weil der Fuss abrollt, und zieht das
    # Signal damit vor dem eigentlichen Abheben nach unten. Gemessen legte sie
    # den Absprung sieben Frames zu frueh und blies dadurch E2 auf 90 cm auf -
    # über E1 von 82 cm, was für einen freien Flug unmöglich ist (R2 verlangt
    # die umgekehrte Reihenfolge).
    #
    # Je Streifen einzeln normiert, dann das Maximum: Kontakt heisst "irgendwo
    # steht ein Fuss". Gemeinsam normiert drueckt der kontrastreichere Streifen
    # den anderen unter die Schwelle.
    c = np.ones(sig.t.size)
    c[lo:hi] = fuse(
        {"occupancy": np.max([normalise(o[lo:hi]) for o in sig.occupancy], axis=0),
         "gap": np.max([normalise(g[lo:hi]) for g in sig.gap], axis=0)},
        {"occupancy": 0.75, "gap": 0.25})
    states = viterbi(c)
    n0, n1 = flight_window(states)

    n = min(info.n_frames, c.size)
    conf = np.nanmean(track.vis[:n][:, list(posemod.HIP) + list(posemod.ANKLE)], axis=1)
    drift = _scale_drift(track, stance, c_n0, c_n1)
    return TrackData(
        t=info.t[:n],
        xy=track.xy[:n],
        width=track.width, height=track.height,
        hip_y=track.pt(posemod.HIP)[:n, 1],
        com_y=track.com()[:n, 1],
        ankle_y=track.pt(posemod.ANKLE)[:n, 1],
        contact=c[:n], confidence=conf,
        n0=n0, n1=n1, coarse_n0=c_n0, coarse_n1=c_n1,
        stance_start=stance.start, stance_end=stance.stop,
        fps=info.measured_fps, shift_residual=sig.shift_residual,
        body_scale_px=track.body_scale(), scale_drift=drift,
    )


def measure(td: TrackData, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, n0: int | None = None, n1: int | None = None) -> Result:
    """Der billige Teil: Fit, Schätzer, Güte. Läuft in Millisekunden.

    `n0`/`n1` überschreiben das Flugfenster - so rechnet eine +/-1-Frame-Korrektur
    aus dem Review-Screen neu, ohne dass die Pose neu bestimmt werden muss.
    """
    # Zuerst die Perspektive. Aendert sich der Massstab mitten im Sprung, sind
    # alle folgenden Pruefungen Folgefehler - eine unplausible Flugzeit ist dann
    # das Symptom, nicht die Ursache, und "verschiebe die Ereignisframes" waere
    # ein irrefuehrender Rat: von Hand ist da nichts zu retten.
    if abs(td.scale_drift) > MAX_SCALE_DRIFT:
        raise Refusal(
            f"Die Person wird während des Flugs {td.scale_drift*100:+.0f} % grösser "
            f"im Bild - gemessen an Ober- und Unterschenkel, und Knochen wachsen "
            f"nicht. Du springst also auf die Kamera zu oder von ihr weg. Damit "
            f"ändert sich der Massstab mitten im Sprung und die Höhe ist nicht "
            f"bestimmbar. Stelle die Kamera seitlich auf, 3-4 m entfernt, und "
            f"springe senkrecht zur Blickrichtung - ohne vor- oder zurückzuspringen."
        )

    # Gegenprobe des Timings, solange nichts von Hand gesetzt wurde.
    body = td.coarse_n1 - td.coarse_n0
    foot = td.n1 - td.n0
    if n0 is None and n1 is None and body > 0:
        ratio = foot / body
        if not (CONTACT_VS_BODY_RANGE[0] <= ratio <= CONTACT_VS_BODY_RANGE[1]):
            raise Refusal(
                f"Kontaktsignal und Körperbewegung widersprechen sich: der Fuss "
                f"meldet {foot} Flugframes, der Knöchelhub {body}. Typisch für "
                f"einen Sprung mit Anlauf - der Messstreifen liegt dort, wo der "
                f"Fuss abspringt, und ist beim Anlaufen schon leer, bevor der "
                f"Sprung beginnt. Springe aus dem Stand, oder verschiebe Absprung "
                f"und Landung im Review-Screen von Hand anhand der Kurve."
            )

    corrected = n0 is not None or n1 is not None
    n0 = td.n0 if n0 is None else int(n0)
    n1 = td.n1 if n1 is None else int(n1)
    n_air = n1 - n0
    if n_air < MIN_FLIGHT_FRAMES:
        raise Refusal(
            f"Nur {n_air} Flugframes erfasst (mindestens {MIN_FLIGHT_FRAMES} nötig) - "
            f"bitte in Slo-Mo mit mindestens 120 fps aufnehmen."
        )

    t = td.t
    t0 = crossing_time(t, td.contact, n0)
    t1 = crossing_time(t, td.contact, n1)
    if corrected:
        # Bei einer manuellen Korrektur zaehlt der gesetzte Frame, nicht der
        # interpolierte Schnittpunkt eines Signals, dem der Nutzer widersprochen hat.
        t0, t1 = float(t[min(n0, t.size - 1)]), float(t[min(n1, t.size - 1)])
    t_f = t1 - t0
    if not (MIN_FLIGHT_S <= t_f <= MAX_FLIGHT_S):
        raise Refusal(
            f"Flugzeit {t_f*1000:.0f} ms liegt ausserhalb des Plausiblen "
            f"({MIN_FLIGHT_S*1000:.0f}-{MAX_FLIGHT_S*1000:.0f} ms) - das wären "
            f"{G*t_f*t_f/8*100:.0f} cm. Absprung oder Landung wurden falsch erkannt. "
            f"Im Review-Screen lassen sich die beiden Ereignisframes von Hand "
            f"verschieben; die Kontaktsignal-Kurve zeigt, wo sie hingehören."
        )

    # R4 - Randframes raus: dort ist der Punkt weder Kontakt noch sauberer Flug,
    # und auf echtem Material zusaetzlich bewegungsunscharf. Auf dem synthetischen
    # Video war ein einziger solcher Frame 0.14 % Höhenfehler.
    #
    # Bewusst **ungeglaettet** in den Fit: der Least-Squares-Fit mittelt das
    # Rauschen bereits über alle Flugframes, eine Vorglaettung bringt an der
    # Höhe nachweislich nichts (gemessen 71.81 gegen 71.84 cm), korreliert aber
    # die Residuen (0.75 -> 0.90) und macht die Gütemasse unbrauchbar.
    sel = slice(n0 + 1, n1 - 1)
    tt = t[sel]
    fit_hip = fit_parabola(tt, td.hip_y[sel], seed=seed)
    fit_com = fit_parabola(tt, td.com_y[sel], seed=seed)

    s = fit_hip.scale                       # R5 - Massstab aus der Physik
    h_e2 = (float(fit_hip.y_at(t0)) - fit_hip.y_apex) * s
    h_e3 = (float(fit_com.y_at(t0)) - fit_com.y_apex) * fit_com.scale
    h_e1 = G * t_f * t_f / 8.0
    h_e4 = 0.5 * G * max(t1 - fit_hip.t_apex, 0.0) ** 2

    v0 = float(fit_hip.v_up_px(t0)) * s
    v_land = abs(float(fit_hip.v_up_px(t1))) * s
    delta = (v_land ** 2 - v0 ** 2) / (2 * G)

    hs = [h_e1, h_e2, h_e3, h_e4]
    spread = float(max(hs) - min(hs))
    disagreement = abs(h_e2 - h_e3)

    # Kontaktzeit: nur beim Drop Jump eine Messgroesse. Beim CMJ steht die Person
    # vorher einfach da, und der Block wäre die gesamte Stehzeit - gemessen
    # 7525 ms, was als "Kontaktzeit" ausgegeben schlicht falsch wäre.
    k = n0 - 1
    while k > 0 and td.contact[k] >= 0.5:
        k -= 1
    ct = float(t0 - t[k + 1]) if n0 - (k + 1) >= 2 else None
    contact_time = ct if (ct is not None and ct <= 1.5) else None
    rsi = (h_e2 / contact_time) if (jump_type == "drop" and contact_time) else None

    # Abdruckdauer: vom tiefsten Punkt der Gegenbewegung bis zum Absprung -
    # anders als die Kontaktzeit auch beim CMJ definiert.
    lo_i = max(0, min(td.stance_start, n0 - 1))
    seg = td.hip_y[lo_i:n0]
    low = (int(np.nanargmax(seg)) + lo_i) if seg.size else lo_i
    push_time = float(t0 - t[low])

    # R6 - Impuls und mittlere Kraft nur mit Masse und Abdruckdauer. Keine Watt:
    # Peak Power braucht den Kraft-Zeit-Verlauf einer Kraftmessplatte.
    impulse = mean_force = None
    if mass_kg:
        impulse = mass_kg * v0
        if push_time > 0.05:
            mean_force = mass_kg * v0 / push_time + mass_kg * G

    conf = float(np.nanmean(td.confidence[sel]))

    warn: list[str] = []
    if td.fps < 120:
        warn.append(f"Nur {td.fps:.0f} fps - für eine Fortschrittsmessung "
                    f"grenzwertig, bitte in Slo-Mo aufnehmen.")
    if td.body_scale_px < MIN_BODY_PX:
        warn.append(f"Die Person ist nur {td.body_scale_px:.0f} px gross im Bild. "
                    f"Der Abstand als solcher stört nicht - die Selbstkalibrierung "
                    f"gleicht ihn aus - aber die Keypoints werden unsicher. "
                    f"Näher herangehen oder höher auflösen.")
    if np.isfinite(td.shift_residual) and td.shift_residual > MAX_SHIFT_RESID_PX:
        warn.append(f"Kamera wackelt (Restversatz {td.shift_residual:.2f} px) - das "
                    f"Kontaktsignal wird dadurch unsicherer.")
    if corrected:
        warn.append("Ereignisframes wurden manuell korrigiert.")

    if fit_hip.r2 < MIN_R2:
        raise Refusal(
            f"Die Hüfte folgt keiner sauberen Wurfparabel (R2 = {fit_hip.r2:.4f}, "
            f"nötig > {MIN_R2}). Meist heisst das weite Kleidung oder eine teilweise "
            f"verdeckte Person."
        )
    if disagreement > MAX_DISAGREE_M:
        raise Refusal(
            f"Hüfte und Körperschwerpunkt kommen auf {h_e2*100:.1f} cm gegen "
            f"{h_e3*100:.1f} cm - {disagreement*100:.1f} cm auseinander (Grenze "
            f"{MAX_DISAGREE_M*100:.0f} cm). Die Keypoints sind zu unsicher für eine "
            f"Messung."
        )
    if h_e1 < h_e2 - 0.005:
        # R2 verlangt, dass die Flugzeitmethode *über*schätzt: man springt
        # gestreckt ab und landet gebeugt, die Fallphase dauert also länger als
        # die Steigphase. Ist E1 kleiner als E2, ist die Hüfte im freien Flug
        # angeblich höher gestiegen, als die Flugzeit erlaubt - unmöglich. In
        # der Praxis heisst es, der Absprung liegt zu frueh und die Parabel wird
        # davor zurück-extrapoliert. Genau so kam auf dem ersten echten Video
        # E2 = 90 cm gegen E1 = 82 cm zustande.
        raise Refusal(
            f"Widerspruch in der Physik: die Flugzeit erlaubt höchstens "
            f"{h_e1*100:.1f} cm, die Hüfte steigt aber auf {h_e2*100:.1f} cm. Der "
            f"Absprungzeitpunkt liegt zu früh."
        )
    if fit_hip.cubic_t > MAX_CUBIC_T:
        raise Refusal(
            f"Die Flugbahn ist keine reine Parabel (kubischer Anteil t = "
            f"{fit_hip.cubic_t:.1f}). Typisch für Objektivverzeichnung bei tief "
            f"stehender Kamera oder für eine Bewegung auf die Kamera zu. Stelle das "
            f"Handy höher und springe senkrecht zur Kameraachse."
        )

    quality = float(np.clip(
        0.35 * min(fit_hip.r2 / MIN_R2, 1.0)
        + 0.30 * (1.0 - min(disagreement / MAX_DISAGREE_M, 1.0))
        + 0.20 * min(n_air / 40.0, 1.0)
        + 0.15 * (conf if np.isfinite(conf) else 0.0), 0.0, 1.0))

    last = t.size - 1
    n1c = min(n1, last)
    return Result(
        h_e1=h_e1, h_e2=h_e2, h_e3=h_e3, h_e4=h_e4,
        spread=spread, disagreement=disagreement,
        t_f=t_f, v0=v0, v_land=v_land, delta=delta,
        t_takeoff=t0, t_land=t1,
        takeoff_frame=n0 + (t0 - t[n0]) / max(t[min(n0 + 1, last)] - t[n0], 1e-9),
        landing_frame=n1c + (t1 - t[n1c]) / max(t[min(n1c + 1, last)] - t[n1c], 1e-9),
        contact_time=contact_time, push_time=push_time, rsi=rsi,
        impulse=impulse, mean_force=mean_force, mass_kg=mass_kg,
        scale=s, a_px=fit_hip.a_px, r2_hip=fit_hip.r2, r2_com=fit_com.r2,
        rms_hip_px=fit_hip.rms_resid, cubic_t=fit_hip.cubic_t,
        n_flight_frames=n_air, mean_confidence=conf,
        shift_residual_px=td.shift_residual, fps_measured=td.fps,
        fit_c0=fit_hip.c0, fit_c1=fit_hip.c1, fit_c2=fit_hip.c2,
        fit_t_ref=fit_hip.t_ref,
        jump_type=jump_type, quality=quality, corrected=corrected, warnings=warn,
    )


def analyze(path: str | Path, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, scale: float | None = None, model: str = "full",
            track_out: str | Path | None = None) -> Result:
    """Bequemlichkeit für die CLI: extrahieren und messen in einem Zug."""
    td = extract(path, scale=scale, model=model)
    if track_out is not None:
        td.save(track_out)
    return measure(td, jump_type=jump_type, mass_kg=mass_kg, seed=seed)
