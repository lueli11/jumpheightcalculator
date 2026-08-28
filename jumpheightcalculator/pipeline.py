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
from .physics import FitError, fit_parabola

MIN_FLIGHT_S, MAX_FLIGHT_S = 0.15, 1.10
MIN_FLIGHT_FRAMES = 10
MIN_R2 = 0.98
# Schulter bis Knoechel, aus dem Massstab der Selbstkalibrierung zurueckgerechnet.
# Weit gefasst - das ist eine Absurditaetspruefung gegen eine gestreckte
# Zeitachse, keine Groessenmessung.
BODY_LEN_BOUNDS_M = (0.5, 3.0)

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
# Eine echte Entfernungsaenderung skaliert alle starren Segmente gleich. Streuen
# die vier Werte weiter auseinander, ist die Ursache Haltung oder Pose-Rauschen,
# und eine Verweigerung waere ein Fehlalarm. Gemessen: saubere Referenz 5 Punkte,
# IMG_0366 (schraege Kamera, Hocke) 15 Punkte.
MAX_SCALE_SPREAD = 0.10

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


class TrajectoryUnusable(Refusal):
    """Die **Bahn** taugt nicht - das Timing kann trotzdem stimmen.

    Der Unterschied entscheidet, ob es eine Zahl gibt. E2 kommt aus der Form der
    Flugkurve und faellt mit ihr; E1 kommt aus `h = g*t^2/8` und braucht nur die
    beiden Kontaktframes, keinen Massstab und keine einzige Pose-Koordinate.
    Eine unbrauchbare Huefte macht also E2 unmoeglich und E1 nicht.

    Umgekehrt gilt das nicht: steht das Timing in Frage (zu wenige Flugframes,
    unplausible Flugzeit, Kontaktsignal gegen Koerperbewegung, gestreckte
    Zeitachse), faellt E1 mit - dann bleibt es bei der Verweigerung.
    """


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
    scale_drift: float       # Groessenaenderung starrer Segmente im Flug
    scale_spread: float = 0.0  # Spanne derselben Aenderung ueber die vier Segmente
    # Faktor, mit dem die Container-PTS gestaucht wurden (angegebene Aufnahmerate).
    # 1.0 heisst unveraendert. Das Vorschauvideo behaelt die **Container**-Achse -
    # wer im Review-Screen einen Frame anspringt, muss t / time_scale nehmen.
    time_scale: float = 1.0

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
            scale_spread=float(d["scale_spread"]) if "scale_spread" in d else 0.0,
            time_scale=float(d["time_scale"]) if "time_scale" in d else 1.0,
        )


@dataclass(frozen=True)
class Result:
    h_e1: float
    h_e2: float | None
    h_e3: float | None
    h_e4: float | None
    spread: float | None
    disagreement: float | None
    t_f: float
    v0: float | None
    v_land: float | None
    delta: float | None
    t_takeoff: float
    t_land: float
    takeoff_frame: float
    landing_frame: float
    contact_time: float | None     # nur beim Drop Jump definiert
    push_time: float | None               # tiefster Punkt der Gegenbewegung bis Absprung
    rsi: float | None
    impulse: float | None
    mean_force: float | None
    # Die Masse gehört ins Ergebnis, nicht nur in die Rechnung: ohne sie lässt
    # sich später nicht sagen, mit welchem Gewicht Impuls und Kraft entstanden.
    mass_kg: float | None
    scale: float | None
    a_px: float | None
    r2_hip: float | None
    r2_com: float | None
    rms_hip_px: float | None
    cubic_t: float | None
    n_flight_frames: int
    mean_confidence: float
    shift_residual_px: float
    fps_measured: float
    # Gefittete Parabel in Arbeitspixeln: y(t) = c0 + c1*(t-t_ref) + c2*(t-t_ref)^2.
    # Das Overlay zeichnet damit genau die Kurve, aus der die Zahl kommt - wer sie
    # sieht, kann der Zahl widersprechen.
    fit_c0: float | None
    fit_c1: float | None
    fit_c2: float | None
    fit_t_ref: float | None
    jump_type: str
    quality: float | None
    # "parabel" = der Referenzwert E2 aus dem Huefte-Fit. "flugzeit" = die Bahn
    # war unbrauchbar, geblieben ist E1 aus den zwei Kontaktframes. E1 braucht
    # **keinen Massstab** (h = g*t^2/8) und ist damit gegen Pose-Rauschen,
    # Kamerawinkel und Perspektive immun - aber es ueberschaetzt bauartbedingt
    # (R2), und um wie viel, sagt nur die Parabel. Darum zweite Wahl.
    mode: str = "parabel"
    # Warum die Bahn verworfen wurde - nur im Modus "flugzeit" gesetzt.
    fallback_reason: str | None = None
    corrected: bool = False
    warnings: list[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _coarse_air(track: posemod.PoseTrack, t: np.ndarray,
                time_scale: float = 1.0) -> tuple[int, int]:
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
    zu_lang = None                            # laengster Block ueber der Grenze
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
            if dur > MAX_FLIGHT_S and (zu_lang is None or dur > zu_lang[2]):
                zu_lang = (i, j, dur)
            i = j
        else:
            i += 1

    # Ein Knoechelhub, der laenger dauert als physikalisch moeglich, **und** dabei
    # einer sauberen Wurfparabel folgt, deren Massstab eine unmoegliche
    # Koerpergroesse ergibt: dann ist nicht die Bewegung falsch, sondern die
    # Zeitachse. Auf IMG_4160 (iPhone-Slo-Mo, bereits verlangsamt exportiert)
    # meldet der Container ehrlich 30 fps, die Zeitleiste ist aber vierfach
    # gestreckt - der Sprung sieht 2.8 s lang aus und faellt hier heraus, uebrig
    # bleibt als "Sprung" ein Laufschritt aus dem Anlauf.
    #
    # Alle drei Bedingungen braucht es. Ein zu langer Block allein ist der
    # Normalfall: nach der Landung geht die Person weg. Auf der sauberen Referenz
    # dauert dieses Weggehen 2.58 s - es ist aber keine Parabel (R2 0.956 gegen
    # 0.996), und daran wird es unterschieden.
    if zu_lang is not None:
        _refuse_if_stretched(t, track.pt(posemod.HIP)[:, 1], track.xy,
                             zu_lang[0], zu_lang[1], zu_lang[2], time_scale)
    if best is None:
        raise Refusal(
            "Kein Sprung gefunden: die Knöchel heben nie für 0.15-1.10 s deutlich "
            "vom Boden ab. Ist die Person im Bild und der ganze Sprung aufgenommen?"
        )
    return best


# Bildraten, die es an Kameras wirklich gibt. Die Schaetzung aus der Geometrie
# ist ungenau (sie unterstellt 1.35 m Schulter-Knoechel), und wer den rohen Wert
# abtippt, rechnet quadratisch daneben: 129 statt 120 sind 13 % zu wenig Hoehe.
# Darum wird die Schaetzung auf die naechste **echte** Rate gerundet und die
# rohe Zahl nur noch als Herkunft genannt.
CAMERA_FPS = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 100.0, 120.0, 200.0, 240.0,
              480.0, 960.0)


def _nearest_camera_fps(fps: float) -> float:
    """Die naechste uebliche Aufnahmerate - relativ gemessen, nicht absolut.

    Absolut waere 129 gleich weit von 120 wie von 138; relativ zaehlt aber das
    Verhaeltnis, denn genau das geht quadratisch in die Hoehe ein.
    """
    return min(CAMERA_FPS, key=lambda f: abs(np.log(f / max(fps, 1e-9))))


def _timebase_advice(time_scale: float, fps: float, k: float) -> str:
    """Der Rat haengt davon ab, ob schon eine Aufnahmerate angegeben wurde.

    Ohne Angabe ist der naechste Schritt der Export mit Originalgeschwindigkeit.
    Mit Angabe waere derselbe Rat zirkulaer - dann ist die **Angabe** falsch, und
    zwar um genau diesen Faktor.
    """
    roh = fps * k
    echt = _nearest_camera_fps(roh)
    if time_scale == 1.0:
        return (f"Das Video ist also verlangsamt exportiert und dürfte mit "
                f"{echt:.0f} fps aufgenommen sein statt der {fps:.0f} fps, die "
                f"im Container stehen (aus der Bahn geschätzt: {roh:.0f}). "
                f"Trag {echt:.0f} ein, oder exportiere das Video mit "
                f"Originalgeschwindigkeit.")
    return (f"Die angegebene Aufnahmerate von {fps:.0f} fps kann nicht stimmen - "
            f"nach der Bahn wären es {roh:.0f}, also vermutlich {echt:.0f} fps. "
            f"Trag den korrigierten Wert ein.")


def _refuse_if_stretched(t: np.ndarray, hip_y: np.ndarray, xy: np.ndarray,
                         n0: int, n1: int, dur: float,
                         time_scale: float = 1.0) -> None:
    """Zu lang **und** saubere Parabel **und** unmoegliche Groesse -> Zeitachse.

    Alle drei Bedingungen braucht es. Ein zu langes Luftfenster allein ist der
    Normalfall: nach der Landung geht die Person weg, auf der sauberen Referenz
    2.58 s lang. Das ist aber keine Parabel (R2 0.956 gegen 0.996), und daran
    wird es unterschieden. Die Groessenprobe ist die dritte Sicherung.
    """
    sel = slice(n0 + 1, n1 - 1)
    if sel.stop - sel.start < 8:
        return
    try:
        fit = fit_parabola(t[sel], hip_y[sel])
    except FitError:
        return
    if fit.r2 < MIN_R2 or fit.a_px <= 0:
        return
    laenge = _shoulder_ankle_px(xy) * fit.scale
    if BODY_LEN_BOUNDS_M[0] <= laenge <= BODY_LEN_BOUNDS_M[1]:
        return
    k = (laenge / 1.35) ** 0.5
    fps = 1.0 / max(float(np.median(np.diff(t))), 1e-9)
    raise Refusal(
        f"Die Person ist {dur:.1f} s am Stück in der Luft und fliegt dabei eine "
        f"saubere Wurfparabel - beides zusammen geht nicht (möglich sind "
        f"{MAX_FLIGHT_S:.2f} s). Der Massstab daraus ergäbe eine Person von "
        f"{laenge:.0f} m. Nicht die Aufnahme ist schuld, sondern die Zeitachse: "
        f"sie ist rund {k:.0f}-fach gestreckt. "
        + _timebase_advice(time_scale, fps, k)
    )


def _shoulder_ankle_px(xy: np.ndarray) -> float:
    """Schulter-Knoechel in Pixeln, p90 - also nahe der gestreckten Haltung."""
    sch = np.nanmean(xy[:, [11, 12]], axis=1)
    ank = np.nanmean(xy[:, [27, 28]], axis=1)
    with np.errstate(invalid="ignore"):
        return float(np.nanpercentile(np.linalg.norm(sch - ank, axis=1), 90))


def _check_scale_plausible(td: "TrackData", s: float) -> None:
    """Gegenprobe zur Selbstkalibrierung: wie gross waere die Person demnach?

    R5 leitet den Massstab aus der Kruemmung ab, `s = g / a_px`. Ist die Zeitachse
    um k gestreckt, faellt a_px um k^2 und der Massstab waechst um k^2 - lautlos,
    denn die Bahn bleibt eine saubere Parabel (auf IMG_4160 R2 = 0.997). Die
    Flugzeitgrenze faengt grosse k ab, ein doppelt gestrecktes Video aber nicht:
    aus 0.45 s werden 0.9 s, und das ist eine voellig normale Flugzeit.

    Was auffliegt, ist die Groesse. Schulter-Knoechel sind bei einem Menschen rund
    1.0-1.8 m. Gemessen: saubere Referenz 1.42 m, IMG_0366 1.80 m,
    **IMG_4160 24.06 m**. Die Grenzen sitzen bewusst weit, das ist eine
    Absurditaetspruefung und keine Anthropometrie.
    """
    px = _shoulder_ankle_px(td.xy)
    if not np.isfinite(px) or px <= 0:
        return
    laenge = px * s
    if BODY_LEN_BOUNDS_M[0] <= laenge <= BODY_LEN_BOUNDS_M[1]:
        return
    k = (laenge / 1.35) ** 0.5
    richtung = "gestreckt" if k > 1 else "gestaucht"
    raise Refusal(
        f"Der Massstab aus der Physik ergibt eine Person von {laenge:.1f} m "
        f"(Schulter bis Knöchel) - das kann nicht sein. Nicht die Aufnahme ist "
        f"schuld, sondern die Zeitachse: sie ist um rund das "
        f"{k:.1f}-fache {richtung}. " + _timebase_advice(td.time_scale, td.fps, k)
    )


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


def _scale_drift(track: posemod.PoseTrack, n0: int, n1: int) -> tuple[float, float]:
    """Groessenaenderung starrer Segmente **innerhalb des Flugs**.

    Gemessen wird, was die Verweigerung auch behauptet: ob die Person waehrend
    des Flugs groesser wird. Vorher stand hier Stand gegen Flug - das misst zu
    einem grossen Teil die **Haltung**, nicht die Entfernung. In der Hocke zeigen
    gebeugte Glieder bei nicht exakt seitlicher Kamera aus der Bildebene heraus
    und sind dadurch verkuerzt; auf IMG_0366 kam so +19 % heraus, obwohl die
    Person vor dem Sprung (Schulter-Knoechel 150.1 px) und nach der Landung
    (143.5 px) nachweislich gleich weit weg steht.

    Zurueckgegeben wird das Paar (Aenderung, Streuung):

    * **Aenderung** ist der Median ueber die vier Segmente von erstem gegen
      letztes Flugdrittel. Median gegen Median, damit weder der Hocker in der
      Mitte noch ein einzelner kaputter Frame durchschlaegt - auf IMG_0366 sind
      Frame 101 und 109 Pose-Muell (Oberschenkel 22 px statt 65).
    * **Streuung** ist die Spanne derselben vier Werte. Eine echte
      Entfernungsaenderung skaliert **alle** Segmente gleich; streuen sie, ist es
      Haltung oder Pose-Rauschen. Gemessen: IMG_0366 [7.1, 7.6, 19.1, 22.0] -
      15 Punkte auseinander, waehrend die saubere Referenz IMG_5418 auf 5 Punkte
      kommt. Das deckt sich mit der dokumentierten Beobachtung, dass ein sauberes
      Video um 3 % streut.

    Die Koerperlaenge Schulter-Knoechel taugt hier nicht: sie aendert sich schon
    dadurch, dass man die Beine anzieht (auf IMG_0366 Stand 150 gegen Flug 220 px,
    +47 % rein aus Streckung).
    """
    pairs = ((23, 25), (24, 26), (25, 27), (26, 28))
    lens = np.stack([np.linalg.norm(track.xy[:, a] - track.xy[:, b], axis=1)
                     for a, b in pairs])
    k = (n1 - n0) // 3
    if k < 3:
        return 0.0, 0.0
    with np.errstate(invalid="ignore"):
        frueh = np.nanmedian(lens[:, n0:n0 + k], axis=1)
        spaet = np.nanmedian(lens[:, n1 - k:n1], axis=1)
    ok = np.isfinite(frueh) & np.isfinite(spaet) & (frueh > 0)
    if ok.sum() < 2:
        return 0.0, 0.0
    r = spaet[ok] / frueh[ok] - 1.0
    return float(np.median(r)), float(r.max() - r.min())


def extract(path: str | Path, *, scale: float | None = None,
            model: str = "full", capture_fps: float | None = None) -> TrackData:
    """Der teure Teil: Decode, Pose, Pixelzweig, Fusion. Rund 2-3 Minuten auf 4K.

    `scale` wird aus der Bildbreite bestimmt, wenn nichts angegeben ist.
    `capture_fps` korrigiert eine gestreckte Zeitachse - siehe `ingest.probe`.
    """
    info = probe(path, capture_fps=capture_fps)
    if scale is None:
        scale = pose_scale(info.width)
    track = posemod.track(info, model=model, scale=scale)
    if track.found.mean() < 0.8:
        raise Refusal(
            f"Person nur in {track.found.mean()*100:.0f} % der Frames erkannt. "
            f"Bitte den ganzen Körper im Bild halten und enge Kleidung tragen."
        )

    c_n0, c_n1 = _coarse_air(track, info.t, info.time_scale)

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
    drift, spread = _scale_drift(track, n0, n1)
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
        scale_spread=spread, time_scale=info.time_scale,
    )


def measure(td: TrackData, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, n0: int | None = None, n1: int | None = None) -> Result:
    """Der billige Teil: Fit, Schätzer, Güte. Läuft in Millisekunden.

    `n0`/`n1` überschreiben das Flugfenster - so rechnet eine +/-1-Frame-Korrektur
    aus dem Review-Screen neu, ohne dass die Pose neu bestimmt werden muss.
    """
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
        # Wer die Frames von Hand setzt, kommt hier an, ohne dass _coarse_air je
        # gelaufen waere - dieselbe Pruefung also noch einmal, sonst bekaeme er
        # bei einem gestreckten Video den Rat, die Marker zu verschieben.
        _refuse_if_stretched(t, td.hip_y, td.xy, n0, n1, t_f, td.time_scale)
        raise Refusal(
            f"Flugzeit {t_f*1000:.0f} ms liegt ausserhalb des Plausiblen "
            f"({MIN_FLIGHT_S*1000:.0f}-{MAX_FLIGHT_S*1000:.0f} ms) - das wären "
            f"{G*t_f*t_f/8*100:.0f} cm. Absprung oder Landung wurden falsch erkannt. "
            f"Im Review-Screen lassen sich die beiden Ereignisframes von Hand "
            f"verschieben; die Kontaktsignal-Kurve zeigt, wo sie hingehören."
        )

    # Ab hier haengt alles an der **Bahn**. Geht sie nicht, bleiben die zwei
    # Kontaktframes - und die tragen h = g*t^2/8 ohne einen einzigen Pixel.
    try:
        return _measure_trajectory(
            td, n0=n0, n1=n1, t0=t0, t1=t1, t_f=t_f, jump_type=jump_type,
            mass_kg=mass_kg, seed=seed, corrected=corrected)
    except (TrajectoryUnusable, FitError) as e:
        return _measure_flight_time(
            td, n0=n0, n1=n1, t0=t0, t1=t1, t_f=t_f, jump_type=jump_type,
            mass_kg=mass_kg, corrected=corrected, grund=str(e))


def _warnings(td: TrackData, *, corrected: bool) -> list[str]:
    """Hinweise, die keine Verweigerung sind.

    Geteilt zwischen beiden Messwegen: sie haengen an der Aufnahme, nicht an
    der Auswertung. Die Flugzeit-Ebene braucht dieselben - wackelige Kamera
    und eine erklaerte Zeitachse betreffen sie genauso.
    """
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
    # Ueber der Schwelle, aber die Segmente widersprechen sich: dann ist es keine
    # Entfernungsaenderung (die skaliert alle gleich), sondern Haltung oder
    # Pose-Rauschen. Kein Grund, die Messung zu verweigern - aber einer, es
    # danebenzuschreiben. Die eigentliche Absicherung uebernehmen hier R2 und der
    # Abstand E2 zu E3, die beide auf einen falschen Massstab ansprechen.
    if abs(td.scale_drift) > MAX_SCALE_DRIFT and td.scale_spread > MAX_SCALE_SPREAD:
        warn.append(f"Die Beinlängen ändern sich im Flug um {td.scale_drift*100:+.0f} %, "
                    f"die vier Segmente aber uneinheitlich (Spanne "
                    f"{td.scale_spread*100:.0f} Prozentpunkte). Das spricht für eine "
                    f"schräge Kameraachse oder unsichere Keypoints, nicht für eine "
                    f"Bewegung zur Kamera. Filme genau von der Seite.")
    if td.time_scale != 1.0:
        warn.append(f"Die Zeitachse stammt aus der von dir angegebenen Aufnahmerate "
                    f"({td.fps:.0f} fps), nicht aus dem Video "
                    f"({td.fps * td.time_scale:.0f} fps im Container). Stimmt die "
                    f"Angabe nicht, ist die Höhe quadratisch daneben.")
    if corrected:
        warn.append("Ereignisframes wurden manuell korrigiert.")
    return warn


def _measure_trajectory(td: TrackData, *, n0: int, n1: int, t0: float, t1: float,
                        t_f: float, jump_type: str, mass_kg: float | None,
                        seed: int, corrected: bool) -> Result:
    """Der Normalfall: E2 aus dem Scheitel der gefitteten Huefte-Parabel.

    Wirft `TrajectoryUnusable`, wenn die Bahn selbst nicht taugt - dann faengt
    `measure` das ab und liefert die Flugzeit-Hoehe. Ein `Refusal`, das **kein**
    `TrajectoryUnusable` ist (gestreckte Zeitachse ueber
    `_check_scale_plausible`), geht bewusst durch: dort steht die Zeitachse in
    Frage, und daran haengt E1 genauso.
    """
    t = td.t
    n_air = n1 - n0
    # Perspektive. Steht sie schief, ist der **Massstab** hin und damit jede
    # Hoehe aus der Bahn - die Flugzeit merkt davon nichts, denn h = g*t^2/8
    # kennt keine Pixel. Darum kein hartes Nein, sondern der Rueckfall auf E1.
    # Nach den Timing-Pruefungen, weil die von der Perspektive unberuehrt sind.
    if abs(td.scale_drift) > MAX_SCALE_DRIFT and td.scale_spread <= MAX_SCALE_SPREAD:
        raise TrajectoryUnusable(
            f"Die Person wird während des Flugs {td.scale_drift*100:+.0f} % grösser "
            f"im Bild - gemessen an Ober- und Unterschenkel, und Knochen wachsen "
            f"nicht. Alle vier Segmente ändern sich dabei im Gleichschritt "
            f"(Spanne {td.scale_spread*100:.0f} Prozentpunkte), das ist eine "
            f"Entfernungsänderung. Du springst also auf die Kamera zu oder von ihr "
            f"weg. Damit ändert sich der Massstab mitten im Sprung und die Höhe ist "
            f"nicht bestimmbar. Stelle die Kamera seitlich auf, 3-4 m entfernt, und "
            f"springe senkrecht zur Blickrichtung - ohne vor- oder zurückzuspringen."
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
    _check_scale_plausible(td, s)
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

    warn = _warnings(td, corrected=corrected)

    if fit_hip.r2 < MIN_R2:
        raise TrajectoryUnusable(
            f"Die Hüfte folgt keiner sauberen Wurfparabel (R2 = {fit_hip.r2:.4f}, "
            f"nötig > {MIN_R2}). Meist heisst das weite Kleidung oder eine teilweise "
            f"verdeckte Person."
        )
    if disagreement > MAX_DISAGREE_M:
        raise TrajectoryUnusable(
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
        raise TrajectoryUnusable(
            f"Widerspruch in der Physik: die Flugzeit erlaubt höchstens "
            f"{h_e1*100:.1f} cm, die Hüfte steigt aber auf {h_e2*100:.1f} cm. Der "
            f"Absprungzeitpunkt liegt zu früh."
        )
    if fit_hip.cubic_t > MAX_CUBIC_T:
        raise TrajectoryUnusable(
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


def _measure_flight_time(td: TrackData, *, n0: int, n1: int, t0: float, t1: float,
                         t_f: float, jump_type: str, mass_kg: float | None,
                         corrected: bool, grund: str) -> Result:
    """Zweite Wahl: die Hoehe allein aus der Flugzeit, `h = g*t_f^2/8`.

    Das ist die Methode, die jede andere Sprung-App benutzt, und sie hat genau
    eine Voraussetzung: gleiche Schwerpunkthoehe bei Absprung und Landung. Die
    ist verletzt (man springt gestreckt ab und landet gebeugt), **deshalb
    ueberschaetzt E1 systematisch** - auf der sauberen Referenz um 3.0 cm, auf
    einem Anlaufsprung um 24.3 cm. Um wie viel es im Einzelfall ist, sagt nur
    die Parabel, und die ist hier gerade weggefallen. Der Wert kommt darum mit
    Ansage und nicht als Referenzwert.

    Was E1 dafuer **nicht** braucht: einen Massstab, eine Pose, eine Perspektive.
    In der Formel steht keine einzige Pixelkoordinate. Genau deshalb ueberlebt
    sie, was E2 umbringt - verrauschte Keypoints, schraege Kameraachse,
    Massstabsaenderung im Flug. Was sie **doch** braucht, sind die beiden
    Kontaktframes und eine richtige Zeitachse; steht davon etwas in Frage, hat
    `measure` schon vorher verweigert.

    Die Kontaktzeit bleibt (sie kommt aus demselben Signal), die Abdruckdauer
    faellt weg: die haengt am tiefsten Punkt der Huefte - also an genau der
    Spur, die gerade als unbrauchbar gilt. Ohne sie gibt es keine mittlere
    Kraft. Der Impuls bleibt, weil v0 hier aus der Flugzeit kommt.
    """
    t = td.t
    h_e1 = G * t_f * t_f / 8.0
    v0 = G * t_f / 2.0          # dieselbe Symmetrieannahme wie E1, nicht mehr
    sel = slice(n0 + 1, n1 - 1)
    with np.errstate(invalid="ignore"):
        conf = (float(np.nanmean(td.confidence[sel]))
                if sel.stop > sel.start else float("nan"))

    k = n0 - 1
    while k > 0 and td.contact[k] >= 0.5:
        k -= 1
    ct = float(t0 - t[k + 1]) if n0 - (k + 1) >= 2 else None
    contact_time = ct if (ct is not None and ct <= 1.5) else None

    warn = _warnings(td, corrected=corrected)
    warn.insert(0, f"Die Flugbahn liess sich nicht auswerten: {grund} "
                   f"Die Höhe kommt deshalb allein aus der Flugzeit "
                   f"(h = g·t²/8). Sie überschätzt bauartbedingt, weil man "
                   f"gestreckt abspringt und gebeugt landet - auf sauberen "
                   f"Aufnahmen um 3 bis 6 cm, bei tiefer Landung deutlich mehr. "
                   f"Absprung, Landung und Flugzeit stimmen davon unberührt.")

    last = t.size - 1
    n1c = min(n1, last)
    return Result(
        h_e1=h_e1, h_e2=None, h_e3=None, h_e4=None,
        spread=None, disagreement=None,
        t_f=t_f, v0=v0, v_land=None, delta=None,
        t_takeoff=t0, t_land=t1,
        takeoff_frame=n0 + (t0 - t[n0]) / max(t[min(n0 + 1, last)] - t[n0], 1e-9),
        landing_frame=n1c + (t1 - t[n1c]) / max(t[min(n1c + 1, last)] - t[n1c], 1e-9),
        contact_time=contact_time, push_time=None, rsi=None,
        impulse=(mass_kg * v0) if mass_kg else None, mean_force=None,
        mass_kg=mass_kg,
        scale=None, a_px=None, r2_hip=None, r2_com=None, rms_hip_px=None,
        cubic_t=None, n_flight_frames=n1 - n0, mean_confidence=conf,
        shift_residual_px=td.shift_residual, fps_measured=td.fps,
        fit_c0=None, fit_c1=None, fit_c2=None, fit_t_ref=None,
        jump_type=jump_type, quality=None, mode="flugzeit", fallback_reason=grund,
        corrected=corrected, warnings=warn,
    )


def analyze(path: str | Path, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, scale: float | None = None, model: str = "full",
            track_out: str | Path | None = None,
            capture_fps: float | None = None) -> Result:
    """Bequemlichkeit für die CLI: extrahieren und messen in einem Zug."""
    td = extract(path, scale=scale, model=model, capture_fps=capture_fps)
    if track_out is not None:
        td.save(track_out)
    return measure(td, jump_type=jump_type, mass_kg=mass_kg, seed=seed)
