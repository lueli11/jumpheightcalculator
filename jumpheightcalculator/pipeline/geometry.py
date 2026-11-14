"""Geometrie vor dem Fit: Flugfenster, Standphase, Grössenänderung.

Diese Helfer bestimmen, **wo** gerechnet wird - der Fit und die Schätzer
stehen in `measurement`. Alles hier läuft nach der Pose, vor dem ersten
fit_parabola, und entscheidet darüber, ob es überhaupt ein Ergebnis gibt.
"""

from __future__ import annotations

import numpy as np

from .. import pose as posemod
from .limits import MAX_FLIGHT_S, MIN_FLIGHT_S
from .timebase import Refusal, _refuse_if_stretched

# Zielbreite für den Pose-Zweig. Ein 4K-Video wird darauf heruntergerechnet,
# weil MediaPipe intern ohnehin kleiner arbeitet und die Rohpipe sonst der
# Flaschenhals ist. Ein **fester** Faktor wäre falsch: ein über WhatsApp
# geschicktes Video kommt mit 464 px Breite an, ein Viertel davon sind 116 px,
# und darauf findet das Modell keine brauchbaren Keypoints mehr. Hochskaliert
# wird nie - erfundene Pixel sind keine Information.
TARGET_POSE_WIDTH = 540


def pose_scale(width: int) -> float:
    return min(1.0, TARGET_POSE_WIDTH / max(width, 1))


def _coarse_air(track: posemod.PoseTrack, t: np.ndarray,
                time_scale: float = 1.0) -> tuple[int, int]:
    """Grobes Flugfenster aus der Knöchelhöhe.

    Nur zur Platzierung des ROI und als Referenz für den leeren Boden -
    **nicht** als Timing (R3). Der Knöchel steigt beim Abdruck bereits, während
    die Zehe noch aufliegt, und meldet den Absprung dadurch deutlich zu früh.
    """
    ank = track.pt(posemod.ANKLE)[:, 1]
    bs = track.body_scale()
    base = float(np.nanmedian(ank[:max(10, len(ank) // 5)]))
    lift = np.nan_to_num(base - ank, nan=-1e9)
    air = lift > 0.05 * bs

    # Genommen wird der Abschnitt mit dem **höchsten** Knöchelhub, nicht der
    # längste. Nach der Landung geht die Person weg, und dieses Weggehen ist
    # leicht länger als der Sprung selbst - auf einem Ausschnitt wurde so das
    # Fenster 252-360 statt 148-241 gewählt und der Fit krümmte sich nach oben.
    # Hoch springt man nur einmal; herumlaufen kann man beliebig lange.
    best, best_lift = None, -np.inf
    zu_lang = None                            # längster Block über der Grenze
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

    # Ein Knöchelhub, der länger dauert als physikalisch möglich, **und** dabei
    # einer sauberen Wurfparabel folgt, deren Massstab eine unmögliche
    # Körpergrösse ergibt: dann ist nicht die Bewegung falsch, sondern die
    # Zeitachse. Auf IMG_4160 (iPhone-Slo-Mo, bereits verlangsamt exportiert)
    # meldet der Container ehrlich 30 fps, die Zeitleiste ist aber vierfach
    # gestreckt - der Sprung sieht 2.8 s lang aus und fällt hier heraus, übrig
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


def _stance_window(track: posemod.PoseTrack, until: int) -> slice:
    """Die letzte Standphase vor dem Absprung: wo der Fuss zuletzt stillstand.

    Gesucht ist das **längste** Fenster vor `until`, in dem die Fuss-x-Position
    innerhalb einer halben Fusslänge bleibt. Bei Gleichstand gewinnt das
    spätere - der Fuss soll dort liegen, wo er beim Abdruck liegt.

    Vorher wurde blind das Fenster [t0-1.0s, t0-0.5s] genommen. Das stimmt nur,
    wenn die Person die ganze Zeit stillsteht. Auf IMG_0366 läuft sie ins Bild
    und stellt sich erst 0.25 s vor dem Sprung hin: der Streifen landete bei
    Fuss-x 38 statt 204 px - sechs Fusslängen daneben. Er lief damit leer,
    sobald die Person daran vorbeiging, und der Absprung wurde 41 Frames
    (342 ms) zu früh gemeldet.

    **Nicht** zu verwechseln mit dem verworfenen Ansatz "Streifen an den Fuss im
    Absprungmoment": gemittelt wird weiterhin über die ganze Standphase, nur
    über die richtige.
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

    # Hinten auf eine Sekunde begrenzt. Der Streifen ist der Median über das
    # Fenster, und wer lange dasteht, scharrt: auf IMG_5418 lief das Fenster
    # über 5.2 s, der Streifen wanderte dadurch und das Flugfenster fiel auf
    # 88 statt 93 Frames. Mit Deckel sind es 92 - ein Frame von der
    # dokumentierten Referenz. Auf IMG_0366 ändert der Deckel nichts, dort
    # steht die Person ohnehin nur 33 Frames.
    dt = float(np.median(np.diff(track.t))) if track.t.size > 1 else 1 / 120
    max_len = max(5, int(round(1.0 / max(dt, 1e-6))))
    return slice(max(best[0], best[1] + 1 - max_len), best[1] + 1)


def _scale_drift(track: posemod.PoseTrack, n0: int, n1: int) -> tuple[float, float]:
    """Grössenänderung starrer Segmente **innerhalb des Flugs**.

    Gemessen wird, was die Verweigerung auch behauptet: ob die Person während
    des Flugs grösser wird. Vorher stand hier Stand gegen Flug - das misst zu
    einem grossen Teil die **Haltung**, nicht die Entfernung. In der Hocke zeigen
    gebeugte Glieder bei nicht exakt seitlicher Kamera aus der Bildebene heraus
    und sind dadurch verkürzt; auf IMG_0366 kam so +19 % heraus, obwohl die
    Person vor dem Sprung (Schulter-Knöchel 150.1 px) und nach der Landung
    (143.5 px) nachweislich gleich weit weg steht.

    Zurückgegeben wird das Paar (Aenderung, Streuung):

    * **Aenderung** ist der Median über die vier Segmente von erstem gegen
      letztes Flugdrittel. Median gegen Median, damit weder der Hocker in der
      Mitte noch ein einzelner kaputter Frame durchschlägt - auf IMG_0366 sind
      Frame 101 und 109 Pose-Müll (Oberschenkel 22 px statt 65).
    * **Streuung** ist die Spanne derselben vier Werte. Eine echte
      Entfernungsänderung skaliert **alle** Segmente gleich; streuen sie, ist es
      Haltung oder Pose-Rauschen. Gemessen: IMG_0366 [7.1, 7.6, 19.1, 22.0] -
      15 Punkte auseinander, während die saubere Referenz IMG_5418 auf 5 Punkte
      kommt. Das deckt sich mit der dokumentierten Beobachtung, dass ein sauberes
      Video um 3 % streut.

    Die Körperlänge Schulter-Knöchel taugt hier nicht: sie ändert sich schon
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
        early = np.nanmedian(lens[:, n0:n0 + k], axis=1)
        late = np.nanmedian(lens[:, n1 - k:n1], axis=1)
    ok = np.isfinite(early) & np.isfinite(late) & (early > 0)
    if ok.sum() < 2:
        return 0.0, 0.0
    r = late[ok] / early[ok] - 1.0
    return float(np.median(r)), float(r.max() - r.min())
