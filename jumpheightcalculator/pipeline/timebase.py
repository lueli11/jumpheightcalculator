"""Verweigerungen (R7) und die Prüfungen, die hinter der Zeitachse stehen.

Hier stehen alle Checks, die eine falsche **Zeitachse** aufdecken: gestreckte
Slo-Mo-Exports, falsch angegebene Aufnahmeraten, daraus folgende unmögliche
Körpergrössen. Die Bahn-Verweigerungen (R2 der Parabel, E2/E3-Abstand, ...)
gehören dagegen zu `measurement` - dort, wo der Fit läuft.
"""

from __future__ import annotations

import numpy as np

from ..physics import FitError, fit_parabola
from .limits import BODY_LEN_BOUNDS_M, MAX_FLIGHT_S, MIN_R2
from .track import TrackData


class Refusal(RuntimeError):
    """R7 - lieber keinen Wert als einen falschen, und immer mit Grund."""


class TrajectoryUnusable(Refusal):
    """Die **Bahn** taugt nicht - das Timing kann trotzdem stimmen.

    Der Unterschied entscheidet, ob es eine Zahl gibt. E2 kommt aus der Form der
    Flugkurve und fällt mit ihr; E1 kommt aus `h = g*t^2/8` und braucht nur die
    beiden Kontaktframes, keinen Massstab und keine einzige Pose-Koordinate.
    Eine unbrauchbare Hüfte macht also E2 unmöglich und E1 nicht.

    Umgekehrt gilt das nicht: steht das Timing in Frage (zu wenige Flugframes,
    unplausible Flugzeit, Kontaktsignal gegen Körperbewegung, gestreckte
    Zeitachse), fällt E1 mit - dann bleibt es bei der Verweigerung.
    """


# Bildraten, die es an Kameras wirklich gibt. Die Schätzung aus der Geometrie
# ist ungenau (sie unterstellt 1.35 m Schulter-Knöchel), und wer den rohen Wert
# abtippt, rechnet quadratisch daneben: 129 statt 120 sind 13 % zu wenig Höhe.
# Darum wird die Schätzung auf die nächste **echte** Rate gerundet und die
# rohe Zahl nur noch als Herkunft genannt.
CAMERA_FPS = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 100.0, 120.0, 200.0, 240.0,
              480.0, 960.0)


def _nearest_camera_fps(fps: float) -> float:
    """Die nächste übliche Aufnahmerate - relativ gemessen, nicht absolut.

    Absolut wäre 129 gleich weit von 120 wie von 138; relativ zählt aber das
    Verhältnis, denn genau das geht quadratisch in die Höhe ein.
    """
    return min(CAMERA_FPS, key=lambda f: abs(np.log(f / max(fps, 1e-9))))


def _timebase_advice(time_scale: float, fps: float, k: float) -> str:
    """Der Rat hängt davon ab, ob schon eine Aufnahmerate angegeben wurde.

    Ohne Angabe ist der nächste Schritt der Export mit Originalgeschwindigkeit.
    Mit Angabe wäre derselbe Rat zirkulär - dann ist die **Angabe** falsch, und
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


def _shoulder_ankle_px(xy: np.ndarray) -> float:
    """Schulter-Knöchel in Pixeln, p90 - also nahe der gestreckten Haltung."""
    sch = np.nanmean(xy[:, [11, 12]], axis=1)
    ank = np.nanmean(xy[:, [27, 28]], axis=1)
    with np.errstate(invalid="ignore"):
        return float(np.nanpercentile(np.linalg.norm(sch - ank, axis=1), 90))


def _refuse_if_stretched(t: np.ndarray, hip_y: np.ndarray, xy: np.ndarray,
                         n0: int, n1: int, dur: float,
                         time_scale: float = 1.0) -> None:
    """Zu lang **und** saubere Parabel **und** unmögliche Grösse -> Zeitachse.

    Alle drei Bedingungen braucht es. Ein zu langes Luftfenster allein ist der
    Normalfall: nach der Landung geht die Person weg, auf der sauberen Referenz
    2.58 s lang. Das ist aber keine Parabel (R2 0.956 gegen 0.996), und daran
    wird es unterschieden. Die Grössenprobe ist die dritte Sicherung.
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
    len_m = _shoulder_ankle_px(xy) * fit.scale
    if BODY_LEN_BOUNDS_M[0] <= len_m <= BODY_LEN_BOUNDS_M[1]:
        return
    k = (len_m / 1.35) ** 0.5
    fps = 1.0 / max(float(np.median(np.diff(t))), 1e-9)
    raise Refusal(
        f"Die Person ist {dur:.1f} s am Stück in der Luft und fliegt dabei eine "
        f"saubere Wurfparabel - beides zusammen geht nicht (möglich sind "
        f"{MAX_FLIGHT_S:.2f} s). Der Massstab daraus ergäbe eine Person von "
        f"{len_m:.0f} m. Nicht die Aufnahme ist schuld, sondern die Zeitachse: "
        f"sie ist rund {k:.0f}-fach gestreckt. "
        + _timebase_advice(time_scale, fps, k)
    )


def _check_scale_plausible(td: TrackData, s: float) -> None:
    """Gegenprobe zur Selbstkalibrierung: wie gross wäre die Person demnach?

    R5 leitet den Massstab aus der Krümmung ab, `s = g / a_px`. Ist die Zeitachse
    um k gestreckt, fällt a_px um k^2 und der Massstab wächst um k^2 - lautlos,
    denn die Bahn bleibt eine saubere Parabel (auf IMG_4160 R2 = 0.997). Die
    Flugzeitgrenze fängt grosse k ab, ein doppelt gestrecktes Video aber nicht:
    aus 0.45 s werden 0.9 s, und das ist eine völlig normale Flugzeit.

    Was auffliegt, ist die Grösse. Schulter-Knöchel sind bei einem Menschen rund
    1.0-1.8 m. Gemessen: saubere Referenz 1.42 m, IMG_0366 1.80 m,
    **IMG_4160 24.06 m**. Die Grenzen sitzen bewusst weit, das ist eine
    Absurditätsprüfung und keine Anthropometrie.
    """
    px = _shoulder_ankle_px(td.xy)
    if not np.isfinite(px) or px <= 0:
        return
    len_m = px * s
    if BODY_LEN_BOUNDS_M[0] <= len_m <= BODY_LEN_BOUNDS_M[1]:
        return
    k = (len_m / 1.35) ** 0.5
    richtung = "gestreckt" if k > 1 else "gestaucht"
    raise Refusal(
        f"Der Massstab aus der Physik ergibt eine Person von {len_m:.1f} m "
        f"(Schulter bis Knöchel) - das kann nicht sein. Nicht die Aufnahme ist "
        f"schuld, sondern die Zeitachse: sie ist um rund das "
        f"{k:.1f}-fache {richtung}. " + _timebase_advice(td.time_scale, td.fps, k)
    )
