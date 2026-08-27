"""Kugel-Tracker.

Dient zwei Zwecken: der Kernphysik-Prüfung auf dem synthetischen Testvideo (P0)
und dem Ballwurf-Test auf echtem Material (P1.4) - laut Konzept die beste
verfuegbare Referenz, weil sie die Pipeline gegen reine Physik prüft, ohne dass
jemand springen muss. Kein ML noetig: Schwellwert und Schwerpunkt reichen.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import G, PIPELINE_VERSION
import dataclasses

from .ingest import VideoInfo, iter_frames, probe
from .physics import FitError, ParabolaFit, fit_parabola, longest_run

# Dieselben Grenzen wie in der Sprung-Pipeline - ein Ballwurf-Test, der lockerer
# geprüft wird als der Sprung, taugt nicht als Referenz für den Sprung.
MIN_R2 = 0.98
MIN_FLIGHT_S, MAX_FLIGHT_S = 0.15, 1.10


@dataclass(frozen=True)
class BallResult:
    h_e1: float          # aus der Flugzeit [m]
    h_e2: float          # aus dem Scheitel der Parabel [m] - Referenzwert
    t_f: float           # Flugzeit [s]
    v0: float            # Absprunggeschwindigkeit [m/s]
    a_px: float          # gefittete Krümmung [px/s^2]
    scale: float         # m/px, aus R5 selbstkalibriert
    t_takeoff: float
    t_land: float
    r2: float
    rms_resid_px: float
    cubic_t: float
    n_flight_frames: int
    n_detected: int
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def track(info: VideoInfo, *, dark: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schwerpunkt der Kugel je Frame. Gibt (t, y, gefunden) zurück.

    Der Schwerpunkt wird intensitaetsgewichtet berechnet, nicht als Mittelpunkt
    der Bounding Box: eine kantengeglaettete Kugel laesst sich damit auf deutlich
    unter einem Pixel genau lokalisieren, und genau davon haengt die Krümmung ab.
    """
    ys = np.full(info.n_frames, np.nan)
    xs = np.full(info.n_frames, np.nan)

    for i, frame in enumerate(iter_frames(info)):
        if i >= info.n_frames:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if not dark:
            g = 255.0 - g

        # Otsu taugt hier nicht: es trennt die zwei *groessten* Klassen, und das
        # ist bei einer Bodenlinie im Bild der Boden gegen den Hintergrund - die
        # Kugel fällt dann mit dem Boden in dieselbe Klasse. Sie ist aber das
        # dunkelste Objekt im Bild, also wird relativ zu Minimum und Median
        # geschwellt.
        lo = float(np.percentile(g, 0.02))
        mid = float(np.median(g))
        thr = lo + 0.40 * (mid - lo)
        mask = (g < thr).astype(np.uint8) * 255

        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lab < 2:
            continue
        cand = [k for k in range(1, n_lab) if stats[k, cv2.CC_STAT_AREA] >= 12]
        if not cand:
            continue
        # Nicht die groesste Komponente, sondern die dunkelste: eine grosse
        # Schattenflaeche wäre sonst staerker als die Kugel.
        k = min(cand, key=lambda j: float(g[labels == j].mean()))

        x0, y0 = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        p_ = 2
        sl = (slice(max(0, y0 - p_), min(info.height, y0 + h + p_)),
              slice(max(0, x0 - p_), min(info.width, x0 + w + p_)))
        patch = g[sl]
        wgt = np.clip(thr - patch, 0, None)
        tot = float(wgt.sum())
        if tot <= 0:
            continue
        rr, cc = np.mgrid[sl[0].start:sl[0].stop, sl[1].start:sl[1].stop]
        ys[i] = float((wgt * rr).sum() / tot)
        xs[i] = float((wgt * cc).sum() / tot)

    return info.t, ys, np.isfinite(ys)


def analyze(path: str | Path, *, seed: int = 0,
            assume_fps: float | None = None) -> BallResult:
    """Kernphysik: Kugel finden, Parabel fitten, selbstkalibrieren.

    `assume_fps` ersetzt die echten Containerzeitstempel durch ein gleichmaessiges
    Raster `n/fps` - also genau das, was ein Werkzeug tut, das der Aufnahmemodus-
    Angabe glaubt statt in den Container zu schauen. Das ist **kein** Betriebsmodus,
    sondern die Negativprobe zu R1: sie muss sichtbar danebenliegen, sonst ist
    nicht belegt, dass die Zeitstempel überhaupt tragen.
    """
    info = probe(path)
    if assume_fps is not None:
        info = dataclasses.replace(
            info, t=np.arange(info.n_frames, dtype=np.float64) / assume_fps)
    t, y, ok = track(info)
    if ok.sum() < 10:
        raise FitError(
            f"Kugel nur in {int(ok.sum())} von {info.n_frames} Frames gefunden - "
            f"zu wenig für eine Messung."
        )

    tv_all, yv_all = t[ok], y[ok]

    # Erster Durchgang nur, um das Flugfenster zu finden: die Ruheframes liegen
    # nicht auf der Parabel und fallen als Ausreisser heraus.
    coarse = fit_parabola(tv_all, yv_all, seed=seed)
    i0, i1 = longest_run(coarse.inliers)

    # R4 - die beiden Randframes raus. Am Übergang ist der Punkt weder Ruhe noch
    # sauberer Flug (auf echtem Material zusaetzlich bewegungsunscharf), und ein
    # einziger solcher Punkt kippt die Krümmung: gemessen 9.797 statt 9.810 m/s^2,
    # also 0.14 % Höhenfehler aus einem einzigen Frame.
    sel = slice(i0 + 1, i1 - 1)
    if (i1 - 1) - (i0 + 1) < 8:
        raise FitError(
            f"Nur {max(0, (i1-1)-(i0+1))} saubere Flugframes erfasst - zu wenig für "
            f"einen Fit. Bitte in Slo-Mo mit mindestens 120 fps aufnehmen."
        )
    fit = fit_parabola(tv_all[sel], yv_all[sel], seed=seed)

    # Ruheniveau: die Kugel liegt vor und nach dem Flug am Boden, also am
    # groessten y. Das ist das Bezugsniveau, über dem die Höhe gemessen wird -
    # nicht der tiefste Punkt der Parabel.
    yv = yv_all
    y_rest = float(np.median(yv[yv >= yv.max() - 2.0]))

    t0, t1 = fit.roots_at(y_rest)
    t_f = t1 - t0
    s = fit.scale

    # E1 - Flugzeit. Setzt gleiche Höhe bei Absprung und Landung voraus; bei
    # einer Kugel gilt das exakt, bei einem Menschen nie (R2).
    h_e1 = G * t_f * t_f / 8.0
    # E2 - Scheitel der gefitteten Parabel über dem Bezugsniveau. Referenzwert.
    h_e2 = (y_rest - fit.y_apex) * s
    v0 = float(fit.v_up_px(t0)) * s

    # R7 - lieber keinen Wert als einen falschen.
    #
    # Ohne diese Prüfungen gab ein absichtlich schlechtes Video (30 fps, unscharf,
    # verrauscht) glatte 113.67 cm statt der wahren 40.00 aus - eine Zahl, die man
    # ohne Gegenprobe geglaubt haette. Sichtbar war der Fehler die ganze Zeit: R2
    # 0.61 gegen 1.00000 beim sauberen Video, und eine rekonstruierte
    # Erdbeschleunigung von 2.28 statt 9.81 m/s^2.
    if fit.r2 < MIN_R2:
        raise FitError(
            f"Die Bahn folgt keiner sauberen Wurfparabel (R2 = {fit.r2:.3f}, nötig "
            f"> {MIN_R2}; Abweichung {fit.rms_resid:.1f} px je Punkt). Meist heisst "
            f"das Bewegungsunschärfe: mit kürzerer Belichtung und in Slo-Mo mit "
            f"mindestens 120 fps aufnehmen."
        )
    if not (MIN_FLIGHT_S <= t_f <= MAX_FLIGHT_S):
        raise FitError(
            f"Flugzeit {t_f*1000:.0f} ms liegt ausserhalb des Plausiblen "
            f"({MIN_FLIGHT_S*1000:.0f}-{MAX_FLIGHT_S*1000:.0f} ms). Entweder wurde "
            f"die Flugphase falsch abgegrenzt, oder die Zeitbasis stimmt nicht."
        )

    return BallResult(
        h_e1=h_e1, h_e2=h_e2, t_f=t_f, v0=v0,
        a_px=fit.a_px, scale=s, t_takeoff=t0, t_land=t1,
        r2=fit.r2, rms_resid_px=fit.rms_resid, cubic_t=fit.cubic_t,
        n_flight_frames=fit.n_inliers, n_detected=int(ok.sum()),
    )
