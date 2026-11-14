"""`extract` - der teure Teil: Decode, Pose, Pixelzweig, Fusion.

Das Ergebnis ist ein `TrackData`: daraus rechnet `measurement.measure` in
Millisekunden. Eine Korrektur der Ereignisframes im Review-Screen darf darum
nicht bedeuten, dass MediaPipe erneut über das ganze Video läuft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import pixels as pixmod
from .. import pose as posemod
from ..fusion import flight_window, fuse, normalise, viterbi
from ..ingest import probe
from .geometry import _coarse_air, _scale_drift, _stance_window, pose_scale
from .timebase import Refusal
from .track import TrackData


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
    # den Absprung sieben Frames zu früh und blies dadurch E2 auf 90 cm auf -
    # über E1 von 82 cm, was für einen freien Flug unmöglich ist (R2 verlangt
    # die umgekehrte Reihenfolge).
    #
    # Je Streifen einzeln normiert, dann das Maximum: Kontakt heisst "irgendwo
    # steht ein Fuss". Gemeinsam normiert drückt der kontrastreichere Streifen
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
