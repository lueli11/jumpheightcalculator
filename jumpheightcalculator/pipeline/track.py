"""TrackData - alles, was `measure` braucht, ohne Video und ohne Pose-Modell."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrackData:
    """Wird als NPZ neben der Analyse abgelegt und speist sowohl das Overlay im
    Review-Screen als auch die Neuberechnung nach einer Korrektur."""
    t: np.ndarray            # (N,) Zeitstempel [s], aus dem Container (R1)
    xy: np.ndarray           # (N, 33, 2) alle Landmarks [px] - für das Skelett-Overlay
    width: int               # Arbeitsauflösung, in der xy gilt
    height: int
    hip_y: np.ndarray        # (N,) Hüfthöhe [px], Bildkoordinaten
    com_y: np.ndarray        # (N,) Schwerpunkthöhe [px]
    ankle_y: np.ndarray      # (N,) für die Anzeige
    contact: np.ndarray      # (N,) Kontaktsignal c(n) in [0,1], 1 = Kontakt
    confidence: np.ndarray   # (N,) mittlere Keypoint-Konfidenz
    n0: int                  # Flugfenster aus dem HMM (Kontaktsignal, R3)
    n1: int
    coarse_n0: int           # Flugfenster aus dem Knöchelhub - Gegenprobe
    coarse_n1: int
    stance_start: int        # letzte echte Standphase vor dem Absprung
    stance_end: int
    fps: float
    shift_residual: float
    body_scale_px: float
    scale_drift: float       # Grössenänderung starrer Segmente im Flug
    scale_spread: float = 0.0  # Spanne derselben Aenderung über die vier Segmente
    # Faktor, mit dem die Container-PTS gestaucht wurden (angegebene Aufnahmerate).
    # 1.0 heisst unverändert. Das Vorschauvideo behält die **Container**-Achse -
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
