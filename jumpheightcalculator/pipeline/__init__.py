"""S4/S5 - die Pipeline-Fassade: extract, measure, TrackData, Result, Verweigerungen.

Die vier Höhenschätzer und was sie unterscheidet:

  E1  g*t_f^2/8, t_f aus dem **Kontaktsignal**. Setzt gleiche Schwerpunkthöhe
      bei Absprung und Landung voraus - man springt aber gestreckt ab und landet
      gebeugt (R2). Überschätzt darum systematisch; dient als Vergleichswert.
  E2  Scheitel der Hüfte-Parabel über der Hüfthöhe zum Absprungzeitpunkt.
      **Referenzwert der App.**
  E3  Dasselbe auf einem eigenen Fit des Ganzkörper-Schwerpunkts (Winter 2009).
  E4  0.5*g*t_fall^2, nur die Fallphase (Scheitel bis Landung aus dem
      Kontaktsignal). Liegt systematisch bei E2 + Delta.

Der teure Teil (`extract`) und der billige (`measure`) sind getrennt: eine
Korrektur der Ereignisframes um +/-1 im Review-Screen darf nicht bedeuten, dass
MediaPipe erneut über das ganze Video läuft.

Die Untermodule:

  limits.py       alle Schwellen mit ihrer gemessenen Begründung
  track.py        TrackData (NPZ: Keypoints, Kontaktsignal, Zeitstempel)
  result.py       Result (Messwerte, Güte, Warnungen)
  timebase.py     Verweigerungsklassen und alle Zeitachsen-Checks
  geometry.py     Flugfenster, Standphase, Skalen-Drift (vor dem Fit)
  extraction.py   extract - Decode, Pose, Pixelzweig, Fusion
  measurement.py  measure - Fit, Schätzer E1-E4, Güte, Rückfall auf die Flugzeit

Die Oberfläche ist dieselbe wie vor dem Split: `pipeline.measure`,
`pipeline.TrackData`, `pipeline.Refusal`, ... funktionieren weiter, und die
Untermodule tragen die Begründungen dorthin, wo man sie braucht.
"""

from .. import PIPELINE_VERSION
from .extraction import extract
from .geometry import TARGET_POSE_WIDTH, _coarse_air, _scale_drift, _stance_window, pose_scale
from .limits import (
                       BODY_LEN_BOUNDS_M,
                       CONTACT_VS_BODY_RANGE,
                       MAX_CUBIC_T,
                       MAX_DISAGREE_M,
                       MAX_FLIGHT_S,
                       MAX_SCALE_DRIFT,
                       MAX_SCALE_SPREAD,
                       MAX_SHIFT_RESID_PX,
                       MIN_BODY_PX,
                       MIN_FLIGHT_FRAMES,
                       MIN_FLIGHT_S,
                       MIN_R2,
)
from .measurement import _measure_flight_time, _measure_trajectory, _warnings, analyze, measure
from .result import Result
from .timebase import (
                       CAMERA_FPS,
                       Refusal,
                       TrajectoryUnusable,
                       _check_scale_plausible,
                       _nearest_camera_fps,
                       _refuse_if_stretched,
                       _shoulder_ankle_px,
                       _timebase_advice,
)
from .track import TrackData

__all__ = [
    "PIPELINE_VERSION",
    "TrackData", "Result",
    "Refusal", "TrajectoryUnusable",
    "extract", "measure", "analyze",
    "pose_scale", "TARGET_POSE_WIDTH",
    "MIN_FLIGHT_S", "MAX_FLIGHT_S", "MIN_FLIGHT_FRAMES", "MIN_R2",
    "BODY_LEN_BOUNDS_M", "MAX_DISAGREE_M", "MAX_CUBIC_T", "MAX_SHIFT_RESID_PX",
    "MAX_SCALE_DRIFT", "MAX_SCALE_SPREAD", "CONTACT_VS_BODY_RANGE",
    "MIN_BODY_PX", "CAMERA_FPS",
    "_coarse_air", "_stance_window", "_scale_drift", "_shoulder_ankle_px",
    "_refuse_if_stretched", "_check_scale_plausible", "_timebase_advice",
    "_nearest_camera_fps", "_warnings", "_measure_trajectory",
    "_measure_flight_time",
]
