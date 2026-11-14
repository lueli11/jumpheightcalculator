"""S1 - Pose-Zweig.

Liefert Trajektorien und die Bezugsgrössen, mit denen alle Schwellwerte
normiert werden. Das Timing kommt ausdrücklich **nicht** von hier (R3): beim
Abdruck streckt sich der Fuss, der Knöchel steigt also bereits, während die
Zehe noch aufliegt. Der Pose-Zweig sagt, *wo* man hinschauen muss.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .ingest import VideoInfo, iter_frames, scaled_size

# MediaPipe-Landmark-Indizes
NOSE = 0
EAR = (7, 8)
SHOULDER = (11, 12)
ELBOW = (13, 14)
WRIST = (15, 16)
HIP = (23, 24)
KNEE = (25, 26)
ANKLE = (27, 28)
HEEL = (29, 30)
TOE = (31, 32)

# Segmentmassen und Schwerpunktlagen nach Winter (2009), Anteile der Körpermasse.
# Gebraucht für E3: im Flug folgt nur der *Schwerpunkt* exakt einer Parabel,
# jeder einzelne Landmark ist Schwerpunkt plus Relativbewegung.
_SEGMENTS = (
    (0.081, "ear", None, 0.0),
    (0.497, "shoulder", "hip", 0.50),
    (0.056, "shoulder", "elbow", 0.436),
    (0.032, "elbow", "wrist", 0.430),
    (0.012, "wrist", None, 0.0),
    (0.200, "hip", "knee", 0.433),
    (0.093, "knee", "ankle", 0.433),
    (0.029, "heel", "toe", 0.50),
)


@dataclass(frozen=True)
class PoseTrack:
    t: np.ndarray             # (N,) Zeitstempel [s]
    xy: np.ndarray            # (N, 33, 2) Bildkoordinaten [px], NaN wo nichts erkannt
    vis: np.ndarray           # (N, 33) Sichtbarkeit [0,1]
    width: int
    height: int

    def pt(self, idx) -> np.ndarray:
        """Mittelpunkt eines Landmark-Paars (oder eines einzelnen), (N,2)."""
        j = list(idx) if isinstance(idx, (tuple, list)) else [idx]
        return np.nanmean(self.xy[:, j, :], axis=1)

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.xy[:, HIP[0], 0])

    def com(self) -> np.ndarray:
        """Ganzkörper-Schwerpunkt nach Winter, (N,2)."""
        named = {"ear": self.pt(EAR), "shoulder": self.pt(SHOULDER),
                 "elbow": self.pt(ELBOW), "wrist": self.pt(WRIST),
                 "hip": self.pt(HIP), "knee": self.pt(KNEE),
                 "ankle": self.pt(ANKLE), "heel": self.pt(HEEL), "toe": self.pt(TOE)}
        acc = np.zeros_like(named["hip"])
        for w, a, b, f in _SEGMENTS:
            p = named[a] if b is None else named[a] + (named[b] - named[a]) * f
            acc += w * p
        return acc / sum(s[0] for s in _SEGMENTS)

    def body_scale(self) -> float:
        """Körperlänge Schulter-Knöchel [px]. Normiert alle Schwellwerte,
        damit nichts von Auflösung oder Kameraabstand abhängt."""
        d = np.linalg.norm(self.pt(SHOULDER) - self.pt(ANKLE), axis=1)
        return float(np.nanmedian(d))

    def ground_line(self, before: int) -> float:
        """Bodenlinie = Median der Knöchelhöhe über die Standphase.

        `before` ist der Frame, vor dem gemittelt wird - die Gegenbewegung und
        alles danach bleiben draussen.
        """
        ank = self.pt(ANKLE)[:before, 1]
        ank = ank[np.isfinite(ank)]
        if ank.size < 5:
            raise ValueError("Standphase zu kurz für eine Bodenlinie.")
        return float(np.median(ank))


def smooth(y: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay statt gleitendem Mittel.

    Ein gleitendes Mittel ist ein Tiefpass und dämpft genau die Krümmung, die
    später gefittet wird - es würde die Schwerkraft kleiner erscheinen lassen
    und über R5 den Massstab verfälschen. Savitzky-Golay legt lokal ein Polynom
    und lässt die zweite Ableitung stehen.
    """
    n = y.size
    w = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    if w <= order + 1:
        return y.copy()
    return savgol_filter(y, w, order)


def track(info: VideoInfo, *, model: str = "full", scale: float = 0.25,
          min_conf: float = 0.3) -> PoseTrack:
    """Keypoints je Frame.

    Bewusst im IMAGE-Modus, nicht im VIDEO-Modus: der VIDEO-Modus glättet
    zeitlich (`smooth_landmarks`), und das ist ein Tiefpass auf der Krümmung,
    die R4 fittet. Ausserdem nimmt `detect_for_video` ganzzahlige Millisekunden -
    bei 240 fps sind das 4.17 ms je Frame. Unsere Zeitachse kommt ausschliesslich
    aus den Container-PTS.
    """
    from pathlib import Path

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    asset = Path(__file__).parent / "models" / f"pose_landmarker_{model}.task"
    if not asset.exists():
        raise FileNotFoundError(
            f"Pose-Modell fehlt: {asset}. Einmalig herunterladen von "
            f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{model}/float16/latest/pose_landmarker_{model}.task"
        )

    w, h = scaled_size(info, scale)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf)

    n = info.n_frames
    xy = np.full((n, 33, 2), np.nan, np.float32)
    vis = np.full((n, 33), np.nan, np.float32)
    with vision.PoseLandmarker.create_from_options(opts) as det:
        for i, frame in enumerate(iter_frames(info, scale=scale)):
            if i >= n:
                break
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks[0]):
                    # Bild-Landmarks, nicht World-Landmarks: gemessen wird in
                    # Pixeln, und der Massstab kommt aus R5.
                    xy[i, j, 0] = lm.x * w
                    xy[i, j, 1] = lm.y * h
                    vis[i, j] = lm.visibility
    return PoseTrack(t=info.t, xy=xy, vis=vis, width=w, height=h)
