"""S2 - Pixel-Zweig.

Hier entsteht das Timing (R3). Der Pose-Zweig kann es nicht liefern: beim
Abdruck streckt sich der Fuss, der Knöchel steigt also schon, während die Zehe
noch aufliegt - bei 240 fps liegen dazwischen zwanzig Frames und mehr. Gemessen
wird deshalb am Auflagepunkt selbst, in einem schmalen Streifen unter dem
Fussballen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ingest import VideoInfo, iter_frames
from .pose import HEEL, HIP, TOE, PoseTrack


@dataclass(frozen=True)
class ContactSignals:
    t: np.ndarray
    occupancy: np.ndarray     # Fuss im Streifen vorhanden? (gegen leeren Boden)
    diff_prev: np.ndarray     # Bewegungsenergie zum stabilisierten Vorframe
    flow_y: np.ndarray        # vertikaler Optical Flow im Streifen
    gap: np.ndarray           # Anteil "Fuss" im Zeilenprofil unter der Zehe
    roi: tuple[int, int, int, int]
    shift_residual: float     # Restversatz des Hintergrunds [px]


def estimate_shift(frames: list[np.ndarray], person_box) -> tuple[np.ndarray, float]:
    """Globale Kamerabewegung über Hintergrund-Features (ORB + RANSAC-Median).

    Bewusst nur Translation statt voller Homographie: auf texturarmem Hintergrund
    ist eine Homographie unterbestimmt und erzeugt dann scheinbare Vertikal-
    bewegung - also genau das Signal, das hier gemessen werden soll. Der
    Restversatz wird zurückgegeben und oben geprüft.
    """
    orb = cv2.ORB_create(1200)
    ref = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask = np.full(ref.shape, 255, np.uint8)
    x0, y0, x1, y1 = person_box
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    k_ref, d_ref = orb.detectAndCompute(ref, mask)
    shifts = np.zeros((len(frames), 2), np.float64)
    if d_ref is None or len(k_ref) < 12:
        # Ohne Maske erneut versuchen. Auf einem texturarmen Hintergrund (Turnhalle,
        # glatte Wand) bleiben sonst zu wenige Ecken uebrig und die Schätzung fällt
        # ganz aus - die Wackel-Warnung koennte dann nie ausloesen. Die Person ist
        # ein kleiner Teil des Bildes, ihre Bewegung fällt beim Median heraus.
        mask = None
        k_ref, d_ref = orb.detectAndCompute(ref, None)
        if d_ref is None or len(k_ref) < 12:
            return shifts, float("nan")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    resid = []
    for i, fr in enumerate(frames[1:], 1):
        k, d = orb.detectAndCompute(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), mask)
        if d is None or len(k) < 12:
            shifts[i] = shifts[i - 1]
            continue
        m = sorted(bf.match(d_ref, d), key=lambda x: x.distance)[:200]
        if len(m) < 8:
            shifts[i] = shifts[i - 1]
            continue
        delta = (np.float32([k[x.trainIdx].pt for x in m])
                 - np.float32([k_ref[x.queryIdx].pt for x in m]))
        good = np.all(np.abs(delta - np.median(delta, 0)) < 3.0, axis=1)
        if good.sum() < 6:
            shifts[i] = shifts[i - 1]
            continue
        shifts[i] = np.median(delta[good], 0)
        resid.append(float(np.median(np.abs(delta[good] - shifts[i]))))
    return shifts, float(np.mean(resid)) if resid else float("nan")


def _roi_from_stance(track: PoseTrack, st: slice) -> tuple[int, int, int, int]:
    """Streifen unter dem Fussballen, aus einer Standphase abgeleitet."""
    toe, heel = track.pt(TOE), track.pt(HEEL)
    foot_len = float(np.nanmedian(np.linalg.norm(toe[st] - heel[st], axis=1)))
    if not np.isfinite(foot_len) or foot_len < 4:
        foot_len = track.body_scale() * 0.15
    ball = np.nanmedian(heel[st] + 0.78 * (toe[st] - heel[st]), axis=0)
    ground = float(np.nanmedian(np.nanmax(np.stack([toe[st, 1], heel[st, 1]]), axis=0)))
    w = max(8, int(1.5 * foot_len))
    h = max(4, int(0.3 * foot_len))
    x0 = max(0, min(track.width - w, int(round(ball[0] - w / 2))))
    y0 = max(0, min(track.height - h, int(round(ground - h / 2))))
    return x0, y0, w, h


def contact_signals(info: VideoInfo, track: PoseTrack, *, stance_end: int,
                    air_frames: np.ndarray, landing_stance: slice | None = None,
                    scale: float = 0.25) -> ContactSignals:
    """Vier Signale im Streifen unter dem Fussballen.

    Ausgewertet werden **zwei** Streifen, einer an der Absprung- und einer an der
    Landestelle. Gemessen an einem echten Sprung zum Korb landet die Person rund
    eine Fusslänge weiter vorne: der Absprung-Streifen kommt nach der Landung
    nur noch auf 16 statt 96 Belegungseinheiten, und der Wiederkontakt wird
    dadurch viel zu spaet erkannt - gemessen 1167 ms Flugzeit statt rund 680 ms.
    Das Kontaktsignal ist deshalb das Maximum beider Streifen: Kontakt heisst
    "irgendwo steht ein Fuss", nicht "an genau dieser Stelle".
    """
    # Der Streifen sitzt an der Standposition vor der Gegenbewegung.
    #
    # Versucht und wieder verworfen: ihn an den Fuss im Moment des Absprungs zu
    # legen (0.15 s davor). Das sollte Spruenge mit Anlauf retten, half dort aber
    # nicht - beim Anlauf ist der Streifen auch vor dem Aufsetzen leer, das Signal
    # bleibt mehrdeutig - und verschlechterte den Normalfall messbar: auf einem
    # sauberen Sprung kippte damit E1 unter E2, was fuer einen freien Flug
    # unmoeglich ist. Spruenge mit Anlauf faengt stattdessen die Gegenprobe
    # zwischen Kontaktsignal und Knoechelhub in der Pipeline ab.
    pre_from = max(0, stance_end - max(10, int(0.5 * info.measured_fps)))
    rois = [_roi_from_stance(track, slice(pre_from, max(pre_from + 1, stance_end)))]
    if landing_stance is not None and landing_stance.stop > landing_stance.start:
        rois.append(_roi_from_stance(track, landing_stance))

    # R3 - das Timing entsteht hier, also wird hier auch in voller Auflösung
    # gelesen. Der Pose-Zweig arbeitet auf einem Viertel, weil ihm das reicht;
    # der Spalt zwischen Zehe und Boden ist dort aber nur noch Bruchteile eines
    # Pixels breit. Statt des ganzen 4K-Bildes wandert nur der Streifen durch die
    # Pipe - gemessen 4.4 s je 200 Frames.
    inv = 1.0 / max(scale, 1e-6)
    full = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            for (x, y, w, h) in rois]
    pad = max(4, int(0.5 * full[0][3]))
    cx = max(0, min(f[0] for f in full) - pad)
    cy = max(0, min(f[1] for f in full) - pad)
    cw = max(f[0] + f[2] for f in full) - cx + pad
    ch = max(f[1] + f[3] for f in full) - cy + pad
    cw, ch = max(2, cw // 2 * 2), max(2, ch // 2 * 2)
    cx = max(0, min(info.width - cw, cx))
    cy = max(0, min(info.height - ch, cy))

    per_roi: list[list[np.ndarray]] = [[] for _ in full]
    for frame in iter_frames(info, crop=(cx, cy, cw, ch)):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for k, (x, y, w, h) in enumerate(full):
            per_roi[k].append(g[y - cy:y - cy + h, x - cx:x - cx + w])
    n = len(per_roi[0])

    probe = []
    probe_at = set(np.linspace(0, info.n_frames - 1, min(info.n_frames, 40))
                   .astype(int).tolist())
    for i, fr in enumerate(iter_frames(info, scale=scale)):
        if i in probe_at:
            probe.append(fr)

    hip = track.pt(HIP)[max(0, stance_end - 200):]
    bs = track.body_scale()
    box = (int(np.nanmin(hip[:, 0]) - 0.6 * bs), 0,
           int(np.nanmax(hip[:, 0]) + 0.6 * bs), track.height)
    _, resid = estimate_shift(probe, box)

    air = np.zeros(n, bool)
    air[air_frames[air_frames < n]] = True

    occ_all, gap_all = [], []
    for strips in per_roi:
        ref = (np.median(np.stack([strips[i] for i in np.flatnonzero(air)]), 0)
               if air.any() else strips[0])
        occ_all.append(np.array([float(np.mean(np.abs(s - ref))) for s in strips]))
        thr = float(np.median(ref)) - 2.0 * float(np.std(ref) + 1e-6)
        gap_all.append(np.array([float(np.mean(np.min(s, axis=1) < thr)) for s in strips]))
    occupancy = np.max(np.stack(occ_all), axis=0)
    gap = np.max(np.stack(gap_all), axis=0)

    strips = per_roi[0]
    diff_prev = np.zeros(n)
    diff_prev[1:] = [float(np.mean(np.abs(strips[i] - strips[i - 1]))) for i in range(1, n)]

    flow_y = np.zeros(n)
    w, h = full[0][2], full[0][3]
    if w >= 16 and h >= 8:
        for i in range(1, n):
            fl = cv2.calcOpticalFlowFarneback(
                strips[i - 1].astype(np.uint8), strips[i].astype(np.uint8),
                None, 0.5, 2, 7, 2, 5, 1.1, 0)
            flow_y[i] = float(np.mean(fl[..., 1]))

    return ContactSignals(t=info.t[:n], occupancy=occupancy, diff_prev=diff_prev,
                          flow_y=flow_y, gap=gap, roi=full[0], shift_residual=resid)
