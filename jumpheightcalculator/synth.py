"""P0 - Synthetisches Testvideo mit bekannter Wahrheit.

Eine Kugel auf einer exakten Wurfparabel. Es gibt noch keine echten Sprungvideos,
und ohne Ground Truth lässt sich nicht sagen, ob die Pipeline richtig rechnet
oder nur plausibel.

Struktur bewusst wie ein Sprung: Ruhephase - Flug - Ruhephase. Damit hat die
spätere Kontakt-/Flug-Segmentierung etwas zu tun und der robuste Fit muss die
Ruheframes als Ausreisser verwerfen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from . import G

# Zeitbasis der geschriebenen Datei: 1 us. Bei 240 fps sind das 0.02 % eines
# Frame-Abstands, also weit unter dem Jitter, den P0.3 prüft.
TIME_BASE = Fraction(1, 1_000_000)


@dataclass(frozen=True)
class Truth:
    """Ground Truth. Alles, was die Pipeline zurückrechnen können muss."""
    h: float                 # Scheitelhöhe über Absprungniveau [m]
    t_f: float               # Flugzeit [s]
    v0: float                # Absprunggeschwindigkeit [m/s]
    t_takeoff: float         # [s], relativ zum ersten Frame
    t_land: float            # [s]
    px_per_m: float          # der Massstab, den R5 aus der Physik zurückgewinnen muss
    g: float
    fps_nominal: float
    jitter: float
    noise: float
    n_frames: int


def generate(
    out: str | Path,
    *,
    h: float = 0.40,
    fps: float = 240.0,
    px_per_m: float = 400.0,
    width: int = 640,
    height: int = 480,
    jitter: float = 0.0,
    noise: float = 0.0,
    blur: float = 0.0,
    rest_frames: int = 10,
    ball_radius: float = 9.0,
    seed: int = 0,
    crf: int = 0,
) -> Truth:
    """Schreibt Video + `<name>.truth.json` und gibt die Wahrheit zurück.

    `jitter` streut die Frame-Abstände um +/- diesen Anteil - damit lässt sich
    prüfen, ob R1 wirklich verdrahtet ist. `blur` ist die Belichtungsdauer als
    Anteil des Frame-Abstands.
    """
    out = Path(out)
    rng = np.random.default_rng(seed)

    v0 = float(np.sqrt(2 * G * h))
    t_f = 2 * v0 / G
    a_px = G * px_per_m
    v0_px = v0 * px_per_m

    dt = 1.0 / fps
    n_flight = int(np.ceil(t_f / dt))
    n = rest_frames + n_flight + rest_frames

    # --- Zeitachse. Das ist der eigentliche Gegenstand des Tests. ---
    k = np.arange(n, dtype=np.float64)
    if jitter > 0:
        # Unabhängiger Versatz je Frame; bei jitter < 0.5 bleibt die Folge monoton.
        k = k + rng.uniform(-jitter, jitter, size=n)
        k -= k[0]
    t = k * dt
    t_takeoff = float(t[rest_frames - 1] + 0.5 * dt) if rest_frames else 0.0
    t_land = t_takeoff + t_f

    ground_y = height - 60.0
    y_rest = ground_y - ball_radius          # Kugelmittelpunkt in Ruhe
    x = width * 0.5

    def centre_at(tt: float) -> tuple[float, float]:
        if tt <= t_takeoff or tt >= t_land:
            return x, y_rest
        u = tt - t_takeoff
        return x, y_rest - (v0_px * u - 0.5 * a_px * u * u)

    bg = _background(width, height, ground_y, seed)

    # --- Frames rendern und mit exakten PTS muxen. ---
    #
    # Nicht cv2.VideoWriter: das kennt nur eine feste Framerate und würde die
    # Jitter-Zeitachse stillschweigend auf ein gleichmässiges Raster legen -
    # P0.3 prüfte dann nichts. Nicht der ffmpeg-concat-Demuxer: dessen
    # `duration`-Angaben werden bei Bild-Input ignoriert, gemessen kollabiert
    # alles auf das 25-fps-Standardraster. PyAV setzt die PTS je Frame direkt.
    out.parent.mkdir(parents=True, exist_ok=True)
    n_sub = 8 if blur > 0 else 1
    with av.open(str(out), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(1_000_000, 1))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = TIME_BASE
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for i in range(n):
            acc = np.zeros((height, width), np.float64)
            for s_i in range(n_sub):
                tt = t[i] + (blur * dt) * (s_i / n_sub if n_sub > 1 else 0.0)
                acc += _render(bg, *centre_at(tt), ball_radius)
            frame = acc / n_sub
            if noise > 0:
                frame += rng.normal(0.0, noise, frame.shape)
            img = np.clip(frame, 0, 255).astype(np.uint8)
            vf = av.VideoFrame.from_ndarray(
                np.repeat(img[:, :, None], 3, axis=2), format="bgr24")
            vf.pts = int(round(t[i] / TIME_BASE))
            vf.time_base = TIME_BASE
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)

    truth = Truth(
        h=h, t_f=t_f, v0=v0, t_takeoff=t_takeoff, t_land=t_land,
        px_per_m=px_per_m, g=G, fps_nominal=fps, jitter=jitter, noise=noise,
        n_frames=n,
    )
    out.with_suffix(".truth.json").write_text(json.dumps(asdict(truth), indent=2))
    return truth


def _background(w: int, h: int, ground_y: float, seed: int) -> np.ndarray:
    """Statischer, texturierter Hintergrund - gibt ORB später etwas zu greifen."""
    rng = np.random.default_rng(seed + 991)
    bg = np.full((h, w), 205.0)
    bg += rng.normal(0, 7, (h, w))
    speck = rng.integers(0, w * h, size=(w * h) // 220)
    flat = bg.reshape(-1)
    flat[speck] -= 55
    bg[int(ground_y):, :] = 150.0 + rng.normal(0, 5, (h - int(ground_y), w))
    return np.clip(bg, 0, 255)


def _render(bg: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Kugel mit Sub-Pixel-Mittelpunkt.

    cv2.circle mit `shift` und LINE_AA zeichnet kantengeglättet an einer
    gebrochenen Position. Eine hart gerasterte Kugel würde den Schwerpunkt auf
    ganze Pixel quantisieren - das Rauschen des Tests käme dann aus dem
    Generator statt aus dem Algorithmus.
    """
    img = bg.copy().astype(np.uint8)
    sh = 4
    f = 1 << sh
    cv2.circle(img, (int(round(cx * f)), int(round(cy * f))), int(round(r * f)),
               35, -1, lineType=cv2.LINE_AA, shift=sh)
    return img.astype(np.float64)
