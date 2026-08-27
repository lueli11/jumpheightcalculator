"""S0 - Ingest und Normalisierung.

R1: Die Zeitachse kommt aus den Praesentationszeitstempeln (PTS) des Containers,
niemals aus einer angenommenen Framerate. Smartphone-Videos sind fast nie exakt
konstant getaktet, und Slo-Mo wird vom Betriebssystem oft schon verlangsamt
exportiert. Der Fehler schlaegt quadratisch in die Höhe durch, weil

    h = g * v0_px^2 / (2 * a_px^2)

unter einer Zeitskalierung t -> k*t mit k^2 skaliert.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# B2 - Plausibilitaetsfenster für das mediane Frame-Intervall.
# PTS zu lesen ist notwendig, aber nicht hinreichend: eine bereits verlangsamt
# exportierte Slo-Mo-Aufnahme traegt PTS über der 8-fachen Realzeit und liefert
# damit lautlos die 64-fache Höhe. Unter 20 fps ist nichts messbar, über
# 1000 fps existiert keine Handykamera - beides ist also ein Zeitbasis-Fehler.
MIN_FPS = 20.0
MAX_FPS = 1000.0


class IngestError(RuntimeError):
    """R7: konkrete Diagnose statt geratener Zahl."""


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise IngestError(f"{cmd[0]} fehlgeschlagen: {p.stderr.strip()[:400]}")
    return p.stdout


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    t: np.ndarray          # (N,) Praesentationszeitstempel [s], aus dem Container
    width: int             # Ausgabebreite nach Rotation
    height: int
    rotation: int          # Grad, aus der Display-Matrix
    nominal_fps: float     # gemeldete Framerate - nur für Anzeige und Warnungen

    @property
    def n_frames(self) -> int:
        return int(self.t.size)

    @property
    def measured_fps(self) -> float:
        """Aus den echten Zeitstempeln, nicht aus dem Containerfeld."""
        return float(1.0 / np.median(np.diff(self.t))) if self.t.size > 1 else float("nan")

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Liest Zeitstempel und Geometrie. Wirft bei unplausibler Zeitbasis (B2)."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"Datei nicht gefunden: {path}")

    stream = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(path),
    ]))["streams"]
    if not stream:
        raise IngestError(f"Kein Videostream in {path.name}")
    st = stream[0]

    rotation = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if rotation == 0 and "rotate" in (st.get("tags") or {}):
        rotation = int(round(float(st["tags"]["rotate"])))

    w, h = int(st["width"]), int(st["height"])
    # ffmpeg dreht beim Decodieren standardmaessig mit (autorotate), die
    # Ausgabegeometrie ist bei +/-90 Grad also getauscht.
    if rotation % 180 != 0:
        w, h = h, w

    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    nominal = float(num) / float(den) if float(den) else 0.0

    vals = _pts(path)
    if len(vals) < 3:
        raise IngestError(
            f"Nur {len(vals)} Frame-Zeitstempel lesbar - der Container trägt keine "
            f"brauchbaren PTS. Ohne echte Zeitstempel ist keine Messung möglich (R1)."
        )
    t = np.asarray(vals, dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        # Nicht sortieren, sondern melden: nicht-monotone PTS heissen meist
        # B-Frames ohne DTS-Korrektur, und das verschiebt die Zeitachse.
        raise IngestError("Zeitstempel sind nicht streng monoton steigend (R1).")

    info = VideoInfo(path, t, w, h, rotation, nominal)
    _check_timebase(info)
    return info


def _pts(path: Path) -> list[float]:
    """Die Praesentationszeitstempel, aus dem Container-Index statt aus dem Decoder.

    `frame=pts_time` **decodiert das ganze Video**, um Zahlen zu lesen, die im
    Index schon stehen: gemessen 16.5 s fuer 41 MB und 58.1 s fuer 140 MB, bei
    rund 43 ms je Frame in 4K. Das lief bisher im Upload-Request, der Nutzer sah
    dafuer eine Minute "laedt hoch". `packet=pts_time` liefert dieselben Werte in
    0.17 s - auf beiden Videos Wert fuer Wert identisch (402 bzw. 1339 Stueck).

    Pakete kommen allerdings in **Decode**-Reihenfolge. Ohne B-Frames ist das
    dasselbe wie die Anzeigereihenfolge (auf beiden Testvideos schon unsortiert
    streng monoton), mit B-Frames nicht. Sortiert wird hier bewusst nicht: eine
    nicht-monotone Liste ist der Hinweis, den `probe` melden soll. Sie ist aber
    kein Grund, ein sonst brauchbares Video abzulehnen - darum in dem Fall der
    langsame, exakte Weg ueber die Frames, und dessen Ergebnis entscheidet.
    """
    def lies(entity: str) -> list[float]:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", f"{entity}=pts_time", "-of", "default=nw=1:nk=1",
            str(path),
        ])
        return [float(x) for x in raw.split() if x not in ("N/A", "")]

    vals = lies("packet")
    if len(vals) >= 3 and all(b > a for a, b in zip(vals, vals[1:])):
        return vals
    return lies("frame")


def _check_timebase(info: VideoInfo) -> None:
    """B2 - erste Haelfte: liegt das Frame-Intervall überhaupt im Bereich einer Kamera?"""
    fps = info.measured_fps
    if not np.isfinite(fps) or not (MIN_FPS <= fps <= MAX_FPS):
        raise IngestError(
            f"Gemessene Framerate {fps:.1f} fps liegt ausserhalb {MIN_FPS:.0f}-{MAX_FPS:.0f} fps. "
            f"Das ist keine Aufnahme, sondern eine kaputte Zeitbasis - typisch für ein "
            f"Slo-Mo, das bereits verlangsamt exportiert wurde. Exportiere das Video mit "
            f"Originalgeschwindigkeit."
        )


def scaled_size(info: VideoInfo, scale: float) -> tuple[int, int]:
    """Geradzahlige Zielgroesse - libav mag ungerade Dimensionen nicht."""
    if scale >= 1.0:
        return info.width, info.height
    return max(2, int(info.width * scale) // 2 * 2), max(2, int(info.height * scale) // 2 * 2)


def iter_frames(info: VideoInfo, *, scale: float = 1.0,
                crop: tuple[int, int, int, int] | None = None) -> Iterator[np.ndarray]:
    """Frames als BGR-Arrays, streamend.

    Bewusst ffmpeg-Rohpipe statt cv2.VideoCapture: VideoCapture normalisiert bei
    variabler Framerate auf ein festes Raster und dupliziert oder verwirft dabei
    Frames. Die Zuordnung Frame-Index <-> PTS wäre damit still verschoben und R1
    ausgehebelt, ohne dass es irgendwo auffällt.

    `-fps_mode passthrough` ist dabei nicht optional, sondern der ganze Punkt:
    ohne die Angabe wandelt ffmpeg selbst nach CFR und dupliziert Frames, bis das
    Raster voll ist. Gemessen an einer 135-Frame-Datei kamen so **224422** Frames
    heraus, alle Duplikate des ersten - dieselbe stille Verschiebung, wegen der
    VideoCapture ausscheidet. Der Abgleich gegen die PTS-Liste unten faengt es ab.
    """
    if crop is not None:
        # Ausschnitt in Originalpixeln, vor dem Skalieren. So laesst sich ein
        # schmaler Streifen in voller Auflösung lesen, ohne das ganze 4K-Bild
        # durch die Pipe zu schieben.
        cx, cy, cw, ch = crop
        cw, ch = max(2, cw // 2 * 2), max(2, ch // 2 * 2)
        cx = max(0, min(info.width - cw, cx))
        cy = max(0, min(info.height - ch, cy))
        w, h = cw, ch
    else:
        w, h = scaled_size(info, scale)
    fsize = w * h * 3
    cmd = ["ffmpeg", "-v", "error", "-i", str(info.path)]
    if crop is not None:
        cmd += ["-vf", f"crop={cw}:{ch}:{cx}:{cy}"]
    elif (w, h) != (info.width, info.height):
        # Herunterskalieren schon im Decoder: ein 4K-Rohframe ist 25 MB, die
        # Pipe ist sonst der Flaschenhals. Frames werden dabei keine verworfen,
        # die Zuordnung zu den PTS bleibt also unberuehrt.
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += [
        "-fps_mode", "passthrough",           # Ausgabeoption - muss hinter -i stehen
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.wait()
    if n != info.n_frames:
        raise IngestError(
            f"{n} decodierte Frames gegen {info.n_frames} Zeitstempel - die Zuordnung "
            f"Frame zu Zeit wäre verschoben. {err.strip()[:200]}"
        )
