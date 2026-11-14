"""arq-Worker.

Eine Analyse dauert Minuten - das darf keinen HTTP-Request blockieren. Der
Worker ist bewusst dünn: er ruft dieselbe `pipeline`-Funktion wie die CLI und
schreibt das Ergebnis in dieselbe Datenbank. Ein zweiter Codepfad wäre eine
zweite Fehlerquelle, und Gate P2.1 verlangt ohnehin identische Zahlen.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from arq.connections import RedisSettings

from . import PIPELINE_VERSION, db, pipeline

DATA_DIR = Path(os.environ.get("JUMPHEIGHTCALCULATOR_DATA", "data"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_proxy(src: str, dest: Path, width: int, height: int) -> None:
    """Vorschauvideo in Arbeitsauflösung.

    Im Browser lässt sich ein 4K-Video nicht frame-genau steppen. Das Proxy hat
    dieselbe Frameanzahl und dieselben Zeitstempel (`-fps_mode passthrough`) und
    liegt in genau der Auflösung, in der auch die Landmarks gelten - das Overlay
    braucht dadurch keine Umrechnung, in der sich ein Fehler verstecken könnte.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-vf", f"scale={width}:{height}", "-fps_mode", "passthrough",
             "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)],
            check=True, capture_output=True, timeout=1800)
    except Exception:
        # Ohne Proxy bleibt der Review-Screen ohne Video, die Zahlen stimmen
        # trotzdem. Kein Grund, die ganze Analyse zu verwerfen.
        dest.unlink(missing_ok=True)


def run_analysis(analysis_id: str, *, data_dir: Path | None = None) -> str:
    """Die eigentliche Arbeit. Synchron, damit sie sich auch direkt testen lässt."""
    root = Path(data_dir) if data_dir else DATA_DIR
    con = db.connect(root / "jumpheightcalculator.sqlite")
    try:
        row = con.execute(
            "SELECT a.*, v.path AS video_path, v.capture_fps AS capture_fps "
            "FROM analysis a "
            "JOIN video v ON v.id = a.video_id WHERE a.id = ?", (analysis_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Analyse {analysis_id} unbekannt")

        # Job atomar übernehmen. Ohne die Bedingung auf `queued` können zwei
        # Worker dieselbe Zeile greifen und beide rechnen - der langsamere
        # überschreibt danach das Ergebnis des schnelleren. Genau das ist beim
        # Testen passiert: dieselbe Datei kam einmal als 74.0 cm und einmal als
        # Verweigerung heraus, obwohl `extract` für sich reproduzierbar ist.
        claimed = con.execute(
            "UPDATE analysis SET status='running' WHERE id=? AND status IN "
            "('queued','failed')", (analysis_id,))
        con.commit()
        if claimed.rowcount != 1:
            return "skipped"

        track_path = root / "tracks" / f"{analysis_id}.npz"
        track_path.parent.mkdir(parents=True, exist_ok=True)
        proxy_path = root / "proxies" / f"{analysis_id}.mp4"
        proxy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            td = pipeline.extract(row["video_path"],
                                  capture_fps=row["capture_fps"])
            td.save(track_path)
            _make_proxy(row["video_path"], proxy_path, td.width, td.height)
            res = pipeline.measure(td, jump_type=row["jump_type"],
                                   mass_kg=row["mass_kg"])
        except Exception as e:                      # R7 und alles andere
            # Track und Vorschau auch im Fehlerfall vermerken, sofern sie
            # entstanden sind. Sonst ist eine Verweigerung eine Sackgasse: die
            # teure Extraktion ist gelaufen, das Kontaktsignal liegt vor, aber der
            # Review-Screen bekäme nichts zu sehen und könnte die
            # Ereignisframes nicht von Hand korrigieren - genau wofür er da ist.
            con.execute(
                "UPDATE analysis SET status='failed', error=?, finished_at=?, "
                "pipeline_version=?, track_path=?, proxy_path=? WHERE id=?",
                (str(e), _now(), PIPELINE_VERSION,
                 str(track_path) if track_path.exists() else None,
                 str(proxy_path) if proxy_path.exists() else None,
                 analysis_id))
            con.commit()
            return "failed"

        con.execute(
            "UPDATE analysis SET status='ready', result=?, quality=?, h_e2=?, v0=?, "
            "track_path=?, proxy_path=?, pipeline_version=?, finished_at=?, "
            "error=NULL WHERE id=?",
            (json.dumps(res.to_dict()), res.quality, res.h_e2, res.v0,
             str(track_path), str(proxy_path) if proxy_path.exists() else None,
             res.pipeline_version, _now(), analysis_id))
        con.commit()
        return "ready"
    finally:
        con.close()


async def analyse_job(ctx, analysis_id: str) -> str:
    """arq ruft asynchron; die Arbeit selbst ist CPU-gebunden und läuft synchron.

    Der Worker fährt mit `max_jobs=1` - zwei parallele Analysen würden sich auf
    einer CPU nur gegenseitig ausbremsen. Ein zweiter Upload blockiert trotzdem
    nicht: er landet in der Warteschlange und die API antwortet sofort (P2.2).
    """
    return run_analysis(analysis_id)


class WorkerSettings:
    functions = [analyse_job]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = 1
    job_timeout = 3600
    keep_result = 3600
