"""SQLite. Videos und Keypoint-Arrays liegen als Dateien auf Disk, in der DB nur
Pfade - Blobs in SQLite bringen hier nichts ausser einer grossen Datei.

Keine Abstraktionsschicht über der DB: es gibt genau eine Implementierung, und
eine Schnittstelle mit einem Implementierer ist nur Verpackung.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS video (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    width       INTEGER, height INTEGER, rotation INTEGER,
    n_frames    INTEGER, duration REAL, fps_measured REAL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis (
    id           TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES video(id),
    jump_type    TEXT NOT NULL,
    mass_kg      REAL,
    status       TEXT NOT NULL,           -- queued | running | ready | failed
    error        TEXT,
    track_path   TEXT,
    proxy_path   TEXT,
    -- Pipeline-Version an jedem Ergebnis: sonst sieht ein Algorithmus-Update im
    -- Verlauf spaeter aus wie ein Formhoch.
    pipeline_version TEXT,
    result       TEXT,                    -- vollstaendiges Result als JSON
    quality      REAL,
    h_e2         REAL,                    -- ausgelagert für Verlauf und CSV
    v0           REAL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_analysis_created ON analysis(created_at);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False, weil FastAPI Sync-Dependencies im Threadpool
    # erzeugt, der async-Endpunkt danach aber im Event-Loop-Thread läuft. Sicher
    # ist das, weil jede Anfrage ihre eigene Verbindung bekommt und wieder
    # schliesst - geteilt wird nichts, die Verbindung wechselt nur einmal den
    # Thread innerhalb derselben Anfrage.
    con = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # WAL, damit ein zweiter Upload nicht blockiert, während der Worker schreibt.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def row_to_analysis(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d
