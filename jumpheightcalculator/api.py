"""FastAPI vor der CLI-Logik.

Kein Auth, kein Multi-Tenancy: ein Nutzer im eigenen Netz (Nicht-Ziel Nr. 1).
Die Analyse läuft im arq-Worker, nie im Request.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from arq.connections import RedisSettings, create_pool
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import PIPELINE_VERSION, db, pipeline
from .ingest import MAX_FPS, MIN_FPS, IngestError, probe

DATA_DIR = Path(os.environ.get("JUMPHEIGHTCALCULATOR_DATA", "data"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
JUMP_TYPES = ("cmj", "sj", "drop", "hops")

# Unter dieser Bildrate ist der Fehler laut Fehlerrechnung mehrere Zentimeter -
# für eine Fortschrittsmessung unbrauchbar. Der Upload wird trotzdem angenommen,
# aber der Nutzer erfährt es sofort statt erst am Ergebnis.
FPS_WARN = 120.0

app = FastAPI(title="jumpheightcalculator", version=PIPELINE_VERSION)

# Der Browser lädt das Video **direkt** hierher, nicht über den Next-Proxy.
# Gemessen riss ein 134-MB-Upload durch den Proxy mit ECONNRESET ab: die Datei
# kam an, der Browser sah trotzdem einen Fehler und startete die Analyse nie.
# Ein Puffer weniger, ein Fehlerfall weniger. Offene Herkunft ist vertretbar -
# ein Nutzer im eigenen Netz, keine Anmeldung, keine Cookies (Nicht-Ziel Nr. 1).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    allow_credentials=False,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    con = db.connect(DATA_DIR / "jumpheightcalculator.sqlite")
    try:
        yield con
    finally:
        con.close()


async def enqueue(analysis_id: str) -> None:
    """In die Warteschlange legen.

    Ist Redis nicht erreichbar, bleibt die Analyse auf `queued` stehen und der
    Upload gilt trotzdem als angenommen. Das ist Absicht: das Video liegt schon
    auf der Platte, und ein Worker, der später hochkommt, holt die Zeile ab. Ein
    500 an dieser Stelle würde den Nutzer den Upload wiederholen lassen, obwohl
    nichts verloren ist. Was es **nicht** tut, ist `ready` zu melden.
    """
    try:
        # Mit Frist: `create_pool` versucht es sonst fünfmal und blockiert den
        # Request rund 25 Sekunden, bevor es aufgibt. Gemessen genau das.
        pool = await asyncio.wait_for(
            create_pool(RedisSettings.from_dsn(REDIS_URL)), timeout=3.0)
    except Exception:
        return
    try:
        await pool.enqueue_job("analyse_job", analysis_id)
    finally:
        await pool.aclose()


class VideoOut(BaseModel):
    video_id: str
    filename: str
    width: int
    height: int
    rotation: int
    n_frames: int
    duration: float
    fps_measured: float
    capture_fps: float | None = None    # vom Nutzer angegeben, None = aus dem Container
    warning: str | None = None


class AnalysisIn(BaseModel):
    video_id: str
    jump_type: str = "cmj"
    mass_kg: float | None = Field(default=None, gt=0, lt=300)


class CaptureFpsIn(BaseModel):
    capture_fps: float | None = Field(default=None, ge=MIN_FPS, le=MAX_FPS)


class CorrectIn(BaseModel):
    takeoff_frame: int | None = None
    landing_frame: int | None = None


# Cloudflare (Free) nimmt höchstens 100 MB pro Anfrage an; ein 4K/120-fps-Video
# hat 140 MB. Grössere Dateien schickt der Browser darum in Stücken. Nicht
# transkodieren: die Zeitachse kommt aus den Container-Zeitstempeln (R1), ein
# Re-Encode im Browser würde sie und die Kanten für MediaPipe verändern.
MAX_UPLOAD = 2 * 1024**3
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


def _part(upload_id: str) -> Path:
    if not UPLOAD_ID.fullmatch(upload_id):
        raise HTTPException(status_code=422, detail="Ungültige Upload-ID.")
    return DATA_DIR / "uploads" / f"{upload_id}.part"


@app.post("/api/uploads/{upload_id}")
def upload_chunk(upload_id: str, offset: int = Form(...),
                 chunk: UploadFile = File(...)):
    # ponytail: abgebrochene .part-Dateien bleiben liegen; Aufräumen, wenn es je stört.
    part = _part(upload_id)
    part.parent.mkdir(parents=True, exist_ok=True)
    size = part.stat().st_size if part.exists() else 0
    # Stücke kommen der Reihe nach. Ein falscher Offset hiesse Lücke oder Doppel.
    if offset != size:
        raise HTTPException(status_code=409, detail=f"Offset {offset}, erwartet {size}.")
    with part.open("ab") as f:
        shutil.copyfileobj(chunk.file, f)
        if f.tell() > MAX_UPLOAD:
            f.close()
            part.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail="Video grösser als 2 GB.")
    return {"size": part.stat().st_size}


@app.post("/api/videos", response_model=VideoOut)
def upload_video(file: UploadFile | None = File(default=None),
                 upload_id: str | None = Form(default=None),
                 filename: str | None = Form(default=None),
                 capture_fps: float | None = Form(default=None),
                 con=Depends(get_db)):
    # Bewusst **kein** `async def`: die Funktion schreibt eine Datei und ruft
    # ffprobe, beides blockierend. In einer Coroutine liefe das im Event-Loop und
    # legte die ganze API still - auch das Statuspolling anderer Nutzer. Als
    # gewöhnliche Funktion schiebt FastAPI sie in den Threadpool.
    if file is None and upload_id is None:
        raise HTTPException(status_code=422, detail="Kein Video.")
    name = (file.filename if file else filename) or ""
    vid = uuid.uuid4().hex
    dest = DATA_DIR / "videos" / f"{vid}{Path(name).suffix or '.mp4'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if file is not None:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        part = _part(upload_id)
        if not part.exists():
            raise HTTPException(status_code=404, detail="Upload unbekannt.")
        part.replace(dest)
    file_name = name or dest.name

    # Die Aufnahmerate gehört in **dieselbe** Anfrage. `probe` läuft hier, und
    # bei einer unplausiblen Zeitbasis fliegt die Datei gleich wieder von der
    # Platte - ein später nachgereichter Wert käme zu spät.
    if capture_fps is not None and not (MIN_FPS <= capture_fps <= MAX_FPS):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=(f"Aufnahmerate {capture_fps:g} fps liegt ausserhalb "
                    f"{MIN_FPS:.0f}-{MAX_FPS:.0f} fps. Die Höhe hängt quadratisch "
                    f"daran - ein Tippfehler um Faktor 10 gibt Faktor 100."))
    try:
        info = probe(dest, capture_fps=capture_fps)
    except IngestError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e)) from e

    con.execute(
        "INSERT INTO video (id, path, filename, width, height, rotation, n_frames,"
        " duration, fps_measured, capture_fps, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (vid, str(dest), file_name, info.width, info.height,
         info.rotation, info.n_frames, info.duration, info.measured_fps,
         capture_fps, _now()))
    con.commit()

    warning = None
    if info.measured_fps < FPS_WARN * 0.99:
        warning = (f"Nur {info.measured_fps:.0f} fps gemessen. Für eine "
                   f"Fortschrittsmessung sind mindestens {FPS_WARN:.0f} fps nötig - "
                   f"bitte in Slo-Mo aufnehmen.")
    return VideoOut(video_id=vid, capture_fps=capture_fps,
                    filename=file_name,
                    width=info.width, height=info.height, rotation=info.rotation,
                    n_frames=info.n_frames, duration=info.duration,
                    fps_measured=info.measured_fps, warning=warning)


@app.post("/api/analyses")
async def create_analysis(body: AnalysisIn, con=Depends(get_db)):
    if body.jump_type not in JUMP_TYPES:
        raise HTTPException(422, f"jump_type muss eines von {JUMP_TYPES} sein")
    if con.execute("SELECT 1 FROM video WHERE id=?", (body.video_id,)).fetchone() is None:
        raise HTTPException(404, "Video unbekannt")

    aid = uuid.uuid4().hex
    con.execute(
        "INSERT INTO analysis (id, video_id, jump_type, mass_kg, status, created_at)"
        " VALUES (?,?,?,?, 'queued', ?)",
        (aid, body.video_id, body.jump_type, body.mass_kg, _now()))
    con.commit()
    await enqueue(aid)
    return {"job_id": aid, "status": "queued"}


def _typical_seconds(con) -> float | None:
    """Wie lange dauert eine Analyse üblicherweise?

    Aus den letzten fertigen Läufen statt aus einer geratenen Konstante - die
    Laufzeit hängt an Videolänge, Auflösung und Maschine, und ein Wert aus der
    eigenen Vergangenheit trifft das besser als jede Schätzung im Code.
    """
    rows = con.execute(
        "SELECT created_at, finished_at FROM analysis WHERE status='ready'"
        " AND finished_at IS NOT NULL ORDER BY created_at DESC LIMIT 10").fetchall()
    secs = []
    for r in rows:
        try:
            a = datetime.fromisoformat(r["created_at"])
            b = datetime.fromisoformat(r["finished_at"])
        except (TypeError, ValueError):
            continue
        d = (b - a).total_seconds()
        if 5 <= d <= 3600:
            secs.append(d)
    if not secs:
        return None
    secs.sort()
    return secs[len(secs) // 2]


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str, con=Depends(get_db)):
    row = con.execute("SELECT * FROM analysis WHERE id=?", (aid,)).fetchone()
    if row is None:
        raise HTTPException(404, "Analyse unbekannt")
    d = db.row_to_analysis(row)
    vid = con.execute("SELECT capture_fps, fps_measured FROM video WHERE id=?",
                      (d["video_id"],)).fetchone()
    out = {"id": d["id"], "video_id": d["video_id"], "status": d["status"],
           "jump_type": d["jump_type"], "error": d["error"],
           "pipeline_version": d["pipeline_version"], "result": d["result"],
           "capture_fps": vid["capture_fps"] if vid else None,
           "fps_measured": vid["fps_measured"] if vid else None}

    if d["status"] in ("queued", "running"):
        # Wie viele sind vorher dran? Der Worker arbeitet die Warteschlange nach
        # Eingang ab (max_jobs=1), also zählen genau die früher angelegten
        # Wartenden plus die eine gerade laufende.
        ahead = con.execute(
            "SELECT COUNT(*) c FROM analysis WHERE status='queued' AND created_at < ?",
            (d["created_at"],)).fetchone()["c"]
        running = con.execute(
            "SELECT COUNT(*) c FROM analysis WHERE status='running' AND id != ?",
            (aid,)).fetchone()["c"]
        out["queue_ahead"] = 0 if d["status"] == "running" else ahead + running
        typ = _typical_seconds(con)
        if typ is not None:
            out["typical_seconds"] = round(typ)
            out["eta_seconds"] = round(typ * (out["queue_ahead"] + 1))
    return out


@app.post("/api/analyses/{aid}/capture-fps")
async def set_capture_fps(aid: str, body: CaptureFpsIn, con=Depends(get_db)):
    """Aufnahmerate nachtragen und dieselbe Analyse neu rechnen.

    Ohne das ist eine Zeitachsen-Verweigerung eine Sackgasse: die App sagt dem
    Nutzer, dass sein Slo-Mo verlangsamt exportiert ist und mit welcher Rate er
    gefilmt hat - und der einzige Ausweg wäre gewesen, dieselben 20-140 MB noch
    einmal hochzuladen. Die Datei liegt schon auf der Platte.

    Der Wert hängt am **Video**, nicht am Lauf: die Zeitachse ist eine
    Eigenschaft der Datei, und zwei Analysen desselben Videos dürfen sich darin
    nicht widersprechen. Track und Vorschau werden verworfen, weil beide auf der
    alten Achse entstanden sind - der Track trägt sie in `t`, das Proxy in
    seinen Zeitstempeln.
    """
    row = con.execute("SELECT video_id, track_path, proxy_path FROM analysis "
                      "WHERE id=?", (aid,)).fetchone()
    if row is None:
        raise HTTPException(404, "Analyse unbekannt")

    con.execute("UPDATE video SET capture_fps=? WHERE id=?",
                (body.capture_fps, row["video_id"]))
    con.execute("UPDATE analysis SET status='queued', error=NULL, result=NULL,"
                " quality=NULL, h_e2=NULL, v0=NULL, track_path=NULL,"
                " proxy_path=NULL, finished_at=NULL WHERE id=?", (aid,))
    con.commit()
    for alt_pfad in (row["track_path"], row["proxy_path"]):
        if alt_pfad:
            Path(alt_pfad).unlink(missing_ok=True)
    await enqueue(aid)
    return {"job_id": aid, "status": "queued", "capture_fps": body.capture_fps}


@app.get("/api/analyses/{aid}/track")
def get_track(aid: str, con=Depends(get_db)):
    """Keypoints und Kontaktsignal je Frame - daraus rendert das Overlay."""
    row = con.execute("SELECT track_path FROM analysis WHERE id=?", (aid,)).fetchone()
    if row is None or not row["track_path"]:
        raise HTTPException(404, "Kein Track vorhanden")
    td = pipeline.TrackData.load(row["track_path"])
    return {
        "t": td.t.tolist(),
        "xy": np.nan_to_num(td.xy, nan=-1.0).round(2).tolist(),
        "width": td.width, "height": td.height,
        "hip_y": np.nan_to_num(td.hip_y, nan=-1.0).tolist(),
        "com_y": np.nan_to_num(td.com_y, nan=-1.0).tolist(),
        "ankle_y": np.nan_to_num(td.ankle_y, nan=-1.0).tolist(),
        "contact": td.contact.tolist(),
        "confidence": np.nan_to_num(td.confidence, nan=0.0).tolist(),
        "n0": td.n0, "n1": td.n1, "fps": td.fps,
        # Das Vorschauvideo behält die Container-Zeitstempel. Ist die Achse per
        # Aufnahmerate gestaucht, muss der Client zum Suchen t / time_scale
        # nehmen, sonst springt er um genau diesen Faktor daneben.
        "time_scale": td.time_scale,
        "body_scale_px": td.body_scale_px,
    }


@app.get("/api/analyses/{aid}/video")
def get_proxy_video(aid: str, con=Depends(get_db)):
    """Herunterskaliertes Vorschauvideo für den Review-Screen.

    Das Original ist 4K und 139 MB - im Browser ist damit kein frame-genaues
    Steppen möglich. Das Proxy hat dieselbe Frameanzahl und dieselben
    Zeitstempel, nur weniger Pixel, und liegt in derselben Auflösung wie die
    Landmarks: das Overlay passt damit ohne Umrechnung.
    """
    row = con.execute("SELECT proxy_path FROM analysis WHERE id=?", (aid,)).fetchone()
    if row is None or not row["proxy_path"] or not Path(row["proxy_path"]).exists():
        raise HTTPException(404, "Kein Vorschauvideo vorhanden")
    return FileResponse(row["proxy_path"], media_type="video/mp4")


@app.post("/api/analyses/{aid}/correct")
def correct_analysis(aid: str, body: CorrectIn, con=Depends(get_db)):
    """Korrigierte Ereignisframes - rechnet neu und markiert das Ergebnis.

    Läuft ohne Worker: der teure Teil (Pose) steckt im gespeicherten Track, die
    Neuberechnung dauert Millisekunden.
    """
    row = con.execute("SELECT * FROM analysis WHERE id=?", (aid,)).fetchone()
    if row is None:
        raise HTTPException(404, "Analyse unbekannt")
    if not row["track_path"]:
        raise HTTPException(409, "Kein Track vorhanden - die Extraktion ist gescheitert")

    td = pipeline.TrackData.load(row["track_path"])
    try:
        res = pipeline.measure(td, jump_type=row["jump_type"], mass_kg=row["mass_kg"],
                               n0=body.takeoff_frame, n1=body.landing_frame)
    except pipeline.Refusal as e:
        raise HTTPException(422, str(e)) from e

    # status auf ready setzen und error löschen: eine Verweigerung, die der
    # Nutzer von Hand korrigiert hat, ist keine Verweigerung mehr.
    # finished_at bleibt stehen: es misst, wie lange der Lauf gedauert hat, und
    # daraus schätzt _typical_seconds die Wartezeit. Eine Korrektur kommt oft
    # erst Minuten später - mitgeschrieben würde daraus eine erfundene Laufzeit.
    con.execute("UPDATE analysis SET status='ready', error=NULL, result=?, quality=?,"
                " h_e2=?, v0=? WHERE id=?",
                (json.dumps(res.to_dict()), res.quality, res.h_e2, res.v0, aid))
    con.commit()
    return {"id": aid, "status": "ready", "result": res.to_dict()}


@app.get("/api/sessions")
def sessions(jump_type: str | None = None, as_csv: bool = False, con=Depends(get_db)):
    """Verlauf. Die Pipeline-Version steht bewusst in jeder Zeile: nach einem
    Algorithmus-Update sind ältere Werte nicht mehr vergleichbar."""
    sql = ("SELECT a.id, a.created_at, a.jump_type, a.status, a.quality, a.h_e2, a.v0,"
           " a.pipeline_version, v.filename FROM analysis a"
           " JOIN video v ON v.id = a.video_id WHERE a.status='ready'")
    args: list = []
    if jump_type:
        sql += " AND a.jump_type=?"
        args.append(jump_type)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY a.created_at DESC", args)]

    if not as_csv:
        return rows
    fields = list(rows[0].keys()) if rows else [
        "id", "created_at", "jump_type", "status", "quality", "h_e2", "v0",
        "pipeline_version", "filename"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jumpheightcalculator.csv"})


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.get("/")
def root():
    """Auskunft auf der Wurzel.

    Ohne diese Route antwortet FastAPI hier mit {"detail":"Not Found"} - formal
    richtig, aber wer die API-Adresse im Browser aufruft, will wissen, ob sie
    läuft, und nicht raten müssen, ob der 404 vom Dienst oder vom Proxy kommt.
    """
    return {
        "name": "jumpheightcalculator",
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "hinweis": "Das ist die API, nicht die Weboberfläche.",
        "endpunkte": {"doku": "/docs", "health": "/api/health",
                      "verlauf": "/api/sessions"},
    }
