"""Gate P2 - API und Worker.

Der Redis-Rundlauf selbst wird hier **nicht** geprueft: auf dieser Maschine gibt
es kein Redis (kein Docker, kein WSL). Getestet wird alles, was danach kommt -
die Warteschlange ist durch ein aufgezeichnetes `enqueue` ersetzt, und die
Worker-Funktion wird direkt gerufen. Es ist dieselbe Funktion, die arq aufruft
(`analyse_job` delegiert unveraendert an `run_analysis`), der Unterschied ist
also nur der Transport. Der echte Redis-Weg gehoert zu Gate P4 auf dem Server.
"""

from __future__ import annotations

import math
from pathlib import Path

import httpx
import pytest

from jumpheightcalculator import api, db, pipeline, worker

REAL_VIDEO = Path("IMG_5418.MOV")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def queued(monkeypatch):
    """`enqueue` aufzeichnen statt Redis zu brauchen."""
    seen: list[str] = []

    async def fake(analysis_id: str) -> None:
        seen.append(analysis_id)

    monkeypatch.setattr(api, "enqueue", fake)
    return seen


@pytest.fixture
async def client(data_dir):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


def _same(a: dict, b: dict) -> bool:
    """Bitgleich, aber NaN-bewusst.

    `float('nan') != float('nan')` - ohne diese Behandlung meldet ein Vergleich
    Unterschiede, wo keine sind. `shift_residual_px` ist NaN, wenn der Hintergrund
    zu texturarm fuer eine Bewegungsschaetzung war.
    """
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) \
                and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["pipeline_version"] == pipeline.PIPELINE_VERSION


@pytest.mark.anyio
async def test_root_says_what_this_is(client):
    """Die Wurzel darf nicht mit einem nackten 404 antworten - wer die
    API-Adresse im Browser aufruft, will wissen, ob der Dienst laeuft."""
    r = await client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["endpunkte"]["doku"] == "/docs"


@pytest.mark.anyio
async def test_upload_rejects_unreadable_file(client):
    r = await client.post("/api/videos",
                          files={"file": ("kaputt.mp4", b"nicht wirklich ein video")})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_analysis_of_unknown_video_is_404(client, queued):
    r = await client.post("/api/analyses", json={"video_id": "gibtsnicht"})
    assert r.status_code == 404


@pytest.mark.anyio
@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="kein echtes Sprungvideo vorhanden")
async def test_second_upload_does_not_block(client, queued):
    """P2.2 - der zweite Upload darf nicht auf den ersten warten.

    Die Analyse laeuft im Worker; der Request legt nur eine Zeile an und
    antwortet. Geprueft wird, dass beide Uploads durchgehen und beide Analysen in
    der Warteschlange landen, ohne dass eine davon gerechnet haette.
    """
    ids = []
    for _ in range(2):
        with REAL_VIDEO.open("rb") as f:
            up = await client.post("/api/videos", files={"file": (REAL_VIDEO.name, f)})
        assert up.status_code == 200
        an = await client.post("/api/analyses",
                               json={"video_id": up.json()["video_id"]})
        assert an.status_code == 200
        ids.append(an.json()["job_id"])

    assert queued == ids
    for aid in ids:
        assert (await client.get(f"/api/analyses/{aid}")).json()["status"] == "queued"


def _fake_track(n0=40, n1=140, fps=120.0, h_m=0.45, px_per_m=400.0):
    """Kuenstlicher Track: Huefte auf exakter Parabel, Kontaktsignal als Rechteck."""
    import numpy as np
    from jumpheightcalculator import G, pipeline as pl

    n = 220
    t = np.arange(n) / fps
    v0 = float(np.sqrt(2 * G * h_m))
    y = np.full(n, 1000.0)
    u = t[n0:n1] - t[n0]
    y[n0:n1] = 1000.0 - (v0 * u - 0.5 * G * u * u) * px_per_m
    xy = np.zeros((n, 33, 2), np.float32)
    xy[:, 23, 1] = xy[:, 24, 1] = y
    xy[:, 11, 1] = xy[:, 12, 1] = y - 250
    xy[:, 27, 1] = xy[:, 28, 1] = y + 150
    contact = np.ones(n)
    contact[n0:n1] = 0.0
    return pl.TrackData(
        t=t, xy=xy, width=540, height=960, hip_y=y, com_y=y, ankle_y=y + 150,
        contact=contact, confidence=np.full(n, 0.9), n0=n0, n1=n1,
        stance_end=max(0, n0 - 30), fps=fps, shift_residual=0.1,
        body_scale_px=400.0, scale_drift=0.0, coarse_n0=n0, coarse_n1=n1)


@pytest.mark.anyio
async def test_queue_position_is_reported(client, data_dir):
    """"queued" allein sagt dem Nutzer nichts - er will wissen, wie viele vor ihm
    sind. Der Worker arbeitet der Reihe nach, also zaehlen die frueher angelegten
    Wartenden plus die eine gerade laufende."""
    import uuid
    from jumpheightcalculator import db as dbmod

    con = dbmod.connect(data_dir / "jumpheightcalculator.sqlite")
    vid = uuid.uuid4().hex
    con.execute("INSERT INTO video (id, path, filename, created_at) VALUES (?,?,?,?)",
                (vid, "x.mp4", "x.mp4", "2026-08-27T10:00:00+00:00"))
    ids = []
    for i, st in enumerate(["running", "queued", "queued", "queued"]):
        aid = uuid.uuid4().hex; ids.append(aid)
        con.execute("INSERT INTO analysis (id, video_id, jump_type, status, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (aid, vid, "cmj", st, f"2026-08-27T10:0{i}:00+00:00"))
    # eine fertige Analyse als Grundlage fuer die Zeitschaetzung
    con.execute("INSERT INTO analysis (id, video_id, jump_type, status, created_at,"
                " finished_at) VALUES (?,?,?,'ready',?,?)",
                (uuid.uuid4().hex, vid, "cmj", "2026-08-27T09:00:00+00:00",
                 "2026-08-27T09:01:00+00:00"))
    con.commit(); con.close()

    laufend = (await client.get(f"/api/analyses/{ids[0]}")).json()
    assert laufend["queue_ahead"] == 0            # laeuft schon

    erster = (await client.get(f"/api/analyses/{ids[1]}")).json()
    assert erster["queue_ahead"] == 1             # nur der laufende

    dritter = (await client.get(f"/api/analyses/{ids[3]}")).json()
    assert dritter["queue_ahead"] == 3            # laufender + zwei Wartende
    assert dritter["typical_seconds"] == 60
    assert dritter["eta_seconds"] == 240          # 4 mal 60 s


@pytest.mark.anyio
async def test_refused_analysis_can_be_corrected_by_hand(client, data_dir):
    """Eine Verweigerung darf keine Sackgasse sein.

    Der teure Teil ist gelaufen und der Track liegt vor - der Review-Screen muss
    die Ereignisframes verschieben koennen und damit aus einem `failed` ein
    `ready` machen. Ohne das ist R7 zwar ehrlich, aber unbenutzbar.
    """
    import uuid
    from jumpheightcalculator import db as dbmod

    aid, vid = uuid.uuid4().hex, uuid.uuid4().hex
    track_path = data_dir / "tracks" / f"{aid}.npz"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_track().save(track_path)

    con = dbmod.connect(data_dir / "jumpheightcalculator.sqlite")
    con.execute("INSERT INTO video (id, path, filename, created_at)"
                " VALUES (?,?,?,?)", (vid, "x.mp4", "x.mp4", "t"))
    con.execute("INSERT INTO analysis (id, video_id, jump_type, status, error,"
                " track_path, created_at) VALUES (?,?,?, 'failed', ?, ?, 't')",
                (aid, vid, "cmj", "Flugzeit ausserhalb des Plausiblen", str(track_path)))
    con.commit(); con.close()

    # Der Track ist trotz Fehlschlag abrufbar - sonst kaeme der Screen nicht an
    # das Kontaktsignal heran.
    tr = await client.get(f"/api/analyses/{aid}/track")
    assert tr.status_code == 200

    r = await client.post(f"/api/analyses/{aid}/correct",
                          json={"takeoff_frame": 41, "landing_frame": 139})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["corrected"] is True

    after = (await client.get(f"/api/analyses/{aid}")).json()
    assert after["status"] == "ready"
    assert after["error"] is None


@pytest.mark.slow
@pytest.mark.anyio
@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="kein echtes Sprungvideo vorhanden")
async def test_api_result_matches_pipeline(client, data_dir, queued):
    """P2.1 - was die API ausliefert, ist Zahl fuer Zahl das, was die Pipeline rechnet.

    Verglichen wird gegen `measure` auf dem gespeicherten Track statt gegen einen
    zweiten vollen Durchlauf: `extract` ist deterministisch (kein RNG), und der
    NPZ-Rundlauf ist separat als bitgleich geprueft. Ein zweiter voller Lauf
    kostete vier Minuten und pruefte nichts zusaetzlich.
    """
    with REAL_VIDEO.open("rb") as f:
        up = await client.post("/api/videos", files={"file": (REAL_VIDEO.name, f)})
    vid = up.json()
    assert vid["fps_measured"] > 100

    an = await client.post("/api/analyses",
                           json={"video_id": vid["video_id"], "jump_type": "cmj",
                                 "mass_kg": 65})
    aid = an.json()["job_id"]

    assert worker.run_analysis(aid, data_dir=data_dir) == "ready"

    got = (await client.get(f"/api/analyses/{aid}")).json()
    assert got["status"] == "ready"
    assert got["pipeline_version"] == pipeline.PIPELINE_VERSION

    con = db.connect(data_dir / "jumpheightcalculator.sqlite")
    track_path = con.execute("SELECT track_path FROM analysis WHERE id=?",
                             (aid,)).fetchone()["track_path"]
    con.close()
    direct = pipeline.measure(pipeline.TrackData.load(track_path),
                              jump_type="cmj", mass_kg=65).to_dict()
    assert _same(got["result"], direct)

    # /track speist das Overlay und muss zur selben Zeitachse gehoeren.
    tr = (await client.get(f"/api/analyses/{aid}/track")).json()
    assert len(tr["t"]) == len(tr["contact"]) == len(tr["hip_y"])
    assert tr["n0"] < tr["n1"]

    # Eine +/-1-Frame-Korrektur muss den Wert nachweislich veraendern (Basis fuer P3.1).
    before = got["result"]["h_e2"]
    cor = await client.post(f"/api/analyses/{aid}/correct",
                            json={"takeoff_frame": tr["n0"] + 1})
    assert cor.status_code == 200
    after = cor.json()["result"]
    assert after["corrected"] is True
    assert abs(after["h_e2"] - before) > 1e-4

    rows = (await client.get("/api/sessions")).json()
    assert len(rows) == 1 and rows[0]["pipeline_version"] == pipeline.PIPELINE_VERSION

    csv_resp = await client.get("/api/sessions", params={"as_csv": True})
    assert csv_resp.status_code == 200
    assert "pipeline_version" in csv_resp.text.splitlines()[0]
