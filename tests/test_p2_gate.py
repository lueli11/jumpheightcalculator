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
