"""Stückweiser Upload (Cloudflare-Grenze 100 MB): kommt die Datei bitgenau an?"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from jumpheightcalculator import api


def test_chunked_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "probe", lambda p, capture_fps=None: SimpleNamespace(
        width=1, height=1, rotation=0, n_frames=1, duration=1.0, measured_fps=240.0))
    c = TestClient(api.app)
    data = bytes(range(256)) * 1000
    uid = "a" * 32

    for off in range(0, len(data), 100_000):
        r = c.post(f"/api/uploads/{uid}", data={"offset": off},
                   files={"chunk": ("blob", data[off:off + 100_000])})
        assert r.status_code == 200, r.text

    # Lücke oder Doppel wird abgelehnt, ungültige ID ebenso
    assert c.post(f"/api/uploads/{uid}", data={"offset": 0},
                  files={"chunk": ("blob", b"x")}).status_code == 409
    assert c.post("/api/uploads/..%2Fx", data={"offset": 0},
                  files={"chunk": ("blob", b"x")}).status_code in (404, 422)

    r = c.post("/api/videos", data={"upload_id": uid, "filename": "sprung.MOV"})
    assert r.status_code == 200, r.text
    vid = r.json()["video_id"]
    assert (tmp_path / "videos" / f"{vid}.MOV").read_bytes() == data
    assert not (tmp_path / "uploads" / f"{uid}.part").exists()
    assert r.json()["filename"] == "sprung.MOV"
