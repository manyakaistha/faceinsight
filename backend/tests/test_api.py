"""API tests."""

from fastapi.testclient import TestClient
from backend.app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_model_info():
    with TestClient(app) as client:
        r = client.get("/api/v1/model/info")
        assert r.status_code == 200
        d = r.json()
        assert "arch" in d
        assert "input_shape" in d


def test_predict_rejects_bad_content_type():
    with TestClient(app) as client:
        r = client.post("/api/v1/predict", files={"file": ("test.txt", b"notimage", "text/plain")})
        assert r.status_code == 400


def test_predict_rejects_oversize():
    with TestClient(app) as client:
        big = b"x" * (11 * 1024 * 1024)
        r = client.post("/api/v1/predict", files={"file": ("big.jpg", big, "image/jpeg")})
        assert r.status_code == 413
