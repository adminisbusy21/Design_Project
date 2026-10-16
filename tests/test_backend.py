import sys
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app


client = TestClient(app)

def test_review_inspection_not_found():
    response = client.put(
        "/inspections/999999/review",
        json={
            "review_status": "APPROVED",
            "review_notes": "Test review",
        },
    )

    assert response.status_code == 404
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
def test_invalid_review_status():
    response = client.put(
        "/inspections/999999/review",
        json={
            "review_status": "INVALID",
            "review_notes": "Test review",
        },
    )

    assert response.status_code == 400