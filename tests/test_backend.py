
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@pytest.fixture
def fake_inspection():
    return SimpleNamespace(
        id=1,
        filename="test.jpg",
        image_format="JPEG",
        width=224,
        height=224,
        size_bytes=1024,
        prediction="Crack",
        confidence=0.95,
        priority="P1",
        priority_reason="High-confidence crack classification",
        review_required=False,
        review_status="PENDING",
        review_notes=None,
        created_at=datetime(2026, 1, 1),
    )


def make_mock_db(inspection=None):
    db = MagicMock()
    query = db.query.return_value
    query.order_by.return_value.all.return_value = (
        [inspection] if inspection is not None else []
    )
    query.filter.return_value.first.return_value = inspection
    return db


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == (
        "Distributed AI Damage Classification Service"
    )
    assert "/upload" in response.json()["available_endpoints"]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert len(response.json()["classes"]) == 5
    assert response.json()["model_version"] == "resnet18-damage-clean-v2"


def test_get_inspections_returns_records(fake_inspection):
    db = make_mock_db(fake_inspection)

    with patch("backend.main.SessionLocal", return_value=db):
        response = client.get("/inspections")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == 1
    assert body[0]["prediction"] == "Crack"
    assert body[0]["review_status"] == "PENDING"
    db.close.assert_called_once()


def test_get_inspections_returns_empty_list():
    db = make_mock_db()

    with patch("backend.main.SessionLocal", return_value=db):
        response = client.get("/inspections")

    assert response.status_code == 200
    assert response.json() == []
    db.close.assert_called_once()


def test_review_inspection_approves_record(fake_inspection):
    db = make_mock_db(fake_inspection)

    with patch("backend.main.SessionLocal", return_value=db):
        response = client.put(
            "/inspections/1/review",
            json={
                "review_status": "APPROVED",
                "review_notes": "Reviewed by inspector",
            },
        )

    assert response.status_code == 200
    assert fake_inspection.review_status == "APPROVED"
    assert fake_inspection.review_notes == "Reviewed by inspector"
    db.commit.assert_called_once()
    db.close.assert_called_once()


def test_review_inspection_rejects_record(fake_inspection):
    db = make_mock_db(fake_inspection)

    with patch("backend.main.SessionLocal", return_value=db):
        response = client.put(
            "/inspections/1/review",
            json={
                "review_status": "REJECTED",
                "review_notes": "Requires further inspection",
            },
        )

    assert response.status_code == 200
    assert fake_inspection.review_status == "REJECTED"
    db.commit.assert_called_once()


def test_review_inspection_returns_404_for_missing_record():
    db = make_mock_db(inspection=None)

    with patch("backend.main.SessionLocal", return_value=db):
        response = client.put(
            "/inspections/999/review",
            json={"review_status": "APPROVED"},
        )

    assert response.status_code == 404
    db.close.assert_called_once()


def test_review_inspection_rejects_invalid_status():
    response = client.put(
        "/inspections/1/review",
        json={"review_status": "PENDING"},
    )

    assert response.status_code == 422


def test_predict_damage_rejects_unsupported_content_type():
    response = client.post(
        "/predict_damage",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_upload_rejects_unsupported_content_type():
    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
