import pytest
from src.api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_positive(client):
    response = client.post(
        "/predict",
        json={"text": "Air Paradis is amazing, best flight ever!"}
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "proba_pos" in data
    assert "proba_neg" in data
    assert "pred_label" in data
    assert "bad_buzz" in data


def test_predict_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400