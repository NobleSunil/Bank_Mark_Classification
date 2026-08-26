from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "Bank Marketing Prediction API is running" in response.json()["message"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    customer = {
        "age": 58,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 2143,
        "housing": "yes",
        "loan": "no",
        "contact": "unknown",
        "day": 5,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result
    assert "threshold" in result

    assert result["prediction"] in ["yes", "no"]
    assert 0 <= result["probability"] <= 1
    assert result["threshold"] == 0.55