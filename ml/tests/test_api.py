import pytest
from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))

from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/ml/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["models"]["pq16_screening"] is True
    assert data["models"]["mdq_screening"] is True


def test_model_info_endpoint(client):
    res = client.get("/api/ml/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["framework"] == "XGBoost"
    assert "dataset_attribution" in data
    assert "models" in data
    assert "pq16" in data["models"]
    assert "mdq" in data["models"]


def test_predict_pq16_endpoint(client):
    payload = {
        "assessment_type": "schizophrenia",
        "features": {f"PRO16_{i}": 1 for i in range(1, 17)},
        "age_group": "19-25",
        "profession": "student"
    }
    res = client.post("/api/ml/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["assessment_type"] == "schizophrenia"
    assert data["algorithm"] == "XGBoost (XGBClassifier)"
    assert data["screening_positive"] is True
    assert data["probability"] > 0.8
    assert "disclaimer" in data


def test_predict_mdq_endpoint(client):
    payload = {
        "assessment_type": "bipolar",
        "features": {
            **{f"MDQ1_{i}": 1 for i in range(1, 14)},
            "MDQ_Frequency": 1,
            "MDQ3_Severity": 3
        },
        "age_group": "19-25",
        "profession": "student"
    }
    res = client.post("/api/ml/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["assessment_type"] == "bipolar"
    assert data["algorithm"] == "XGBoost (XGBClassifier)"
    assert data["screening_positive"] is True
    assert data["probability"] > 0.8


def test_predict_invalid_assessment_type(client):
    payload = {
        "assessment_type": "invalid_type",
        "features": {}
    }
    res = client.post("/api/ml/predict", json=payload)
    assert res.status_code == 400


def test_predict_structured_answers(client):
    # Test passing frontend answers array structure
    answers = [
        {"id": i, "answer": "True" if i <= 10 else "False"}
        for i in range(1, 17)
    ]
    payload = {
        "assessment_type": "schizophrenia",
        "answers": answers,
        "age_group": "19-25",
        "profession": "student"
    }
    res = client.post("/api/ml/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["assessment_type"] == "schizophrenia"
    assert "risk_level" in data
