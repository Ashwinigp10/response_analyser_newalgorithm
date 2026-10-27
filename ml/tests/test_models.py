import pytest
import json
from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR / "src"))
sys.path.insert(0, str(ML_DIR))

from predict import MindWavePredictor


@pytest.fixture(scope="module")
def predictor():
    models_dir = ML_DIR / "models"
    return MindWavePredictor(models_dir)


def test_models_loaded(predictor):
    assert predictor.is_loaded is True
    assert predictor.pq16_model is not None
    assert predictor.mdq_model is not None


def test_pq16_prediction_schema(predictor):
    features = {f"PRO16_{i}": 1 for i in range(1, 17)}
    res = predictor.predict_pq16(features)

    assert res["assessment_type"] == "schizophrenia"
    assert res["model_name"] == "pq16_xgboost_screening"
    assert res["algorithm"] == "XGBoost (XGBClassifier)"
    assert res["screening_positive"] is True
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in ["Elevated", "Moderate", "Mild", "Minimal"]
    assert "disclaimer" in res


def test_pq16_low_risk(predictor):
    features = {f"PRO16_{i}": 0 for i in range(1, 17)}
    res = predictor.predict_pq16(features)
    assert res["screening_positive"] is False
    assert res["probability"] < 0.20
    assert res["risk_level"] == "Minimal"


def test_mdq_prediction_schema(predictor):
    features = {f"MDQ1_{i}": 1 for i in range(1, 14)}
    features["MDQ_Frequency"] = 1
    features["MDQ3_Severity"] = 3
    res = predictor.predict_mdq(features)

    assert res["assessment_type"] == "bipolar"
    assert res["model_name"] == "mdq_xgboost_screening"
    assert res["algorithm"] == "XGBoost (XGBClassifier)"
    assert res["screening_positive"] is True
    assert 0.0 <= res["probability"] <= 1.0
    assert "disclaimer" in res


def test_mdq_low_risk(predictor):
    features = {f"MDQ1_{i}": 0 for i in range(1, 14)}
    features["MDQ_Frequency"] = 0
    features["MDQ3_Severity"] = 1
    res = predictor.predict_mdq(features)

    assert res["screening_positive"] is False
    assert res["probability"] < 0.20
    assert res["risk_level"] == "Minimal"


def test_invalid_input_handling(predictor):
    with pytest.raises(TypeError):
        predictor.predict_pq16("not_a_dict_or_list")
