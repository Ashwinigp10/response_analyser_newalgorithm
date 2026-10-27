import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import xgboost as xgb

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
MODELS_DIR = ML_DIR / "models"

PQ16_FEATURES = [f"PRO16_{i}" for i in range(1, 17)]
MDQ_FEATURES = [f"MDQ1_{i}" for i in range(1, 14)] + ["MDQ_Frequency", "MDQ3_Severity"]


class MindWavePredictor:
    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.pq16_model = None
        self.mdq_model = None
        self.pq16_meta = None
        self.mdq_meta = None
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        pq_model_file = self.models_dir / "pq16_xgboost_model.json"
        pq_meta_file = self.models_dir / "pq16_xgboost_pipeline.json"

        mdq_model_file = self.models_dir / "mdq_xgboost_model.json"
        mdq_meta_file = self.models_dir / "mdq_xgboost_pipeline.json"

        missing = []
        if not pq_model_file.exists():
            missing.append(str(pq_model_file.name))
        if not mdq_model_file.exists():
            missing.append(str(mdq_model_file.name))

        if missing:
            raise FileNotFoundError(
                f"Missing required XGBoost model artifact(s): {missing}. "
                f"Please run 'python ml/src/train.py' to generate and save models."
            )

        # Load PQ-16
        self.pq16_model = xgb.XGBClassifier()
        self.pq16_model.load_model(str(pq_model_file))
        if pq_meta_file.exists():
            with open(pq_meta_file, "r", encoding="utf-8") as f:
                self.pq16_meta = json.load(f)

        # Load MDQ
        self.mdq_model = xgb.XGBClassifier()
        self.mdq_model.load_model(str(mdq_model_file))
        if mdq_meta_file.exists():
            with open(mdq_meta_file, "r", encoding="utf-8") as f:
                self.mdq_meta = json.load(f)

        self.is_loaded = True
        print("[MindWavePredictor] Successfully loaded PQ-16 and MDQ XGBoost models.")

    def predict_pq16(self, input_features: Union[Dict[str, Any], List[Union[int, float]]]) -> Dict[str, Any]:
        """
        Expects 16 item answers (PRO16_1 to PRO16_16) as 0/1 (or False/True).
        """
        if not self.is_loaded:
            raise RuntimeError("Models are not loaded.")

        row = {}
        if isinstance(input_features, list):
            if len(input_features) < 16:
                raise ValueError(f"PQ-16 requires 16 item inputs, received {len(input_features)}")
            for i, val in enumerate(input_features[:16]):
                row[f"PRO16_{i+1}"] = 1 if val in (1, True, "True", "true", "1") else 0
        elif isinstance(input_features, dict):
            for i in range(1, 17):
                key = f"PRO16_{i}"
                alt_key = f"q{i}"
                val = input_features.get(key, input_features.get(alt_key, 0))
                row[key] = 1 if val in (1, True, "True", "true", "1") else 0
        else:
            raise TypeError("input_features must be a list or dict")

        X = pd.DataFrame([row])[PQ16_FEATURES]
        pred_label = int(self.pq16_model.predict(X)[0])
        pred_proba = float(self.pq16_model.predict_proba(X)[0][1])

        symptom_count = sum(row.values())

        if pred_proba >= 0.70 or symptom_count > 12:
            risk_level = "Elevated"
        elif pred_proba >= 0.45 or symptom_count >= 8:
            risk_level = "Moderate"
        elif pred_proba >= 0.20 or symptom_count >= 4:
            risk_level = "Mild"
        else:
            risk_level = "Minimal"

        return {
            "assessment_type": "schizophrenia",
            "model_name": "pq16_xgboost_screening",
            "algorithm": "XGBoost (XGBClassifier)",
            "screening_positive": bool(pred_label == 1),
            "probability": round(pred_proba, 4),
            "risk_level": risk_level,
            "symptom_count": symptom_count,
            "threshold_criteria": "Ising et al. PQ-16 Ultra-High Risk Cutoff (Score >= 6)",
            "disclaimer": "This is an experimental questionnaire screening risk estimation, not an independent clinical diagnosis."
        }

    def predict_mdq(self, input_features: Union[Dict[str, Any], List[Union[int, float]]]) -> Dict[str, Any]:
        """
        Expects 13 symptom answers (MDQ1_1 to MDQ1_13) as 0/1,
        plus MDQ_Frequency (0/1) and MDQ3_Severity (1 to 4).
        """
        if not self.is_loaded:
            raise RuntimeError("Models are not loaded.")

        row = {}
        if isinstance(input_features, list):
            if len(input_features) < 13:
                raise ValueError(f"MDQ requires at least 13 symptom inputs, received {len(input_features)}")
            for i, val in enumerate(input_features[:13]):
                row[f"MDQ1_{i+1}"] = 1 if val in (1, True, "True", "true", "1") else 0
            row["MDQ_Frequency"] = 1 if len(input_features) > 13 and input_features[13] in (1, True, "True", "1") else 0
            row["MDQ3_Severity"] = int(input_features[14]) if len(input_features) > 14 else 1
        elif isinstance(input_features, dict):
            for i in range(1, 14):
                key = f"MDQ1_{i}"
                alt_key = f"q{i}"
                val = input_features.get(key, input_features.get(alt_key, 0))
                row[key] = 1 if val in (1, True, "True", "true", "1") else 0

            # Frequency (Question 14 in MindWave)
            freq_val = input_features.get("MDQ_Frequency", input_features.get("q14", 0))
            row["MDQ_Frequency"] = 1 if freq_val in (1, True, "True", "1") else 0

            # Severity (Question 11 or Question 15 / MDQ3_Severity)
            sev_val = input_features.get("MDQ3_Severity", input_features.get("severity", input_features.get("q11", 1)))
            try:
                row["MDQ3_Severity"] = int(sev_val)
            except (ValueError, TypeError):
                row["MDQ3_Severity"] = 1
        else:
            raise TypeError("input_features must be a list or dict")

        # Clamp severity between 1 and 4
        row["MDQ3_Severity"] = max(1, min(4, row["MDQ3_Severity"]))

        X = pd.DataFrame([row])[MDQ_FEATURES]
        pred_label = int(self.mdq_model.predict(X)[0])
        pred_proba = float(self.mdq_model.predict_proba(X)[0][1])

        symptom_count = sum(row[f"MDQ1_{i}"] for i in range(1, 14))

        if pred_proba >= 0.70 or symptom_count > 12:
            risk_level = "Elevated"
        elif pred_proba >= 0.45 or symptom_count >= 8:
            risk_level = "Moderate"
        elif pred_proba >= 0.20 or symptom_count >= 4:
            risk_level = "Mild"
        else:
            risk_level = "Minimal"

        return {
            "assessment_type": "bipolar",
            "model_name": "mdq_xgboost_screening",
            "algorithm": "XGBoost (XGBClassifier)",
            "screening_positive": bool(pred_label == 1),
            "probability": round(pred_proba, 4),
            "risk_level": risk_level,
            "symptom_count": symptom_count,
            "threshold_criteria": "Hirschfeld et al. MDQ Screening Positive (7+ co-occurring symptoms, moderate/severe impairment)",
            "disclaimer": "This is an experimental questionnaire screening risk estimation, not an independent clinical diagnosis."
        }


def main():
    print("Testing MindWavePredictor inference on example records...")
    predictor = MindWavePredictor()

    # Test PQ-16
    sample_pq_high = {f"PRO16_{i}": 1 for i in range(1, 17)}
    sample_pq_low = {f"PRO16_{i}": 0 for i in range(1, 17)}
    res_high = predictor.predict_pq16(sample_pq_high)
    res_low = predictor.predict_pq16(sample_pq_low)

    print("\n[PQ-16 High Risk Example]:")
    print(json.dumps(res_high, indent=2))
    assert res_high["screening_positive"] is True

    print("\n[PQ-16 Low Risk Example]:")
    print(json.dumps(res_low, indent=2))
    assert res_low["screening_positive"] is False

    # Test MDQ
    sample_mdq_high = {f"MDQ1_{i}": 1 for i in range(1, 14)}
    sample_mdq_high["MDQ_Frequency"] = 1
    sample_mdq_high["MDQ3_Severity"] = 3

    sample_mdq_low = {f"MDQ1_{i}": 0 for i in range(1, 14)}
    sample_mdq_low["MDQ_Frequency"] = 0
    sample_mdq_low["MDQ3_Severity"] = 1

    res_mdq_high = predictor.predict_mdq(sample_mdq_high)
    res_mdq_low = predictor.predict_mdq(sample_mdq_low)

    print("\n[MDQ High Risk Example]:")
    print(json.dumps(res_mdq_high, indent=2))
    assert res_mdq_high["screening_positive"] is True

    print("\n[MDQ Low Risk Example]:")
    print(json.dumps(res_mdq_low, indent=2))
    assert res_mdq_low["screening_positive"] is False

    print("\n[SUCCESS] Predictor verified successfully for all test cases!")


if __name__ == "__main__":
    main()
