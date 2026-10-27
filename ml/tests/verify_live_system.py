import urllib.request
import json
import sys

# Test Backend Health
print("Testing Backend Health (http://127.0.0.1:5000/api/ml/health)...")
try:
    with urllib.request.urlopen("http://127.0.0.1:5000/api/ml/health") as res:
        assert res.getcode() == 200
        health_data = json.loads(res.read().decode("utf-8"))
        print("  [PASS] Backend Health:", health_data)
        assert health_data["status"] == "ok"
        assert health_data["models_loaded"] is True
except Exception as e:
    print("  [FAIL] Backend Health Error:", e)
    sys.exit(1)

# Test Backend Model Info
print("\nTesting Model Info (http://127.0.0.1:5000/api/ml/model-info)...")
try:
    with urllib.request.urlopen("http://127.0.0.1:5000/api/ml/model-info") as res:
        assert res.getcode() == 200
        info_data = json.loads(res.read().decode("utf-8"))
        print("  [PASS] Framework:", info_data.get("framework"))
        print("  [PASS] PQ-16 Test Accuracy:", info_data["models"]["pq16"]["test_metrics"]["accuracy"])
        print("  [PASS] MDQ Test Accuracy:", info_data["models"]["mdq"]["test_metrics"]["accuracy"])
except Exception as e:
    print("  [FAIL] Model Info Error:", e)
    sys.exit(1)

# Test PQ-16 Prediction
print("\nTesting PQ-16 Live Prediction (http://127.0.0.1:5000/api/ml/predict)...")
try:
    payload = {
        "assessment_type": "schizophrenia",
        "answers": [{"id": i, "answer": "True" if i <= 8 else "False"} for i in range(1, 17)],
        "age_group": "19-25",
        "profession": "student"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/ml/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        assert res.getcode() == 200
        pq_res = json.loads(res.read().decode("utf-8"))
        print("  [PASS] Prediction Result:", pq_res)
        assert pq_res["algorithm"] == "XGBoost (XGBClassifier)"
        assert pq_res["assessment_type"] == "schizophrenia"
        assert 0.0 <= pq_res["probability"] <= 1.0
except Exception as e:
    print("  [FAIL] PQ-16 Prediction Error:", e)
    sys.exit(1)

# Test MDQ Prediction
print("\nTesting MDQ Live Prediction (http://127.0.0.1:5000/api/ml/predict)...")
try:
    payload = {
        "assessment_type": "bipolar",
        "answers": [
            {"id": i, "answer": "True" if i <= 9 else "False"} for i in range(1, 16)
        ],
        "age_group": "19-25",
        "profession": "student"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/ml/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        assert res.getcode() == 200
        mdq_res = json.loads(res.read().decode("utf-8"))
        print("  [PASS] Prediction Result:", mdq_res)
        assert mdq_res["algorithm"] == "XGBoost (XGBClassifier)"
        assert mdq_res["assessment_type"] == "bipolar"
        assert 0.0 <= mdq_res["probability"] <= 1.0
except Exception as e:
    print("  [FAIL] MDQ Prediction Error:", e)
    sys.exit(1)

print("\n[ALL LIVE ENDPOINT CHECKS PASSED SUCCESSFULLY]")
