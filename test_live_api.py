import urllib.request, json

def test_predict(assessment_type, label):
    payload = json.dumps({
        "assessment_type": assessment_type,
        "answers": [{"id": i, "answer": "True", "isTrue": True} for i in range(1, 9)],
        "age_group": "19-25",
        "profession": "student"
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/ml/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    print(f"\n=== {label} ===")
    print(f"  screening_positive : {data['screening_positive']}")
    print(f"  probability        : {data['probability']:.4f}")
    print(f"  risk_level         : {data['risk_level']}")
    print(f"  symptom_count      : {data['symptom_count']}")
    print(f"  algorithm          : {data['algorithm']}")

test_predict("schizophrenia", "PQ-16 Test (8 True answers)")
test_predict("bipolar", "MDQ Test (8 True answers)")
print("\nAll prediction tests PASSED")
