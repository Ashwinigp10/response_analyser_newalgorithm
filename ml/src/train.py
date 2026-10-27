import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
PROCESSED_DIR = ML_DIR / "data" / "processed"
MODELS_DIR = ML_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

PQ16_FEATURES = [f"PRO16_{i}" for i in range(1, 17)]
MDQ_FEATURES = [f"MDQ1_{i}" for i in range(1, 14)] + ["MDQ_Frequency", "MDQ3_Severity"]


def train_pq16_model():
    print("=" * 60)
    print("TRAINING XGBOOST MODEL: PQ-16 Psychosis Prodrome Screening")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "pq16_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "pq16_val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "pq16_test.csv")

    X_train, y_train = train_df[PQ16_FEATURES], train_df["target"]
    X_val, y_val = val_df[PQ16_FEATURES], val_df["target"]
    X_test, y_test = test_df[PQ16_FEATURES], test_df["target"]

    # Calculate scale_pos_weight for class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(1, num_pos))
    print(f"PQ-16 Training samples: {len(X_train)} (Neg: {num_neg}, Pos: {num_pos}, scale_pos_weight: {scale_pos_weight:.2f})")

    # Hyperparameters
    params = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 15
    }

    # Stratified 5-Fold Cross Validation on Train Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )
    cv_scores = cross_validate(
        cv_clf, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    print("\n5-Fold Cross Validation Results (Train set):")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cv_scores[f"test_{metric}"]
        print(f"  CV {metric.capitalize()}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Final Model Training with Validation early stopping
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    best_iteration = model.best_iteration
    print(f"\nTraining completed. Best iteration: {best_iteration}")

    # Evaluate on held-out test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Recall/Sensitivity: {test_metrics['recall']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Save artifacts
    model_json_path = MODELS_DIR / "pq16_xgboost_model.json"
    model.save_model(str(model_json_path))

    pipeline_bundle = {
        "model_type": "XGBClassifier",
        "xgboost_version": xgb.__version__,
        "task": "pq16_screening",
        "features": PQ16_FEATURES,
        "target": "target",
        "hyperparameters": {k: v for k, v in params.items() if k != "early_stopping_rounds"},
        "best_iteration": int(best_iteration),
        "scale_pos_weight": scale_pos_weight,
        "cv_metrics": {m: {"mean": float(cv_scores[f"test_{m}"].mean()), "std": float(cv_scores[f"test_{m}"].std())} for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "test_metrics": test_metrics,
        "saved_model_file": "pq16_xgboost_model.json"
    }
    with open(MODELS_DIR / "pq16_xgboost_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_bundle, f, indent=2)

    joblib.dump(model, MODELS_DIR / "pq16_pipeline.joblib")
    print(f"[SAVED] Artifacts saved to {MODELS_DIR}")
    return pipeline_bundle


def train_mdq_model():
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL: MDQ Bipolar Screening")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "mdq_train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "mdq_val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "mdq_test.csv")

    X_train, y_train = train_df[MDQ_FEATURES], train_df["target"]
    X_val, y_val = val_df[MDQ_FEATURES], val_df["target"]
    X_test, y_test = test_df[MDQ_FEATURES], test_df["target"]

    # Calculate scale_pos_weight for class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(1, num_pos))
    print(f"MDQ Training samples: {len(X_train)} (Neg: {num_neg}, Pos: {num_pos}, scale_pos_weight: {scale_pos_weight:.2f})")

    # Hyperparameters tuned for imbalanced tabular screening data
    params = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.2,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 15
    }

    # Stratified 5-Fold Cross Validation on Train Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )
    cv_scores = cross_validate(
        cv_clf, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    print("\n5-Fold Cross Validation Results (Train set):")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cv_scores[f"test_{metric}"]
        print(f"  CV {metric.capitalize()}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Final Model Training with Validation early stopping
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    best_iteration = model.best_iteration
    print(f"\nTraining completed. Best iteration: {best_iteration}")

    # Evaluate on held-out test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Recall/Sensitivity: {test_metrics['recall']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Save artifacts
    model_json_path = MODELS_DIR / "mdq_xgboost_model.json"
    model.save_model(str(model_json_path))

    pipeline_bundle = {
        "model_type": "XGBClassifier",
        "xgboost_version": xgb.__version__,
        "task": "mdq_screening",
        "features": MDQ_FEATURES,
        "target": "target",
        "hyperparameters": {k: v for k, v in params.items() if k != "early_stopping_rounds"},
        "best_iteration": int(best_iteration),
        "scale_pos_weight": scale_pos_weight,
        "cv_metrics": {m: {"mean": float(cv_scores[f"test_{m}"].mean()), "std": float(cv_scores[f"test_{m}"].std())} for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "test_metrics": test_metrics,
        "saved_model_file": "mdq_xgboost_model.json"
    }
    with open(MODELS_DIR / "mdq_xgboost_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_bundle, f, indent=2)

    joblib.dump(model, MODELS_DIR / "mdq_pipeline.joblib")
    print(f"[SAVED] Artifacts saved to {MODELS_DIR}")
    return pipeline_bundle


def main():
    print("STARTING XGBOOST MODEL TRAINING FOR MINDWAVE...")
    pq_res = train_pq16_model()
    mdq_res = train_mdq_model()

    summary = {
        "trained_timestamp": pd.Timestamp.now().isoformat(),
        "xgboost_version": xgb.__version__,
        "pq16": pq_res,
        "mdq": mdq_res
    }
    with open(MODELS_DIR / "models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("[ALL DONE] Both PQ-16 and MDQ XGBoost models trained and verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
