import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
RAW_DIR = ML_DIR / "data" / "raw"
PROCESSED_DIR = ML_DIR / "data" / "processed"

RAW_EXCEL_FILE = RAW_DIR / "Akram et al 2023 Student MH Data UK SciData.xlsx"
RANDOM_SEED = 42

PQ16_FEATURE_COLS = [f"PRO16_{i}" for i in range(1, 17)]
MDQ_FEATURE_COLS = [f"MDQ1_{i}" for i in range(1, 14)] + ["MDQ_Frequency", "MDQ3_Severity"]


def load_raw_data() -> pd.DataFrame:
    if not RAW_EXCEL_FILE.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {RAW_EXCEL_FILE}")
    print(f"Loading raw dataset from {RAW_EXCEL_FILE}...")
    df = pd.read_excel(RAW_EXCEL_FILE)
    print(f"Loaded raw dataset with shape: {df.shape}")
    return df


def preprocess_pq16(df: pd.DataFrame):
    print("\n[Preprocessing PQ-16 Data]")
    # Extract feature columns and target
    subset = df[PQ16_FEATURE_COLS + ["P16_Psychotic_Exp_Sum"]].copy()

    # Calculate target: standard Ultra-High Risk (UHR) cutoff is score >= 6
    # (Ising et al. 2012 Schizophrenia Research)
    subset["target"] = (subset["P16_Psychotic_Exp_Sum"] >= 6).astype(int)

    # Missing value handling: Impute NaN in PRO16 items with 0 (unendorsed)
    missing_before = subset[PQ16_FEATURE_COLS].isna().sum().sum()
    subset[PQ16_FEATURE_COLS] = subset[PQ16_FEATURE_COLS].fillna(0).astype(int)
    print(f"Imputed {missing_before} missing entries in PQ-16 items with 0.")

    X = subset[PQ16_FEATURE_COLS]
    y = subset["target"]

    print(f"PQ-16 Class distribution: {y.value_counts().to_dict()} (Positive rate: {y.mean()*100:.2f}%)")

    # Stratified split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"PQ-16 Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess_mdq(df: pd.DataFrame):
    print("\n[Preprocessing MDQ Data]")
    subset = df[MDQ_FEATURE_COLS + ["MDQ5"]].copy()

    # MDQ Hirschfeld et al. (2000) Clinical Positive Screen:
    # 1. 7 or more endorsed items in MDQ1_1..13
    # 2. Symptoms occurred during the same period (MDQ_Frequency == 1)
    # 3. Caused moderate or serious problem (MDQ3_Severity >= 3)
    mdq1_cols = [f"MDQ1_{i}" for i in range(1, 14)]
    mdq_sum = subset[mdq1_cols].sum(axis=1)

    # Impute missing in MDQ_Frequency with 0 (if no symptoms, frequency is 0)
    subset["MDQ_Frequency"] = subset["MDQ_Frequency"].fillna(0).astype(int)
    subset["MDQ3_Severity"] = subset["MDQ3_Severity"].fillna(1).astype(int)
    subset[mdq1_cols] = subset[mdq1_cols].fillna(0).astype(int)

    # Create target
    subset["target"] = (
        (mdq_sum >= 7) &
        (subset["MDQ_Frequency"] == 1) &
        (subset["MDQ3_Severity"] >= 3)
    ).astype(int)

    X = subset[MDQ_FEATURE_COLS]
    y = subset["target"]

    print(f"MDQ Class distribution: {y.value_counts().to_dict()} (Positive rate: {y.mean()*100:.2f}%)")

    # Stratified split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"MDQ Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()

    # PQ-16
    pq_train, pq_val, pq_test = preprocess_pq16(df)
    pq_train[0].assign(target=pq_train[1]).to_csv(PROCESSED_DIR / "pq16_train.csv", index=False)
    pq_val[0].assign(target=pq_val[1]).to_csv(PROCESSED_DIR / "pq16_val.csv", index=False)
    pq_test[0].assign(target=pq_test[1]).to_csv(PROCESSED_DIR / "pq16_test.csv", index=False)

    # MDQ
    mdq_train, mdq_val, mdq_test = preprocess_mdq(df)
    mdq_train[0].assign(target=mdq_train[1]).to_csv(PROCESSED_DIR / "mdq_train.csv", index=False)
    mdq_val[0].assign(target=mdq_val[1]).to_csv(PROCESSED_DIR / "mdq_val.csv", index=False)
    mdq_test[0].assign(target=mdq_test[1]).to_csv(PROCESSED_DIR / "mdq_test.csv", index=False)

    # Save metadata schema
    schema = {
        "pq16": {
            "features": PQ16_FEATURE_COLS,
            "target": "target",
            "target_meaning": "PQ-16 Prodromal Psychosis Screening Positive (Score >= 6)",
            "train_size": len(pq_train[0]),
            "val_size": len(pq_val[0]),
            "test_size": len(pq_test[0]),
            "positive_rate": float(pq_train[1].mean())
        },
        "mdq": {
            "features": MDQ_FEATURE_COLS,
            "target": "target",
            "target_meaning": "MDQ Hirschfeld Bipolar Positive Screen (7+ symptoms, co-occurring, moderate/serious severity)",
            "train_size": len(mdq_train[0]),
            "val_size": len(mdq_val[0]),
            "test_size": len(mdq_test[0]),
            "positive_rate": float(mdq_train[1].mean())
        },
        "random_seed": RANDOM_SEED
    }
    with open(PROCESSED_DIR / "dataset_splits_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"\n[SUCCESS] Preprocessing completed. Processed files saved in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
