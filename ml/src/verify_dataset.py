import os
import sys
import json
from pathlib import Path

# Fix Windows console encoding issues
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import pypdf
import pyreadstat


SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
RAW_DATA_DIR = ML_DIR / "data" / "raw"

XLSX_PATH = RAW_DATA_DIR / "Akram et al 2023 Student MH Data UK SciData.xlsx"
SAV_PATH = RAW_DATA_DIR / "Akram et al 2023 Student MH Data UK SciData.sav"
PDF_PATH = RAW_DATA_DIR / "Variable codes and abbreviations .pdf"


def inspect_documentation():
    print("=" * 60)
    print("INSPECTING VARIABLE CODES AND DOCUMENTATION")
    print("=" * 60)
    if PDF_PATH.exists():
        reader = pypdf.PdfReader(str(PDF_PATH))
        full_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text.append(f"--- PAGE {i+1} ---\n" + text)
        doc_str = "\n".join(full_text)
        print(f"Read {len(reader.pages)} pages from PDF.")
        doc_dump_path = RAW_DATA_DIR / "variable_codes_extracted.txt"
        with open(doc_dump_path, "w", encoding="utf-8") as f:
            f.write(doc_str)
        print(f"Extracted PDF text saved to {doc_dump_path}")
    else:
        print("PDF not found!")


def inspect_spss_metadata():
    print("=" * 60)
    print("INSPECTING SPSS SAV METADATA")
    print("=" * 60)
    if SAV_PATH.exists():
        df_sav, meta = pyreadstat.read_sav(str(SAV_PATH), metadataonly=True)
        print(f"Total variables in SAV: {len(meta.column_names)}")
        print("\nRelevant variable labels:")
        for col_name, col_label in meta.column_names_to_labels.items():
            if any(term in col_name.upper() for term in ["MDQ", "PRO", "P16", "AGE", "SEX", "ETH"]):
                val_labels = meta.variable_value_labels.get(col_name, {})
                print(f"  {col_name}: {col_label} | Value Labels: {val_labels}")


def inspect_dataset_integrity():
    print("=" * 60)
    print("INSPECTING DATASET INTEGRITY AND DISTRIBUTIONS")
    print("=" * 60)
    df = pd.read_excel(str(XLSX_PATH))
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Check PQ-16 items
    pq16_cols = [f"PRO16_{i}" for i in range(1, 17)]
    missing_pq = [c for c in pq16_cols if c not in df.columns]
    print(f"\nPQ-16 items present: {len(pq16_cols) - len(missing_pq)}/16 (Missing: {missing_pq})")
    print("PQ-16 sum column stats (P16_Psychotic_Exp_Sum):")
    print(df["P16_Psychotic_Exp_Sum"].describe())

    # Check MDQ items
    mdq_cols = [f"MDQ1_{i}" for i in range(1, 14)]
    missing_mdq = [c for c in mdq_cols if c not in df.columns]
    print(f"\nMDQ items present: {len(mdq_cols) - len(missing_mdq)}/13 (Missing: {missing_mdq})")
    print("MDQ Mania score stats (MDQ_Mania):")
    print(df["MDQ_Mania"].describe())
    print("MDQ_Frequency value counts:")
    print(df["MDQ_Frequency"].value_counts(dropna=False))
    print("MDQ3_Severity value counts:")
    print(df["MDQ3_Severity"].value_counts(dropna=False))
    print("MDQ4 (family history) counts:")
    print(df["MDQ4"].value_counts(dropna=False))
    print("MDQ5 (prior bipolar diagnosis) counts:")
    print(df["MDQ5"].value_counts(dropna=False))


if __name__ == "__main__":
    inspect_documentation()
    inspect_spss_metadata()
    inspect_dataset_integrity()
