import pandas as pd
from pathlib import Path

data_path = Path("ml/data/raw/Akram et al 2023 Student MH Data UK SciData.xlsx")
df = pd.read_excel(data_path)

print("PRO16_1 unique values:", df["PRO16_1"].unique())
print("PRO16_1 value counts:\n", df["PRO16_1"].value_counts(dropna=False))

print("\nMDQ1_1 unique values:", df["MDQ1_1"].unique())
print("MDQ1_1 value counts:\n", df["MDQ1_1"].value_counts(dropna=False))

print("\nMDQ_Frequency value counts:\n", df["MDQ_Frequency"].value_counts(dropna=False))
print("\nMDQ3_Severity value counts:\n", df["MDQ3_Severity"].value_counts(dropna=False))
print("\nMDQ5 (bipolar diagnosis) value counts:\n", df["MDQ5"].value_counts(dropna=False))

# Check missing values across PRO16 columns
pro_cols = [f"PRO16_{i}" for i in range(1, 17)]
print("\nMissing values in PRO16 items:", df[pro_cols].isna().sum().to_dict())

# Check missing values across MDQ columns
mdq_cols = [f"MDQ1_{i}" for i in range(1, 14)]
print("Missing values in MDQ1 items:", df[mdq_cols].isna().sum().to_dict())
print("Missing in MDQ_Frequency:", df["MDQ_Frequency"].isna().sum())
print("Missing in MDQ3_Severity:", df["MDQ3_Severity"].isna().sum())
