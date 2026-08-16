from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NDA_DIR = PROJECT_ROOT / "data" / "raw" / "nda_2024_25"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Locate NDA workbook
# --------------------------------------------------

excel_files = list(NDA_DIR.glob("*.xlsx"))

if len(excel_files) != 1:
    raise RuntimeError(
        f"Expected exactly one NDA Excel file, "
        f"but found {len(excel_files)}."
    )

nda_file = excel_files[0]


# --------------------------------------------------
# 3. Load raw Type 2 / other diabetes worksheet
# --------------------------------------------------

raw = pd.read_excel(
    nda_file,
    sheet_name="Type 2 and other CP_TT",
    header=None
)


# --------------------------------------------------
# 4. Extract the 42 statutory ICB rows
# --------------------------------------------------

data = raw.iloc[9:].copy()

icb_mask = (
    data[0].astype("string").str.fullmatch(
        r"Q[A-Z0-9]{2}",
        na=False
    )
    &
    data[[2, 3, 4, 5]].isna().all(axis=1)
    &
    (data[0] != "Q99")
)

icb = data.loc[icb_mask].copy()

if len(icb) != 42:
    raise RuntimeError(
        f"Expected 42 statutory ICB rows, found {len(icb)}."
    )


# --------------------------------------------------
# 5. Find all percentage measure columns automatically
# --------------------------------------------------

percentage_columns = []

for col in raw.columns:
    if raw.iloc[6, col] == "Percentage":
        percentage_columns.append(col)

print(f"Percentage measures detected: {len(percentage_columns)}")


# --------------------------------------------------
# 6. Function to create safe column names
# --------------------------------------------------

def clean_column_name(text):
    text = str(text).strip().lower()

    text = text.replace("<=", "le")
    text = text.replace("%", "pct")

    text = re.sub(r"[^a-z0-9]+", "_", text)

    return text.strip("_")


# --------------------------------------------------
# 7. Build clean analysis table
# --------------------------------------------------

clean = pd.DataFrame()

clean["icb_code"] = icb[0].astype("string")
clean["icb_name"] = icb[1].astype("string").str.strip()

# Forward-fill the measure labels across the merged Excel header cells
measure_labels = raw.iloc[5].ffill()

for col in percentage_columns:
    measure_name = measure_labels[col]

    clean_name = clean_column_name(measure_name)

    clean[clean_name] = pd.to_numeric(
        icb[col],
        errors="coerce"
    )

# --------------------------------------------------
# 8. Basic quality checks
# --------------------------------------------------

print(f"Clean dataset shape: {clean.shape}")
print(f"Unique ICB codes: {clean['icb_code'].nunique()}")
print(f"Duplicate ICB codes: {clean['icb_code'].duplicated().sum()}")

print("\nMissing values by variable:")
print(clean.isna().sum())


# --------------------------------------------------
# 9. Save analysis-ready NDA table
# --------------------------------------------------

output_file = OUTPUT_DIR / "nda_type2_icb_2024_25.csv"

clean.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")