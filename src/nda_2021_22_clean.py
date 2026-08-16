from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NDA_DIR = PROJECT_ROOT / "data" / "raw" / "nda_2021_22"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Locate the historical NDA workbook
# --------------------------------------------------

excel_files = list(NDA_DIR.glob("*.xlsx"))

if len(excel_files) != 1:
    raise RuntimeError(
        f"Expected exactly one NDA 2021-22 workbook, "
        f"but found {len(excel_files)}."
    )

nda_file = excel_files[0]

print(f"NDA 2021-22 file: {nda_file.name}")


# --------------------------------------------------
# 3. Load Type 2 and other care-process data
# --------------------------------------------------

raw = pd.read_excel(
    nda_file,
    sheet_name="Type 2 and other CP_TT",
    header=None
)

print(f"Raw worksheet shape: {raw.shape}")


# --------------------------------------------------
# 4. Extract the 42 ICB summary rows
# --------------------------------------------------

data = raw.iloc[3:].copy()

icb_mask = (
    data[1].astype("string").str.fullmatch(
        r"Q[A-Z0-9]{2}",
        na=False
    )
    &
    (
        data[1].astype("string")
        == data[2].astype("string")
    )
    &
    (
        data[1].astype("string")
        == data[3].astype("string")
    )
)

icb = data.loc[icb_mask].copy()

print(f"Candidate ICB rows: {len(icb)}")
print(f"Unique ICB codes: {icb[1].nunique()}")
print(f"Duplicate ICB codes: {icb[1].duplicated().sum()}")

if len(icb) != 42:
    raise RuntimeError(
        f"Expected 42 ICB rows, found {len(icb)}."
    )

if icb[1].duplicated().any():
    raise RuntimeError(
        "Duplicate ICB codes detected."
    )


# --------------------------------------------------
# 5. Detect all percentage-measure columns
# --------------------------------------------------

percentage_columns = [
    col
    for col in raw.columns
    if raw.iloc[2, col] == "Percentage"
]

print(
    f"Percentage measures detected: "
    f"{len(percentage_columns)}"
)


# --------------------------------------------------
# 6. Prepare measure names
# --------------------------------------------------

# Excel uses merged cells for measure headings.
# Forward-fill makes each numerator/denominator/
# percentage column inherit the correct measure name.

measure_labels = raw.iloc[1].ffill()


def clean_column_name(text):
    """
    Convert NHS measure names into safe,
    consistent Python column names.
    """

    text = str(text).strip().lower()

    text = text.replace("<=", "le")
    text = text.replace("%", "pct")

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# --------------------------------------------------
# 7. Build clean historical predictor table
# --------------------------------------------------

clean = pd.DataFrame()

clean["audit_year"] = (
    icb[0]
    .astype("string")
    .str.strip()
)

clean["icb_code"] = (
    icb[1]
    .astype("string")
    .str.strip()
)


for col in percentage_columns:

    measure_name = measure_labels[col]

    clean_name = clean_column_name(
        measure_name
    )

    clean[clean_name] = pd.to_numeric(
        icb[col],
        errors="coerce"
    )


# --------------------------------------------------
# 8. Sort consistently
# --------------------------------------------------

clean = (
    clean
    .sort_values("icb_code")
    .reset_index(drop=True)
)


# --------------------------------------------------
# 9. Structural and missing-value checks
# --------------------------------------------------

print(
    f"Clean historical NDA shape: "
    f"{clean.shape}"
)

print(
    f"Unique ICB codes: "
    f"{clean['icb_code'].nunique()}"
)

print(
    f"Duplicate ICB codes: "
    f"{clean['icb_code'].duplicated().sum()}"
)

print("\nAudit-year values:")
print(
    clean["audit_year"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nMissing values:")
print(
    clean
    .isna()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 10. Check numeric ranges and variation
# --------------------------------------------------

numeric = clean.select_dtypes(
    include="number"
)

summary = pd.DataFrame({
    "min": numeric.min(),
    "max": numeric.max(),
    "mean": numeric.mean().round(2),
    "unique": numeric.nunique()
})

print("\nVariable summary:")
print(summary.to_string())


# --------------------------------------------------
# 11. Validation rules
# --------------------------------------------------

if clean["icb_code"].nunique() != 42:
    raise RuntimeError(
        "Expected 42 unique ICBs."
    )

if clean["icb_code"].duplicated().any():
    raise RuntimeError(
        "Duplicate ICB codes detected."
    )

if set(clean["audit_year"].dropna()) != {"2021_22"}:
    raise RuntimeError(
        "Unexpected audit-year value detected."
    )

# Percentage measures should not be outside 0-100.
for col in numeric.columns:

    if (
        (clean[col] < 0).any()
        or
        (clean[col] > 100).any()
    ):
        raise RuntimeError(
            f"Invalid percentage range in {col}."
        )


print(
    "\nNDA 2021-22 historical predictor "
    "structure check: PASSED"
)


# --------------------------------------------------
# 12. Save cleaned historical predictor table
# --------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "nda_type2_icb_2021_22.csv"
)

clean.to_csv(
    output_file,
    index=False
)

print(f"Saved to: {output_file}")