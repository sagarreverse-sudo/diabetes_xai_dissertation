from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nda_complications"
    / "Open data - Chronic kidney disease.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load raw CKD complication dataset
# --------------------------------------------------

raw = pd.read_csv(RAW_FILE)

print(f"Raw CKD dataset shape: {raw.shape}")


# --------------------------------------------------
# 3. Select Type 2 diabetes CKD outcome
#    at statutory English ICB level
# --------------------------------------------------

ckd = raw[
    (raw["Organisation type"] == "ICB/LHB")
    & (raw["Country name"] == "England")
    & (raw["Diabetes type"] == "Type 2")
    & (raw["Sex"] == "All")
    & (raw["Demographic"] == "All")
    & (raw["Complication"] == "Chronic kidney disease")
    & (raw["Admission type"] == "Any")
].copy()


# --------------------------------------------------
# 4. Keep only fields required for the outcome panel
# --------------------------------------------------

ckd = ckd[
    [
        "Organisation code",
        "Organisation name",
        "Year",
        "Cases",
        "Population",
        "Risk rate per 1,000 people",
    ]
].copy()


# --------------------------------------------------
# 5. Rename columns
# --------------------------------------------------

ckd = ckd.rename(
    columns={
        "Organisation code": "icb_code",
        "Organisation name": "icb_name",
        "Year": "year",
        "Cases": "ckd_cases",
        "Population": "diabetes_population",
        "Risk rate per 1,000 people": "ckd_risk_rate_per_1000",
    }
)


# --------------------------------------------------
# 6. Clean organisation names
# --------------------------------------------------

ckd["icb_code"] = (
    ckd["icb_code"]
    .astype("string")
    .str.strip()
)

ckd["icb_name"] = (
    ckd["icb_name"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# --------------------------------------------------
# 7. Convert numeric fields safely
# --------------------------------------------------

numeric_columns = [
    "year",
    "ckd_cases",
    "diabetes_population",
    "ckd_risk_rate_per_1000",
]

for col in numeric_columns:
    ckd[col] = pd.to_numeric(
        ckd[col],
        errors="coerce"
    )


# --------------------------------------------------
# 8. Sort panel consistently
# --------------------------------------------------

ckd = (
    ckd
    .sort_values(
        ["year", "icb_code"]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 9. Basic structural checks
# --------------------------------------------------

print(f"Filtered CKD panel shape: {ckd.shape}")
print(f"Unique ICBs: {ckd['icb_code'].nunique()}")
print(f"Years: {ckd['year'].nunique()}")

duplicate_count = ckd.duplicated(
    subset=["icb_code", "year"]
).sum()

print(f"Duplicate ICB-year rows: {duplicate_count}")


# --------------------------------------------------
# 10. Missing-value audit
# --------------------------------------------------

print("\nMissing values:")
print(ckd.isna().sum())


# --------------------------------------------------
# 11. Check number of ICBs per year
# --------------------------------------------------

year_counts = (
    ckd
    .groupby("year")["icb_code"]
    .nunique()
)

print("\nICBs per year:")
print(year_counts.to_string())


# --------------------------------------------------
# 12. Outcome summary
# --------------------------------------------------

print("\nCKD risk-rate summary:")
print(
    ckd["ckd_risk_rate_per_1000"]
    .describe()
    .round(2)
)


# --------------------------------------------------
# 13. Validation rules
# --------------------------------------------------

if len(ckd) != 630:
    raise RuntimeError(
        f"Expected 630 ICB-year rows, "
        f"but found {len(ckd)}."
    )

if ckd["icb_code"].nunique() != 42:
    raise RuntimeError(
        f"Expected 42 unique ICBs, "
        f"but found {ckd['icb_code'].nunique()}."
    )

if ckd["year"].nunique() != 15:
    raise RuntimeError(
        f"Expected 15 years, "
        f"but found {ckd['year'].nunique()}."
    )

if duplicate_count != 0:
    raise RuntimeError(
        "Duplicate ICB-year rows detected."
    )

if not (year_counts == 42).all():
    raise RuntimeError(
        "Not every year contains all 42 ICBs."
    )

if ckd["ckd_cases"].isna().any():
    raise RuntimeError(
        "Missing CKD case counts detected."
    )

if ckd["diabetes_population"].isna().any():
    raise RuntimeError(
        "Missing diabetes population values detected."
    )

if ckd["ckd_risk_rate_per_1000"].isna().any():
    raise RuntimeError(
        "Missing CKD risk-rate values detected."
    )

if (ckd["ckd_cases"] < 0).any():
    raise RuntimeError(
        "Negative CKD case counts detected."
    )

if (ckd["diabetes_population"] <= 0).any():
    raise RuntimeError(
        "Invalid diabetes population values detected."
    )

if (ckd["ckd_risk_rate_per_1000"] < 0).any():
    raise RuntimeError(
        "Negative CKD risk-rate values detected."
    )


print("\nCKD outcome panel structure check: PASSED")


# --------------------------------------------------
# 14. Save cleaned outcome panel
# --------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)

ckd.to_csv(
    output_file,
    index=False
)

print(f"Saved to: {output_file}")