from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QOF_DIR = PROJECT_ROOT / "data" / "raw" / "qof_2024_25"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Locate QOF workbook
# --------------------------------------------------

excel_files = list(QOF_DIR.glob("*.xlsx"))

if len(excel_files) != 1:
    raise RuntimeError(
        f"Expected exactly one QOF Excel file, "
        f"but found {len(excel_files)}."
    )

qof_file = excel_files[0]


# --------------------------------------------------
# 3. Load Diabetes Mellitus worksheet
# --------------------------------------------------

raw = pd.read_excel(
    qof_file,
    sheet_name="DM",
    header=None
)


# --------------------------------------------------
# 4. Extract 42 statutory ICB rows
# --------------------------------------------------

data = raw.iloc[12:].copy()

icb_mask = (
    data[0].astype("string").str.fullmatch(
        r"Q[A-Z0-9]{2}",
        na=False
    )
    &
    data[1].astype("string").str.fullmatch(
        r"E54[0-9]+",
        na=False
    )
)

icb = data.loc[icb_mask].copy()

if len(icb) != 42:
    raise RuntimeError(
        f"Expected 42 statutory ICB rows, found {len(icb)}."
    )


# --------------------------------------------------
# 5. Build clean candidate-variable table
# --------------------------------------------------

clean = pd.DataFrame()

clean["icb_code"] = icb[0].astype("string")
clean["icb_ons_code"] = icb[1].astype("string")
clean["icb_name"] = icb[2].astype("string").str.strip()

# Diabetes prevalence
clean["diabetes_prevalence_2023_24"] = pd.to_numeric(
    icb[6],
    errors="coerce"
)

clean["diabetes_prevalence_2024_25"] = pd.to_numeric(
    icb[11],
    errors="coerce"
)

clean["diabetes_prevalence_yoy_change"] = pd.to_numeric(
    icb[12],
    errors="coerce"
)

# QOF diabetes indicators:
# use "Patients receiving Intervention (%)"
clean["dm006_nephropathy_acei_arb"] = pd.to_numeric(
    icb[34],
    errors="coerce"
)

clean["dm012_foot_exam_risk_classification"] = pd.to_numeric(
    icb[42],
    errors="coerce"
)

clean["dm014_structured_education_referral"] = pd.to_numeric(
    icb[50],
    errors="coerce"
)

clean["dm020_hba1c_control_no_frailty"] = pd.to_numeric(
    icb[58],
    errors="coerce"
)

clean["dm021_hba1c_control_with_frailty"] = pd.to_numeric(
    icb[66],
    errors="coerce"
)

clean["dm022_primary_prevention_statin"] = pd.to_numeric(
    icb[74],
    errors="coerce"
)

clean["dm023_secondary_prevention_statin"] = pd.to_numeric(
    icb[82],
    errors="coerce"
)

clean["dm033_bp_control"] = pd.to_numeric(
    icb[90],
    errors="coerce"
)


# --------------------------------------------------
# 6. Quality checks
# --------------------------------------------------

print(f"Clean QOF dataset shape: {clean.shape}")
print(f"Unique ICB codes: {clean['icb_code'].nunique()}")
print(f"Duplicate ICB codes: {clean['icb_code'].duplicated().sum()}")

print("\nMissing values:")
print(clean.isna().sum())

numeric = clean.select_dtypes(include="number")

summary = pd.DataFrame({
    "min": numeric.min(),
    "max": numeric.max(),
    "mean": numeric.mean().round(2),
    "unique": numeric.nunique()
})

print("\nVariable summary:")
print(summary.to_string())


# --------------------------------------------------
# 7. Save interim table
# --------------------------------------------------

output_file = OUTPUT_DIR / "qof_diabetes_icb_2024_25.csv"

clean.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")