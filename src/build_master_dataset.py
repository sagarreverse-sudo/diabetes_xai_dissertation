from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
INPUT_DIR = Path("data/interim")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "nda_predictors_long.csv"


# ---------------------------------------------------------
# Historical NDA files
# ---------------------------------------------------------
FILES = {
    "2018_19": INPUT_DIR / "nda_type2_icb_2018_19.csv",
    "2019_20": INPUT_DIR / "nda_type2_icb_2019_20.csv",
    "2020_21": INPUT_DIR / "nda_type2_icb_2020_21.csv",
    "2021_22": INPUT_DIR / "nda_type2_icb_2021_22.csv",
}


# ---------------------------------------------------------
# 19 predictors available in ALL four years
# ---------------------------------------------------------
COMMON_PREDICTORS = [
    "hba1c",
    "blood_pressure",
    "cholesterol",
    "serum_creatinine",
    "urine_albumin",
    "foot_surveillance",
    "bmi",
    "smoking",
    "all_eight_care_processes",
    "hba1c_le_48_mmol_mol_6_5pct",
    "hba1c_le_53_mmol_mol_7_0pct",
    "hba1c_le_58_mmol_mol_7_5pct",
    "hba1c_le_75_mmol_mol_9_0pct",
    "hba1c_le_86_mmol_mol_10_0pct",
    "blood_pressure_le_140_80",
    "all_three_treatment_targets",
    "primary_prevention_on_statins_without_cvd_history",
    "secondary_prevention_on_statins_with_cvd_history",
    "combined_prevention_on_statins",
]


# ---------------------------------------------------------
# Target year corresponding to each NDA predictor period
# ---------------------------------------------------------
TARGET_YEAR_MAP = {
    "2018_19": 2020,
    "2019_20": 2021,
    "2020_21": 2022,
    "2021_22": 2023,
}


# ---------------------------------------------------------
# Load and standardise each annual dataset
# ---------------------------------------------------------
yearly_data = []

for period, file_path in FILES.items():

    print(f"\nLoading: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = ["icb_code"] + COMMON_PREDICTORS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{period} is missing required columns: {missing_columns}"
        )

    # Keep only the common variables
    df = df[required_columns].copy()

    # Standardise ICB codes
    df["icb_code"] = (
        df["icb_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Add explicit time variables
    df["predictor_period"] = period
    df["target_year"] = TARGET_YEAR_MAP[period]

    # Check expected number of ICBs
    if len(df) != 42:
        raise ValueError(
            f"{period}: expected 42 rows but found {len(df)}"
        )

    # Check duplicate ICBs within each period
    duplicate_count = df["icb_code"].duplicated().sum()

    if duplicate_count != 0:
        raise ValueError(
            f"{period}: found {duplicate_count} duplicate ICB codes"
        )

    yearly_data.append(df)

    print(f"{period}: PASSED")
    print(f"Rows: {len(df)}")
    print(f"Unique ICBs: {df['icb_code'].nunique()}")


# ---------------------------------------------------------
# Combine all four years vertically
# ---------------------------------------------------------
master = pd.concat(
    yearly_data,
    ignore_index=True
)


# Put identifiers first
column_order = (
    ["icb_code", "predictor_period", "target_year"]
    + COMMON_PREDICTORS
)

master = master[column_order]


# ---------------------------------------------------------
# Final validation checks
# ---------------------------------------------------------
expected_rows = 42 * 4

if len(master) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows but found {len(master)}"
    )

duplicate_pairs = master.duplicated(
    subset=["icb_code", "predictor_period"]
).sum()

if duplicate_pairs != 0:
    raise ValueError(
        f"Found {duplicate_pairs} duplicate ICB-period rows"
    )

missing_predictor_values = (
    master[COMMON_PREDICTORS]
    .isna()
    .sum()
    .sum()
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
master = master.sort_values(
    ["target_year", "icb_code"]
).reset_index(drop=True)

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Audit output
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MASTER NDA PREDICTOR DATASET CREATED")
print("=" * 60)

print(f"Output: {OUTPUT_FILE}")
print(f"Shape: {master.shape}")
print(f"Rows: {len(master)}")
print(f"Unique ICBs: {master['icb_code'].nunique()}")
print(f"Predictor periods: {master['predictor_period'].nunique()}")
print(f"Common predictors: {len(COMMON_PREDICTORS)}")
print(f"Duplicate ICB-period rows: {duplicate_pairs}")
print(f"Missing predictor values: {missing_predictor_values}")

print("\nRows by predictor period:")
print(
    master.groupby("predictor_period")
    .size()
    .sort_index()
)

print("\nTarget years:")
print(
    master.groupby("target_year")
    .size()
    .sort_index()
)

print("\nExpected final shape: (168, 22)")
print(f"Actual final shape:   {master.shape}")

if (
    master.shape == (168, 22)
    and duplicate_pairs == 0
    and missing_predictor_values == 0
):
    print("\nMASTER DATASET AUDIT: PASSED")
else:
    print("\nMASTER DATASET AUDIT: REVIEW REQUIRED")