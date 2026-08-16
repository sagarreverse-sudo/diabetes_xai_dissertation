from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_FILE = (
    INTERIM_DIR
    / "forecasting_panel_2019_20_to_2023.csv"
)


# --------------------------------------------------
# 2. Input files
# --------------------------------------------------

PREDICTOR_FILES = {
    "2019_20": INTERIM_DIR / "nda_type2_icb_2019_20.csv",
    "2020_21": INTERIM_DIR / "nda_type2_icb_2020_21.csv",
    "2021_22": INTERIM_DIR / "nda_type2_icb_2021_22.csv",
}

CKD_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)


# --------------------------------------------------
# 3. Forecasting alignment
#
# Predictor audit period -> later CKD calendar year
# --------------------------------------------------

TARGET_YEAR_MAP = {
    "2019_20": 2021,
    "2020_21": 2022,
    "2021_22": 2023,
}


# --------------------------------------------------
# 4. Common predictor set
#
# These 21 variables exist consistently across
# all three historical NDA releases.
# --------------------------------------------------

COMMON_FEATURES = [
    "hba1c",
    "blood_pressure",
    "cholesterol",
    "serum_creatinine",
    "urine_albumin",
    "retinal_screening",
    "foot_surveillance",
    "bmi",
    "smoking",
    "all_eight_care_processes",
    "all_nine_care_processes",
    "hba1c_le_48_mmol_mol_6_5pct",
    "hba1c_le_53_mmol_mol_7_0pct",
    "hba1c_le_58_mmol_mol_7_5pct",
    "hba1c_le_75_mmol_mol_9_0pct",
    "hba1c_le_86_mmol_mol_10_0pct",
    "blood_pressure_le_140_80",
    "primary_prevention_on_statins_without_cvd_history",
    "secondary_prevention_on_statins_with_cvd_history",
    "combined_prevention_on_statins",
    "all_three_treatment_targets",
]


# --------------------------------------------------
# 5. Check source files exist
# --------------------------------------------------

for audit_year, file_path in PREDICTOR_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing predictor file for "
            f"{audit_year}: {file_path}"
        )


if not CKD_FILE.exists():
    raise FileNotFoundError(
        f"Missing CKD outcome file: {CKD_FILE}"
    )


# --------------------------------------------------
# 6. Load CKD outcome panel
# --------------------------------------------------

ckd = pd.read_csv(CKD_FILE)

print(
    f"CKD outcome dataset shape: "
    f"{ckd.shape}"
)


required_ckd_columns = {
    "icb_code",
    "icb_name",
    "year",
    "ckd_cases",
    "diabetes_population",
    "ckd_risk_rate_per_1000",
}


missing_ckd_columns = (
    required_ckd_columns
    - set(ckd.columns)
)


if missing_ckd_columns:
    raise RuntimeError(
        "Missing required CKD columns: "
        f"{sorted(missing_ckd_columns)}"
    )


# --------------------------------------------------
# 7. Build each forecasting year separately
# --------------------------------------------------

forecasting_frames = []


for audit_year, predictor_file in PREDICTOR_FILES.items():

    print(
        f"\nProcessing predictor year "
        f"{audit_year}..."
    )

    predictors = pd.read_csv(
        predictor_file
    )

    target_year = (
        TARGET_YEAR_MAP[audit_year]
    )


    # ----------------------------------------------
    # Validate predictor structure
    # ----------------------------------------------

    required_predictor_columns = (
        {"audit_year", "icb_code"}
        | set(COMMON_FEATURES)
    )


    missing_predictor_columns = (
        required_predictor_columns
        - set(predictors.columns)
    )


    if missing_predictor_columns:
        raise RuntimeError(
            f"{audit_year} is missing "
            f"predictors: "
            f"{sorted(missing_predictor_columns)}"
        )


    if len(predictors) != 42:
        raise RuntimeError(
            f"{audit_year}: expected "
            f"42 predictor rows, "
            f"found {len(predictors)}."
        )


    if (
        predictors["icb_code"]
        .nunique()
        != 42
    ):
        raise RuntimeError(
            f"{audit_year}: expected "
            f"42 unique ICB codes."
        )


    if (
        predictors["icb_code"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{audit_year}: duplicate "
            f"ICB predictor rows detected."
        )


    # Keep only the common feature set.
    predictors = predictors[
        [
            "audit_year",
            "icb_code",
        ]
        + COMMON_FEATURES
    ].copy()


    # ----------------------------------------------
    # Select the later CKD outcome year
    # ----------------------------------------------

    outcome = ckd[
        ckd["year"] == target_year
    ][
        [
            "icb_code",
            "icb_name",
            "year",
            "ckd_cases",
            "diabetes_population",
            "ckd_risk_rate_per_1000",
        ]
    ].copy()


    print(
        f"Target CKD year: "
        f"{target_year}"
    )

    print(
        f"Predictor ICBs: "
        f"{predictors['icb_code'].nunique()}"
    )

    print(
        f"Outcome ICBs: "
        f"{outcome['icb_code'].nunique()}"
    )


    # ----------------------------------------------
    # Validate exact ICB code match
    # ----------------------------------------------

    predictor_codes = set(
        predictors["icb_code"]
    )

    outcome_codes = set(
        outcome["icb_code"]
    )


    only_predictors = sorted(
        predictor_codes
        - outcome_codes
    )

    only_outcome = sorted(
        outcome_codes
        - predictor_codes
    )


    if only_predictors or only_outcome:
        raise RuntimeError(
            f"ICB mismatch for {audit_year} "
            f"-> {target_year}.\n"
            f"Only predictors: "
            f"{only_predictors}\n"
            f"Only outcome: "
            f"{only_outcome}"
        )


    # ----------------------------------------------
    # Merge predictors with future CKD outcome
    # ----------------------------------------------

    merged = predictors.merge(
        outcome,
        on="icb_code",
        how="inner",
        validate="one_to_one"
    )


    merged = merged.rename(
        columns={
            "year": "target_year"
        }
    )


    # ----------------------------------------------
    # Validate merged forecasting block
    # ----------------------------------------------

    if len(merged) != 42:
        raise RuntimeError(
            f"{audit_year} -> "
            f"{target_year}: expected "
            f"42 merged rows, "
            f"found {len(merged)}."
        )


    if merged.isna().any().any():
        missing = (
            merged
            .isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        raise RuntimeError(
            f"Missing values after merging "
            f"{audit_year} -> "
            f"{target_year}:\n"
            f"{missing}"
        )


    forecasting_frames.append(
        merged
    )


# --------------------------------------------------
# 8. Combine all three forecasting periods
# --------------------------------------------------

panel = pd.concat(
    forecasting_frames,
    ignore_index=True
)


# --------------------------------------------------
# 9. Sort consistently
# --------------------------------------------------

panel = (
    panel
    .sort_values(
        [
            "target_year",
            "icb_code",
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 10. Reorder columns
# --------------------------------------------------

panel = panel[
    [
        "audit_year",
        "target_year",
        "icb_code",
        "icb_name",
    ]
    + COMMON_FEATURES
    + [
        "ckd_cases",
        "diabetes_population",
        "ckd_risk_rate_per_1000",
    ]
]


# --------------------------------------------------
# 11. Final panel audit
# --------------------------------------------------

print(
    f"\nFinal forecasting panel shape: "
    f"{panel.shape}"
)

print(
    f"Unique ICBs: "
    f"{panel['icb_code'].nunique()}"
)

print(
    f"Target years: "
    f"{sorted(panel['target_year'].unique())}"
)


print("\nRows per target year:")

print(
    panel
    .groupby("target_year")
    .size()
    .to_string()
)


print("\nUnique ICBs per target year:")

print(
    panel
    .groupby("target_year")["icb_code"]
    .nunique()
    .to_string()
)


print("\nMissing values:")

missing = (
    panel
    .isna()
    .sum()
)

print(
    missing.to_string()
)


print("\nCKD target summary by year:")

target_summary = (
    panel
    .groupby("target_year")[
        "ckd_risk_rate_per_1000"
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max",
        ]
    )
    .round(2)
)

print(
    target_summary.to_string()
)


# --------------------------------------------------
# 12. Final validation rules
# --------------------------------------------------

if len(panel) != 126:
    raise RuntimeError(
        f"Expected 126 forecasting rows, "
        f"found {len(panel)}."
    )


if panel["icb_code"].nunique() != 42:
    raise RuntimeError(
        "Expected 42 unique ICBs "
        "in final forecasting panel."
    )


expected_target_years = {
    2021,
    2022,
    2023,
}


if set(
    panel["target_year"].unique()
) != expected_target_years:

    raise RuntimeError(
        "Unexpected CKD target years."
    )


rows_per_year = (
    panel
    .groupby("target_year")
    .size()
)


if not (
    rows_per_year == 42
).all():

    raise RuntimeError(
        "Expected exactly 42 rows "
        "for every target year."
    )


icbs_per_year = (
    panel
    .groupby("target_year")[
        "icb_code"
    ]
    .nunique()
)


if not (
    icbs_per_year == 42
).all():

    raise RuntimeError(
        "Every target year must contain "
        "all 42 ICBs."
    )


duplicate_count = (
    panel
    .duplicated(
        subset=[
            "icb_code",
            "target_year",
        ]
    )
    .sum()
)


if duplicate_count != 0:
    raise RuntimeError(
        "Duplicate ICB-target-year "
        "rows detected."
    )


if panel.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in final forecasting panel."
    )


if (
    panel[
        "ckd_risk_rate_per_1000"
    ] < 0
).any():

    raise RuntimeError(
        "Negative CKD risk-rate "
        "values detected."
    )


print(
    "\nForecasting panel validation: "
    "PASSED"
)


# --------------------------------------------------
# 13. Save final forecasting panel
# --------------------------------------------------

panel.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved to: {OUTPUT_FILE}"
)