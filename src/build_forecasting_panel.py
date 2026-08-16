from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

OUTPUT_FILE = (
    INTERIM_DIR
    / "forecasting_panel_2018_19_to_2023.csv"
)


# --------------------------------------------------
# 2. Historical predictor files
# --------------------------------------------------

PREDICTOR_FILES = {
    "2018_19":
        INTERIM_DIR
        / "nda_type2_icb_2018_19.csv",

    "2019_20":
        INTERIM_DIR
        / "nda_type2_icb_2019_20.csv",

    "2020_21":
        INTERIM_DIR
        / "nda_type2_icb_2020_21.csv",

    "2021_22":
        INTERIM_DIR
        / "nda_type2_icb_2021_22.csv",
}


CKD_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)


# --------------------------------------------------
# 3. Forecasting alignment
#
# Historical NDA predictor period
#          ↓
# Later CKD calendar-year outcome
# --------------------------------------------------

TARGET_YEAR_MAP = {
    "2018_19": 2020,
    "2019_20": 2021,
    "2020_21": 2022,
    "2021_22": 2023,
}


# --------------------------------------------------
# 4. Common four-year feature set
#
# These 19 predictors are consistently available
# across ALL FOUR historical NDA periods.
#
# Retinal screening and all-nine-care-processes
# are excluded because they are unavailable
# in 2018-19.
# --------------------------------------------------

COMMON_FEATURES = [
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

    "primary_prevention_on_statins_without_cvd_history",
    "secondary_prevention_on_statins_with_cvd_history",
    "combined_prevention_on_statins",

    "all_three_treatment_targets",
]


# --------------------------------------------------
# 5. Validate input files
# --------------------------------------------------

for audit_year, file_path in PREDICTOR_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing predictor file for "
            f"{audit_year}: {file_path}"
        )


if not CKD_FILE.exists():
    raise FileNotFoundError(
        f"Missing CKD outcome file: "
        f"{CKD_FILE}"
    )


# --------------------------------------------------
# 6. Load CKD outcome panel
# --------------------------------------------------

ckd = pd.read_csv(
    CKD_FILE
)


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


# Standardise ICB codes.
ckd["icb_code"] = (
    ckd["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 7. Validate CKD outcome uniqueness
# --------------------------------------------------

ckd_duplicate_count = (
    ckd
    .duplicated(
        subset=[
            "icb_code",
            "year",
        ]
    )
    .sum()
)


if ckd_duplicate_count != 0:
    raise RuntimeError(
        "Duplicate CKD ICB-year "
        "observations detected."
    )


# --------------------------------------------------
# 8. Build each forecasting block
# --------------------------------------------------

forecasting_frames = []


for audit_year, predictor_file in PREDICTOR_FILES.items():

    print(
        "\n----------------------------------------"
    )

    print(
        f"Processing predictor year: "
        f"{audit_year}"
    )

    print(
        "----------------------------------------"
    )


    predictors = pd.read_csv(
        predictor_file
    )


    # ----------------------------------------------
    # Standardise ICB codes
    # ----------------------------------------------

    predictors["icb_code"] = (
        predictors["icb_code"]
        .astype("string")
        .str.strip()
    )


    # ----------------------------------------------
    # Confirm correct audit year
    # ----------------------------------------------

    observed_audit_years = set(
        predictors[
            "audit_year"
        ]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
    )


    if observed_audit_years != {
        audit_year
    }:
        raise RuntimeError(
            f"{audit_year}: unexpected "
            f"audit-year values: "
            f"{observed_audit_years}"
        )


    # ----------------------------------------------
    # Identify later target year
    # ----------------------------------------------

    target_year = (
        TARGET_YEAR_MAP[
            audit_year
        ]
    )


    print(
        f"Target CKD year: "
        f"{target_year}"
    )


    # ----------------------------------------------
    # Validate required predictors
    # ----------------------------------------------

    required_predictor_columns = (
        {
            "audit_year",
            "icb_code",
        }
        |
        set(
            COMMON_FEATURES
        )
    )


    missing_predictor_columns = (
        required_predictor_columns
        - set(
            predictors.columns
        )
    )


    if missing_predictor_columns:
        raise RuntimeError(
            f"{audit_year} is missing "
            f"required predictors: "
            f"{sorted(missing_predictor_columns)}"
        )


    # ----------------------------------------------
    # Validate predictor geography
    # ----------------------------------------------

    if len(predictors) != 42:
        raise RuntimeError(
            f"{audit_year}: expected "
            f"42 predictor rows, "
            f"found {len(predictors)}."
        )


    if (
        predictors[
            "icb_code"
        ]
        .nunique()
        != 42
    ):
        raise RuntimeError(
            f"{audit_year}: expected "
            f"42 unique ICB codes."
        )


    if (
        predictors[
            "icb_code"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{audit_year}: duplicate "
            f"ICB predictor rows detected."
        )


    # ----------------------------------------------
    # Keep only common four-year feature core
    # ----------------------------------------------

    predictors = predictors[
        [
            "audit_year",
            "icb_code",
        ]
        +
        COMMON_FEATURES
    ].copy()


    # ----------------------------------------------
    # Check missing predictor values
    # ----------------------------------------------

    if (
        predictors[
            COMMON_FEATURES
        ]
        .isna()
        .any()
        .any()
    ):

        missing = (
            predictors[
                COMMON_FEATURES
            ]
            .isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        raise RuntimeError(
            f"{audit_year}: missing "
            f"predictor values detected:\n"
            f"{missing}"
        )


    # ----------------------------------------------
    # Select future CKD outcome
    # ----------------------------------------------

    outcome = (
        ckd[
            ckd["year"]
            == target_year
        ][
            [
                "icb_code",
                "icb_name",
                "year",
                "ckd_cases",
                "diabetes_population",
                "ckd_risk_rate_per_1000",
            ]
        ]
        .copy()
    )


    print(
        f"Predictor ICBs: "
        f"{predictors['icb_code'].nunique()}"
    )

    print(
        f"Outcome ICBs: "
        f"{outcome['icb_code'].nunique()}"
    )


    if len(outcome) != 42:
        raise RuntimeError(
            f"CKD {target_year}: expected "
            f"42 outcome rows, "
            f"found {len(outcome)}."
        )


    if (
        outcome[
            "icb_code"
        ]
        .nunique()
        != 42
    ):
        raise RuntimeError(
            f"CKD {target_year}: "
            f"expected 42 unique ICBs."
        )


    # ----------------------------------------------
    # Exact geography comparison
    # ----------------------------------------------

    predictor_codes = set(
        predictors[
            "icb_code"
        ]
    )

    outcome_codes = set(
        outcome[
            "icb_code"
        ]
    )


    only_predictors = sorted(
        predictor_codes
        - outcome_codes
    )

    only_outcome = sorted(
        outcome_codes
        - predictor_codes
    )


    if (
        only_predictors
        or
        only_outcome
    ):
        raise RuntimeError(
            f"ICB mismatch for "
            f"{audit_year} -> "
            f"{target_year}.\n"
            f"Only predictors: "
            f"{only_predictors}\n"
            f"Only outcomes: "
            f"{only_outcome}"
        )


    print(
        "ICB geography match: PASSED"
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
            "year":
                "target_year"
        }
    )


    # ----------------------------------------------
    # Validate forecasting block
    # ----------------------------------------------

    if len(merged) != 42:
        raise RuntimeError(
            f"{audit_year} -> "
            f"{target_year}: expected "
            f"42 merged rows, "
            f"found {len(merged)}."
        )


    if (
        merged["target_year"]
        .nunique()
        != 1
    ):
        raise RuntimeError(
            f"{audit_year}: multiple "
            f"target years detected."
        )


    if (
        merged["target_year"]
        .iloc[0]
        != target_year
    ):
        raise RuntimeError(
            f"{audit_year}: incorrect "
            f"target-year assignment."
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
            f"Missing values after "
            f"{audit_year} -> "
            f"{target_year} merge:\n"
            f"{missing}"
        )


    forecasting_frames.append(
        merged
    )


# --------------------------------------------------
# 9. Combine all four forecasting blocks
# --------------------------------------------------

panel = pd.concat(
    forecasting_frames,
    ignore_index=True
)


# --------------------------------------------------
# 10. Sort final panel
# --------------------------------------------------

panel = (
    panel
    .sort_values(
        [
            "target_year",
            "icb_code",
        ]
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# 11. Reorder final columns
# --------------------------------------------------

panel = panel[
    [
        "audit_year",
        "target_year",
        "icb_code",
        "icb_name",
    ]
    +
    COMMON_FEATURES
    +
    [
        "ckd_cases",
        "diabetes_population",
        "ckd_risk_rate_per_1000",
    ]
]


# --------------------------------------------------
# 12. Final forecasting-panel audit
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL FOUR-YEAR FORECASTING PANEL"
)

print(
    "========================================"
)


print(
    f"\nFinal forecasting panel shape: "
    f"{panel.shape}"
)

print(
    f"Common predictors: "
    f"{len(COMMON_FEATURES)}"
)

print(
    f"Unique ICBs: "
    f"{panel['icb_code'].nunique()}"
)

print(
    f"Target years: "
    f"{sorted(panel['target_year'].unique())}"
)


# --------------------------------------------------
# 13. Rows per target year
# --------------------------------------------------

rows_per_year = (
    panel
    .groupby(
        "target_year"
    )
    .size()
)


print(
    "\nRows per target year:"
)

print(
    rows_per_year.to_string()
)


# --------------------------------------------------
# 14. Unique ICBs per target year
# --------------------------------------------------

icbs_per_year = (
    panel
    .groupby(
        "target_year"
    )[
        "icb_code"
    ]
    .nunique()
)


print(
    "\nUnique ICBs per target year:"
)

print(
    icbs_per_year.to_string()
)


# --------------------------------------------------
# 15. Missing-value audit
# --------------------------------------------------

print(
    "\nMissing values:"
)

print(
    panel
    .isna()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 16. CKD outcome summary by target year
# --------------------------------------------------

target_summary = (
    panel
    .groupby(
        "target_year"
    )[
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
    "\nCKD target summary by year:"
)

print(
    target_summary.to_string()
)


# --------------------------------------------------
# 17. Validate exact four-year panel structure
# --------------------------------------------------

if len(panel) != 168:
    raise RuntimeError(
        f"Expected 168 forecasting rows, "
        f"found {len(panel)}."
    )


if len(panel.columns) != 26:
    raise RuntimeError(
        f"Expected 26 columns, "
        f"found {len(panel.columns)}."
    )


if (
    panel[
        "icb_code"
    ]
    .nunique()
    != 42
):
    raise RuntimeError(
        "Expected 42 unique ICBs."
    )


expected_target_years = {
    2020,
    2021,
    2022,
    2023,
}


observed_target_years = set(
    panel[
        "target_year"
    ]
    .unique()
)


if (
    observed_target_years
    != expected_target_years
):
    raise RuntimeError(
        "Unexpected target-year set: "
        f"{sorted(observed_target_years)}"
    )


if not (
    rows_per_year == 42
).all():
    raise RuntimeError(
        "Every target year must contain "
        "exactly 42 rows."
    )


if not (
    icbs_per_year == 42
).all():
    raise RuntimeError(
        "Every target year must contain "
        "all 42 ICBs."
    )


# --------------------------------------------------
# 18. Check same ICB membership every year
# --------------------------------------------------

year_icb_sets = [
    set(
        panel.loc[
            panel[
                "target_year"
            ]
            == year,
            "icb_code"
        ]
    )
    for year
    in sorted(
        expected_target_years
    )
]


if not all(
    current_set
    == year_icb_sets[0]
    for current_set
    in year_icb_sets[1:]
):
    raise RuntimeError(
        "ICB membership differs "
        "between target years."
    )


# --------------------------------------------------
# 19. Duplicate check
# --------------------------------------------------

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


# --------------------------------------------------
# 20. Missing-value check
# --------------------------------------------------

if panel.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in final forecasting panel."
    )


# --------------------------------------------------
# 21. Target validity checks
# --------------------------------------------------

if (
    panel[
        "ckd_risk_rate_per_1000"
    ]
    < 0
).any():
    raise RuntimeError(
        "Negative CKD risk-rate "
        "values detected."
    )


if (
    panel[
        "ckd_cases"
    ]
    < 0
).any():
    raise RuntimeError(
        "Negative CKD case counts detected."
    )


if (
    panel[
        "diabetes_population"
    ]
    <= 0
).any():
    raise RuntimeError(
        "Invalid diabetes-population "
        "values detected."
    )


# --------------------------------------------------
# 22. Explicit temporal alignment check
# --------------------------------------------------

observed_alignment = (
    panel[
        [
            "audit_year",
            "target_year",
        ]
    ]
    .drop_duplicates()
)


expected_alignment = {
    (
        audit_year,
        target_year
    )
    for audit_year, target_year
    in TARGET_YEAR_MAP.items()
}


observed_alignment_set = set(
    map(
        tuple,
        observed_alignment[
            [
                "audit_year",
                "target_year",
            ]
        ]
        .to_numpy()
    )
)


if (
    observed_alignment_set
    != expected_alignment
):
    raise RuntimeError(
        "Final temporal alignment "
        "does not match intended design."
    )


print(
    "\nForecasting panel validation: "
    "PASSED"
)


# --------------------------------------------------
# 23. Save final four-year forecasting panel
# --------------------------------------------------

panel.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved to: {OUTPUT_FILE}"
)