from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ckd_forecasting_master.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "validation_split_manifest.csv"
)


# --------------------------------------------------
# 2. Load forecasting panel
# --------------------------------------------------

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting panel not found: "
        f"{PANEL_FILE}"
    )


panel = pd.read_csv(PANEL_FILE)

print(
    f"Forecasting panel shape: "
    f"{panel.shape}"
)


# --------------------------------------------------
# 3. Required columns
# --------------------------------------------------

REQUIRED_COLUMNS = {
    "audit_year",
    "target_year",
    "icb_code",
    "icb_name",
    "ckd_cases",
    "diabetes_population",
    "ckd_risk_rate_per_1000",
}


missing_columns = (
    REQUIRED_COLUMNS
    - set(panel.columns)
)


if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# --------------------------------------------------
# 4. Expected forecasting alignment
# --------------------------------------------------

EXPECTED_ALIGNMENT = {
    "2018_19": 2020,
    "2019_20": 2021,
    "2020_21": 2022,
    "2021_22": 2023,
}


print(
    "\nExpected forecasting alignment:"
)

for audit_year, target_year in EXPECTED_ALIGNMENT.items():
    print(
        f"{audit_year} -> {target_year}"
    )


# --------------------------------------------------
# 5. Validate temporal alignment
# --------------------------------------------------

observed_alignment = (
    panel[
        [
            "audit_year",
            "target_year",
        ]
    ]
    .drop_duplicates()
    .sort_values("target_year")
)


print(
    "\nObserved forecasting alignment:"
)

print(
    observed_alignment.to_string(
        index=False
    )
)


expected_pairs = {
    (audit_year, target_year)
    for audit_year, target_year
    in EXPECTED_ALIGNMENT.items()
}


observed_pairs = set(
    map(
        tuple,
        observed_alignment[
            [
                "audit_year",
                "target_year",
            ]
        ].to_numpy()
    )
)


if observed_pairs != expected_pairs:
    raise RuntimeError(
        "Forecasting-year alignment "
        "does not match intended design."
    )


print(
    "\nTemporal alignment check: PASSED"
)


# --------------------------------------------------
# 6. Define final validation protocol
#
# 2020 + 2021:
#     development training
#
# 2022:
#     model validation / selection
#
# 2023:
#     untouched final temporal test
# --------------------------------------------------

def assign_split(target_year):

    if target_year in {
        2020,
        2021,
    }:
        return "development_train"

    if target_year == 2022:
        return "validation"

    if target_year == 2023:
        return "final_test"

    return None


panel["split"] = (
    panel["target_year"]
    .apply(assign_split)
)


if panel["split"].isna().any():

    unexpected_years = sorted(
        panel.loc[
            panel["split"].isna(),
            "target_year"
        ].unique()
    )

    raise RuntimeError(
        "Unexpected target years found: "
        f"{unexpected_years}"
    )


# --------------------------------------------------
# 7. Split summary
# --------------------------------------------------

split_summary = (
    panel
    .groupby("split")
    .agg(
        rows=(
            "icb_code",
            "size"
        ),

        unique_icbs=(
            "icb_code",
            "nunique"
        ),

        target_years=(
            "target_year",
            "nunique"
        ),

        min_target_year=(
            "target_year",
            "min"
        ),

        max_target_year=(
            "target_year",
            "max"
        ),
    )
)


split_order = [
    "development_train",
    "validation",
    "final_test",
]


split_summary = (
    split_summary
    .reindex(split_order)
)


print(
    "\n========================================"
)

print(
    "VALIDATION SPLIT SUMMARY"
)

print(
    "========================================"
)


print(
    split_summary.to_string()
)


# --------------------------------------------------
# 8. Expected split sizes
# --------------------------------------------------

EXPECTED_SPLIT_ROWS = {
    "development_train": 84,
    "validation": 42,
    "final_test": 42,
}


EXPECTED_SPLIT_YEARS = {
    "development_train": 2,
    "validation": 1,
    "final_test": 1,
}


for split_name in split_order:

    actual_rows = (
        split_summary.loc[
            split_name,
            "rows"
        ]
    )

    expected_rows = (
        EXPECTED_SPLIT_ROWS[
            split_name
        ]
    )


    if actual_rows != expected_rows:
        raise RuntimeError(
            f"{split_name}: expected "
            f"{expected_rows} rows, "
            f"found {actual_rows}."
        )


    actual_year_count = (
        split_summary.loc[
            split_name,
            "target_years"
        ]
    )

    expected_year_count = (
        EXPECTED_SPLIT_YEARS[
            split_name
        ]
    )


    if (
        actual_year_count
        != expected_year_count
    ):
        raise RuntimeError(
            f"{split_name}: expected "
            f"{expected_year_count} "
            f"target year(s)."
        )


# --------------------------------------------------
# 9. Validate number of ICBs
# --------------------------------------------------

for split_name in split_order:

    unique_icbs = (
        panel.loc[
            panel["split"]
            == split_name,
            "icb_code"
        ]
        .nunique()
    )

    if unique_icbs != 42:
        raise RuntimeError(
            f"{split_name}: expected "
            f"42 unique ICBs."
        )


print(
    "\nICB count within all splits: PASSED"
)


# --------------------------------------------------
# 10. Check each individual target year
# --------------------------------------------------

rows_per_year = (
    panel
    .groupby("target_year")
    .size()
)


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
    "\nRows per target year:"
)

print(
    rows_per_year.to_string()
)


print(
    "\nUnique ICBs per target year:"
)

print(
    icbs_per_year.to_string()
)


if not (
    rows_per_year == 42
).all():
    raise RuntimeError(
        "Every target year must contain "
        "42 observations."
    )


if not (
    icbs_per_year == 42
).all():
    raise RuntimeError(
        "Every target year must contain "
        "all 42 ICBs."
    )


# --------------------------------------------------
# 11. Same ICB membership across all four years
# --------------------------------------------------

year_icb_sets = {
    year: set(
        panel.loc[
            panel["target_year"]
            == year,
            "icb_code"
        ]
    )
    for year
    in sorted(
        panel["target_year"]
        .unique()
    )
}


reference_set = (
    year_icb_sets[
        min(year_icb_sets)
    ]
)


for year, codes in year_icb_sets.items():

    if codes != reference_set:
        raise RuntimeError(
            f"ICB membership differs "
            f"in target year {year}."
        )


print(
    "\nSame 42 ICBs across "
    "all four target years: PASSED"
)


# --------------------------------------------------
# 12. Duplicate check
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


print(
    f"\nDuplicate ICB-target-year rows: "
    f"{duplicate_count}"
)


if duplicate_count != 0:
    raise RuntimeError(
        "Duplicate ICB-target-year "
        "rows detected."
    )


# --------------------------------------------------
# 13. Check chronology
# --------------------------------------------------

development_years = sorted(
    panel.loc[
        panel["split"]
        == "development_train",
        "target_year"
    ]
    .unique()
)


validation_years = sorted(
    panel.loc[
        panel["split"]
        == "validation",
        "target_year"
    ]
    .unique()
)


test_years = sorted(
    panel.loc[
        panel["split"]
        == "final_test",
        "target_year"
    ]
    .unique()
)


print(
    "\nChronological design:"
)

print(
    f"Development train : "
    f"{development_years}"
)

print(
    f"Validation        : "
    f"{validation_years}"
)

print(
    f"Final test        : "
    f"{test_years}"
)


if max(
    development_years
) >= min(
    validation_years
):

    raise RuntimeError(
        "Development data overlaps or "
        "occurs after validation."
    )


if max(
    validation_years
) >= min(
    test_years
):

    raise RuntimeError(
        "Validation data overlaps or "
        "occurs after final test."
    )


print(
    "\nChronology check: PASSED"
)


# --------------------------------------------------
# 14. Define fields NEVER used as model predictors
# --------------------------------------------------

NON_FEATURE_COLUMNS = {
    "audit_year",
    "predictor_period",
    "target_year",
    "icb_code",
    "icb_name",
    "split",
    "ckd_cases",
    "diabetes_population",
    "ckd_risk_rate_per_1000",
    "target_ckd_rate_per_1000",
    "previous_year_ckd_rate",
}


candidate_features = [
    column
    for column in panel.columns
    if column
    not in NON_FEATURE_COLUMNS
]


print(
    f"\nCandidate predictors available: "
    f"{len(candidate_features)}"
)


if len(
    candidate_features
) != 19:

    raise RuntimeError(
        f"Expected 19 candidate predictors, "
        f"found {len(candidate_features)}."
    )


print(
    "\nIdentifiers / outcomes excluded "
    "from model predictors:"
)

for column in sorted(
    NON_FEATURE_COLUMNS
):
    print(
        f"- {column}"
    )


# --------------------------------------------------
# 15. Final-refit definition
#
# After model and hyperparameters are locked
# using validation year 2022:
#
# Final fitting data =
# target years 2020 + 2021 + 2022
#
# Final untouched test =
# target year 2023
# --------------------------------------------------

final_refit_mask = (
    panel[
        "target_year"
    ]
    .isin(
        [
            2020,
            2021,
            2022,
        ]
    )
)


final_refit_rows = (
    final_refit_mask.sum()
)


final_test_rows = (
    (
        panel[
            "target_year"
        ]
        == 2023
    )
    .sum()
)


print(
    "\nFinal model refit plan:"
)

print(
    f"Refit rows "
    f"(2020 + 2021 + 2022): "
    f"{final_refit_rows}"
)

print(
    f"Final test rows "
    f"(2023): "
    f"{final_test_rows}"
)


if final_refit_rows != 126:
    raise RuntimeError(
        "Expected 126 rows for "
        "final model refit."
    )


if final_test_rows != 42:
    raise RuntimeError(
        "Expected 42 rows for "
        "final temporal test."
    )


# --------------------------------------------------
# 16. Create split manifest
#
# No target values are stored here.
# --------------------------------------------------

manifest = panel[
    [
        "audit_year",
        "target_year",
        "icb_code",
        "icb_name",
        "split",
    ]
].copy()


manifest = (
    manifest
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
# 17. Final manifest checks
# --------------------------------------------------

if len(manifest) != 168:
    raise RuntimeError(
        f"Expected 168 manifest rows, "
        f"found {len(manifest)}."
    )


if manifest.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in split manifest."
    )


actual_split_counts = (
    manifest[
        "split"
    ]
    .value_counts()
    .to_dict()
)


if (
    actual_split_counts
    != EXPECTED_SPLIT_ROWS
):
    raise RuntimeError(
        "Unexpected split counts: "
        f"{actual_split_counts}"
    )


# --------------------------------------------------
# 18. Print final validation protocol
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL VALIDATION PROTOCOL"
)

print(
    "========================================"
)


print(
    """
Stage 1 — Development
Use CKD target years 2020 and 2021
(84 ICB-year observations) for initial
model fitting and development.

Stage 2 — Validation
Evaluate candidate models on CKD target
year 2022 (42 ICB observations).

The 2022 validation year may be used for:
- model comparison
- feature-set decisions
- hyperparameter selection

Stage 3 — Lock model specification
After validation decisions are complete,
freeze the chosen features, algorithm and
hyperparameters.

Stage 4 — Final refit
Refit the locked model using CKD target
years 2020, 2021 and 2022
(126 ICB-year observations).

Stage 5 — Final temporal test
Evaluate once on CKD target year 2023
(42 ICB observations).

The 2023 outcome must not be used for
feature selection, model selection,
hyperparameter tuning or model redesign.
""".strip()
)


print(
    "\nValidation split audit: PASSED"
)


# --------------------------------------------------
# 19. Save split manifest
# --------------------------------------------------

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved to: {OUTPUT_FILE}"
)