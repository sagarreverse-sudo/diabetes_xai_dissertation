from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "forecasting_panel_2018_19_to_2023.csv"
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

AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "temporal_predictor_audit.csv"
)

YEAR_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "temporal_predictor_year_summary.csv"
)


# --------------------------------------------------
# 2. Final four-year design
# --------------------------------------------------

AUDIT_YEAR_ORDER = [
    "2018_19",
    "2019_20",
    "2020_21",
    "2021_22",
]


ADJACENT_YEAR_PAIRS = [
    ("2018_19", "2019_20"),
    ("2019_20", "2020_21"),
    ("2020_21", "2021_22"),
]


# --------------------------------------------------
# 3. Final 19-feature common predictor core
# --------------------------------------------------

FEATURES = [
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
# 4. Load final forecasting panel
# --------------------------------------------------

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting panel not found: {PANEL_FILE}"
    )


panel = pd.read_csv(PANEL_FILE)


print(
    f"Forecasting panel shape: {panel.shape}"
)


# --------------------------------------------------
# 5. Validate required columns
# --------------------------------------------------

required_columns = {
    "audit_year",
    "target_year",
    "icb_code",
    "icb_name",
} | set(FEATURES)


missing_columns = (
    required_columns
    - set(panel.columns)
)


if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if len(panel) != 168:
    raise RuntimeError(
        f"Expected 168 rows, found {len(panel)}."
    )


if len(FEATURES) != 19:
    raise RuntimeError(
        f"Expected 19 features, found {len(FEATURES)}."
    )


# --------------------------------------------------
# 6. Standardise key fields
# --------------------------------------------------

panel["audit_year"] = (
    panel["audit_year"]
    .astype("string")
    .str.strip()
)


panel["icb_code"] = (
    panel["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 7. Confirm four predictor periods
# --------------------------------------------------

observed_years = set(
    panel["audit_year"].unique()
)


expected_years = set(
    AUDIT_YEAR_ORDER
)


if observed_years != expected_years:
    raise RuntimeError(
        "Unexpected audit-year set: "
        f"{sorted(observed_years)}"
    )


print(
    "\nPredictor periods:"
)

for year in AUDIT_YEAR_ORDER:
    print(
        f"- {year}"
    )


# --------------------------------------------------
# 8. Check 42 ICBs per predictor year
# --------------------------------------------------

rows_per_year = (
    panel
    .groupby("audit_year")
    .size()
    .reindex(AUDIT_YEAR_ORDER)
)


icbs_per_year = (
    panel
    .groupby("audit_year")["icb_code"]
    .nunique()
    .reindex(AUDIT_YEAR_ORDER)
)


print(
    "\nRows per predictor year:"
)

print(
    rows_per_year.to_string()
)


print(
    "\nUnique ICBs per predictor year:"
)

print(
    icbs_per_year.to_string()
)


if not (
    rows_per_year == 42
).all():
    raise RuntimeError(
        "Every predictor year must contain "
        "42 rows."
    )


if not (
    icbs_per_year == 42
).all():
    raise RuntimeError(
        "Every predictor year must contain "
        "42 unique ICBs."
    )


# --------------------------------------------------
# 9. Confirm identical ICB membership
# --------------------------------------------------

icb_sets = {
    year: set(
        panel.loc[
            panel["audit_year"] == year,
            "icb_code"
        ]
    )
    for year in AUDIT_YEAR_ORDER
}


reference_set = (
    icb_sets[
        AUDIT_YEAR_ORDER[0]
    ]
)


for year in AUDIT_YEAR_ORDER[1:]:

    if icb_sets[year] != reference_set:
        raise RuntimeError(
            f"ICB membership differs in {year}."
        )


print(
    "\nSame 42 ICBs across all four "
    "predictor periods: PASSED"
)


# --------------------------------------------------
# 10. Missing-value audit
# --------------------------------------------------

feature_missing = (
    panel[FEATURES]
    .isna()
    .sum()
)


if (
    feature_missing > 0
).any():

    raise RuntimeError(
        "Missing feature values detected:\n"
        f"{feature_missing[feature_missing > 0]}"
    )


print(
    "Predictor missing-value check: PASSED"
)


# --------------------------------------------------
# 11. Create descriptive summary by year
# --------------------------------------------------

summary_rows = []


for feature in FEATURES:

    for year in AUDIT_YEAR_ORDER:

        values = panel.loc[
            panel["audit_year"] == year,
            feature
        ]


        summary_rows.append({
            "feature": feature,
            "audit_year": year,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
        })


year_summary = pd.DataFrame(
    summary_rows
)


year_summary[
    [
        "mean",
        "std",
        "min",
        "median",
        "max",
    ]
] = (
    year_summary[
        [
            "mean",
            "std",
            "min",
            "median",
            "max",
        ]
    ]
    .round(3)
)


# --------------------------------------------------
# 12. Helper for aligned ICB comparison
# --------------------------------------------------

def get_aligned_values(
    feature,
    year_a,
    year_b
):

    a = (
        panel.loc[
            panel["audit_year"] == year_a,
            [
                "icb_code",
                feature,
            ]
        ]
        .rename(
            columns={
                feature: "value_a"
            }
        )
    )


    b = (
        panel.loc[
            panel["audit_year"] == year_b,
            [
                "icb_code",
                feature,
            ]
        ]
        .rename(
            columns={
                feature: "value_b"
            }
        )
    )


    merged = a.merge(
        b,
        on="icb_code",
        how="inner",
        validate="one_to_one"
    )


    if len(merged) != 42:
        raise RuntimeError(
            f"{feature}: expected 42 aligned "
            f"ICBs for {year_a} vs {year_b}, "
            f"found {len(merged)}."
        )


    return merged


# --------------------------------------------------
# 13. Temporal stability audit
#
# For every feature we calculate:
#
# - mean in each year
# - adjacent-year absolute mean shift
# - adjacent-year Spearman correlation
# - largest absolute mean shift
# - weakest adjacent Spearman correlation
#
# Spearman measures whether ICB geographic
# ranking is preserved even when national
# levels change.
# --------------------------------------------------

audit_rows = []


for feature in FEATURES:

    year_means = {}


    for year in AUDIT_YEAR_ORDER:

        year_means[year] = (
            panel.loc[
                panel["audit_year"] == year,
                feature
            ]
            .mean()
        )


    row = {
        "feature": feature,
    }


    # Store annual means.
    for year in AUDIT_YEAR_ORDER:

        row[
            f"mean_{year}"
        ] = year_means[year]


    adjacent_shifts = []
    adjacent_rhos = []


    for year_a, year_b in ADJACENT_YEAR_PAIRS:

        aligned = get_aligned_values(
            feature,
            year_a,
            year_b
        )


        mean_shift = (
            year_means[year_b]
            - year_means[year_a]
        )


        abs_mean_shift = abs(
            mean_shift
        )


        rho, p_value = spearmanr(
            aligned["value_a"],
            aligned["value_b"]
        )


        pair_name = (
            f"{year_a}_to_{year_b}"
        )


        row[
            f"mean_shift_{pair_name}"
        ] = mean_shift


        row[
            f"abs_mean_shift_{pair_name}"
        ] = abs_mean_shift


        row[
            f"spearman_{pair_name}"
        ] = rho


        row[
            f"spearman_p_{pair_name}"
        ] = p_value


        adjacent_shifts.append(
            abs_mean_shift
        )


        adjacent_rhos.append(
            rho
        )


    # ----------------------------------------------
    # First-to-last change
    # ----------------------------------------------

    first_year = (
        AUDIT_YEAR_ORDER[0]
    )

    last_year = (
        AUDIT_YEAR_ORDER[-1]
    )


    first_last = get_aligned_values(
        feature,
        first_year,
        last_year
    )


    first_last_rho, first_last_p = (
        spearmanr(
            first_last["value_a"],
            first_last["value_b"]
        )
    )


    row[
        "mean_shift_2018_19_to_2021_22"
    ] = (
        year_means["2021_22"]
        - year_means["2018_19"]
    )


    row[
        "abs_mean_shift_2018_19_to_2021_22"
    ] = abs(
        row[
            "mean_shift_2018_19_to_2021_22"
        ]
    )


    row[
        "spearman_2018_19_to_2021_22"
    ] = first_last_rho


    row[
        "spearman_p_2018_19_to_2021_22"
    ] = first_last_p


    # ----------------------------------------------
    # Overall stability indicators
    # ----------------------------------------------

    row[
        "max_adjacent_abs_mean_shift"
    ] = max(
        adjacent_shifts
    )


    row[
        "min_adjacent_spearman"
    ] = min(
        adjacent_rhos
    )


    # Large national-level movement.
    row[
        "large_mean_shift_flag"
    ] = (
        row[
            "max_adjacent_abs_mean_shift"
        ]
        >= 10
    )


    # Weak geographic-rank stability.
    row[
        "weak_rank_stability_flag"
    ] = (
        row[
            "min_adjacent_spearman"
        ]
        < 0.50
    )


    audit_rows.append(
        row
    )


audit = pd.DataFrame(
    audit_rows
)


# --------------------------------------------------
# 14. Round numerical output
# --------------------------------------------------

numeric_columns = (
    audit
    .select_dtypes(
        include="number"
    )
    .columns
)


audit[numeric_columns] = (
    audit[
        numeric_columns
    ]
    .round(3)
)


# --------------------------------------------------
# 15. Sort most unstable variables first
# --------------------------------------------------

audit = (
    audit
    .sort_values(
        by=[
            "large_mean_shift_flag",
            "weak_rank_stability_flag",
            "max_adjacent_abs_mean_shift",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 16. Print concise four-year audit
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FOUR-YEAR TEMPORAL PREDICTOR AUDIT"
)

print(
    "========================================"
)


display_columns = [
    "feature",
    "mean_2018_19",
    "mean_2019_20",
    "mean_2020_21",
    "mean_2021_22",
    "max_adjacent_abs_mean_shift",
    "min_adjacent_spearman",
    "large_mean_shift_flag",
    "weak_rank_stability_flag",
]


print(
    audit[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 17. Large mean-shift features
# --------------------------------------------------

large_shift = audit[
    audit[
        "large_mean_shift_flag"
    ]
]


print(
    "\nFeatures with >=10 percentage-point "
    "adjacent-year mean shift:"
)


if len(large_shift) == 0:

    print(
        "None"
    )

else:

    print(
        large_shift[
            [
                "feature",
                "max_adjacent_abs_mean_shift",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 18. Weak rank-stability features
# --------------------------------------------------

weak_rank = audit[
    audit[
        "weak_rank_stability_flag"
    ]
]


print(
    "\nFeatures with minimum adjacent-year "
    "Spearman < 0.50:"
)


if len(weak_rank) == 0:

    print(
        "None"
    )

else:

    print(
        weak_rank[
            [
                "feature",
                "min_adjacent_spearman",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 19. Important interpretation note
# --------------------------------------------------

print(
    """
Interpretation:
- A large mean shift indicates that the
  national level of a measure changed
  substantially between adjacent periods.

- A high Spearman correlation means the
  relative ordering of ICBs remained
  similar despite that overall shift.

- Therefore, a feature should NOT be
  automatically removed solely because
  its national mean changed.

- Variables with both a large mean shift
  and weak rank stability deserve the
  strongest caution during feature
  selection and robustness analysis.
""".strip()
)


# --------------------------------------------------
# 20. Final validation
# --------------------------------------------------

if len(audit) != 19:
    raise RuntimeError(
        f"Expected 19 audited features, "
        f"found {len(audit)}."
    )


if audit["feature"].nunique() != 19:
    raise RuntimeError(
        "Duplicate feature rows detected."
    )


if audit.isna().any().any():
    raise RuntimeError(
        "Missing values detected in "
        "temporal audit output."
    )


print(
    "\nTemporal predictor audit: PASSED"
)


# --------------------------------------------------
# 21. Save outputs
# --------------------------------------------------

audit.to_csv(
    AUDIT_OUTPUT,
    index=False
)


year_summary.to_csv(
    YEAR_SUMMARY_OUTPUT,
    index=False
)


print(
    f"Saved audit to: {AUDIT_OUTPUT}"
)

print(
    f"Saved yearly summary to: "
    f"{YEAR_SUMMARY_OUTPUT}"
)