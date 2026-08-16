from pathlib import Path
from itertools import combinations

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

PAIRS_OUTPUT = (
    OUTPUT_DIR
    / "feature_redundancy_pairs.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "feature_redundancy_summary.csv"
)

MATRIX_OUTPUT = (
    OUTPUT_DIR
    / "feature_spearman_matrix.csv"
)

BY_YEAR_OUTPUT = (
    OUTPUT_DIR
    / "feature_spearman_by_year.csv"
)


# --------------------------------------------------
# 2. Final four-year predictor periods
# --------------------------------------------------

AUDIT_YEAR_ORDER = [
    "2018_19",
    "2019_20",
    "2020_21",
    "2021_22",
]


# --------------------------------------------------
# 3. Final 19 common predictors
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
# 4. Correlation thresholds
#
# These are audit flags, NOT automatic
# feature-removal rules.
# --------------------------------------------------

HIGH_POOLED_THRESHOLD = 0.90

VERY_HIGH_POOLED_THRESHOLD = 0.95

CONSISTENT_WITHIN_YEAR_THRESHOLD = 0.80

POOLED_ONLY_WEAK_THRESHOLD = 0.60


# --------------------------------------------------
# 5. Load final forecasting panel
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
# 6. Validate final panel
# --------------------------------------------------

required_columns = {
    "audit_year",
    "icb_code",
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
        f"Expected 19 predictors, found {len(FEATURES)}."
    )


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


observed_years = set(
    panel["audit_year"].unique()
)


if observed_years != set(AUDIT_YEAR_ORDER):
    raise RuntimeError(
        "Unexpected predictor periods: "
        f"{sorted(observed_years)}"
    )


if panel[FEATURES].isna().any().any():
    raise RuntimeError(
        "Missing predictor values detected."
    )


print(
    "Final predictor set validation: PASSED"
)


# --------------------------------------------------
# 7. Check 42 ICBs per year
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


if not (
    rows_per_year == 42
).all():
    raise RuntimeError(
        "Every year must contain 42 rows."
    )


if not (
    icbs_per_year == 42
).all():
    raise RuntimeError(
        "Every year must contain 42 ICBs."
    )


# --------------------------------------------------
# 8. Confirm same ICB membership
# --------------------------------------------------

year_sets = {
    year: set(
        panel.loc[
            panel["audit_year"] == year,
            "icb_code"
        ]
    )
    for year in AUDIT_YEAR_ORDER
}


reference_set = (
    year_sets[
        AUDIT_YEAR_ORDER[0]
    ]
)


for year in AUDIT_YEAR_ORDER[1:]:

    if year_sets[year] != reference_set:
        raise RuntimeError(
            f"ICB membership differs in {year}."
        )


print(
    "Same 42 ICBs across all four years: PASSED"
)


# --------------------------------------------------
# 9. Number of feature pairs
#
# 19 choose 2 = 171
# --------------------------------------------------

feature_pairs = list(
    combinations(
        FEATURES,
        2
    )
)


print(
    f"\nPredictors: {len(FEATURES)}"
)

print(
    f"Feature pairs: {len(feature_pairs)}"
)


if len(feature_pairs) != 171:
    raise RuntimeError(
        "Expected exactly 171 feature pairs."
    )


# --------------------------------------------------
# 10. Pooled Spearman matrix
#
# This uses all 168 ICB-year observations.
#
# Important:
# pooled correlation can sometimes be inflated
# by common year-to-year movements.
# Therefore we ALSO calculate correlations
# separately inside each year.
# --------------------------------------------------

pooled_matrix = (
    panel[FEATURES]
    .corr(
        method="spearman"
    )
)


if pooled_matrix.isna().any().any():
    raise RuntimeError(
        "Missing values found in pooled "
        "Spearman matrix."
    )


# --------------------------------------------------
# 11. Calculate pairwise pooled and
#     within-year correlations
# --------------------------------------------------

pair_rows = []

by_year_rows = []


for feature_a, feature_b in feature_pairs:

    # ----------------------------------------------
    # Pooled correlation
    # ----------------------------------------------

    pooled_rho, pooled_p = spearmanr(
        panel[feature_a],
        panel[feature_b]
    )


    yearly_rhos = {}


    # ----------------------------------------------
    # Correlation separately within each year
    # ----------------------------------------------

    for year in AUDIT_YEAR_ORDER:

        year_data = panel.loc[
            panel["audit_year"] == year,
            [
                feature_a,
                feature_b,
            ]
        ]


        if len(year_data) != 42:
            raise RuntimeError(
                f"{year}: expected 42 rows."
            )


        rho, p_value = spearmanr(
            year_data[feature_a],
            year_data[feature_b]
        )


        if pd.isna(rho):
            raise RuntimeError(
                f"Undefined Spearman correlation "
                f"for {feature_a} vs {feature_b} "
                f"in {year}."
            )


        yearly_rhos[year] = rho


        by_year_rows.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "audit_year": year,
            "spearman_rho": rho,
            "abs_spearman_rho": abs(rho),
            "p_value": p_value,
        })


    # ----------------------------------------------
    # Within-year stability statistics
    # ----------------------------------------------

    yearly_abs_rhos = [
        abs(
            yearly_rhos[year]
        )
        for year in AUDIT_YEAR_ORDER
    ]


    min_within_abs = min(
        yearly_abs_rhos
    )


    max_within_abs = max(
        yearly_abs_rhos
    )


    median_within_abs = (
        pd.Series(
            yearly_abs_rhos
        )
        .median()
    )


    mean_within_abs = (
        pd.Series(
            yearly_abs_rhos
        )
        .mean()
    )


    # ----------------------------------------------
    # Descriptive redundancy flags
    # ----------------------------------------------

    high_pooled_flag = (
        abs(pooled_rho)
        >= HIGH_POOLED_THRESHOLD
    )


    very_high_pooled_flag = (
        abs(pooled_rho)
        >= VERY_HIGH_POOLED_THRESHOLD
    )


    consistently_strong_flag = (
        high_pooled_flag
        and
        min_within_abs
        >= CONSISTENT_WITHIN_YEAR_THRESHOLD
    )


    # A high pooled correlation but at least one
    # weak within-year relationship suggests the
    # pooled relationship may partly reflect a
    # common time/year effect.
    pooled_year_effect_caution_flag = (
        high_pooled_flag
        and
        min_within_abs
        < POOLED_ONLY_WEAK_THRESHOLD
    )


    pair_rows.append({
        "feature_a": feature_a,
        "feature_b": feature_b,

        "pooled_spearman": pooled_rho,
        "abs_pooled_spearman": abs(pooled_rho),
        "pooled_p_value": pooled_p,

        "spearman_2018_19":
            yearly_rhos["2018_19"],

        "spearman_2019_20":
            yearly_rhos["2019_20"],

        "spearman_2020_21":
            yearly_rhos["2020_21"],

        "spearman_2021_22":
            yearly_rhos["2021_22"],

        "min_within_year_abs_spearman":
            min_within_abs,

        "median_within_year_abs_spearman":
            median_within_abs,

        "mean_within_year_abs_spearman":
            mean_within_abs,

        "max_within_year_abs_spearman":
            max_within_abs,

        "high_pooled_correlation_flag":
            high_pooled_flag,

        "very_high_pooled_correlation_flag":
            very_high_pooled_flag,

        "consistently_strong_redundancy_flag":
            consistently_strong_flag,

        "pooled_year_effect_caution_flag":
            pooled_year_effect_caution_flag,
    })


pairs = pd.DataFrame(
    pair_rows
)


by_year = pd.DataFrame(
    by_year_rows
)


# --------------------------------------------------
# 12. Round numerical columns
# --------------------------------------------------

for dataframe in [
    pairs,
    by_year,
]:

    numeric_columns = (
        dataframe
        .select_dtypes(
            include="number"
        )
        .columns
    )


    dataframe[
        numeric_columns
    ] = (
        dataframe[
            numeric_columns
        ]
        .round(3)
    )


pooled_matrix = (
    pooled_matrix
    .round(3)
)


# --------------------------------------------------
# 13. Sort strongest relationships first
# --------------------------------------------------

pairs = (
    pairs
    .sort_values(
        by=[
            "consistently_strong_redundancy_flag",
            "very_high_pooled_correlation_flag",
            "high_pooled_correlation_flag",
            "abs_pooled_spearman",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 14. Build per-feature redundancy summary
#
# This shows how often each predictor appears
# in strong correlation pairs.
# --------------------------------------------------

summary_rows = []


for feature in FEATURES:

    related = pairs[
        (
            pairs["feature_a"] == feature
        )
        |
        (
            pairs["feature_b"] == feature
        )
    ]


    high_pooled_count = int(
        related[
            "high_pooled_correlation_flag"
        ]
        .sum()
    )


    very_high_count = int(
        related[
            "very_high_pooled_correlation_flag"
        ]
        .sum()
    )


    consistent_count = int(
        related[
            "consistently_strong_redundancy_flag"
        ]
        .sum()
    )


    year_effect_count = int(
        related[
            "pooled_year_effect_caution_flag"
        ]
        .sum()
    )


    max_pooled_abs = (
        related[
            "abs_pooled_spearman"
        ]
        .max()
    )


    summary_rows.append({
        "feature": feature,

        "pairs_with_abs_pooled_rho_ge_0_90":
            high_pooled_count,

        "pairs_with_abs_pooled_rho_ge_0_95":
            very_high_count,

        "consistently_strong_pairs":
            consistent_count,

        "pooled_year_effect_caution_pairs":
            year_effect_count,

        "maximum_abs_pooled_spearman":
            max_pooled_abs,
    })


summary = pd.DataFrame(
    summary_rows
)


summary[
    "maximum_abs_pooled_spearman"
] = (
    summary[
        "maximum_abs_pooled_spearman"
    ]
    .round(3)
)


summary = (
    summary
    .sort_values(
        by=[
            "consistently_strong_pairs",
            "pairs_with_abs_pooled_rho_ge_0_95",
            "pairs_with_abs_pooled_rho_ge_0_90",
            "maximum_abs_pooled_spearman",
        ],
        ascending=False
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 15. Overall redundancy counts
# --------------------------------------------------

high_pooled = pairs[
    pairs[
        "high_pooled_correlation_flag"
    ]
]


very_high_pooled = pairs[
    pairs[
        "very_high_pooled_correlation_flag"
    ]
]


consistent = pairs[
    pairs[
        "consistently_strong_redundancy_flag"
    ]
]


year_effect_caution = pairs[
    pairs[
        "pooled_year_effect_caution_flag"
    ]
]


# --------------------------------------------------
# 16. Print audit summary
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FOUR-YEAR FEATURE REDUNDANCY AUDIT"
)

print(
    "========================================"
)


print(
    f"\nTotal feature pairs: "
    f"{len(pairs)}"
)

print(
    f"Pairs with pooled |Spearman| >= 0.90: "
    f"{len(high_pooled)}"
)

print(
    f"Pairs with pooled |Spearman| >= 0.95: "
    f"{len(very_high_pooled)}"
)

print(
    "Pairs with pooled |Spearman| >= 0.90 "
    "AND every yearly |Spearman| >= 0.80: "
    f"{len(consistent)}"
)

print(
    "High pooled pairs with at least one "
    "yearly |Spearman| < 0.60: "
    f"{len(year_effect_caution)}"
)


# --------------------------------------------------
# 17. Display high pooled correlations
# --------------------------------------------------

print(
    "\nPairs with pooled |Spearman| >= 0.90:"
)


if len(high_pooled) == 0:

    print(
        "None"
    )

else:

    print(
        high_pooled[
            [
                "feature_a",
                "feature_b",
                "pooled_spearman",
                "spearman_2018_19",
                "spearman_2019_20",
                "spearman_2020_21",
                "spearman_2021_22",
                "min_within_year_abs_spearman",
                "consistently_strong_redundancy_flag",
                "pooled_year_effect_caution_flag",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 18. Display consistently strong redundancy
# --------------------------------------------------

print(
    "\nConsistently strong redundancy pairs:"
)


if len(consistent) == 0:

    print(
        "None"
    )

else:

    print(
        consistent[
            [
                "feature_a",
                "feature_b",
                "pooled_spearman",
                "min_within_year_abs_spearman",
                "median_within_year_abs_spearman",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 19. Display possible pooled year-effect pairs
# --------------------------------------------------

print(
    "\nPooled-correlation year-effect caution pairs:"
)


if len(year_effect_caution) == 0:

    print(
        "None"
    )

else:

    print(
        year_effect_caution[
            [
                "feature_a",
                "feature_b",
                "pooled_spearman",
                "spearman_2018_19",
                "spearman_2019_20",
                "spearman_2020_21",
                "spearman_2021_22",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 20. Most redundancy-connected features
# --------------------------------------------------

print(
    "\nFeature redundancy summary:"
)


print(
    summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# 21. Interpretation
# --------------------------------------------------

print(
    """
Interpretation:
- Pooled Spearman uses all 168 ICB-year rows.

- Within-year Spearman evaluates the
  relationship across the 42 ICBs separately
  in each historical period.

- A consistently strong pair is much stronger
  evidence of genuine feature redundancy than
  a high pooled correlation alone.

- A high pooled correlation with weak
  within-year correlation may partly reflect
  common year-to-year shifts rather than true
  redundancy between the measures.

- These flags are diagnostic only.
  No predictor should be removed solely because
  it crosses one correlation threshold.

- Final feature selection must also consider
  temporal stability, clinical meaning,
  information leakage, model complexity and
  validation performance.
""".strip()
)


# --------------------------------------------------
# 22. Final validation
# --------------------------------------------------

if len(pairs) != 171:
    raise RuntimeError(
        f"Expected 171 pair rows, "
        f"found {len(pairs)}."
    )


if len(summary) != 19:
    raise RuntimeError(
        f"Expected 19 summary rows, "
        f"found {len(summary)}."
    )


if pooled_matrix.shape != (
    19,
    19,
):
    raise RuntimeError(
        "Expected a 19 x 19 "
        "correlation matrix."
    )


if len(by_year) != (
    171 * 4
):
    raise RuntimeError(
        f"Expected {171 * 4} "
        "within-year rows, "
        f"found {len(by_year)}."
    )


if pairs.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in redundancy pair output."
    )


if summary.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in redundancy summary."
    )


print(
    "\nFeature redundancy audit: PASSED"
)


# --------------------------------------------------
# 23. Save outputs
# --------------------------------------------------

pairs.to_csv(
    PAIRS_OUTPUT,
    index=False
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


pooled_matrix.to_csv(
    MATRIX_OUTPUT
)


by_year.to_csv(
    BY_YEAR_OUTPUT,
    index=False
)


print(
    f"Saved pairs to: {PAIRS_OUTPUT}"
)

print(
    f"Saved summary to: {SUMMARY_OUTPUT}"
)

print(
    f"Saved matrix to: {MATRIX_OUTPUT}"
)

print(
    f"Saved within-year correlations to: "
    f"{BY_YEAR_OUTPUT}"
)