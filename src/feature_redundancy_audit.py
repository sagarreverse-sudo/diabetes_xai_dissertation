from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "forecasting_panel_2019_20_to_2023.csv"
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


# --------------------------------------------------
# 2. Common predictor set
# --------------------------------------------------

FEATURES = [
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
# 3. Load forecasting panel
# --------------------------------------------------

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting panel not found: "
        f"{PANEL_FILE}"
    )


panel = pd.read_csv(
    PANEL_FILE
)


print(
    f"Forecasting panel shape: "
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


# --------------------------------------------------
# 4. Validate predictor structure
# --------------------------------------------------

missing_features = (
    set(FEATURES)
    - set(panel.columns)
)


if missing_features:
    raise RuntimeError(
        "Missing predictors: "
        f"{sorted(missing_features)}"
    )


if len(FEATURES) != 21:
    raise RuntimeError(
        f"Expected 21 features, "
        f"found {len(FEATURES)}."
    )


if panel[FEATURES].isna().any().any():
    raise RuntimeError(
        "Missing predictor values detected."
    )


print(
    "\nFeature structure check: PASSED"
)


# --------------------------------------------------
# 5. Correlation matrices
#
# We use both:
#
# Pearson  -> linear association
# Spearman -> rank/monotonic association
#
# Spearman is particularly useful here because
# many percentage indicators are not guaranteed
# to have perfectly linear relationships.
# --------------------------------------------------

pearson_matrix = (
    panel[FEATURES]
    .corr(method="pearson")
)

spearman_matrix = (
    panel[FEATURES]
    .corr(method="spearman")
)


# --------------------------------------------------
# 6. Within-year correlation matrices
#
# This prevents us from relying only on pooled
# correlations that could be driven by year effects.
# --------------------------------------------------

target_years = sorted(
    panel["target_year"]
    .unique()
)


within_year_spearman = {}


for year in target_years:

    year_data = panel[
        panel["target_year"] == year
    ]

    within_year_spearman[year] = (
        year_data[FEATURES]
        .corr(method="spearman")
    )


# --------------------------------------------------
# 7. Build pairwise redundancy audit
# --------------------------------------------------

pair_rows = []


for feature_1, feature_2 in combinations(
    FEATURES,
    2
):

    pooled_pearson = (
        pearson_matrix.loc[
            feature_1,
            feature_2
        ]
    )

    pooled_spearman = (
        spearman_matrix.loc[
            feature_1,
            feature_2
        ]
    )


    year_correlations = {}


    for year in target_years:

        rho = (
            within_year_spearman[year]
            .loc[
                feature_1,
                feature_2
            ]
        )

        year_correlations[year] = rho


    abs_year_correlations = [
        abs(value)
        for value
        in year_correlations.values()
        if pd.notna(value)
    ]


    mean_abs_within_year = (
        np.mean(abs_year_correlations)
        if abs_year_correlations
        else np.nan
    )


    max_abs_within_year = (
        np.max(abs_year_correlations)
        if abs_year_correlations
        else np.nan
    )


    # Count how many individual years have
    # a very strong relationship.
    years_above_080 = sum(
        abs(value) >= 0.80
        for value
        in year_correlations.values()
        if pd.notna(value)
    )


    # ----------------------------------------------
    # Diagnostic redundancy flags
    #
    # These do NOT automatically remove variables.
    # ----------------------------------------------

    high_pooled_correlation = (
        abs(pooled_spearman) >= 0.85
    )

    persistent_within_year_correlation = (
        years_above_080 >= 2
    )

    very_high_correlation = (
        abs(pooled_spearman) >= 0.95
    )


    potential_redundancy = (
        high_pooled_correlation
        or
        persistent_within_year_correlation
    )


    row = {
        "feature_1": feature_1,
        "feature_2": feature_2,

        "pooled_pearson":
            pooled_pearson,

        "pooled_spearman":
            pooled_spearman,

        "abs_pooled_spearman":
            abs(pooled_spearman),

        "mean_abs_within_year_spearman":
            mean_abs_within_year,

        "max_abs_within_year_spearman":
            max_abs_within_year,

        "years_abs_spearman_ge_0_80":
            years_above_080,

        "high_pooled_corr_flag":
            high_pooled_correlation,

        "persistent_within_year_corr_flag":
            persistent_within_year_correlation,

        "very_high_corr_flag":
            very_high_correlation,

        "potential_redundancy_flag":
            potential_redundancy,
    }


    for year in target_years:

        row[
            f"spearman_target_{year}"
        ] = year_correlations[year]


    pair_rows.append(row)


pairwise = pd.DataFrame(
    pair_rows
)


# --------------------------------------------------
# 8. Round numeric columns
# --------------------------------------------------

numeric_columns = (
    pairwise
    .select_dtypes(
        include="number"
    )
    .columns
)


pairwise[numeric_columns] = (
    pairwise[numeric_columns]
    .round(3)
)


# --------------------------------------------------
# 9. Sort strongest relationships first
# --------------------------------------------------

pairwise = (
    pairwise
    .sort_values(
        [
            "potential_redundancy_flag",
            "abs_pooled_spearman",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 10. Print strongest pooled correlations
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "STRONGEST PREDICTOR CORRELATIONS"
)

print(
    "========================================"
)


strongest = pairwise[
    [
        "feature_1",
        "feature_2",
        "pooled_pearson",
        "pooled_spearman",
        "mean_abs_within_year_spearman",
        "years_abs_spearman_ge_0_80",
    ]
].head(30)


print(
    strongest.to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Potential redundant pairs
# --------------------------------------------------

redundant_pairs = pairwise[
    pairwise[
        "potential_redundancy_flag"
    ]
].copy()


print(
    "\n========================================"
)

print(
    "POTENTIALLY REDUNDANT PAIRS"
)

print(
    "========================================"
)


if len(redundant_pairs) == 0:

    print(
        "No predictor pairs triggered "
        "the redundancy rules."
    )

else:

    print(
        redundant_pairs[
            [
                "feature_1",
                "feature_2",
                "pooled_spearman",
                "spearman_target_2021",
                "spearman_target_2022",
                "spearman_target_2023",
                "years_abs_spearman_ge_0_80",
                "very_high_corr_flag",
            ]
        ].to_string(
            index=False
        )
    )


# --------------------------------------------------
# 12. Feature-level redundancy summary
# --------------------------------------------------

feature_summary_rows = []


for feature in FEATURES:

    related = pairwise[
        (
            pairwise["feature_1"]
            == feature
        )
        |
        (
            pairwise["feature_2"]
            == feature
        )
    ].copy()


    flagged = related[
        related[
            "potential_redundancy_flag"
        ]
    ]


    # Find strongest partner
    strongest_row = (
        related
        .sort_values(
            "abs_pooled_spearman",
            ascending=False
        )
        .iloc[0]
    )


    if (
        strongest_row["feature_1"]
        == feature
    ):

        strongest_partner = (
            strongest_row["feature_2"]
        )

    else:

        strongest_partner = (
            strongest_row["feature_1"]
        )


    feature_summary_rows.append({
        "feature": feature,

        "redundant_partner_count":
            len(flagged),

        "strongest_partner":
            strongest_partner,

        "max_abs_pooled_spearman":
            strongest_row[
                "abs_pooled_spearman"
            ],
    })


feature_summary = pd.DataFrame(
    feature_summary_rows
)


feature_summary = (
    feature_summary
    .sort_values(
        [
            "redundant_partner_count",
            "max_abs_pooled_spearman",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


print(
    "\n========================================"
)

print(
    "FEATURE-LEVEL REDUNDANCY SUMMARY"
)

print(
    "========================================"
)


print(
    feature_summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# 13. Overall counts
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "REDUNDANCY AUDIT SUMMARY"
)

print(
    "========================================"
)


print(
    f"Total predictors: "
    f"{len(FEATURES)}"
)

print(
    f"Total predictor pairs: "
    f"{len(pairwise)}"
)

print(
    f"Potentially redundant pairs: "
    f"{len(redundant_pairs)}"
)

print(
    "Pairs with pooled "
    "|Spearman| >= 0.95: "
    f"{pairwise['very_high_corr_flag'].sum()}"
)


# --------------------------------------------------
# 14. Save outputs
# --------------------------------------------------

PAIR_OUTPUT = (
    OUTPUT_DIR
    / "feature_redundancy_pairs.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "feature_redundancy_summary.csv"
)

SPEARMAN_OUTPUT = (
    OUTPUT_DIR
    / "feature_spearman_matrix.csv"
)


pairwise.to_csv(
    PAIR_OUTPUT,
    index=False
)

feature_summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

spearman_matrix.to_csv(
    SPEARMAN_OUTPUT
)


print(
    "\nFeature redundancy audit: PASSED"
)

print(
    f"Pairwise audit saved to: "
    f"{PAIR_OUTPUT}"
)

print(
    f"Feature summary saved to: "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Spearman matrix saved to: "
    f"{SPEARMAN_OUTPUT}"
)