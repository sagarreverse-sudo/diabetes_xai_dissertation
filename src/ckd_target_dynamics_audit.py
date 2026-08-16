from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CKD_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "nda_ckd_type2_icb_2009_2023.csv"
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

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "ckd_target_dynamics_summary.csv"
)

ICB_OUTPUT = (
    OUTPUT_DIR
    / "ckd_target_dynamics_by_icb.csv"
)


# --------------------------------------------------
# 2. Pre-final transitions
#
# We inspect:
#
# 2019 -> 2020
# 2020 -> 2021
# 2021 -> 2022
#
# 2023 remains untouched.
# --------------------------------------------------

TRANSITIONS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
]

FINAL_TEST_YEAR = 2023

TARGET = "ckd_risk_rate_per_1000"


# --------------------------------------------------
# 3. Load CKD history
# --------------------------------------------------

if not CKD_FILE.exists():
    raise FileNotFoundError(
        f"CKD file not found: {CKD_FILE}"
    )


ckd = pd.read_csv(CKD_FILE)


print(
    f"CKD history shape: {ckd.shape}"
)


# --------------------------------------------------
# 4. Standardise identifiers
# --------------------------------------------------

ckd["icb_code"] = (
    ckd["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 5. Validate required columns
# --------------------------------------------------

required_columns = {
    "icb_code",
    "icb_name",
    "year",
    TARGET,
}


missing_columns = (
    required_columns
    - set(ckd.columns)
)


if missing_columns:
    raise RuntimeError(
        "Missing required CKD columns: "
        f"{sorted(missing_columns)}"
    )


if (
    ckd
    .duplicated(
        subset=[
            "icb_code",
            "year",
        ]
    )
    .any()
):
    raise RuntimeError(
        "Duplicate ICB-year CKD rows detected."
    )


# --------------------------------------------------
# 6. Protect final test year
#
# Create a separate audit dataframe containing
# only the years needed for this diagnostic.
# --------------------------------------------------

AUDIT_YEARS = {
    year
    for pair in TRANSITIONS
    for year in pair
}


audit_data = (
    ckd.loc[
        ckd["year"]
        .isin(AUDIT_YEARS)
    ]
    .copy()
)


if FINAL_TEST_YEAR in set(
    audit_data["year"]
):
    raise RuntimeError(
        "FINAL TEST LEAKAGE: "
        "2023 entered target-dynamics audit."
    )


print(
    "\nYears used in dynamics audit:"
)

print(
    sorted(
        audit_data["year"].unique()
    )
)


# --------------------------------------------------
# 7. Validate 42 ICBs in every audit year
# --------------------------------------------------

rows_per_year = (
    audit_data
    .groupby("year")
    .size()
)


icbs_per_year = (
    audit_data
    .groupby("year")["icb_code"]
    .nunique()
)


print(
    "\nRows per year:"
)

print(
    rows_per_year.to_string()
)


print(
    "\nUnique ICBs per year:"
)

print(
    icbs_per_year.to_string()
)


if not (
    rows_per_year == 42
).all():
    raise RuntimeError(
        "Every audit year must contain "
        "exactly 42 rows."
    )


if not (
    icbs_per_year == 42
).all():
    raise RuntimeError(
        "Every audit year must contain "
        "all 42 ICBs."
    )


# --------------------------------------------------
# 8. Confirm same ICB membership
# --------------------------------------------------

year_sets = {
    year: set(
        audit_data.loc[
            audit_data["year"] == year,
            "icb_code"
        ]
    )
    for year in sorted(AUDIT_YEARS)
}


reference_set = (
    year_sets[
        min(year_sets)
    ]
)


for year, codes in year_sets.items():

    if codes != reference_set:
        raise RuntimeError(
            f"ICB membership differs "
            f"in year {year}."
        )


print(
    "\nSame 42 ICBs across all audit years: PASSED"
)


# --------------------------------------------------
# 9. Metric helper
# --------------------------------------------------

def persistence_metrics(
    previous,
    current,
):

    mae = mean_absolute_error(
        current,
        previous
    )

    rmse = np.sqrt(
        mean_squared_error(
            current,
            previous
        )
    )

    r2 = r2_score(
        current,
        previous
    )

    return {
        "persistence_mae": mae,
        "persistence_rmse": rmse,
        "persistence_r2": r2,
    }


# --------------------------------------------------
# 10. Analyse each transition
# --------------------------------------------------

summary_rows = []
icb_rows = []


for previous_year, current_year in TRANSITIONS:

    print(
        "\n========================================"
    )

    print(
        f"CKD TRANSITION {previous_year} -> {current_year}"
    )

    print(
        "========================================"
    )


    previous = (
        audit_data.loc[
            audit_data["year"] == previous_year,
            [
                "icb_code",
                "icb_name",
                TARGET,
            ]
        ]
        .rename(
            columns={
                TARGET:
                    "previous_ckd_rate",

                "icb_name":
                    "previous_icb_name",
            }
        )
    )


    current = (
        audit_data.loc[
            audit_data["year"] == current_year,
            [
                "icb_code",
                "icb_name",
                TARGET,
            ]
        ]
        .rename(
            columns={
                TARGET:
                    "current_ckd_rate",

                "icb_name":
                    "current_icb_name",
            }
        )
    )


    transition = previous.merge(
        current,
        on="icb_code",
        how="inner",
        validate="one_to_one"
    )


    if len(transition) != 42:
        raise RuntimeError(
            f"{previous_year}->{current_year}: "
            f"expected 42 aligned ICBs, "
            f"found {len(transition)}."
        )


    # ----------------------------------------------
    # Change statistics
    # ----------------------------------------------

    transition[
        "signed_change"
    ] = (
        transition[
            "current_ckd_rate"
        ]
        -
        transition[
            "previous_ckd_rate"
        ]
    )


    transition[
        "absolute_change"
    ] = (
        transition[
            "signed_change"
        ]
        .abs()
    )


    transition[
        "percent_change"
    ] = (
        transition[
            "signed_change"
        ]
        /
        transition[
            "previous_ckd_rate"
        ]
        *
        100
    )


    transition[
        "direction"
    ] = np.where(
        transition[
            "signed_change"
        ] > 0,
        "increase",
        np.where(
            transition[
                "signed_change"
            ] < 0,
            "decrease",
            "unchanged"
        )
    )


    # ----------------------------------------------
    # Correlations
    # ----------------------------------------------

    spearman_rho, spearman_p = (
        spearmanr(
            transition[
                "previous_ckd_rate"
            ],
            transition[
                "current_ckd_rate"
            ]
        )
    )


    pearson_r, pearson_p = (
        pearsonr(
            transition[
                "previous_ckd_rate"
            ],
            transition[
                "current_ckd_rate"
            ]
        )
    )


    # ----------------------------------------------
    # Persistence performance
    # ----------------------------------------------

    persistence = (
        persistence_metrics(
            transition[
                "previous_ckd_rate"
            ],
            transition[
                "current_ckd_rate"
            ],
        )
    )


    # ----------------------------------------------
    # Direction counts
    # ----------------------------------------------

    increases = int(
        (
            transition[
                "signed_change"
            ] > 0
        )
        .sum()
    )


    decreases = int(
        (
            transition[
                "signed_change"
            ] < 0
        )
        .sum()
    )


    unchanged = int(
        (
            transition[
                "signed_change"
            ] == 0
        )
        .sum()
    )


    # ----------------------------------------------
    # Summary row
    # ----------------------------------------------

    summary_rows.append({
        "previous_year":
            previous_year,

        "current_year":
            current_year,

        "n_icbs":
            len(transition),

        "previous_mean":
            transition[
                "previous_ckd_rate"
            ]
            .mean(),

        "current_mean":
            transition[
                "current_ckd_rate"
            ]
            .mean(),

        "mean_signed_change":
            transition[
                "signed_change"
            ]
            .mean(),

        "median_signed_change":
            transition[
                "signed_change"
            ]
            .median(),

        "std_signed_change":
            transition[
                "signed_change"
            ]
            .std(),

        "mean_absolute_change":
            transition[
                "absolute_change"
            ]
            .mean(),

        "median_absolute_change":
            transition[
                "absolute_change"
            ]
            .median(),

        "max_absolute_change":
            transition[
                "absolute_change"
            ]
            .max(),

        "mean_percent_change":
            transition[
                "percent_change"
            ]
            .mean(),

        "icbs_increased":
            increases,

        "icbs_decreased":
            decreases,

        "icbs_unchanged":
            unchanged,

        "spearman_rho":
            spearman_rho,

        "spearman_p":
            spearman_p,

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        **persistence,
    })


    # ----------------------------------------------
    # Store ICB-level changes
    # ----------------------------------------------

    for _, row in transition.iterrows():

        icb_rows.append({
            "previous_year":
                previous_year,

            "current_year":
                current_year,

            "icb_code":
                row["icb_code"],

            "icb_name":
                row["current_icb_name"],

            "previous_ckd_rate":
                row["previous_ckd_rate"],

            "current_ckd_rate":
                row["current_ckd_rate"],

            "signed_change":
                row["signed_change"],

            "absolute_change":
                row["absolute_change"],

            "percent_change":
                row["percent_change"],

            "direction":
                row["direction"],
        })


    # ----------------------------------------------
    # Print transition results
    # ----------------------------------------------

    print(
        f"Previous mean CKD rate : "
        f"{transition['previous_ckd_rate'].mean():.2f}"
    )

    print(
        f"Current mean CKD rate  : "
        f"{transition['current_ckd_rate'].mean():.2f}"
    )

    print(
        f"Mean signed change     : "
        f"{transition['signed_change'].mean():.2f}"
    )

    print(
        f"Mean absolute change   : "
        f"{transition['absolute_change'].mean():.2f}"
    )

    print(
        f"Median absolute change : "
        f"{transition['absolute_change'].median():.2f}"
    )

    print(
        f"Max absolute change    : "
        f"{transition['absolute_change'].max():.2f}"
    )


    print(
        "\nDirection:"
    )

    print(
        f"Increased : {increases}"
    )

    print(
        f"Decreased : {decreases}"
    )

    print(
        f"Unchanged : {unchanged}"
    )


    print(
        "\nYear-to-year association:"
    )

    print(
        f"Spearman rho = "
        f"{spearman_rho:.3f}"
    )

    print(
        f"Pearson r    = "
        f"{pearson_r:.3f}"
    )


    print(
        "\nPersistence performance:"
    )

    print(
        f"MAE  = "
        f"{persistence['persistence_mae']:.3f}"
    )

    print(
        f"RMSE = "
        f"{persistence['persistence_rmse']:.3f}"
    )

    print(
        f"R2   = "
        f"{persistence['persistence_r2']:.3f}"
    )


# --------------------------------------------------
# 11. Build output tables
# --------------------------------------------------

summary_df = pd.DataFrame(
    summary_rows
)

icb_df = pd.DataFrame(
    icb_rows
)


# --------------------------------------------------
# 12. Round numerical values
# --------------------------------------------------

summary_numeric = (
    summary_df
    .select_dtypes(
        include="number"
    )
    .columns
)


summary_df[
    summary_numeric
] = (
    summary_df[
        summary_numeric
    ]
    .round(3)
)


icb_numeric = [
    "previous_ckd_rate",
    "current_ckd_rate",
    "signed_change",
    "absolute_change",
    "percent_change",
]


icb_df[
    icb_numeric
] = (
    icb_df[
        icb_numeric
    ]
    .round(3)
)


# --------------------------------------------------
# 13. Overall transition comparison
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "PRE-FINAL CKD DYNAMICS SUMMARY"
)

print(
    "========================================"
)


print(
    summary_df[
        [
            "previous_year",
            "current_year",
            "previous_mean",
            "current_mean",
            "mean_signed_change",
            "mean_absolute_change",
            "std_signed_change",
            "icbs_increased",
            "icbs_decreased",
            "spearman_rho",
            "persistence_mae",
            "persistence_rmse",
            "persistence_r2",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 14. Identify most volatile transition
# --------------------------------------------------

most_volatile = (
    summary_df
    .sort_values(
        "mean_absolute_change",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nLargest mean absolute "
    "year-to-year change:"
)

print(
    f"{int(most_volatile['previous_year'])}"
    f" -> "
    f"{int(most_volatile['current_year'])}"
)

print(
    f"Mean absolute change = "
    f"{most_volatile['mean_absolute_change']:.3f}"
)


# --------------------------------------------------
# 15. Identify weakest persistence transition
# --------------------------------------------------

weakest_persistence = (
    summary_df
    .sort_values(
        "persistence_mae",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nWeakest persistence transition:"
)

print(
    f"{int(weakest_persistence['previous_year'])}"
    f" -> "
    f"{int(weakest_persistence['current_year'])}"
)

print(
    f"Persistence MAE = "
    f"{weakest_persistence['persistence_mae']:.3f}"
)


# --------------------------------------------------
# 16. Largest-changing ICBs by transition
# --------------------------------------------------

print(
    "\nLargest absolute ICB changes "
    "within each transition:"
)


for previous_year, current_year in TRANSITIONS:

    largest = (
        icb_df.loc[
            (
                icb_df["previous_year"]
                == previous_year
            )
            &
            (
                icb_df["current_year"]
                == current_year
            )
        ]
        .nlargest(
            5,
            "absolute_change"
        )
    )


    print(
        f"\n{previous_year} -> {current_year}"
    )


    print(
        largest[
            [
                "icb_code",
                "icb_name",
                "previous_ckd_rate",
                "current_ckd_rate",
                "signed_change",
                "absolute_change",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 17. Interpretation guardrail
# --------------------------------------------------

print(
    """
Interpretation rule:
This audit can identify whether one transition
is unusually volatile relative to the other
pre-final transitions.

It does NOT establish the cause of that change.

In particular, we should not label a transition
as a COVID-19 effect without supporting external
evidence and appropriate discussion of changes
in service delivery, recording or population
risk.
""".strip()
)


# --------------------------------------------------
# 18. Final-test protection
# --------------------------------------------------

if FINAL_TEST_YEAR in set(
    audit_data["year"]
):
    raise RuntimeError(
        "2023 entered audit data."
    )


if (
    summary_df["current_year"]
    == FINAL_TEST_YEAR
).any():
    raise RuntimeError(
        "2023 transition was analysed."
    )


if (
    icb_df["current_year"]
    == FINAL_TEST_YEAR
).any():
    raise RuntimeError(
        "2023 ICB changes were analysed."
    )


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 19. Final validation
# --------------------------------------------------

if len(summary_df) != 3:
    raise RuntimeError(
        f"Expected 3 transition summaries, "
        f"found {len(summary_df)}."
    )


if len(icb_df) != 126:
    raise RuntimeError(
        f"Expected 126 ICB-transition rows, "
        f"found {len(icb_df)}."
    )


if summary_df.isna().any().any():
    raise RuntimeError(
        "Missing values detected in summary."
    )


if icb_df.isna().any().any():
    raise RuntimeError(
        "Missing values detected in ICB output."
    )


print(
    "\nCKD target dynamics audit: PASSED"
)


# --------------------------------------------------
# 20. Save outputs
# --------------------------------------------------

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


icb_df.to_csv(
    ICB_OUTPUT,
    index=False
)


print(
    f"\nSaved transition summary to: "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Saved ICB-level changes to: "
    f"{ICB_OUTPUT}"
)