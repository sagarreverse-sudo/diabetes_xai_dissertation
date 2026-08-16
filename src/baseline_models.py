from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
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


PANEL_FILE = (
    INTERIM_DIR
    / "forecasting_panel_2018_19_to_2023.csv"
)

CKD_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)


PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "baseline_predictions_pre_final.csv"
)

METRICS_OUTPUT = (
    OUTPUT_DIR
    / "baseline_metrics_pre_final.csv"
)


# --------------------------------------------------
# 2. Evaluation design
#
# IMPORTANT:
# We deliberately exclude target year 2023.
#
# 2020 + 2021 = development
# 2022        = validation
# 2023        = untouched final test
# --------------------------------------------------

EVALUATION_YEARS = [
    2020,
    2021,
    2022,
]

FINAL_TEST_YEAR = 2023


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting panel not found: {PANEL_FILE}"
    )


if not CKD_FILE.exists():
    raise FileNotFoundError(
        f"CKD history file not found: {CKD_FILE}"
    )


panel = pd.read_csv(PANEL_FILE)
ckd = pd.read_csv(CKD_FILE)


print(
    f"Forecasting panel shape: {panel.shape}"
)

print(
    f"CKD history shape: {ckd.shape}"
)


# --------------------------------------------------
# 4. Standardise identifiers
# --------------------------------------------------

for dataframe in [
    panel,
    ckd,
]:

    dataframe["icb_code"] = (
        dataframe["icb_code"]
        .astype("string")
        .str.strip()
    )


# --------------------------------------------------
# 5. Validate CKD historical data
# --------------------------------------------------

required_ckd_columns = {
    "icb_code",
    "icb_name",
    "year",
    "ckd_cases",
    "diabetes_population",
    "ckd_risk_rate_per_1000",
}


missing_ckd = (
    required_ckd_columns
    - set(ckd.columns)
)


if missing_ckd:
    raise RuntimeError(
        "Missing CKD columns: "
        f"{sorted(missing_ckd)}"
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
        "Duplicate CKD ICB-year rows detected."
    )


print(
    "\nCKD years available:"
)

print(
    sorted(
        ckd["year"].unique()
    )
)


# --------------------------------------------------
# 6. Extract actual outcomes for
#    pre-final evaluation years only
# --------------------------------------------------

evaluation = (
    panel.loc[
        panel["target_year"]
        .isin(EVALUATION_YEARS),
        [
            "target_year",
            "icb_code",
            "icb_name",
            "ckd_risk_rate_per_1000",
        ]
    ]
    .copy()
)


evaluation = evaluation.rename(
    columns={
        "ckd_risk_rate_per_1000":
            "actual_ckd_rate"
    }
)


# --------------------------------------------------
# 7. Explicitly protect final-test year
# --------------------------------------------------

if (
    evaluation["target_year"]
    == FINAL_TEST_YEAR
).any():

    raise RuntimeError(
        "FINAL TEST LEAKAGE: 2023 was included "
        "in baseline evaluation."
    )


if len(evaluation) != 126:
    raise RuntimeError(
        f"Expected 126 pre-final observations, "
        f"found {len(evaluation)}."
    )


print(
    "\nBaseline evaluation years:"
)

print(
    sorted(
        evaluation["target_year"].unique()
    )
)


print(
    f"Evaluation observations: "
    f"{len(evaluation)}"
)


# --------------------------------------------------
# 8. Baseline prediction functions
# --------------------------------------------------

def persistence_prediction(
    icb_code,
    target_year,
):
    """
    Predict future CKD rate using the same
    ICB's CKD rate from exactly one year earlier.

    Example:
    CKD 2022 prediction = observed CKD 2021 rate.
    """

    previous_year = (
        target_year - 1
    )


    history = ckd.loc[
        (
            ckd["icb_code"]
            == icb_code
        )
        &
        (
            ckd["year"]
            == previous_year
        ),
        "ckd_risk_rate_per_1000"
    ]


    if len(history) != 1:
        raise RuntimeError(
            f"Expected exactly one previous-year "
            f"CKD observation for {icb_code}, "
            f"target {target_year}; "
            f"found {len(history)}."
        )


    return float(
        history.iloc[0]
    )


def icb_historical_mean_prediction(
    icb_code,
    target_year,
):
    """
    Predict using this ICB's mean CKD rate
    across ALL years strictly before the
    target year.
    """

    history = ckd.loc[
        (
            ckd["icb_code"]
            == icb_code
        )
        &
        (
            ckd["year"]
            < target_year
        ),
        "ckd_risk_rate_per_1000"
    ]


    if len(history) == 0:
        raise RuntimeError(
            f"No historical CKD data available "
            f"for {icb_code} before "
            f"{target_year}."
        )


    return float(
        history.mean()
    )


def national_historical_mean_prediction(
    target_year,
):
    """
    Predict every ICB using the unweighted mean
    CKD rate across all ICB-year observations
    strictly before the target year.

    This is deliberately a simple sanity-check
    benchmark.
    """

    history = ckd.loc[
        ckd["year"]
        < target_year,
        "ckd_risk_rate_per_1000"
    ]


    if len(history) == 0:
        raise RuntimeError(
            f"No national historical data "
            f"before {target_year}."
        )


    return float(
        history.mean()
    )


# --------------------------------------------------
# 9. Generate leakage-safe predictions
# --------------------------------------------------

evaluation[
    "persistence_prediction"
] = evaluation.apply(
    lambda row:
        persistence_prediction(
            row["icb_code"],
            int(
                row["target_year"]
            ),
        ),
    axis=1,
)


evaluation[
    "icb_historical_mean_prediction"
] = evaluation.apply(
    lambda row:
        icb_historical_mean_prediction(
            row["icb_code"],
            int(
                row["target_year"]
            ),
        ),
    axis=1,
)


evaluation[
    "national_historical_mean_prediction"
] = evaluation[
    "target_year"
].apply(
    lambda year:
        national_historical_mean_prediction(
            int(year)
        )
)


# --------------------------------------------------
# 10. Validate predictions
# --------------------------------------------------

prediction_columns = [
    "persistence_prediction",
    "icb_historical_mean_prediction",
    "national_historical_mean_prediction",
]


if (
    evaluation[
        prediction_columns
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "Missing baseline predictions detected."
    )


print(
    "\nBaseline prediction generation: PASSED"
)


# --------------------------------------------------
# 11. Metric helper
# --------------------------------------------------

def calculate_metrics(
    actual,
    predicted,
):

    mae = mean_absolute_error(
        actual,
        predicted
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


    r2 = r2_score(
        actual,
        predicted
    )


    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# --------------------------------------------------
# 12. Baseline names
# --------------------------------------------------

BASELINES = {
    "persistence":
        "persistence_prediction",

    "icb_historical_mean":
        "icb_historical_mean_prediction",

    "national_historical_mean":
        "national_historical_mean_prediction",
}


# --------------------------------------------------
# 13. Metrics separately for each year
# --------------------------------------------------

metric_rows = []


for target_year in EVALUATION_YEARS:

    year_data = evaluation.loc[
        evaluation["target_year"]
        == target_year
    ]


    if len(year_data) != 42:
        raise RuntimeError(
            f"Expected 42 observations for "
            f"{target_year}, found "
            f"{len(year_data)}."
        )


    if target_year in {
        2020,
        2021,
    }:

        evaluation_stage = (
            "development"
        )

    elif target_year == 2022:

        evaluation_stage = (
            "validation"
        )

    else:

        raise RuntimeError(
            f"Unexpected target year: "
            f"{target_year}"
        )


    for baseline_name, prediction_column in BASELINES.items():

        metrics = calculate_metrics(
            year_data[
                "actual_ckd_rate"
            ],

            year_data[
                prediction_column
            ],
        )


        metric_rows.append({
            "evaluation_scope":
                "single_year",

            "evaluation_stage":
                evaluation_stage,

            "target_year":
                target_year,

            "baseline":
                baseline_name,

            "n":
                len(year_data),

            **metrics,
        })


# --------------------------------------------------
# 14. Overall pre-final metrics
#
# These summarize performance across all
# 126 observations from 2020-2022.
#
# They do NOT include the untouched 2023 test.
# --------------------------------------------------

for baseline_name, prediction_column in BASELINES.items():

    metrics = calculate_metrics(
        evaluation[
            "actual_ckd_rate"
        ],

        evaluation[
            prediction_column
        ],
    )


    metric_rows.append({
        "evaluation_scope":
            "overall_pre_final",

        "evaluation_stage":
            "development_plus_validation",

        "target_year":
            "2020_2022",

        "baseline":
            baseline_name,

        "n":
            len(evaluation),

        **metrics,
    })


metrics_df = pd.DataFrame(
    metric_rows
)


# --------------------------------------------------
# 15. Round metrics for readable output
# --------------------------------------------------

for column in [
    "mae",
    "rmse",
    "r2",
]:

    metrics_df[column] = (
        metrics_df[column]
        .round(3)
    )


# --------------------------------------------------
# 16. Display validation-year performance
#
# 2022 is especially important because this is
# the year later used for model comparison.
# --------------------------------------------------

validation_metrics = (
    metrics_df.loc[
        (
            metrics_df[
                "evaluation_scope"
            ]
            == "single_year"
        )
        &
        (
            metrics_df[
                "target_year"
            ]
            .astype(str)
            == "2022"
        )
    ]
    .sort_values("mae")
)


print(
    "\n========================================"
)

print(
    "BASELINE PERFORMANCE - VALIDATION 2022"
)

print(
    "========================================"
)


print(
    validation_metrics[
        [
            "baseline",
            "n",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 17. Display all year-specific results
# --------------------------------------------------

single_year_metrics = (
    metrics_df.loc[
        metrics_df[
            "evaluation_scope"
        ]
        == "single_year"
    ]
    .sort_values(
        [
            "target_year",
            "mae",
        ]
    )
)


print(
    "\nAll pre-final year-specific baseline results:"
)


print(
    single_year_metrics[
        [
            "target_year",
            "evaluation_stage",
            "baseline",
            "n",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 18. Display overall pre-final results
# --------------------------------------------------

overall_metrics = (
    metrics_df.loc[
        metrics_df[
            "evaluation_scope"
        ]
        == "overall_pre_final"
    ]
    .sort_values("mae")
)


print(
    "\nOverall pre-final baseline results "
    "(2020-2022 only):"
)


print(
    overall_metrics[
        [
            "baseline",
            "n",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 19. Identify validation benchmark
#
# This is descriptive only.
# We are not changing the feature set here.
# --------------------------------------------------

best_validation = (
    validation_metrics
    .iloc[0]
)


print(
    "\nStrongest 2022 baseline by MAE:"
)

print(
    f"{best_validation['baseline']}"
)

print(
    f"MAE  = "
    f"{best_validation['mae']}"
)

print(
    f"RMSE = "
    f"{best_validation['rmse']}"
)

print(
    f"R2   = "
    f"{best_validation['r2']}"
)


# --------------------------------------------------
# 20. Protect final test explicitly
# --------------------------------------------------

if (
    FINAL_TEST_YEAR
    in evaluation[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 final test was accessed."
    )


if (
    FINAL_TEST_YEAR
    in metrics_df[
        "target_year"
    ]
    .astype(str)
    .tolist()
):

    raise RuntimeError(
        "2023 final-test metrics were generated."
    )


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 21. Final output validation
# --------------------------------------------------

# 3 baselines × 3 individual years = 9
# plus 3 overall rows = 12 metric rows.

if len(metrics_df) != 12:
    raise RuntimeError(
        f"Expected 12 metric rows, "
        f"found {len(metrics_df)}."
    )


if len(evaluation) != 126:
    raise RuntimeError(
        f"Expected 126 prediction rows, "
        f"found {len(evaluation)}."
    )


if metrics_df[
    [
        "mae",
        "rmse",
        "r2",
    ]
].isna().any().any():

    raise RuntimeError(
        "Missing metric values detected."
    )


print(
    "\nBaseline model audit: PASSED"
)


# --------------------------------------------------
# 22. Save results
# --------------------------------------------------

evaluation = (
    evaluation
    .sort_values(
        [
            "target_year",
            "icb_code",
        ]
    )
    .reset_index(drop=True)
)


metrics_df = (
    metrics_df
    .sort_values(
        [
            "evaluation_scope",
            "target_year",
            "mae",
        ]
    )
    .reset_index(drop=True)
)


evaluation.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)


metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


print(
    f"\nSaved predictions to: "
    f"{PREDICTIONS_OUTPUT}"
)

print(
    f"Saved metrics to: "
    f"{METRICS_OUTPUT}"
)