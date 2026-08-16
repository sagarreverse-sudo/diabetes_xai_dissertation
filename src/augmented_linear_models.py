from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

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

FEATURE_MANIFEST_FILE = (
    OUTPUT_DIR
    / "feature_set_manifest.csv"
)

BASELINE_METRICS_FILE = (
    OUTPUT_DIR
    / "baseline_metrics_pre_final.csv"
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "augmented_linear_validation_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "augmented_linear_validation_predictions.csv"
)

RIDGE_SEARCH_OUTPUT = (
    OUTPUT_DIR
    / "augmented_ridge_alpha_search.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "augmented_linear_coefficients.csv"
)


# --------------------------------------------------
# 2. Locked temporal design
# --------------------------------------------------

TRAIN_YEARS = [
    2020,
    2021,
]

VALIDATION_YEAR = 2022

FINAL_TEST_YEAR = 2023

TARGET = "ckd_risk_rate_per_1000"

LAG_FEATURE = "lagged_ckd_rate"


# --------------------------------------------------
# 3. Transparent Ridge alpha grid
#
# We include stronger regularisation because
# the previous full 19-feature model showed
# substantial instability.
# --------------------------------------------------

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


# --------------------------------------------------
# 4. Load required data
# --------------------------------------------------

for file_path in [
    PANEL_FILE,
    CKD_FILE,
    FEATURE_MANIFEST_FILE,
    BASELINE_METRICS_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path}"
        )


panel = pd.read_csv(
    PANEL_FILE
)

ckd = pd.read_csv(
    CKD_FILE
)

feature_manifest = pd.read_csv(
    FEATURE_MANIFEST_FILE
)

baseline_metrics = pd.read_csv(
    BASELINE_METRICS_FILE
)


print(
    f"Forecasting panel shape: "
    f"{panel.shape}"
)

print(
    f"CKD history shape: "
    f"{ckd.shape}"
)


# --------------------------------------------------
# 5. Standardise ICB codes
# --------------------------------------------------

panel["icb_code"] = (
    panel["icb_code"]
    .astype("string")
    .str.strip()
)

ckd["icb_code"] = (
    ckd["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 6. Protect final test year
#
# Only target years 2020-2022 enter modelling.
# --------------------------------------------------

model_data = (
    panel.loc[
        panel["target_year"]
        .isin(
            TRAIN_YEARS
            + [
                VALIDATION_YEAR
            ]
        )
    ]
    .copy()
)


if (
    model_data["target_year"]
    == FINAL_TEST_YEAR
).any():

    raise RuntimeError(
        "FINAL TEST LEAKAGE: "
        "2023 entered modelling data."
    )


if len(model_data) != 126:

    raise RuntimeError(
        f"Expected 126 pre-final rows, "
        f"found {len(model_data)}."
    )


print(
    "\nYears available to modelling:"
)

print(
    sorted(
        model_data[
            "target_year"
        ]
        .unique()
    )
)


# --------------------------------------------------
# 7. Recover locked NDA feature sets
# --------------------------------------------------

def to_boolean(series):

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )


feature_manifest[
    "primary_compact_set"
] = to_boolean(
    feature_manifest[
        "primary_compact_set"
    ]
)


feature_manifest[
    "full_sensitivity_set"
] = to_boolean(
    feature_manifest[
        "full_sensitivity_set"
    ]
)


primary_manifest = (
    feature_manifest.loc[
        feature_manifest[
            "primary_compact_set"
        ]
    ]
    .sort_values(
        "primary_order"
    )
)


PRIMARY_FEATURES = (
    primary_manifest[
        "feature"
    ]
    .tolist()
)


FULL_FEATURES = (
    feature_manifest.loc[
        feature_manifest[
            "full_sensitivity_set"
        ],
        "feature"
    ]
    .tolist()
)


if len(PRIMARY_FEATURES) != 5:

    raise RuntimeError(
        f"Expected 5 compact NDA predictors, "
        f"found {len(PRIMARY_FEATURES)}."
    )


if len(FULL_FEATURES) != 19:

    raise RuntimeError(
        f"Expected 19 full NDA predictors, "
        f"found {len(FULL_FEATURES)}."
    )


print(
    f"\nCompact NDA predictors: "
    f"{len(PRIMARY_FEATURES)}"
)

print(
    f"Full NDA predictors: "
    f"{len(FULL_FEATURES)}"
)


# --------------------------------------------------
# 8. Construct lagged CKD predictor
#
# For target year t:
#
# lagged_ckd_rate = CKD rate in t - 1
#
# Examples:
#
# target 2020 -> CKD 2019
# target 2021 -> CKD 2020
# target 2022 -> CKD 2021
#
# Therefore no future outcome information
# is used.
# --------------------------------------------------

lag_source = (
    ckd[
        [
            "icb_code",
            "year",
            "ckd_risk_rate_per_1000",
        ]
    ]
    .copy()
)


lag_source["target_year"] = (
    lag_source["year"]
    + 1
)


lag_source = lag_source.rename(
    columns={
        "ckd_risk_rate_per_1000":
            LAG_FEATURE,

        "year":
            "lag_source_year",
    }
)


# Keep ONLY lag years needed by modelling rows.
lag_source = (
    lag_source.loc[
        lag_source["target_year"]
        .isin(
            TRAIN_YEARS
            + [
                VALIDATION_YEAR
            ]
        )
    ]
    .copy()
)


model_data = model_data.merge(
    lag_source[
        [
            "icb_code",
            "target_year",
            "lag_source_year",
            LAG_FEATURE,
        ]
    ],
    on=[
        "icb_code",
        "target_year",
    ],
    how="left",
    validate="one_to_one"
)


if model_data[
    LAG_FEATURE
].isna().any():

    raise RuntimeError(
        "Missing lagged CKD values detected."
    )


# --------------------------------------------------
# 9. Validate temporal lag explicitly
# --------------------------------------------------

if not (
    model_data[
        "lag_source_year"
    ]
    ==
    (
        model_data[
            "target_year"
        ]
        - 1
    )
).all():

    raise RuntimeError(
        "Lag-year alignment is incorrect."
    )


if (
    model_data[
        "lag_source_year"
    ]
    >=
    model_data[
        "target_year"
    ]
).any():

    raise RuntimeError(
        "Temporal leakage in lagged CKD."
    )


print(
    "\nLagged CKD alignment:"
)


print(
    model_data[
        [
            "lag_source_year",
            "target_year",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "target_year"
    )
    .to_string(
        index=False
    )
)


print(
    "\nLagged CKD temporal check: PASSED"
)


# --------------------------------------------------
# 10. Development and validation split
# --------------------------------------------------

train = (
    model_data.loc[
        model_data[
            "target_year"
        ]
        .isin(
            TRAIN_YEARS
        )
    ]
    .copy()
)


validation = (
    model_data.loc[
        model_data[
            "target_year"
        ]
        == VALIDATION_YEAR
    ]
    .copy()
)


if len(train) != 84:

    raise RuntimeError(
        f"Expected 84 training rows, "
        f"found {len(train)}."
    )


if len(validation) != 42:

    raise RuntimeError(
        f"Expected 42 validation rows, "
        f"found {len(validation)}."
    )


print(
    f"\nTraining rows: "
    f"{len(train)}"
)

print(
    f"Validation rows: "
    f"{len(validation)}"
)

print(
    "Final-test rows accessed: 0"
)


# --------------------------------------------------
# 11. Verify persistence construction
#
# Persistence prediction should simply equal
# lagged_ckd_rate.
#
# Its 2022 MAE must match our previous
# baseline result.
# --------------------------------------------------

persistence_predictions = (
    validation[
        LAG_FEATURE
    ]
)


persistence_mae_direct = (
    mean_absolute_error(
        validation[
            TARGET
        ],
        persistence_predictions,
    )
)


persistence_row = (
    baseline_metrics.loc[
        (
            baseline_metrics[
                "evaluation_scope"
            ]
            == "single_year"
        )
        &
        (
            baseline_metrics[
                "target_year"
            ]
            .astype(str)
            == "2022"
        )
        &
        (
            baseline_metrics[
                "baseline"
            ]
            == "persistence"
        )
    ]
)


if len(persistence_row) != 1:

    raise RuntimeError(
        "Could not recover unique "
        "2022 persistence benchmark."
    )


persistence_mae_saved = float(
    persistence_row[
        "mae"
    ]
    .iloc[0]
)


if not np.isclose(
    persistence_mae_direct,
    persistence_mae_saved,
    atol=0.001,
):

    raise RuntimeError(
        "Lagged CKD does not reproduce "
        "the saved persistence benchmark."
    )


print(
    "\nPersistence reconstruction: PASSED"
)

print(
    f"2022 persistence MAE: "
    f"{persistence_mae_direct:.3f}"
)


# --------------------------------------------------
# 12. Metric helper
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
# 13. Model builders
# --------------------------------------------------

def build_linear_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        ),
    ])


def build_ridge_model(alpha):

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=alpha
            )
        ),
    ])


# --------------------------------------------------
# 14. Model feature sets
#
# lag_only:
#     autoregressive CKD history only
#
# lag_plus_compact:
#     previous CKD + 5 pre-specified NDA features
#
# lag_plus_full:
#     previous CKD + all 19 NDA features
# --------------------------------------------------

FEATURE_SETS = {

    "lag_only": [
        LAG_FEATURE
    ],

    "lag_plus_compact_5": [
        LAG_FEATURE
    ]
    + PRIMARY_FEATURES,

    "lag_plus_full_19": [
        LAG_FEATURE
    ]
    + FULL_FEATURES,
}


print(
    "\nModel feature counts:"
)

for name, features in FEATURE_SETS.items():

    print(
        f"{name}: "
        f"{len(features)}"
    )


# --------------------------------------------------
# 15. Containers
# --------------------------------------------------

metric_rows = []
prediction_rows = []
ridge_search_rows = []
coefficient_rows = []


# --------------------------------------------------
# 16. Ordinary Linear Regression
# --------------------------------------------------

for feature_set_name, features in FEATURE_SETS.items():

    X_train = train[
        features
    ]

    y_train = train[
        TARGET
    ]

    X_validation = validation[
        features
    ]

    y_validation = validation[
        TARGET
    ]


    model = build_linear_model()


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_validation
    )


    metrics = calculate_metrics(
        y_validation,
        predictions
    )


    metric_rows.append({
        "model":
            "linear_regression",

        "feature_set":
            feature_set_name,

        "alpha":
            np.nan,

        "feature_count":
            len(features),

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        **metrics,
    })


    for index, prediction in zip(
        validation.index,
        predictions,
    ):

        prediction_rows.append({
            "target_year":
                VALIDATION_YEAR,

            "icb_code":
                validation.loc[
                    index,
                    "icb_code"
                ],

            "icb_name":
                validation.loc[
                    index,
                    "icb_name"
                ],

            "actual_ckd_rate":
                validation.loc[
                    index,
                    TARGET
                ],

            "model":
                "linear_regression",

            "feature_set":
                feature_set_name,

            "alpha":
                np.nan,

            "prediction":
                prediction,
        })


    estimator = (
        model.named_steps[
            "model"
        ]
    )


    for feature, coefficient in zip(
        features,
        estimator.coef_,
    ):

        coefficient_rows.append({
            "model":
                "linear_regression",

            "feature_set":
                feature_set_name,

            "alpha":
                np.nan,

            "feature":
                feature,

            "standardised_coefficient":
                coefficient,

            "abs_standardised_coefficient":
                abs(coefficient),
        })


# --------------------------------------------------
# 17. Ridge models
#
# Ridge is tested only for the two augmented
# feature sets.
#
# A one-feature lag-only Ridge model adds little
# scientific value, so lag-only is represented
# by the fitted linear model and fixed
# persistence benchmark.
# --------------------------------------------------

RIDGE_FEATURE_SETS = {
    "lag_plus_compact_5":
        FEATURE_SETS[
            "lag_plus_compact_5"
        ],

    "lag_plus_full_19":
        FEATURE_SETS[
            "lag_plus_full_19"
        ],
}


for feature_set_name, features in RIDGE_FEATURE_SETS.items():

    print(
        "\n----------------------------------------"
    )

    print(
        f"Ridge alpha search: "
        f"{feature_set_name}"
    )

    print(
        "----------------------------------------"
    )


    X_train = train[
        features
    ]

    y_train = train[
        TARGET
    ]

    X_validation = validation[
        features
    ]

    y_validation = validation[
        TARGET
    ]


    alpha_results = []


    for alpha in RIDGE_ALPHAS:

        model = build_ridge_model(
            alpha
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_validation
        )


        metrics = calculate_metrics(
            y_validation,
            predictions
        )


        ridge_search_rows.append({
            "feature_set":
                feature_set_name,

            "alpha":
                alpha,

            **metrics,
        })


        alpha_results.append(
            (
                alpha,
                metrics,
                model,
                predictions,
            )
        )


        print(
            f"alpha={alpha:<7} "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.3f}"
        )


    # Choose alpha by validation MAE.
    best_result = min(
        alpha_results,
        key=lambda item:
            item[1]["mae"]
    )


    (
        best_alpha,
        best_metrics,
        best_model,
        best_predictions,
    ) = best_result


    print(
        f"\nBest alpha for "
        f"{feature_set_name}: "
        f"{best_alpha}"
    )


    metric_rows.append({
        "model":
            "ridge_regression",

        "feature_set":
            feature_set_name,

        "alpha":
            best_alpha,

        "feature_count":
            len(features),

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        **best_metrics,
    })


    for index, prediction in zip(
        validation.index,
        best_predictions,
    ):

        prediction_rows.append({
            "target_year":
                VALIDATION_YEAR,

            "icb_code":
                validation.loc[
                    index,
                    "icb_code"
                ],

            "icb_name":
                validation.loc[
                    index,
                    "icb_name"
                ],

            "actual_ckd_rate":
                validation.loc[
                    index,
                    TARGET
                ],

            "model":
                "ridge_regression",

            "feature_set":
                feature_set_name,

            "alpha":
                best_alpha,

            "prediction":
                prediction,
        })


    estimator = (
        best_model.named_steps[
            "model"
        ]
    )


    for feature, coefficient in zip(
        features,
        estimator.coef_,
    ):

        coefficient_rows.append({
            "model":
                "ridge_regression",

            "feature_set":
                feature_set_name,

            "alpha":
                best_alpha,

            "feature":
                feature,

            "standardised_coefficient":
                coefficient,

            "abs_standardised_coefficient":
                abs(coefficient),
        })


# --------------------------------------------------
# 18. Build result tables
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)

ridge_search_df = pd.DataFrame(
    ridge_search_rows
)

coefficients_df = pd.DataFrame(
    coefficient_rows
)


# --------------------------------------------------
# 19. Persistence benchmark
# --------------------------------------------------

persistence_metrics = calculate_metrics(
    validation[
        TARGET
    ],
    validation[
        LAG_FEATURE
    ],
)


# --------------------------------------------------
# 20. Compare every fitted model with
#     fixed persistence
# --------------------------------------------------

metrics_df[
    "mae_improvement_vs_persistence"
] = (
    persistence_metrics[
        "mae"
    ]
    - metrics_df[
        "mae"
    ]
)


metrics_df[
    "rmse_improvement_vs_persistence"
] = (
    persistence_metrics[
        "rmse"
    ]
    - metrics_df[
        "rmse"
    ]
)


metrics_df[
    "beats_persistence_mae"
] = (
    metrics_df[
        "mae"
    ]
    <
    persistence_metrics[
        "mae"
    ]
)


# --------------------------------------------------
# 21. Compare augmented models with
#     fitted lag-only linear model
# --------------------------------------------------

lag_only_row = (
    metrics_df.loc[
        (
            metrics_df[
                "model"
            ]
            == "linear_regression"
        )
        &
        (
            metrics_df[
                "feature_set"
            ]
            == "lag_only"
        )
    ]
)


if len(lag_only_row) != 1:

    raise RuntimeError(
        "Expected one lag-only "
        "linear model row."
    )


lag_only_mae = float(
    lag_only_row[
        "mae"
    ]
    .iloc[0]
)


metrics_df[
    "mae_improvement_vs_lag_only_linear"
] = (
    lag_only_mae
    - metrics_df[
        "mae"
    ]
)


# --------------------------------------------------
# 22. Round outputs
# --------------------------------------------------

for dataframe in [
    metrics_df,
    ridge_search_df,
    coefficients_df,
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
        .round(4)
    )


predictions_df[
    [
        "actual_ckd_rate",
        "prediction",
    ]
] = (
    predictions_df[
        [
            "actual_ckd_rate",
            "prediction",
        ]
    ]
    .round(4)
)


# --------------------------------------------------
# 23. Display results
# --------------------------------------------------

metrics_df = (
    metrics_df
    .sort_values(
        "mae"
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n========================================"
)

print(
    "AUGMENTED MODEL VALIDATION RESULTS - 2022"
)

print(
    "========================================"
)


print(
    metrics_df[
        [
            "model",
            "feature_set",
            "alpha",
            "feature_count",
            "mae",
            "rmse",
            "r2",
            "mae_improvement_vs_persistence",
            "beats_persistence_mae",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nFixed persistence benchmark:"
)

print(
    f"MAE  = "
    f"{persistence_metrics['mae']:.3f}"
)

print(
    f"RMSE = "
    f"{persistence_metrics['rmse']:.3f}"
)

print(
    f"R2   = "
    f"{persistence_metrics['r2']:.3f}"
)


print(
    "\nFitted lag-only linear model:"
)

print(
    f"MAE = "
    f"{lag_only_mae:.3f}"
)


# --------------------------------------------------
# 24. Best augmented model
# --------------------------------------------------

augmented_only = (
    metrics_df.loc[
        metrics_df[
            "feature_set"
        ]
        != "lag_only"
    ]
)


best_augmented = (
    augmented_only
    .iloc[0]
)


print(
    "\nBest augmented model "
    "by 2022 validation MAE:"
)

print(
    f"Model       : "
    f"{best_augmented['model']}"
)

print(
    f"Feature set : "
    f"{best_augmented['feature_set']}"
)

print(
    f"Alpha       : "
    f"{best_augmented['alpha']}"
)

print(
    f"MAE         : "
    f"{best_augmented['mae']}"
)

print(
    f"RMSE        : "
    f"{best_augmented['rmse']}"
)

print(
    f"R2          : "
    f"{best_augmented['r2']}"
)

print(
    f"MAE gain vs persistence: "
    f"{best_augmented['mae_improvement_vs_persistence']}"
)


# --------------------------------------------------
# 25. Prediction range audit
#
# This helps detect severe extrapolation.
# --------------------------------------------------

prediction_ranges = (
    predictions_df
    .groupby(
        [
            "model",
            "feature_set",
        ]
    )[
        "prediction"
    ]
    .agg(
        [
            "min",
            "mean",
            "max",
        ]
    )
    .round(2)
)


print(
    "\nValidation prediction ranges:"
)

print(
    prediction_ranges.to_string()
)


print(
    "\nActual CKD validation range:"
)

print(
    f"{validation[TARGET].min():.1f} "
    f"to "
    f"{validation[TARGET].max():.1f}"
)


# --------------------------------------------------
# 26. Final-test protection
# --------------------------------------------------

if (
    FINAL_TEST_YEAR
    in model_data[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 entered modelling data."
    )


if (
    FINAL_TEST_YEAR
    in predictions_df[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 predictions were generated."
    )


# The latest CKD year used as a lag predictor
# must be 2021 because our latest evaluated
# target is 2022.
if (
    model_data[
        "lag_source_year"
    ]
    .max()
    != 2021
):

    raise RuntimeError(
        "Unexpected CKD lag source year."
    )


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 27. Final output checks
#
# Linear:
#   lag only
#   lag + compact
#   lag + full
# = 3
#
# Selected Ridge:
#   lag + compact
#   lag + full
# = 2
#
# Total selected models = 5
# --------------------------------------------------

if len(metrics_df) != 5:

    raise RuntimeError(
        f"Expected 5 selected model rows, "
        f"found {len(metrics_df)}."
    )


# 2 augmented Ridge feature sets
# x 6 alpha values
if len(ridge_search_df) != 12:

    raise RuntimeError(
        f"Expected 12 Ridge search rows, "
        f"found {len(ridge_search_df)}."
    )


# 5 models x 42 validation ICBs
if len(predictions_df) != 210:

    raise RuntimeError(
        f"Expected 210 prediction rows, "
        f"found {len(predictions_df)}."
    )


if metrics_df[
    [
        "mae",
        "rmse",
        "r2",
    ]
].isna().any().any():

    raise RuntimeError(
        "Missing model metrics detected."
    )


print(
    "\nAugmented linear-model audit: PASSED"
)


# --------------------------------------------------
# 28. Save outputs
# --------------------------------------------------

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

ridge_search_df.to_csv(
    RIDGE_SEARCH_OUTPUT,
    index=False
)

coefficients_df.to_csv(
    COEFFICIENT_OUTPUT,
    index=False
)


print(
    f"\nSaved metrics to: "
    f"{METRICS_OUTPUT}"
)

print(
    f"Saved predictions to: "
    f"{PREDICTIONS_OUTPUT}"
)

print(
    f"Saved Ridge search to: "
    f"{RIDGE_SEARCH_OUTPUT}"
)

print(
    f"Saved coefficients to: "
    f"{COEFFICIENT_OUTPUT}"
)