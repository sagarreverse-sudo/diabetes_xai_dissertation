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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ckd_forecasting_master.csv"
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
    / "linear_model_validation_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "linear_model_validation_predictions.csv"
)

RIDGE_SEARCH_OUTPUT = (
    OUTPUT_DIR
    / "ridge_alpha_search.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "linear_model_coefficients.csv"
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

TARGET = (
    "ckd_risk_rate_per_1000"
)


# --------------------------------------------------
# 3. Pre-specified Ridge alpha values
#
# We use a small transparent grid rather than
# a huge automated search.
#
# 2022 may be used for hyperparameter selection
# according to our locked validation protocol.
# --------------------------------------------------

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

for required_file in [
    PANEL_FILE,
    FEATURE_MANIFEST_FILE,
    BASELINE_METRICS_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found: "
            f"{required_file}"
        )


panel = pd.read_csv(
    PANEL_FILE
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


# --------------------------------------------------
# 5. Protect final test year
#
# We retain the full panel in memory only long
# enough to validate its structure.
#
# Model fitting and prediction below use only
# 2020, 2021 and 2022.
# --------------------------------------------------

if FINAL_TEST_YEAR not in set(
    panel["target_year"]
):

    raise RuntimeError(
        "Expected final test year 2023 "
        "is missing from forecasting panel."
    )


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
# 6. Recover locked feature sets from manifest
#
# This avoids redefining/changing the feature
# strategy after seeing model results.
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


if (
    feature_manifest[
        "primary_compact_set"
    ]
    .isna()
    .any()
):

    raise RuntimeError(
        "Could not interpret "
        "primary_compact_set values."
    )


# Compact features in locked order
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


# Full feature set retains manifest order
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
        f"Expected 5 primary features, "
        f"found {len(PRIMARY_FEATURES)}."
    )


if len(FULL_FEATURES) != 19:

    raise RuntimeError(
        f"Expected 19 full features, "
        f"found {len(FULL_FEATURES)}."
    )


print(
    f"\nPrimary features: "
    f"{len(PRIMARY_FEATURES)}"
)

for feature in PRIMARY_FEATURES:
    print(
        f"- {feature}"
    )


print(
    f"\nFull sensitivity features: "
    f"{len(FULL_FEATURES)}"
)


# --------------------------------------------------
# 7. Validate feature columns
# --------------------------------------------------

all_required_features = set(
    FULL_FEATURES
)


missing_features = (
    all_required_features
    - set(
        model_data.columns
    )
)


if missing_features:

    raise RuntimeError(
        "Missing model features: "
        f"{sorted(missing_features)}"
    )


if (
    model_data[
        FULL_FEATURES
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "Missing feature values detected."
    )


# --------------------------------------------------
# 8. Development and validation datasets
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
    "\nTraining rows: "
    f"{len(train)}"
)

print(
    "Validation rows: "
    f"{len(validation)}"
)

print(
    "Final-test rows accessed: 0"
)


# --------------------------------------------------
# 9. Metric helper
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
# 10. Model helper
#
# StandardScaler is fitted ONLY on training data.
#
# Pipeline prevents scaling information from
# leaking from validation into training.
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


def build_ridge_model(
    alpha,
):

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
# 11. Feature-set definitions
# --------------------------------------------------

FEATURE_SETS = {

    "compact_5":
        PRIMARY_FEATURES,

    "full_19":
        FULL_FEATURES,
}


# --------------------------------------------------
# 12. Containers for results
# --------------------------------------------------

metric_rows = []

prediction_rows = []

ridge_search_rows = []

fitted_models = {}


# --------------------------------------------------
# 13. Fit ordinary linear regression
# --------------------------------------------------

for feature_set_name, features in FEATURE_SETS.items():

    print(
        "\n----------------------------------------"
    )

    print(
        f"Linear Regression: "
        f"{feature_set_name}"
    )

    print(
        "----------------------------------------"
    )


    X_train = (
        train[
            features
        ]
    )

    y_train = (
        train[
            TARGET
        ]
    )


    X_validation = (
        validation[
            features
        ]
    )

    y_validation = (
        validation[
            TARGET
        ]
    )


    model = (
        build_linear_model()
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = (
        model.predict(
            X_validation
        )
    )


    metrics = (
        calculate_metrics(
            y_validation,
            predictions,
        )
    )


    metric_rows.append({
        "model":
            "linear_regression",

        "feature_set":
            feature_set_name,

        "alpha":
            np.nan,

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        **metrics,
    })


    fitted_models[
        (
            "linear_regression",
            feature_set_name
        )
    ] = model


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


# --------------------------------------------------
# 14. Ridge alpha search
#
# Each alpha is trained using 2020+2021 only
# and evaluated on 2022.
#
# 2023 is never accessed.
# --------------------------------------------------

best_ridge_models = {}


for feature_set_name, features in FEATURE_SETS.items():

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


    X_train = (
        train[
            features
        ]
    )

    y_train = (
        train[
            TARGET
        ]
    )


    X_validation = (
        validation[
            features
        ]
    )

    y_validation = (
        validation[
            TARGET
        ]
    )


    alpha_results = []


    for alpha in RIDGE_ALPHAS:

        model = (
            build_ridge_model(
                alpha
            )
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = (
            model.predict(
                X_validation
            )
        )


        metrics = (
            calculate_metrics(
                y_validation,
                predictions,
            )
        )


        result = {
            "feature_set":
                feature_set_name,

            "alpha":
                alpha,

            **metrics,
        }


        ridge_search_rows.append(
            result
        )


        alpha_results.append(
            (
                alpha,
                metrics,
                model,
                predictions,
            )
        )


        print(
            f"alpha={alpha:<6} "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.3f}"
        )


    # ----------------------------------------------
    # Choose alpha using validation MAE
    # ----------------------------------------------

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

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        **best_metrics,
    })


    best_ridge_models[
        feature_set_name
    ] = (
        best_model
    )


    fitted_models[
        (
            "ridge_regression",
            feature_set_name
        )
    ] = (
        best_model
    )


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


# --------------------------------------------------
# 15. Build result tables
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


# --------------------------------------------------
# 16. Add persistence benchmark
#
# Read existing baseline result rather than
# recalculating the final-test year.
# --------------------------------------------------

persistence_2022 = (
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


if len(
    persistence_2022
) != 1:

    raise RuntimeError(
        "Expected exactly one 2022 "
        "persistence baseline row."
    )


persistence_mae = float(
    persistence_2022[
        "mae"
    ]
    .iloc[0]
)

persistence_rmse = float(
    persistence_2022[
        "rmse"
    ]
    .iloc[0]
)

persistence_r2 = float(
    persistence_2022[
        "r2"
    ]
    .iloc[0]
)


# --------------------------------------------------
# 17. Compare models with persistence
# --------------------------------------------------

metrics_df[
    "mae_improvement_vs_persistence"
] = (
    persistence_mae
    - metrics_df[
        "mae"
    ]
)


metrics_df[
    "beats_persistence_mae"
] = (
    metrics_df[
        "mae"
    ]
    < persistence_mae
)


# --------------------------------------------------
# 18. Extract standardised coefficients
#
# Because predictors are scaled before fitting,
# these coefficients are comparable in terms
# of a one-standard-deviation change in predictor.
#
# They are NOT causal effects.
# --------------------------------------------------

coefficient_rows = []


for (
    model_name,
    feature_set_name,
), model in fitted_models.items():

    features = (
        FEATURE_SETS[
            feature_set_name
        ]
    )


    estimator = (
        model.named_steps[
            "model"
        ]
    )


    coefficients = (
        estimator.coef_
    )


    if len(
        coefficients
    ) != len(
        features
    ):

        raise RuntimeError(
            "Coefficient count does not "
            "match feature count."
        )


    if model_name == "ridge_regression":

        selected_alpha = float(
            metrics_df.loc[
                (
                    metrics_df[
                        "model"
                    ]
                    == model_name
                )
                &
                (
                    metrics_df[
                        "feature_set"
                    ]
                    == feature_set_name
                ),
                "alpha"
            ]
            .iloc[0]
        )

    else:

        selected_alpha = np.nan


    for feature, coefficient in zip(
        features,
        coefficients,
    ):

        coefficient_rows.append({
            "model":
                model_name,

            "feature_set":
                feature_set_name,

            "alpha":
                selected_alpha,

            "feature":
                feature,

            "standardised_coefficient":
                coefficient,

            "abs_standardised_coefficient":
                abs(
                    coefficient
                ),
        })


coefficients_df = pd.DataFrame(
    coefficient_rows
)


# --------------------------------------------------
# 19. Round display/output values
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
# 20. Display validation results
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
    "LINEAR MODEL VALIDATION RESULTS - 2022"
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
    "\nPersistence benchmark:"
)

print(
    f"MAE  = {persistence_mae:.3f}"
)

print(
    f"RMSE = {persistence_rmse:.3f}"
)

print(
    f"R2   = {persistence_r2:.3f}"
)


# --------------------------------------------------
# 21. Identify current best linear-family model
# --------------------------------------------------

best_model_row = (
    metrics_df
    .iloc[0]
)


print(
    "\nBest linear-family model "
    "by 2022 validation MAE:"
)

print(
    f"Model       : "
    f"{best_model_row['model']}"
)

print(
    f"Feature set : "
    f"{best_model_row['feature_set']}"
)

print(
    f"Alpha       : "
    f"{best_model_row['alpha']}"
)

print(
    f"MAE         : "
    f"{best_model_row['mae']}"
)

print(
    f"RMSE        : "
    f"{best_model_row['rmse']}"
)

print(
    f"R2          : "
    f"{best_model_row['r2']}"
)


# --------------------------------------------------
# 22. Explicit final-test protection
# --------------------------------------------------

if (
    FINAL_TEST_YEAR
    in train[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 entered training data."
    )


if (
    FINAL_TEST_YEAR
    in validation[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 entered validation data."
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


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 23. Final checks
# --------------------------------------------------

# 2 Linear models + 2 selected Ridge models
if len(metrics_df) != 4:

    raise RuntimeError(
        f"Expected 4 selected model rows, "
        f"found {len(metrics_df)}."
    )


# 5 alphas × 2 feature sets
if len(ridge_search_df) != 10:

    raise RuntimeError(
        f"Expected 10 Ridge search rows, "
        f"found {len(ridge_search_df)}."
    )


# 4 selected models × 42 validation predictions
if len(predictions_df) != 168:

    raise RuntimeError(
        f"Expected 168 validation prediction rows, "
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
    "\nLinear/Ridge model audit: PASSED"
)


# --------------------------------------------------
# 24. Save outputs
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
    f"\nSaved model metrics to: "
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