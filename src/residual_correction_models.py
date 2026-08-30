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

CKD_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)

FEATURE_MANIFEST_FILE = (
    OUTPUT_DIR
    / "feature_set_manifest.csv"
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "residual_correction_validation_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "residual_correction_validation_predictions.csv"
)

RIDGE_SEARCH_OUTPUT = (
    OUTPUT_DIR
    / "residual_correction_ridge_search.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "residual_correction_coefficients.csv"
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

RESIDUAL_TARGET = "ckd_change_from_previous_year"


# --------------------------------------------------
# 3. Ridge alpha grid
#
# 2022 is our designated validation year,
# so it may be used for alpha selection.
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
# 4. Load required files
# --------------------------------------------------

for file_path in [
    PANEL_FILE,
    CKD_FILE,
    FEATURE_MANIFEST_FILE,
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

manifest = pd.read_csv(
    FEATURE_MANIFEST_FILE
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
# 6. Recover pre-specified feature sets
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


manifest[
    "primary_compact_set"
] = to_boolean(
    manifest[
        "primary_compact_set"
    ]
)


manifest[
    "full_sensitivity_set"
] = to_boolean(
    manifest[
        "full_sensitivity_set"
    ]
)


primary_manifest = (
    manifest.loc[
        manifest[
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
    manifest.loc[
        manifest[
            "full_sensitivity_set"
        ],
        "feature"
    ]
    .tolist()
)


if len(PRIMARY_FEATURES) != 5:

    raise RuntimeError(
        f"Expected 5 compact features, "
        f"found {len(PRIMARY_FEATURES)}."
    )


if len(FULL_FEATURES) != 19:

    raise RuntimeError(
        f"Expected 19 full features, "
        f"found {len(FULL_FEATURES)}."
    )


print(
    f"\nCompact predictors: "
    f"{len(PRIMARY_FEATURES)}"
)

print(
    f"Full sensitivity predictors: "
    f"{len(FULL_FEATURES)}"
)


# --------------------------------------------------
# 7. Restrict modelling to pre-final years
# --------------------------------------------------

model_data = (
    panel.loc[
        panel[
            "target_year"
        ]
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
    model_data[
        "target_year"
    ]
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


# --------------------------------------------------
# 8. Create previous-year CKD predictor
# --------------------------------------------------

lag_source = (
    ckd[
        [
            "icb_code",
            "year",
            TARGET,
        ]
    ]
    .copy()
)


lag_source[
    "target_year"
] = (
    lag_source["year"]
    + 1
)


lag_source = lag_source.rename(
    columns={
        "year":
            "lag_source_year",

        TARGET:
            LAG_FEATURE,
    }
)


lag_source = (
    lag_source.loc[
        lag_source[
            "target_year"
        ]
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
        "Missing lagged CKD values."
    )


if not (
    model_data[
        "lag_source_year"
    ]
    ==
    model_data[
        "target_year"
    ]
    - 1
).all():

    raise RuntimeError(
        "Incorrect lag-year alignment."
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


# --------------------------------------------------
# 9. Define residual/change target
#
# Actual CKD(t) - CKD(t-1)
# --------------------------------------------------

model_data[
    RESIDUAL_TARGET
] = (
    model_data[
        TARGET
    ]
    -
    model_data[
        LAG_FEATURE
    ]
)


print(
    "\nResidual target summary by year:"
)


print(
    model_data
    .groupby(
        "target_year"
    )[
        RESIDUAL_TARGET
    ]
    .agg(
        [
            "mean",
            "std",
            "min",
            "median",
            "max",
        ]
    )
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 10. Development / validation split
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
# 11. Metrics
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
# 12. Model builders
# --------------------------------------------------

def build_linear():

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


def build_ridge(alpha):

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
# 13. Feature sets
#
# Residual correction uses NDA features only.
#
# Lagged CKD is NOT an input to the correction
# model because persistence is already added
# explicitly afterwards.
# --------------------------------------------------

FEATURE_SETS = {

    "compact_5":
        PRIMARY_FEATURES,

    "full_19":
        FULL_FEATURES,
}


# --------------------------------------------------
# 14. Fixed persistence benchmark
#
# Correction = zero.
# --------------------------------------------------

persistence_prediction = (
    validation[
        LAG_FEATURE
    ]
    .to_numpy()
)


persistence_metrics = (
    calculate_metrics(
        validation[
            TARGET
        ],
        persistence_prediction,
    )
)


print(
    "\nPersistence benchmark:"
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


# --------------------------------------------------
# 15. Mean-change benchmark
#
# Predict one constant correction equal to the
# average residual observed in training data.
# --------------------------------------------------

training_mean_change = (
    train[
        RESIDUAL_TARGET
    ]
    .mean()
)


mean_change_prediction = (
    validation[
        LAG_FEATURE
    ]
    +
    training_mean_change
)


mean_change_metrics = (
    calculate_metrics(
        validation[
            TARGET
        ],
        mean_change_prediction,
    )
)


print(
    "\nTraining mean residual correction:"
)

print(
    f"{training_mean_change:.3f}"
)


print(
    "\nMean-correction benchmark:"
)

print(
    f"MAE  = "
    f"{mean_change_metrics['mae']:.3f}"
)

print(
    f"RMSE = "
    f"{mean_change_metrics['rmse']:.3f}"
)

print(
    f"R2   = "
    f"{mean_change_metrics['r2']:.3f}"
)


# --------------------------------------------------
# 16. Result containers
# --------------------------------------------------

metric_rows = []
prediction_rows = []
ridge_search_rows = []
coefficient_rows = []


# --------------------------------------------------
# 17. Add fixed benchmarks to results
# --------------------------------------------------

metric_rows.append({
    "model":
        "persistence",

    "feature_set":
        "none",

    "alpha":
        np.nan,

    "feature_count":
        0,

    "mae":
        persistence_metrics["mae"],

    "rmse":
        persistence_metrics["rmse"],

    "r2":
        persistence_metrics["r2"],
})


metric_rows.append({
    "model":
        "mean_residual_correction",

    "feature_set":
        "none",

    "alpha":
        np.nan,

    "feature_count":
        0,

    "mae":
        mean_change_metrics["mae"],

    "rmse":
        mean_change_metrics["rmse"],

    "r2":
        mean_change_metrics["r2"],
})


# --------------------------------------------------
# 18. Ordinary residual linear models
# --------------------------------------------------

for feature_set_name, features in FEATURE_SETS.items():

    X_train = train[
        features
    ]

    y_train = train[
        RESIDUAL_TARGET
    ]

    X_validation = validation[
        features
    ]


    model = build_linear()


    model.fit(
        X_train,
        y_train
    )


    predicted_change = (
        model.predict(
            X_validation
        )
    )


    final_prediction = (
        validation[
            LAG_FEATURE
        ]
        .to_numpy()
        +
        predicted_change
    )


    metrics = (
        calculate_metrics(
            validation[
                TARGET
            ],
            final_prediction,
        )
    )


    metric_rows.append({
        "model":
            "residual_linear",

        "feature_set":
            feature_set_name,

        "alpha":
            np.nan,

        "feature_count":
            len(features),

        **metrics,
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
                "residual_linear",

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


    for index, predicted_delta, prediction in zip(
        validation.index,
        predicted_change,
        final_prediction,
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

            "lagged_ckd_rate":
                validation.loc[
                    index,
                    LAG_FEATURE
                ],

            "actual_change":
                validation.loc[
                    index,
                    RESIDUAL_TARGET
                ],

            "predicted_change":
                predicted_delta,

            "actual_ckd_rate":
                validation.loc[
                    index,
                    TARGET
                ],

            "model":
                "residual_linear",

            "feature_set":
                feature_set_name,

            "alpha":
                np.nan,

            "prediction":
                prediction,
        })


# --------------------------------------------------
# 19. Ridge residual models
# --------------------------------------------------

for feature_set_name, features in FEATURE_SETS.items():

    print(
        "\n----------------------------------------"
    )

    print(
        f"Residual Ridge search: "
        f"{feature_set_name}"
    )

    print(
        "----------------------------------------"
    )


    X_train = train[
        features
    ]

    y_train = train[
        RESIDUAL_TARGET
    ]

    X_validation = validation[
        features
    ]


    alpha_results = []


    for alpha in RIDGE_ALPHAS:

        model = build_ridge(
            alpha
        )


        model.fit(
            X_train,
            y_train
        )


        predicted_change = (
            model.predict(
                X_validation
            )
        )


        final_prediction = (
            validation[
                LAG_FEATURE
            ]
            .to_numpy()
            +
            predicted_change
        )


        metrics = (
            calculate_metrics(
                validation[
                    TARGET
                ],
                final_prediction,
            )
        )


        ridge_search_rows.append({
            "feature_set":
                feature_set_name,

            "alpha":
                alpha,

            "mae":
                metrics["mae"],

            "rmse":
                metrics["rmse"],

            "r2":
                metrics["r2"],
        })


        alpha_results.append(
            (
                alpha,
                metrics,
                model,
                predicted_change,
                final_prediction,
            )
        )


        print(
            f"alpha={alpha:<7} "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.3f}"
        )


    # ----------------------------------------------
    # Select best alpha by validation MAE
    # ----------------------------------------------

    best_result = min(
        alpha_results,
        key=lambda result:
            result[1]["mae"]
    )


    (
        best_alpha,
        best_metrics,
        best_model,
        best_predicted_change,
        best_final_prediction,
    ) = best_result


    print(
        f"\nBest alpha for "
        f"{feature_set_name}: "
        f"{best_alpha}"
    )


    metric_rows.append({
        "model":
            "residual_ridge",

        "feature_set":
            feature_set_name,

        "alpha":
            best_alpha,

        "feature_count":
            len(features),

        **best_metrics,
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
                "residual_ridge",

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


    for (
        index,
        predicted_delta,
        prediction,
    ) in zip(
        validation.index,
        best_predicted_change,
        best_final_prediction,
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

            "lagged_ckd_rate":
                validation.loc[
                    index,
                    LAG_FEATURE
                ],

            "actual_change":
                validation.loc[
                    index,
                    RESIDUAL_TARGET
                ],

            "predicted_change":
                predicted_delta,

            "actual_ckd_rate":
                validation.loc[
                    index,
                    TARGET
                ],

            "model":
                "residual_ridge",

            "feature_set":
                feature_set_name,

            "alpha":
                best_alpha,

            "prediction":
                prediction,
        })


# --------------------------------------------------
# 20. Build output dataframes
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
# 21. Compare against persistence
# --------------------------------------------------

metrics_df[
    "mae_improvement_vs_persistence"
] = (
    persistence_metrics[
        "mae"
    ]
    -
    metrics_df[
        "mae"
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


if len(predictions_df) > 0:

    prediction_numeric = [
        "lagged_ckd_rate",
        "actual_change",
        "predicted_change",
        "actual_ckd_rate",
        "prediction",
    ]


    predictions_df[
        prediction_numeric
    ] = (
        predictions_df[
            prediction_numeric
        ]
        .round(4)
    )


# --------------------------------------------------
# 23. Display validation results
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
    "RESIDUAL-CORRECTION VALIDATION - 2022"
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


# --------------------------------------------------
# 24. Best learned correction model
# --------------------------------------------------

learned_models = (
    metrics_df.loc[
        metrics_df[
            "model"
        ]
        .isin(
            [
                "residual_linear",
                "residual_ridge",
            ]
        )
    ]
)


best_learned = (
    learned_models
    .iloc[0]
)


print(
    "\nBest learned residual-correction model:"
)

print(
    f"Model       : "
    f"{best_learned['model']}"
)

print(
    f"Feature set : "
    f"{best_learned['feature_set']}"
)

print(
    f"Alpha       : "
    f"{best_learned['alpha']}"
)

print(
    f"MAE         : "
    f"{best_learned['mae']}"
)

print(
    f"RMSE        : "
    f"{best_learned['rmse']}"
)

print(
    f"R2          : "
    f"{best_learned['r2']}"
)

print(
    f"MAE gain vs persistence: "
    f"{best_learned['mae_improvement_vs_persistence']}"
)


# --------------------------------------------------
# 25. Prediction-range audit
# --------------------------------------------------

if len(predictions_df) > 0:

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
    "\nActual validation CKD range:"
)

print(
    f"{validation[TARGET].min():.1f} "
    f"to "
    f"{validation[TARGET].max():.1f}"
)


# --------------------------------------------------
# 26. Predicted residual/change ranges
# --------------------------------------------------

if len(predictions_df) > 0:

    change_ranges = (
        predictions_df
        .groupby(
            [
                "model",
                "feature_set",
            ]
        )[
            "predicted_change"
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
        "\nPredicted CKD corrections:"
    )

    print(
        change_ranges.to_string()
    )


print(
    "\nActual 2022 CKD change summary:"
)

print(
    validation[
        RESIDUAL_TARGET
    ]
    .describe()
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 27. Final-test protection
# --------------------------------------------------

if FINAL_TEST_YEAR in set(
    model_data[
        "target_year"
    ]
):
    raise RuntimeError(
        "2023 entered modelling data."
    )


if (
    model_data[
        "lag_source_year"
    ]
    .max()
    != 2021
):
    raise RuntimeError(
        "Unexpected latest lag source year."
    )


if (
    len(predictions_df) > 0
    and FINAL_TEST_YEAR
    in set(
        predictions_df[
            "target_year"
        ]
    )
):
    raise RuntimeError(
        "2023 predictions were generated."
    )


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 28. Final validations
#
# 2 benchmarks
# + 2 OLS residual models
# + 2 selected Ridge residual models
# = 6 metric rows
# --------------------------------------------------

if len(metrics_df) != 6:

    raise RuntimeError(
        f"Expected 6 metric rows, "
        f"found {len(metrics_df)}."
    )


# 2 feature sets × 6 alphas
if len(ridge_search_df) != 12:

    raise RuntimeError(
        f"Expected 12 Ridge search rows, "
        f"found {len(ridge_search_df)}."
    )


# Four learned models × 42 ICBs
if len(predictions_df) != 168:

    raise RuntimeError(
        f"Expected 168 model predictions, "
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
        "Missing validation metrics detected."
    )


print(
    "\nResidual-correction model audit: PASSED"
)


# --------------------------------------------------
# 29. Save outputs
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