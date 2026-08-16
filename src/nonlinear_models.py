from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
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

FEATURE_MANIFEST_FILE = (
    OUTPUT_DIR
    / "feature_set_manifest.csv"
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "nonlinear_validation_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "nonlinear_validation_predictions.csv"
)

IMPORTANCE_OUTPUT = (
    OUTPUT_DIR
    / "nonlinear_feature_importance.csv"
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

RANDOM_STATE = 42


# --------------------------------------------------
# 3. Load files
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
# 4. Standardise identifiers
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
# 5. Recover pre-specified compact feature set
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


if len(PRIMARY_FEATURES) != 5:

    raise RuntimeError(
        f"Expected 5 compact predictors, "
        f"found {len(PRIMARY_FEATURES)}."
    )


print(
    "\nCompact NDA predictors:"
)

for feature in PRIMARY_FEATURES:

    print(
        f"- {feature}"
    )


# --------------------------------------------------
# 6. Restrict modelling to 2020-2022
#
# 2023 is untouched.
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
# 7. Create lagged CKD predictor
#
# target 2020 -> CKD 2019
# target 2021 -> CKD 2020
# target 2022 -> CKD 2021
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
        "Incorrect lagged CKD alignment."
    )


print(
    "\nLagged CKD temporal check: PASSED"
)


# --------------------------------------------------
# 8. Train / validation split
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
# 9. Feature structures
#
# NDA-only:
#     compact 5
#
# Augmented:
#     lagged CKD + compact 5
# --------------------------------------------------

FEATURE_SETS = {

    "compact_5":
        PRIMARY_FEATURES,

    "lag_plus_compact_5":
        [
            LAG_FEATURE
        ]
        + PRIMARY_FEATURES,
}


print(
    "\nFeature-set sizes:"
)

for name, features in FEATURE_SETS.items():

    print(
        f"{name}: "
        f"{len(features)}"
    )


# --------------------------------------------------
# 10. Fixed nonlinear models
#
# Conservative settings are used because:
# - only 84 development observations exist
# - overfitting risk is high
# - this is a limited nonlinear comparison
#
# We are NOT tuning hundreds of combinations.
# --------------------------------------------------

def build_random_forest():

    return RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gradient_boosting():

    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )


MODEL_BUILDERS = {

    "random_forest":
        build_random_forest,

    "gradient_boosting":
        build_gradient_boosting,
}


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
# 12. Fixed persistence benchmark
# --------------------------------------------------

persistence_predictions = (
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
        persistence_predictions,
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
# 13. Result containers
# --------------------------------------------------

metric_rows = []

prediction_rows = []

importance_rows = []


# --------------------------------------------------
# 14. Fit nonlinear models
# --------------------------------------------------

for (
    model_name,
    model_builder,
) in MODEL_BUILDERS.items():

    for (
        feature_set_name,
        features,
    ) in FEATURE_SETS.items():

        print(
            "\n----------------------------------------"
        )

        print(
            f"{model_name}: "
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
            model_builder()
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
                model_name,

            "feature_set":
                feature_set_name,

            "feature_count":
                len(features),

            "train_rows":
                len(train),

            "validation_rows":
                len(validation),

            **metrics,
        })


        print(
            f"MAE  = "
            f"{metrics['mae']:.3f}"
        )

        print(
            f"RMSE = "
            f"{metrics['rmse']:.3f}"
        )

        print(
            f"R2   = "
            f"{metrics['r2']:.3f}"
        )


        # ------------------------------------------
        # Store predictions
        # ------------------------------------------

        for (
            index,
            prediction,
        ) in zip(
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
                    model_name,

                "feature_set":
                    feature_set_name,

                "prediction":
                    prediction,

                "absolute_error":
                    abs(
                        validation.loc[
                            index,
                            TARGET
                        ]
                        - prediction
                    ),
            })


        # ------------------------------------------
        # Store impurity-based feature importance
        #
        # This is descriptive only.
        # SHAP will be performed later on the
        # final selected explainable model.
        # ------------------------------------------

        importances = (
            model.feature_importances_
        )


        if len(
            importances
        ) != len(
            features
        ):

            raise RuntimeError(
                "Feature-importance length mismatch."
            )


        for (
            feature,
            importance,
        ) in zip(
            features,
            importances,
        ):

            importance_rows.append({
                "model":
                    model_name,

                "feature_set":
                    feature_set_name,

                "feature":
                    feature,

                "importance":
                    importance,
            })


# --------------------------------------------------
# 15. Build results
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)

importance_df = pd.DataFrame(
    importance_rows
)


# --------------------------------------------------
# 16. Compare against persistence
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
    "rmse_improvement_vs_persistence"
] = (
    persistence_metrics[
        "rmse"
    ]
    -
    metrics_df[
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
# 17. Add persistence to metrics table
# --------------------------------------------------

persistence_row = pd.DataFrame(
    [
        {
            "model":
                "persistence",

            "feature_set":
                "lag_only",

            "feature_count":
                1,

            "train_rows":
                0,

            "validation_rows":
                len(validation),

            "mae":
                persistence_metrics[
                    "mae"
                ],

            "rmse":
                persistence_metrics[
                    "rmse"
                ],

            "r2":
                persistence_metrics[
                    "r2"
                ],

            "mae_improvement_vs_persistence":
                0.0,

            "rmse_improvement_vs_persistence":
                0.0,

            "beats_persistence_mae":
                False,
        }
    ]
)


metrics_df = pd.concat(
    [
        metrics_df,
        persistence_row,
    ],
    ignore_index=True,
)


# --------------------------------------------------
# 18. Round outputs
# --------------------------------------------------

for dataframe in [
    metrics_df,
    importance_df,
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
        "absolute_error",
    ]
] = (
    predictions_df[
        [
            "actual_ckd_rate",
            "prediction",
            "absolute_error",
        ]
    ]
    .round(4)
)


# --------------------------------------------------
# 19. Display model ranking
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
    "NONLINEAR VALIDATION RESULTS - 2022"
)

print(
    "========================================"
)


print(
    metrics_df[
        [
            "model",
            "feature_set",
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
# 20. Best nonlinear model
# --------------------------------------------------

nonlinear_only = (
    metrics_df.loc[
        metrics_df[
            "model"
        ]
        != "persistence"
    ]
)


best_nonlinear = (
    nonlinear_only
    .iloc[0]
)


print(
    "\nBest nonlinear model "
    "by 2022 validation MAE:"
)

print(
    f"Model       : "
    f"{best_nonlinear['model']}"
)

print(
    f"Feature set : "
    f"{best_nonlinear['feature_set']}"
)

print(
    f"MAE         : "
    f"{best_nonlinear['mae']}"
)

print(
    f"RMSE        : "
    f"{best_nonlinear['rmse']}"
)

print(
    f"R2          : "
    f"{best_nonlinear['r2']}"
)

print(
    f"MAE gain vs persistence: "
    f"{best_nonlinear['mae_improvement_vs_persistence']}"
)


# --------------------------------------------------
# 21. Prediction-range audit
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
    "\nActual validation CKD range:"
)

print(
    f"{validation[TARGET].min():.1f} "
    f"to "
    f"{validation[TARGET].max():.1f}"
)


# --------------------------------------------------
# 22. Feature-importance summary
# --------------------------------------------------

importance_df = (
    importance_df
    .sort_values(
        [
            "model",
            "feature_set",
            "importance",
        ],
        ascending=[
            True,
            True,
            False,
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nTop feature importances:"
)


for (
    model_name,
    feature_set_name,
) in (
    importance_df[
        [
            "model",
            "feature_set",
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None,
    )
):

    subset = (
        importance_df.loc[
            (
                importance_df[
                    "model"
                ]
                == model_name
            )
            &
            (
                importance_df[
                    "feature_set"
                ]
                == feature_set_name
            )
        ]
        .head(6)
    )


    print(
        f"\n{model_name} / "
        f"{feature_set_name}"
    )


    print(
        subset[
            [
                "feature",
                "importance",
            ]
        ]
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 23. Final-test protection
# --------------------------------------------------

if FINAL_TEST_YEAR in set(
    model_data[
        "target_year"
    ]
):

    raise RuntimeError(
        "2023 entered modelling data."
    )


if FINAL_TEST_YEAR in set(
    predictions_df[
        "target_year"
    ]
):

    raise RuntimeError(
        "2023 predictions were generated."
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


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 24. Final validation
#
# Four nonlinear models:
#
# RF compact
# RF lag + compact
# GB compact
# GB lag + compact
#
# + persistence = 5 metric rows
# --------------------------------------------------

if len(metrics_df) != 5:

    raise RuntimeError(
        f"Expected 5 metric rows, "
        f"found {len(metrics_df)}."
    )


# 4 nonlinear models x 42 validation ICBs
if len(predictions_df) != 168:

    raise RuntimeError(
        f"Expected 168 prediction rows, "
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


if importance_df[
    "importance"
].isna().any():

    raise RuntimeError(
        "Missing feature importances detected."
    )


print(
    "\nNonlinear model audit: PASSED"
)


# --------------------------------------------------
# 25. Save outputs
# --------------------------------------------------

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


predictions_df.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)


importance_df.to_csv(
    IMPORTANCE_OUTPUT,
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
    f"Saved feature importance to: "
    f"{IMPORTANCE_OUTPUT}"
)