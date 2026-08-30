from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
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

MODEL_LOCK_FILE = (
    OUTPUT_DIR
    / "final_model_lock.csv"
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "final_test_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "final_model_coefficients.csv"
)


# --------------------------------------------------
# 2. Locked design
# --------------------------------------------------

TRAIN_YEARS = [
    2020,
    2021,
    2022,
]

FINAL_TEST_YEAR = 2023

TARGET = "ckd_risk_rate_per_1000"

LAG_FEATURE = "lagged_ckd_rate"

LOCKED_ALPHA = 1.0


# --------------------------------------------------
# 3. Check required files
# --------------------------------------------------

for file_path in [
    PANEL_FILE,
    CKD_FILE,
    FEATURE_MANIFEST_FILE,
    MODEL_LOCK_FILE,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path}"
        )


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

panel = pd.read_csv(
    PANEL_FILE
)

ckd = pd.read_csv(
    CKD_FILE
)

manifest = pd.read_csv(
    FEATURE_MANIFEST_FILE
)

model_lock = pd.read_csv(
    MODEL_LOCK_FILE
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
# 5. Verify formal model lock
# --------------------------------------------------

if len(model_lock) != 1:
    raise RuntimeError(
        "Expected exactly one model-lock row."
    )


lock = model_lock.iloc[0]


if (
    lock["status"]
    != "LOCKED_BEFORE_FINAL_TEST"
):
    raise RuntimeError(
        "Model lock status is invalid."
    )


if (
    lock["model"]
    != "ridge_regression"
):
    raise RuntimeError(
        "Locked model is not Ridge regression."
    )


if not np.isclose(
    float(lock["alpha"]),
    LOCKED_ALPHA
):
    raise RuntimeError(
        "Locked alpha does not equal 1.0."
    )


if int(
    lock["final_test_target_year"]
) != FINAL_TEST_YEAR:
    raise RuntimeError(
        "Locked final-test year is not 2023."
    )


print(
    "\nFormal model lock: PASSED"
)

print(
    "Model : Ridge regression"
)

print(
    f"Alpha : {LOCKED_ALPHA}"
)


# --------------------------------------------------
# 6. Standardise identifiers
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
# 7. Recover compact 5 features
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


compact_features = (
    manifest.loc[
        manifest[
            "primary_compact_set"
        ]
    ]
    .sort_values(
        "primary_order"
    )[
        "feature"
    ]
    .tolist()
)


if len(compact_features) != 5:
    raise RuntimeError(
        f"Expected 5 compact predictors, "
        f"found {len(compact_features)}."
    )


print(
    "\nLocked compact NDA predictors:"
)

for feature in compact_features:
    print(
        f"- {feature}"
    )


# --------------------------------------------------
# 8. Verify feature lock
# --------------------------------------------------

locked_features = [
    lock[f"nda_feature_{i}"]
    for i in range(1, 6)
]


if compact_features != locked_features:
    raise RuntimeError(
        "Compact feature set differs from "
        "the formal model lock."
    )


MODEL_FEATURES = (
    [LAG_FEATURE]
    + compact_features
)


print(
    "\nFinal model feature count:"
)

print(
    len(MODEL_FEATURES)
)


# --------------------------------------------------
# 9. Build lagged CKD table
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


# --------------------------------------------------
# 10. Build FINAL REFIT data only
#
# IMPORTANT:
# Only target years 2020-2022 are used here.
# --------------------------------------------------

train = (
    panel.loc[
        panel[
            "target_year"
        ]
        .isin(
            TRAIN_YEARS
        )
    ]
    .copy()
)


train = train.merge(
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


if len(train) != 126:
    raise RuntimeError(
        f"Expected 126 final-refit rows, "
        f"found {len(train)}."
    )


if train[
    LAG_FEATURE
].isna().any():
    raise RuntimeError(
        "Missing training lagged CKD values."
    )


if not (
    train[
        "lag_source_year"
    ]
    ==
    train[
        "target_year"
    ]
    - 1
).all():
    raise RuntimeError(
        "Training lag alignment failed."
    )


if set(
    train[
        "target_year"
    ]
) != {
    2020,
    2021,
    2022,
}:
    raise RuntimeError(
        "Incorrect final-refit years."
    )


print(
    "\n========================================"
)

print(
    "FINAL MODEL REFIT"
)

print(
    "========================================"
)

print(
    f"Training target years: "
    f"{TRAIN_YEARS}"
)

print(
    f"Training rows: "
    f"{len(train)}"
)

print(
    f"Unique ICBs: "
    f"{train['icb_code'].nunique()}"
)


# --------------------------------------------------
# 11. Fit LOCKED model
#
# No tuning.
# No feature changes.
# Alpha remains 1.0.
# --------------------------------------------------

final_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge(
            alpha=LOCKED_ALPHA
        )
    ),
])


X_train = train[
    MODEL_FEATURES
]

y_train = train[
    TARGET
]


final_model.fit(
    X_train,
    y_train
)


print(
    "\nLocked Ridge model refitted."
)

print(
    "No 2023 outcome was used for fitting."
)


# --------------------------------------------------
# 12. Recover final model coefficients
# --------------------------------------------------

ridge_step = (
    final_model.named_steps[
        "ridge"
    ]
)


coefficient_df = pd.DataFrame({
    "feature":
        MODEL_FEATURES,

    "standardized_coefficient":
        ridge_step.coef_,
})


coefficient_df[
    "absolute_standardized_coefficient"
] = (
    coefficient_df[
        "standardized_coefficient"
    ]
    .abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "absolute_standardized_coefficient",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# 13. NOW construct untouched 2023 test set
#
# This is the first final-test evaluation.
# --------------------------------------------------

final_test = (
    panel.loc[
        panel[
            "target_year"
        ]
        == FINAL_TEST_YEAR
    ]
    .copy()
)


final_test = final_test.merge(
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


if len(final_test) != 42:
    raise RuntimeError(
        f"Expected 42 final-test rows, "
        f"found {len(final_test)}."
    )


if final_test[
    "icb_code"
].nunique() != 42:
    raise RuntimeError(
        "Expected 42 unique final-test ICBs."
    )


if final_test[
    LAG_FEATURE
].isna().any():
    raise RuntimeError(
        "Missing 2023 lagged CKD values."
    )


if not (
    final_test[
        "lag_source_year"
    ]
    == 2022
).all():
    raise RuntimeError(
        "2023 forecast must use 2022 CKD lag."
    )


print(
    "\n========================================"
)

print(
    "2023 FINAL TEST OPENED"
)

print(
    "========================================"
)

print(
    f"Final-test rows: "
    f"{len(final_test)}"
)

print(
    f"Unique ICBs: "
    f"{final_test['icb_code'].nunique()}"
)

print(
    "Lag source year: 2022"
)


# --------------------------------------------------
# 14. Generate locked model predictions
# --------------------------------------------------

X_test = final_test[
    MODEL_FEATURES
]

y_test = final_test[
    TARGET
]


ridge_predictions = (
    final_model.predict(
        X_test
    )
)


# --------------------------------------------------
# 15. Persistence predictions
# --------------------------------------------------

persistence_predictions = (
    final_test[
        LAG_FEATURE
    ]
    .to_numpy()
)


# --------------------------------------------------
# 16. Metric helper
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
# 17. Final metrics
# --------------------------------------------------

ridge_metrics = (
    calculate_metrics(
        y_test,
        ridge_predictions,
    )
)


persistence_metrics = (
    calculate_metrics(
        y_test,
        persistence_predictions,
    )
)


mae_gain = (
    persistence_metrics[
        "mae"
    ]
    -
    ridge_metrics[
        "mae"
    ]
)


mae_gain_percent = (
    mae_gain
    /
    persistence_metrics[
        "mae"
    ]
    *
    100
)


rmse_gain = (
    persistence_metrics[
        "rmse"
    ]
    -
    ridge_metrics[
        "rmse"
    ]
)


# --------------------------------------------------
# 18. Build metrics table
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "evaluation_year":
            FINAL_TEST_YEAR,

        "model":
            "persistence",

        "feature_set":
            "previous_year_ckd_only",

        "alpha":
            np.nan,

        "train_rows":
            0,

        "test_rows":
            len(final_test),

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
    },

    {
        "evaluation_year":
            FINAL_TEST_YEAR,

        "model":
            "ridge_regression",

        "feature_set":
            "lag_plus_compact_5",

        "alpha":
            LOCKED_ALPHA,

        "train_rows":
            len(train),

        "test_rows":
            len(final_test),

        "mae":
            ridge_metrics[
                "mae"
            ],

        "rmse":
            ridge_metrics[
                "rmse"
            ],

        "r2":
            ridge_metrics[
                "r2"
            ],

        "mae_improvement_vs_persistence":
            mae_gain,

        "rmse_improvement_vs_persistence":
            rmse_gain,
    },
])


# --------------------------------------------------
# 19. Build final predictions table
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "target_year":
        FINAL_TEST_YEAR,

    "icb_code":
        final_test[
            "icb_code"
        ].values,

    "icb_name":
        final_test[
            "icb_name"
        ].values,

    "lag_source_year":
        final_test[
            "lag_source_year"
        ].values,

    "previous_year_ckd_rate":
        persistence_predictions,

    "actual_ckd_rate":
        y_test.values,

    "persistence_prediction":
        persistence_predictions,

    "ridge_prediction":
        ridge_predictions,
})


predictions_df[
    "persistence_absolute_error"
] = (
    predictions_df[
        "actual_ckd_rate"
    ]
    -
    predictions_df[
        "persistence_prediction"
    ]
).abs()


predictions_df[
    "ridge_absolute_error"
] = (
    predictions_df[
        "actual_ckd_rate"
    ]
    -
    predictions_df[
        "ridge_prediction"
    ]
).abs()


predictions_df[
    "ridge_error_improvement"
] = (
    predictions_df[
        "persistence_absolute_error"
    ]
    -
    predictions_df[
        "ridge_absolute_error"
    ]
)


predictions_df[
    "ridge_better_than_persistence"
] = (
    predictions_df[
        "ridge_absolute_error"
    ]
    <
    predictions_df[
        "persistence_absolute_error"
    ]
)


# --------------------------------------------------
# 20. ICB-level comparison
# --------------------------------------------------

ridge_better_count = int(
    predictions_df[
        "ridge_better_than_persistence"
    ]
    .sum()
)


median_error_gain = float(
    predictions_df[
        "ridge_error_improvement"
    ]
    .median()
)


# --------------------------------------------------
# 21. Round saved outputs
# --------------------------------------------------

for dataframe in [
    metrics_df,
    predictions_df,
    coefficient_df,
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


# --------------------------------------------------
# 22. Print FINAL result
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL OUT-OF-SAMPLE TEST - 2023"
)

print(
    "========================================"
)


print(
    "\nPersistence:"
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
    "\nLocked Ridge:"
)

print(
    f"MAE  = "
    f"{ridge_metrics['mae']:.3f}"
)

print(
    f"RMSE = "
    f"{ridge_metrics['rmse']:.3f}"
)

print(
    f"R2   = "
    f"{ridge_metrics['r2']:.3f}"
)


print(
    "\nMAE improvement vs persistence:"
)

print(
    f"{mae_gain:.3f} "
    f"({mae_gain_percent:.2f}%)"
)


print(
    "\nICB-level comparison:"
)

print(
    f"Ridge better in "
    f"{ridge_better_count}/42 ICBs"
)

print(
    f"Median absolute-error improvement: "
    f"{median_error_gain:.3f}"
)


# --------------------------------------------------
# 23. Prediction range audit
# --------------------------------------------------

print(
    "\nPrediction ranges:"
)

print(
    f"Actual 2023:"
    f" {y_test.min():.2f}"
    f" to {y_test.max():.2f}"
)

print(
    f"Persistence:"
    f" {persistence_predictions.min():.2f}"
    f" to {persistence_predictions.max():.2f}"
)

print(
    f"Locked Ridge:"
    f" {ridge_predictions.min():.2f}"
    f" to {ridge_predictions.max():.2f}"
)


# --------------------------------------------------
# 24. Standardized Ridge coefficients
# --------------------------------------------------

print(
    "\nFinal standardized Ridge coefficients:"
)

print(
    coefficient_df[
        [
            "feature",
            "standardized_coefficient",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 25. Evidence-based interpretation
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL INTERPRETATION"
)

print(
    "========================================"
)


if (
    ridge_metrics["mae"]
    < persistence_metrics["mae"]
):

    print(
        "The locked Ridge model outperformed "
        "persistence on the untouched 2023 "
        "final test by MAE."
    )

    print(
        "This provides final-test evidence "
        "that the compact NDA indicators add "
        "incremental predictive value beyond "
        "previous-year CKD burden."
    )

else:

    print(
        "The locked Ridge model did not "
        "outperform persistence on the "
        "untouched 2023 final test."
    )

    print(
        "The evidence therefore does not "
        "support a claim that the compact NDA "
        "indicators consistently add predictive "
        "value beyond previous-year CKD burden."
    )


print(
    "\nNo model, feature or alpha changes "
    "will be made in response to this result."
)


# --------------------------------------------------
# 26. Final integrity checks
# --------------------------------------------------

if len(
    metrics_df
) != 2:
    raise RuntimeError(
        "Expected exactly two final metric rows."
    )


if len(
    predictions_df
) != 42:
    raise RuntimeError(
        "Expected exactly 42 final predictions."
    )


if predictions_df[
    [
        "actual_ckd_rate",
        "persistence_prediction",
        "ridge_prediction",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing final-test values detected."
    )


if not (
    predictions_df[
        "lag_source_year"
    ]
    == 2022
).all():
    raise RuntimeError(
        "Final-test lag source must be 2022."
    )


print(
    "\nFinal-test integrity checks: PASSED"
)


# --------------------------------------------------
# 27. Save outputs
# --------------------------------------------------

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


predictions_df.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)


coefficient_df.to_csv(
    COEFFICIENT_OUTPUT,
    index=False
)


print(
    f"\nSaved final metrics to: "
    f"{METRICS_OUTPUT}"
)

print(
    f"Saved final predictions to: "
    f"{PREDICTIONS_OUTPUT}"
)

print(
    f"Saved final coefficients to: "
    f"{COEFFICIENT_OUTPUT}"
)