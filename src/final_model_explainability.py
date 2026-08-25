from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ImportError:
    raise RuntimeError(
        "SHAP is not installed in the active virtual environment."
    )


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

MODEL_LOCK_FILE = (
    OUTPUT_DIR
    / "final_model_lock.csv"
)

FINAL_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)


GLOBAL_OUTPUT = (
    OUTPUT_DIR
    / "final_model_shap_global.csv"
)

LOCAL_OUTPUT = (
    OUTPUT_DIR
    / "final_model_shap_2023.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "final_model_xai_coefficients.csv"
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


# --------------------------------------------------
# 3. Load required files
# --------------------------------------------------

for file_path in [
    PANEL_FILE,
    CKD_FILE,
    MODEL_LOCK_FILE,
    FINAL_PREDICTIONS_FILE,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


panel = pd.read_csv(
    PANEL_FILE
)

ckd = pd.read_csv(
    CKD_FILE
)

model_lock = pd.read_csv(
    MODEL_LOCK_FILE
)

saved_predictions = pd.read_csv(
    FINAL_PREDICTIONS_FILE
)


print(
    f"Forecasting panel shape: {panel.shape}"
)

print(
    f"CKD history shape: {ckd.shape}"
)


# --------------------------------------------------
# 4. Recover formal model lock
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
        "Invalid model-lock status."
    )


if (
    lock["model"]
    != "ridge_regression"
):
    raise RuntimeError(
        "Locked model is not Ridge regression."
    )


LOCKED_ALPHA = float(
    lock["alpha"]
)


compact_features = [
    lock[f"nda_feature_{i}"]
    for i in range(1, 6)
]


MODEL_FEATURES = (
    [LAG_FEATURE]
    + compact_features
)


if len(MODEL_FEATURES) != 6:
    raise RuntimeError(
        "Expected six locked predictors."
    )


print(
    "\nFormal model lock: PASSED"
)

print(
    f"Locked alpha: {LOCKED_ALPHA}"
)

print(
    "\nLocked model predictors:"
)

for feature in MODEL_FEATURES:
    print(
        f"- {feature}"
    )


# --------------------------------------------------
# 5. Standardise identifiers
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

saved_predictions["icb_code"] = (
    saved_predictions["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 6. Construct lagged CKD
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
# 7. Final refit data
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
    validate="one_to_one",
)


if len(train) != 126:
    raise RuntimeError(
        f"Expected 126 training rows, "
        f"found {len(train)}."
    )


if train[
    MODEL_FEATURES
].isna().any().any():
    raise RuntimeError(
        "Missing training predictors detected."
    )


# --------------------------------------------------
# 8. Final 2023 explanation set
# --------------------------------------------------

test = (
    panel.loc[
        panel[
            "target_year"
        ]
        == FINAL_TEST_YEAR
    ]
    .copy()
)


test = test.merge(
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
    validate="one_to_one",
)


if len(test) != 42:
    raise RuntimeError(
        f"Expected 42 final-test rows, "
        f"found {len(test)}."
    )


if not (
    test[
        "lag_source_year"
    ]
    == 2022
).all():
    raise RuntimeError(
        "2023 explanations must use "
        "2022 lagged CKD."
    )


print(
    "\nFinal-refit rows: 126"
)

print(
    "2023 explanation rows: 42"
)


# --------------------------------------------------
# 9. Refit exact locked model
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


X_test = test[
    MODEL_FEATURES
]


final_model.fit(
    X_train,
    y_train
)


refit_predictions = (
    final_model.predict(
        X_test
    )
)


print(
    "\nLocked model refitted for explanation."
)


# --------------------------------------------------
# 10. Verify predictions against saved
#     final-test predictions
#
# This ensures we are explaining exactly
# the same model specification.
# --------------------------------------------------

prediction_check = pd.DataFrame({
    "icb_code":
        test[
            "icb_code"
        ].values,

    "refit_prediction":
        refit_predictions,
})


prediction_check = prediction_check.merge(
    saved_predictions[
        [
            "icb_code",
            "ridge_prediction",
        ]
    ],
    on="icb_code",
    how="left",
    validate="one_to_one",
)


max_prediction_difference = (
    (
        prediction_check[
            "refit_prediction"
        ]
        -
        prediction_check[
            "ridge_prediction"
        ]
    )
    .abs()
    .max()
)


if (
    max_prediction_difference
    > 0.0001
):
    raise RuntimeError(
        "Refitted predictions do not match "
        "saved final-test predictions."
    )


print(
    "\nPrediction reproducibility check: PASSED"
)

print(
    f"Maximum difference: "
    f"{max_prediction_difference:.6f}"
)


# --------------------------------------------------
# 11. Extract fitted scaler and Ridge model
# --------------------------------------------------

scaler = (
    final_model.named_steps[
        "scaler"
    ]
)

ridge = (
    final_model.named_steps[
        "ridge"
    ]
)


X_train_scaled = pd.DataFrame(
    scaler.transform(
        X_train
    ),
    columns=MODEL_FEATURES,
    index=X_train.index,
)


X_test_scaled = pd.DataFrame(
    scaler.transform(
        X_test
    ),
    columns=MODEL_FEATURES,
    index=X_test.index,
)


# --------------------------------------------------
# 12. Standardized coefficients
# --------------------------------------------------

coefficient_df = pd.DataFrame({
    "feature":
        MODEL_FEATURES,

    "standardized_coefficient":
        ridge.coef_,
})


coefficient_df[
    "absolute_standardized_coefficient"
] = (
    coefficient_df[
        "standardized_coefficient"
    ]
    .abs()
)


coefficient_df[
    "coefficient_direction"
] = np.where(
    coefficient_df[
        "standardized_coefficient"
    ]
    > 0,
    "positive",
    "negative",
)


# --------------------------------------------------
# 13. SHAP LinearExplainer
#
# Background = final-refit training data.
#
# SHAP explanations are descriptive
# model explanations, not causal effects.
# --------------------------------------------------

explainer = shap.LinearExplainer(
    ridge,
    X_train_scaled,
)


shap_explanation = explainer(
    X_test_scaled
)


shap_values = np.asarray(
    shap_explanation.values
)


if shap_values.shape != (
    42,
    6,
):
    raise RuntimeError(
        f"Unexpected SHAP shape: "
        f"{shap_values.shape}"
    )


# --------------------------------------------------
# 14. Verify SHAP additivity
# --------------------------------------------------

base_values = np.asarray(
    shap_explanation.base_values
)


if base_values.ndim == 0:

    base_values = np.repeat(
        float(base_values),
        len(test),
    )


shap_reconstructed_predictions = (
    base_values
    +
    shap_values.sum(
        axis=1
    )
)


max_shap_difference = float(
    np.max(
        np.abs(
            shap_reconstructed_predictions
            -
            refit_predictions
        )
    )
)


if max_shap_difference > 0.0001:

    raise RuntimeError(
        "SHAP values do not reconstruct "
        "the model predictions."
    )


print(
    "\nSHAP additivity check: PASSED"
)

print(
    f"Maximum reconstruction difference: "
    f"{max_shap_difference:.8f}"
)


# --------------------------------------------------
# 15. Global SHAP importance
# --------------------------------------------------

mean_abs_shap = (
    np.abs(
        shap_values
    )
    .mean(
        axis=0
    )
)


global_df = pd.DataFrame({
    "feature":
        MODEL_FEATURES,

    "mean_absolute_shap":
        mean_abs_shap,

    "standardized_coefficient":
        ridge.coef_,
})


global_df[
    "importance_share_percent"
] = (
    global_df[
        "mean_absolute_shap"
    ]
    /
    global_df[
        "mean_absolute_shap"
    ]
    .sum()
    *
    100
)


global_df[
    "coefficient_direction"
] = np.where(
    global_df[
        "standardized_coefficient"
    ]
    > 0,
    "positive",
    "negative",
)


global_df = (
    global_df
    .sort_values(
        "mean_absolute_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


global_df[
    "global_rank"
] = (
    np.arange(
        1,
        len(global_df) + 1
    )
)


# --------------------------------------------------
# 16. Local 2023 SHAP table
# --------------------------------------------------

local_df = pd.DataFrame({
    "target_year":
        FINAL_TEST_YEAR,

    "icb_code":
        test[
            "icb_code"
        ].values,

    "icb_name":
        test[
            "icb_name"
        ].values,

    "actual_ckd_rate":
        test[
            TARGET
        ].values,

    "ridge_prediction":
        refit_predictions,

    "shap_base_value":
        base_values,
})


for index, feature in enumerate(
    MODEL_FEATURES
):

    local_df[
        f"{feature}_value"
    ] = (
        test[
            feature
        ]
        .values
    )

    local_df[
        f"{feature}_shap"
    ] = (
        shap_values[
            :,
            index
        ]
    )


# --------------------------------------------------
# 17. Identify dominant local driver
# --------------------------------------------------

absolute_shap = np.abs(
    shap_values
)


top_feature_index = (
    absolute_shap.argmax(
        axis=1
    )
)


local_df[
    "largest_absolute_shap_feature"
] = [
    MODEL_FEATURES[index]
    for index
    in top_feature_index
]


local_df[
    "largest_absolute_shap_value"
] = [
    shap_values[
        row,
        feature_index,
    ]
    for row, feature_index
    in enumerate(
        top_feature_index
    )
]


# --------------------------------------------------
# 18. Global summary
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL MODEL SHAP EXPLAINABILITY"
)

print(
    "========================================"
)


print(
    "\nGlobal feature importance:"
)


print(
    global_df[
        [
            "global_rank",
            "feature",
            "mean_absolute_shap",
            "importance_share_percent",
            "standardized_coefficient",
            "coefficient_direction",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 19. Dominant driver frequency
# --------------------------------------------------

driver_frequency = (
    local_df[
        "largest_absolute_shap_feature"
    ]
    .value_counts()
)


print(
    "\nLargest local SHAP driver "
    "across 42 ICBs:"
)


print(
    driver_frequency.to_string()
)


# --------------------------------------------------
# 20. Example highest and lowest predictions
# --------------------------------------------------

highest_predictions = (
    local_df
    .sort_values(
        "ridge_prediction",
        ascending=False,
    )
    .head(5)
)


lowest_predictions = (
    local_df
    .sort_values(
        "ridge_prediction",
        ascending=True,
    )
    .head(5)
)


print(
    "\nFive highest predicted "
    "2023 CKD rates:"
)


print(
    highest_predictions[
        [
            "icb_code",
            "icb_name",
            "ridge_prediction",
            "largest_absolute_shap_feature",
            "largest_absolute_shap_value",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\nFive lowest predicted "
    "2023 CKD rates:"
)


print(
    lowest_predictions[
        [
            "icb_code",
            "icb_name",
            "ridge_prediction",
            "largest_absolute_shap_feature",
            "largest_absolute_shap_value",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 21. Interpretation warning
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "INTERPRETATION NOTE"
)

print(
    "========================================"
)


print(
    """
SHAP values describe how the fitted Ridge
model distributes each prediction relative
to its expected prediction.

They do NOT demonstrate causal effects.

The analysis uses aggregated ICB-level data.
Therefore, feature contributions must not be
interpreted as individual-level clinical risk
factors or treatment effects.
""".strip()
)


# --------------------------------------------------
# 22. Round saved outputs
# --------------------------------------------------

for dataframe in [
    global_df,
    local_df,
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
# 23. Integrity checks
# --------------------------------------------------

if len(global_df) != 6:
    raise RuntimeError(
        "Expected six global SHAP rows."
    )


if len(local_df) != 42:
    raise RuntimeError(
        "Expected 42 local explanation rows."
    )


if global_df[
    "mean_absolute_shap"
].isna().any():
    raise RuntimeError(
        "Missing global SHAP values."
    )


if local_df.filter(
    like="_shap"
).isna().any().any():
    raise RuntimeError(
        "Missing local SHAP values."
    )


print(
    "\nFinal explainability audit: PASSED"
)


# --------------------------------------------------
# 24. Save outputs
# --------------------------------------------------

global_df.to_csv(
    GLOBAL_OUTPUT,
    index=False
)


local_df.to_csv(
    LOCAL_OUTPUT,
    index=False
)


coefficient_df.to_csv(
    COEFFICIENT_OUTPUT,
    index=False
)


print(
    f"\nSaved global SHAP summary to: "
    f"{GLOBAL_OUTPUT}"
)

print(
    f"Saved 2023 local SHAP values to: "
    f"{LOCAL_OUTPUT}"
)

print(
    f"Saved coefficient summary to: "
    f"{COEFFICIENT_OUTPUT}"
)