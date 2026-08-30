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
    / "residual_rolling_origin_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "residual_rolling_origin_predictions.csv"
)

COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / "residual_rolling_origin_comparison.csv"
)


# --------------------------------------------------
# 2. Locked residual model
#
# Alpha 10 was selected using the designated
# 2022 validation year.
#
# It is NOT retuned during this robustness audit.
# --------------------------------------------------

RIDGE_ALPHA = 10.0

TARGET = "ckd_risk_rate_per_1000"

LAG_FEATURE = "lagged_ckd_rate"

RESIDUAL_TARGET = (
    "ckd_change_from_previous_year"
)

FINAL_TEST_YEAR = 2023


# --------------------------------------------------
# 3. Rolling-origin design
# --------------------------------------------------

ORIGINS = [
    {
        "origin": "origin_1",
        "train_years": [2020],
        "evaluation_year": 2021,
    },
    {
        "origin": "origin_2",
        "train_years": [2020, 2021],
        "evaluation_year": 2022,
    },
]


# --------------------------------------------------
# 4. Load data
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
# 6. Recover locked compact feature set
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
        f"Expected 5 compact features, "
        f"found {len(PRIMARY_FEATURES)}."
    )


print(
    "\nLocked compact residual predictors:"
)

for feature in PRIMARY_FEATURES:

    print(
        f"- {feature}"
    )


print(
    f"\nFixed Ridge alpha: "
    f"{RIDGE_ALPHA}"
)


# --------------------------------------------------
# 7. Restrict to pre-final target years
# --------------------------------------------------

model_data = (
    panel.loc[
        panel[
            "target_year"
        ]
        .isin(
            [
                2020,
                2021,
                2022,
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
# 8. Construct lagged CKD
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
            [
                2020,
                2021,
                2022,
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
        "Incorrect temporal lag alignment."
    )


print(
    "\nLagged CKD alignment: PASSED"
)


# --------------------------------------------------
# 9. Construct residual target
#
# residual = current CKD - previous CKD
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
    "\nResidual means:"
)

print(
    model_data
    .groupby(
        "target_year"
    )[
        RESIDUAL_TARGET
    ]
    .mean()
    .round(3)
    .to_string()
)


# --------------------------------------------------
# 10. Metric helper
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
# 11. Fixed residual Ridge model
# --------------------------------------------------

def build_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=RIDGE_ALPHA
            )
        ),
    ])


# --------------------------------------------------
# 12. Run rolling origins
# --------------------------------------------------

metric_rows = []

prediction_rows = []


for origin_info in ORIGINS:

    origin = (
        origin_info[
            "origin"
        ]
    )

    train_years = (
        origin_info[
            "train_years"
        ]
    )

    evaluation_year = (
        origin_info[
            "evaluation_year"
        ]
    )


    print(
        "\n========================================"
    )

    print(
        origin.upper()
    )

    print(
        "========================================"
    )


    print(
        f"Train target years: "
        f"{train_years}"
    )

    print(
        f"Evaluation target year: "
        f"{evaluation_year}"
    )


    train = (
        model_data.loc[
            model_data[
                "target_year"
            ]
            .isin(
                train_years
            )
        ]
        .copy()
    )


    evaluation = (
        model_data.loc[
            model_data[
                "target_year"
            ]
            == evaluation_year
        ]
        .copy()
    )


    expected_train_rows = (
        len(train_years)
        * 42
    )


    if len(train) != expected_train_rows:

        raise RuntimeError(
            f"{origin}: expected "
            f"{expected_train_rows} "
            f"training rows, "
            f"found {len(train)}."
        )


    if len(evaluation) != 42:

        raise RuntimeError(
            f"{origin}: expected "
            f"42 evaluation rows."
        )


    # ----------------------------------------------
    # Persistence
    # ----------------------------------------------

    persistence_predictions = (
        evaluation[
            LAG_FEATURE
        ]
        .to_numpy()
    )


    persistence_metrics = (
        calculate_metrics(
            evaluation[
                TARGET
            ],
            persistence_predictions,
        )
    )


    # ----------------------------------------------
    # Residual Ridge
    #
    # Model predicts CKD change.
    # Final forecast =
    # lagged CKD + predicted change.
    # ----------------------------------------------

    model = build_model()


    model.fit(
        train[
            PRIMARY_FEATURES
        ],
        train[
            RESIDUAL_TARGET
        ]
    )


    predicted_change = (
        model.predict(
            evaluation[
                PRIMARY_FEATURES
            ]
        )
    )


    residual_predictions = (
        evaluation[
            LAG_FEATURE
        ]
        .to_numpy()
        +
        predicted_change
    )


    residual_metrics = (
        calculate_metrics(
            evaluation[
                TARGET
            ],
            residual_predictions,
        )
    )


    # ----------------------------------------------
    # Store metrics
    # ----------------------------------------------

    metric_rows.append({
        "origin":
            origin,

        "train_years":
            "_".join(
                str(year)
                for year
                in train_years
            ),

        "evaluation_year":
            evaluation_year,

        "model":
            "persistence",

        "alpha":
            np.nan,

        "train_rows":
            len(train),

        "evaluation_rows":
            len(evaluation),

        **persistence_metrics,
    })


    metric_rows.append({
        "origin":
            origin,

        "train_years":
            "_".join(
                str(year)
                for year
                in train_years
            ),

        "evaluation_year":
            evaluation_year,

        "model":
            "residual_ridge_compact_5",

        "alpha":
            RIDGE_ALPHA,

        "train_rows":
            len(train),

        "evaluation_rows":
            len(evaluation),

        **residual_metrics,
    })


    # ----------------------------------------------
    # Store paired predictions
    # ----------------------------------------------

    for (
        index,
        persistence_prediction,
        predicted_delta,
        residual_prediction,
    ) in zip(
        evaluation.index,
        persistence_predictions,
        predicted_change,
        residual_predictions,
    ):

        actual = float(
            evaluation.loc[
                index,
                TARGET
            ]
        )


        persistence_error = abs(
            actual
            - persistence_prediction
        )


        residual_error = abs(
            actual
            - residual_prediction
        )


        prediction_rows.append({
            "origin":
                origin,

            "evaluation_year":
                evaluation_year,

            "icb_code":
                evaluation.loc[
                    index,
                    "icb_code"
                ],

            "icb_name":
                evaluation.loc[
                    index,
                    "icb_name"
                ],

            "lagged_ckd_rate":
                persistence_prediction,

            "actual_change":
                evaluation.loc[
                    index,
                    RESIDUAL_TARGET
                ],

            "predicted_change":
                predicted_delta,

            "actual_ckd_rate":
                actual,

            "persistence_prediction":
                persistence_prediction,

            "residual_prediction":
                residual_prediction,

            "persistence_abs_error":
                persistence_error,

            "residual_abs_error":
                residual_error,

            "abs_error_improvement":
                (
                    persistence_error
                    - residual_error
                ),

            "residual_model_better":
                (
                    residual_error
                    < persistence_error
                ),
        })


    # ----------------------------------------------
    # Print origin result
    # ----------------------------------------------

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
        "\nFixed residual Ridge:"
    )

    print(
        f"MAE  = "
        f"{residual_metrics['mae']:.3f}"
    )

    print(
        f"RMSE = "
        f"{residual_metrics['rmse']:.3f}"
    )

    print(
        f"R2   = "
        f"{residual_metrics['r2']:.3f}"
    )


    mae_improvement = (
        persistence_metrics["mae"]
        - residual_metrics["mae"]
    )


    improvement_percent = (
        mae_improvement
        /
        persistence_metrics["mae"]
        *
        100
    )


    print(
        "\nResidual-model MAE improvement "
        "vs persistence:"
    )

    print(
        f"{mae_improvement:.3f} "
        f"({improvement_percent:.2f}%)"
    )


# --------------------------------------------------
# 13. Build output tables
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)


# --------------------------------------------------
# 14. Build direct origin comparison
# --------------------------------------------------

comparison_rows = []


for origin_info in ORIGINS:

    origin = (
        origin_info[
            "origin"
        ]
    )


    persistence = (
        metrics_df.loc[
            (
                metrics_df[
                    "origin"
                ]
                == origin
            )
            &
            (
                metrics_df[
                    "model"
                ]
                == "persistence"
            )
        ]
        .iloc[0]
    )


    residual = (
        metrics_df.loc[
            (
                metrics_df[
                    "origin"
                ]
                == origin
            )
            &
            (
                metrics_df[
                    "model"
                ]
                == "residual_ridge_compact_5"
            )
        ]
        .iloc[0]
    )


    paired = (
        predictions_df.loc[
            predictions_df[
                "origin"
            ]
            == origin
        ]
    )


    mae_gain = (
        persistence[
            "mae"
        ]
        -
        residual[
            "mae"
        ]
    )


    comparison_rows.append({
        "origin":
            origin,

        "evaluation_year":
            int(
                persistence[
                    "evaluation_year"
                ]
            ),

        "persistence_mae":
            persistence[
                "mae"
            ],

        "residual_mae":
            residual[
                "mae"
            ],

        "mae_improvement":
            mae_gain,

        "mae_improvement_percent":
            (
                mae_gain
                /
                persistence[
                    "mae"
                ]
                *
                100
            ),

        "persistence_rmse":
            persistence[
                "rmse"
            ],

        "residual_rmse":
            residual[
                "rmse"
            ],

        "persistence_r2":
            persistence[
                "r2"
            ],

        "residual_r2":
            residual[
                "r2"
            ],

        "icbs_residual_better":
            int(
                paired[
                    "residual_model_better"
                ]
                .sum()
            ),

        "n_icbs":
            len(paired),

        "proportion_residual_better":
            paired[
                "residual_model_better"
            ]
            .mean(),

        "median_abs_error_improvement":
            paired[
                "abs_error_improvement"
            ]
            .median(),
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# --------------------------------------------------
# 15. Print summary
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "RESIDUAL ROLLING-ORIGIN SUMMARY"
)

print(
    "========================================"
)


print(
    comparison_df[
        [
            "origin",
            "evaluation_year",
            "persistence_mae",
            "residual_mae",
            "mae_improvement",
            "mae_improvement_percent",
            "persistence_rmse",
            "residual_rmse",
            "persistence_r2",
            "residual_r2",
            "icbs_residual_better",
            "n_icbs",
            "median_abs_error_improvement",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 16. Overall descriptive comparison
# --------------------------------------------------

mean_persistence_mae = (
    comparison_df[
        "persistence_mae"
    ]
    .mean()
)


mean_residual_mae = (
    comparison_df[
        "residual_mae"
    ]
    .mean()
)


print(
    "\nMean MAE across both "
    "pre-final origins:"
)

print(
    f"Persistence      : "
    f"{mean_persistence_mae:.3f}"
)

print(
    f"Residual Ridge   : "
    f"{mean_residual_mae:.3f}"
)

print(
    f"Difference       : "
    f"{mean_persistence_mae - mean_residual_mae:.3f}"
)


# --------------------------------------------------
# 17. Interpretation
# --------------------------------------------------

improvements = (
    comparison_df[
        "mae_improvement"
    ]
)


if (
    improvements > 0
).all():

    interpretation = (
        "The fixed residual Ridge model "
        "improved MAE over persistence at "
        "both pre-final forecast origins."
    )

elif (
    improvements > 0
).any():

    interpretation = (
        "The fixed residual Ridge model "
        "improved MAE at only one pre-final "
        "forecast origin. Incremental NDA "
        "value is therefore not temporally "
        "consistent."
    )

else:

    interpretation = (
        "The fixed residual Ridge model "
        "did not improve MAE over persistence "
        "at either pre-final forecast origin."
    )


print(
    "\nInterpretation:"
)

print(
    interpretation
)


print(
    "\nAlpha=10.0 was kept fixed and "
    "was not retuned using the 2021 origin."
)


# --------------------------------------------------
# 18. Final-test protection
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
        "evaluation_year"
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
# 19. Final validation
# --------------------------------------------------

# 2 origins x 2 models
if len(metrics_df) != 4:

    raise RuntimeError(
        f"Expected 4 metric rows, "
        f"found {len(metrics_df)}."
    )


# 2 origins x 42 ICBs
if len(predictions_df) != 84:

    raise RuntimeError(
        f"Expected 84 prediction rows, "
        f"found {len(predictions_df)}."
    )


if len(comparison_df) != 2:

    raise RuntimeError(
        f"Expected 2 comparison rows, "
        f"found {len(comparison_df)}."
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
    "\nResidual rolling-origin audit: PASSED"
)


# --------------------------------------------------
# 20. Save outputs
# --------------------------------------------------

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


predictions_df.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)


comparison_df.to_csv(
    COMPARISON_OUTPUT,
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
    f"Saved comparison to: "
    f"{COMPARISON_OUTPUT}"
)