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
    / "rolling_origin_metrics.csv"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "rolling_origin_predictions.csv"
)

ICB_COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / "rolling_origin_icb_error_comparison.csv"
)


# --------------------------------------------------
# 2. Locked current candidate specification
#
# alpha=1.0 was selected using the formally
# designated 2022 validation year.
#
# We do NOT retune alpha during this audit.
# --------------------------------------------------

RIDGE_ALPHA = 1.0

TARGET = "ckd_risk_rate_per_1000"
LAG_FEATURE = "lagged_ckd_rate"

FINAL_TEST_YEAR = 2023


# --------------------------------------------------
# 3. Rolling-origin robustness design
#
# Origin 1:
# train target 2020
# evaluate target 2021
#
# Origin 2:
# train targets 2020 + 2021
# evaluate target 2022
#
# 2023 remains completely untouched.
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
# 4. Load required data
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


panel = pd.read_csv(PANEL_FILE)
ckd = pd.read_csv(CKD_FILE)
manifest = pd.read_csv(FEATURE_MANIFEST_FILE)


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


MODEL_FEATURES = (
    [LAG_FEATURE]
    + PRIMARY_FEATURES
)


print(
    "\nLocked augmented feature set:"
)

for feature in MODEL_FEATURES:
    print(
        f"- {feature}"
    )


print(
    f"\nFixed Ridge alpha: "
    f"{RIDGE_ALPHA}"
)


# --------------------------------------------------
# 7. Restrict analysis to pre-final years
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
        "2023 entered robustness audit."
    )


if len(model_data) != 126:
    raise RuntimeError(
        f"Expected 126 pre-final rows, "
        f"found {len(model_data)}."
    )


# --------------------------------------------------
# 8. Construct previous-year CKD predictor
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

        "ckd_risk_rate_per_1000":
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
    "\nLagged CKD temporal alignment: PASSED"
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
# 10. Ridge model builder
#
# Scaling is fitted ONLY on each origin's
# training period.
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
# 11. Run rolling-origin evaluation
# --------------------------------------------------

metric_rows = []
prediction_rows = []


for origin_info in ORIGINS:

    origin = (
        origin_info["origin"]
    )

    train_years = (
        origin_info["train_years"]
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
        f"{origin.upper()}"
    )

    print(
        "========================================"
    )


    print(
        f"Train years: "
        f"{train_years}"
    )

    print(
        f"Evaluation year: "
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
            f"42 evaluation rows, "
            f"found {len(evaluation)}."
        )


    # ----------------------------------------------
    # Persistence benchmark
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
    # Fixed augmented Ridge model
    # ----------------------------------------------

    model = build_model()


    model.fit(
        train[
            MODEL_FEATURES
        ],
        train[
            TARGET
        ]
    )


    augmented_predictions = (
        model.predict(
            evaluation[
                MODEL_FEATURES
            ]
        )
    )


    augmented_metrics = (
        calculate_metrics(
            evaluation[
                TARGET
            ],
            augmented_predictions,
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
            "ridge_lag_plus_compact_5",

        "alpha":
            RIDGE_ALPHA,

        "train_rows":
            len(train),

        "evaluation_rows":
            len(evaluation),

        **augmented_metrics,
    })


    # ----------------------------------------------
    # Store paired ICB predictions
    # ----------------------------------------------

    for (
        index,
        persistence_prediction,
        augmented_prediction,
    ) in zip(
        evaluation.index,
        persistence_predictions,
        augmented_predictions,
    ):

        actual = float(
            evaluation.loc[
                index,
                TARGET
            ]
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

            "actual_ckd_rate":
                actual,

            "persistence_prediction":
                persistence_prediction,

            "augmented_prediction":
                augmented_prediction,

            "persistence_abs_error":
                abs(
                    actual
                    - persistence_prediction
                ),

            "augmented_abs_error":
                abs(
                    actual
                    - augmented_prediction
                ),
        })


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
        "\nFixed augmented Ridge:"
    )

    print(
        f"MAE  = "
        f"{augmented_metrics['mae']:.3f}"
    )

    print(
        f"RMSE = "
        f"{augmented_metrics['rmse']:.3f}"
    )

    print(
        f"R2   = "
        f"{augmented_metrics['r2']:.3f}"
    )


    mae_gain = (
        persistence_metrics["mae"]
        - augmented_metrics["mae"]
    )


    relative_gain = (
        mae_gain
        /
        persistence_metrics["mae"]
        *
        100
    )


    print(
        "\nAugmented MAE improvement "
        "vs persistence:"
    )

    print(
        f"{mae_gain:.3f} "
        f"({relative_gain:.2f}%)"
    )


# --------------------------------------------------
# 12. Build result tables
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)


# --------------------------------------------------
# 13. Paired ICB error comparison
#
# Positive error improvement means the
# augmented model made a smaller absolute error.
# --------------------------------------------------

predictions_df[
    "abs_error_improvement"
] = (
    predictions_df[
        "persistence_abs_error"
    ]
    -
    predictions_df[
        "augmented_abs_error"
    ]
)


predictions_df[
    "augmented_better"
] = (
    predictions_df[
        "augmented_abs_error"
    ]
    <
    predictions_df[
        "persistence_abs_error"
    ]
)


# --------------------------------------------------
# 14. ICB-level summary by origin
# --------------------------------------------------

comparison_rows = []


for origin_info in ORIGINS:

    origin = (
        origin_info["origin"]
    )


    origin_data = (
        predictions_df.loc[
            predictions_df[
                "origin"
            ]
            == origin
        ]
    )


    comparison_rows.append({
        "origin":
            origin,

        "evaluation_year":
            origin_data[
                "evaluation_year"
            ]
            .iloc[0],

        "n_icbs":
            len(origin_data),

        "icbs_augmented_better":
            int(
                origin_data[
                    "augmented_better"
                ]
                .sum()
            ),

        "proportion_augmented_better":
            origin_data[
                "augmented_better"
            ]
            .mean(),

        "mean_abs_error_improvement":
            origin_data[
                "abs_error_improvement"
            ]
            .mean(),

        "median_abs_error_improvement":
            origin_data[
                "abs_error_improvement"
            ]
            .median(),
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# --------------------------------------------------
# 15. Add direct metric comparison
# --------------------------------------------------

pivot = (
    metrics_df
    .pivot(
        index=[
            "origin",
            "evaluation_year",
        ],
        columns="model",
        values=[
            "mae",
            "rmse",
            "r2",
        ]
    )
)


comparison_metric_rows = []


for origin_info in ORIGINS:

    origin = origin_info["origin"]

    evaluation_year = (
        origin_info[
            "evaluation_year"
        ]
    )


    p = (
        metrics_df.loc[
            (
                metrics_df["origin"]
                == origin
            )
            &
            (
                metrics_df["model"]
                == "persistence"
            )
        ]
        .iloc[0]
    )


    a = (
        metrics_df.loc[
            (
                metrics_df["origin"]
                == origin
            )
            &
            (
                metrics_df["model"]
                == "ridge_lag_plus_compact_5"
            )
        ]
        .iloc[0]
    )


    mae_gain = (
        p["mae"]
        - a["mae"]
    )


    comparison_metric_rows.append({
        "origin":
            origin,

        "evaluation_year":
            evaluation_year,

        "persistence_mae":
            p["mae"],

        "augmented_mae":
            a["mae"],

        "mae_improvement":
            mae_gain,

        "mae_improvement_percent":
            (
                mae_gain
                /
                p["mae"]
                *
                100
            ),

        "persistence_rmse":
            p["rmse"],

        "augmented_rmse":
            a["rmse"],

        "persistence_r2":
            p["r2"],

        "augmented_r2":
            a["r2"],
    })


metric_comparison = pd.DataFrame(
    comparison_metric_rows
)


comparison_df = (
    comparison_df
    .merge(
        metric_comparison,
        on=[
            "origin",
            "evaluation_year",
        ],
        how="left",
        validate="one_to_one"
    )
)


# --------------------------------------------------
# 16. Descriptive two-origin summary
#
# This is NOT treated as an independent-sample
# statistical estimate because the same 42 ICBs
# recur across time.
# --------------------------------------------------

mean_persistence_mae = (
    metric_comparison[
        "persistence_mae"
    ]
    .mean()
)


mean_augmented_mae = (
    metric_comparison[
        "augmented_mae"
    ]
    .mean()
)


mean_mae_gain = (
    mean_persistence_mae
    - mean_augmented_mae
)


print(
    "\n========================================"
)

print(
    "ROLLING-ORIGIN ROBUSTNESS SUMMARY"
)

print(
    "========================================"
)


print(
    metric_comparison[
        [
            "origin",
            "evaluation_year",
            "persistence_mae",
            "augmented_mae",
            "mae_improvement",
            "mae_improvement_percent",
            "persistence_rmse",
            "augmented_rmse",
            "persistence_r2",
            "augmented_r2",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\nICB-level paired error comparison:"
)


print(
    comparison_df[
        [
            "origin",
            "evaluation_year",
            "icbs_augmented_better",
            "n_icbs",
            "proportion_augmented_better",
            "mean_abs_error_improvement",
            "median_abs_error_improvement",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\nDescriptive mean MAE across "
    "the two pre-final origins:"
)

print(
    f"Persistence: "
    f"{mean_persistence_mae:.3f}"
)

print(
    f"Augmented Ridge: "
    f"{mean_augmented_mae:.3f}"
)

print(
    f"Difference: "
    f"{mean_mae_gain:.3f}"
)


# --------------------------------------------------
# 17. Interpretation rule
# --------------------------------------------------

origin_improvements = (
    metric_comparison[
        "mae_improvement"
    ]
)


if (
    origin_improvements > 0
).all():

    interpretation = (
        "The fixed augmented Ridge model "
        "improved MAE over persistence at "
        "both pre-final forecasting origins."
    )

elif (
    origin_improvements > 0
).any():

    interpretation = (
        "The fixed augmented Ridge model "
        "improved over persistence at only "
        "one of the two pre-final origins, "
        "so incremental value is not "
        "temporally consistent."
    )

else:

    interpretation = (
        "The fixed augmented Ridge model "
        "did not improve over persistence "
        "at either pre-final origin."
    )


print(
    "\nInterpretation:"
)

print(
    interpretation
)


print(
    "\nImportant: this is a retrospective "
    "robustness audit. Alpha=1.0 is not "
    "retuned using the 2021 origin."
)


# --------------------------------------------------
# 18. Final-test protection
# --------------------------------------------------

if (
    FINAL_TEST_YEAR
    in model_data[
        "target_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 entered robustness data."
    )


if (
    FINAL_TEST_YEAR
    in predictions_df[
        "evaluation_year"
    ]
    .unique()
):

    raise RuntimeError(
        "2023 predictions were generated."
    )


if model_data[
    "lag_source_year"
].max() != 2021:

    raise RuntimeError(
        "Unexpected latest lag source year."
    )


print(
    "\n2023 final test protection: PASSED"
)


# --------------------------------------------------
# 19. Final validation
# --------------------------------------------------

# Two origins × two models
if len(metrics_df) != 4:
    raise RuntimeError(
        f"Expected 4 metric rows, "
        f"found {len(metrics_df)}."
    )


# Two origins × 42 ICBs
if len(predictions_df) != 84:
    raise RuntimeError(
        f"Expected 84 paired predictions, "
        f"found {len(predictions_df)}."
    )


if len(comparison_df) != 2:
    raise RuntimeError(
        f"Expected 2 origin summaries, "
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
        "Missing metric values detected."
    )


print(
    "\nRolling-origin robustness audit: PASSED"
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
    ICB_COMPARISON_OUTPUT,
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
    f"Saved ICB comparison to: "
    f"{ICB_COMPARISON_OUTPUT}"
)