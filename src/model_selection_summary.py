from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FEATURE_MANIFEST_FILE = (
    OUTPUT_DIR
    / "feature_set_manifest.csv"
)

BASELINE_FILE = (
    OUTPUT_DIR
    / "baseline_metrics_pre_final.csv"
)

AUGMENTED_FILE = (
    OUTPUT_DIR
    / "augmented_linear_validation_metrics.csv"
)

ROLLING_FILE = (
    OUTPUT_DIR
    / "rolling_origin_icb_error_comparison.csv"
)

RESIDUAL_FILE = (
    OUTPUT_DIR
    / "residual_correction_validation_metrics.csv"
)

RESIDUAL_ROLLING_FILE = (
    OUTPUT_DIR
    / "residual_rolling_origin_comparison.csv"
)

NONLINEAR_FILE = (
    OUTPUT_DIR
    / "nonlinear_validation_metrics.csv"
)

VALIDATION_MANIFEST_FILE = (
    OUTPUT_DIR
    / "validation_split_manifest.csv"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "model_selection_summary.csv"
)

LOCK_OUTPUT = (
    OUTPUT_DIR
    / "final_model_lock.csv"
)


# --------------------------------------------------
# 2. Locked candidate specification
#
# This is the model selected BEFORE opening
# the final 2023 test outcome.
# --------------------------------------------------

SELECTED_MODEL = "ridge_regression"

SELECTED_FEATURE_SET = "lag_plus_compact_5"

SELECTED_ALPHA = 1.0

FINAL_TEST_YEAR = 2023


# --------------------------------------------------
# 3. Load required evidence
# --------------------------------------------------

required_files = [
    FEATURE_MANIFEST_FILE,
    BASELINE_FILE,
    AUGMENTED_FILE,
    ROLLING_FILE,
    RESIDUAL_FILE,
    RESIDUAL_ROLLING_FILE,
    NONLINEAR_FILE,
    VALIDATION_MANIFEST_FILE,
]


for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required evidence file not found: "
            f"{file_path}"
        )


feature_manifest = pd.read_csv(
    FEATURE_MANIFEST_FILE
)

baseline = pd.read_csv(
    BASELINE_FILE
)

augmented = pd.read_csv(
    AUGMENTED_FILE
)

rolling = pd.read_csv(
    ROLLING_FILE
)

residual = pd.read_csv(
    RESIDUAL_FILE
)

residual_rolling = pd.read_csv(
    RESIDUAL_ROLLING_FILE
)

nonlinear = pd.read_csv(
    NONLINEAR_FILE
)

validation_manifest = pd.read_csv(
    VALIDATION_MANIFEST_FILE
)


print(
    "All model-selection evidence files loaded."
)


# --------------------------------------------------
# 4. Recover pre-specified compact features
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


compact_features = (
    feature_manifest.loc[
        feature_manifest[
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
    "\nPre-specified compact NDA predictors:"
)

for feature in compact_features:
    print(
        f"- {feature}"
    )


# --------------------------------------------------
# 5. Verify temporal split remains locked
# --------------------------------------------------

split_counts = (
    validation_manifest[
        "split"
    ]
    .value_counts()
    .to_dict()
)


expected_split_counts = {
    "development_train": 84,
    "validation": 42,
    "final_test": 42,
}


if split_counts != expected_split_counts:

    raise RuntimeError(
        "Validation manifest does not match "
        "the locked temporal design.\n"
        f"Observed: {split_counts}"
    )


final_test_manifest = (
    validation_manifest.loc[
        validation_manifest[
            "split"
        ]
        == "final_test"
    ]
)


if len(final_test_manifest) != 42:

    raise RuntimeError(
        "Expected 42 final-test ICB rows."
    )


if set(
    final_test_manifest[
        "target_year"
    ]
) != {
    FINAL_TEST_YEAR
}:

    raise RuntimeError(
        "Final test is not exclusively 2023."
    )


print(
    "\nTemporal split lock: PASSED"
)

print(
    "Development train: 84 rows"
)

print(
    "Validation       : 42 rows"
)

print(
    "Final test       : 42 rows"
)


# --------------------------------------------------
# 6. Recover 2022 persistence benchmark
# --------------------------------------------------

persistence_2022 = (
    baseline.loc[
        (
            baseline[
                "evaluation_scope"
            ]
            == "single_year"
        )
        &
        (
            baseline[
                "target_year"
            ]
            .astype(str)
            == "2022"
        )
        &
        (
            baseline[
                "baseline"
            ]
            == "persistence"
        )
    ]
)


if len(persistence_2022) != 1:

    raise RuntimeError(
        "Expected one unique 2022 "
        "persistence benchmark."
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
# 7. Recover selected Ridge candidate
# --------------------------------------------------

selected_candidate = (
    augmented.loc[
        (
            augmented[
                "model"
            ]
            == SELECTED_MODEL
        )
        &
        (
            augmented[
                "feature_set"
            ]
            == SELECTED_FEATURE_SET
        )
        &
        (
            np.isclose(
                augmented[
                    "alpha"
                ],
                SELECTED_ALPHA
            )
        )
    ]
)


if len(selected_candidate) != 1:

    raise RuntimeError(
        "Could not recover exactly one "
        "selected augmented Ridge model."
    )


selected_candidate = (
    selected_candidate
    .iloc[0]
)


selected_mae = float(
    selected_candidate[
        "mae"
    ]
)

selected_rmse = float(
    selected_candidate[
        "rmse"
    ]
)

selected_r2 = float(
    selected_candidate[
        "r2"
    ]
)


# --------------------------------------------------
# 8. Verify designated validation result
# --------------------------------------------------

if selected_mae >= persistence_mae:

    raise RuntimeError(
        "Selected candidate does not beat "
        "persistence on designated 2022 "
        "validation MAE."
    )


mae_gain = (
    persistence_mae
    - selected_mae
)


mae_gain_percent = (
    mae_gain
    /
    persistence_mae
    *
    100
)


# --------------------------------------------------
# 9. Recover rolling-origin robustness evidence
# --------------------------------------------------

if len(rolling) != 2:

    raise RuntimeError(
        "Expected two augmented rolling-origin "
        "comparison rows."
    )


rolling_2021 = (
    rolling.loc[
        rolling[
            "evaluation_year"
        ]
        == 2021
    ]
)


rolling_2022 = (
    rolling.loc[
        rolling[
            "evaluation_year"
        ]
        == 2022
    ]
)


if (
    len(rolling_2021) != 1
    or
    len(rolling_2022) != 1
):

    raise RuntimeError(
        "Missing expected rolling-origin rows."
    )


rolling_2021 = (
    rolling_2021
    .iloc[0]
)

rolling_2022 = (
    rolling_2022
    .iloc[0]
)


# --------------------------------------------------
# 10. Confirm temporal robustness limitation
# --------------------------------------------------

if float(
    rolling_2021[
        "augmented_mae"
    ]
) <= float(
    rolling_2021[
        "persistence_mae"
    ]
):

    raise RuntimeError(
        "Expected earlier-origin evidence "
        "to show poorer augmented performance."
    )


if float(
    rolling_2022[
        "augmented_mae"
    ]
) >= float(
    rolling_2022[
        "persistence_mae"
    ]
):

    raise RuntimeError(
        "Expected 2022 augmented model to "
        "improve over persistence."
    )


print(
    "\nRolling-origin evidence verified:"
)

print(
    "2021: augmented model worse "
    "than persistence"
)

print(
    "2022: augmented model better "
    "than persistence"
)


# --------------------------------------------------
# 11. Recover strongest residual model
# --------------------------------------------------

learned_residual = (
    residual.loc[
        residual[
            "model"
        ]
        .isin(
            [
                "residual_linear",
                "residual_ridge",
            ]
        )
    ]
    .sort_values(
        "mae"
    )
)


best_residual = (
    learned_residual
    .iloc[0]
)


# --------------------------------------------------
# 12. Recover strongest nonlinear model
# --------------------------------------------------

nonlinear_only = (
    nonlinear.loc[
        nonlinear[
            "model"
        ]
        != "persistence"
    ]
    .sort_values(
        "mae"
    )
)


best_nonlinear = (
    nonlinear_only
    .iloc[0]
)


# --------------------------------------------------
# 13. Verify nonlinear models did not beat
#     persistence
# --------------------------------------------------

if float(
    best_nonlinear[
        "mae"
    ]
) < persistence_mae:

    raise RuntimeError(
        "A nonlinear model unexpectedly "
        "beats persistence. Revisit model "
        "selection before locking."
    )


# --------------------------------------------------
# 14. Verify residual model rolling-origin
#     limitation
# --------------------------------------------------

if len(
    residual_rolling
) != 2:

    raise RuntimeError(
        "Expected two residual rolling-origin "
        "comparison rows."
    )


residual_2021 = (
    residual_rolling.loc[
        residual_rolling[
            "evaluation_year"
        ]
        == 2021
    ]
    .iloc[0]
)


if float(
    residual_2021[
        "residual_mae"
    ]
) <= float(
    residual_2021[
        "persistence_mae"
    ]
):

    raise RuntimeError(
        "Expected residual model to be worse "
        "than persistence at 2021 origin."
    )


# --------------------------------------------------
# 15. Build model-selection evidence table
# --------------------------------------------------

summary_rows = []


summary_rows.append({
    "candidate":
        "Persistence benchmark",

    "model":
        "persistence",

    "feature_set":
        "lagged_ckd_only",

    "alpha":
        np.nan,

    "validation_year":
        2022,

    "mae":
        persistence_mae,

    "rmse":
        persistence_rmse,

    "r2":
        persistence_r2,

    "selected":
        False,

    "selection_interpretation":
        (
            "Strong and temporally robust benchmark; "
            "must remain the primary comparator."
        ),
})


summary_rows.append({
    "candidate":
        "Selected augmented Ridge",

    "model":
        SELECTED_MODEL,

    "feature_set":
        SELECTED_FEATURE_SET,

    "alpha":
        SELECTED_ALPHA,

    "validation_year":
        2022,

    "mae":
        selected_mae,

    "rmse":
        selected_rmse,

    "r2":
        selected_r2,

    "selected":
        True,

    "selection_interpretation":
        (
            "Lowest designated 2022 validation MAE "
            "among evaluated primary candidates; "
            "parsimonious and interpretable. "
            "Improvement over persistence was not "
            "temporally consistent at the earlier "
            "rolling origin."
        ),
})


summary_rows.append({
    "candidate":
        "Best residual-correction model",

    "model":
        best_residual[
            "model"
        ],

    "feature_set":
        best_residual[
            "feature_set"
        ],

    "alpha":
        best_residual[
            "alpha"
        ],

    "validation_year":
        2022,

    "mae":
        best_residual[
            "mae"
        ],

    "rmse":
        best_residual[
            "rmse"
        ],

    "r2":
        best_residual[
            "r2"
        ],

    "selected":
        False,

    "selection_interpretation":
        (
            "Competitive 2022 validation performance "
            "but not superior to selected augmented "
            "Ridge and not temporally robust."
        ),
})


summary_rows.append({
    "candidate":
        "Best nonlinear model",

    "model":
        best_nonlinear[
            "model"
        ],

    "feature_set":
        best_nonlinear[
            "feature_set"
        ],

    "alpha":
        np.nan,

    "validation_year":
        2022,

    "mae":
        best_nonlinear[
            "mae"
        ],

    "rmse":
        best_nonlinear[
            "rmse"
        ],

    "r2":
        best_nonlinear[
            "r2"
        ],

    "selected":
        False,

    "selection_interpretation":
        (
            "Did not outperform persistence; "
            "greater complexity was not justified."
        ),
})


summary_df = pd.DataFrame(
    summary_rows
)


# --------------------------------------------------
# 16. Build immutable model-lock record
# --------------------------------------------------

lock_df = pd.DataFrame(
    [
        {
            "status":
                "LOCKED_BEFORE_FINAL_TEST",

            "model":
                SELECTED_MODEL,

            "alpha":
                SELECTED_ALPHA,

            "feature_structure":
                "lagged_ckd_plus_compact_5",

            "lag_feature":
                "previous_year_ckd_rate",

            "nda_feature_1":
                compact_features[0],

            "nda_feature_2":
                compact_features[1],

            "nda_feature_3":
                compact_features[2],

            "nda_feature_4":
                compact_features[3],

            "nda_feature_5":
                compact_features[4],

            "development_target_years":
                "2020,2021",

            "validation_target_year":
                2022,

            "final_refit_target_years":
                "2020,2021,2022",

            "final_test_target_year":
                2023,

            "final_refit_rows":
                126,

            "final_test_rows":
                42,

            "validation_mae":
                selected_mae,

            "validation_rmse":
                selected_rmse,

            "validation_r2":
                selected_r2,

            "persistence_validation_mae":
                persistence_mae,

            "validation_mae_gain_vs_persistence":
                mae_gain,

            "validation_mae_gain_percent":
                mae_gain_percent,

            "robustness_caveat":
                (
                    "The selected model improved over "
                    "persistence on 2022 validation but "
                    "performed worse than persistence "
                    "at the earlier 2021 rolling origin."
                ),

            "final_test_rule":
                (
                    "No feature, algorithm or "
                    "hyperparameter changes are allowed "
                    "after viewing 2023 performance."
                ),
        }
    ]
)


# --------------------------------------------------
# 17. Round numeric output
# --------------------------------------------------

for dataframe in [
    summary_df,
    lock_df,
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
# 18. Print model-selection summary
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL PRE-TEST MODEL SELECTION"
)

print(
    "========================================"
)


print(
    summary_df[
        [
            "candidate",
            "model",
            "feature_set",
            "alpha",
            "mae",
            "rmse",
            "r2",
            "selected",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nSelected final ML candidate:"
)

print(
    f"Model       : "
    f"{SELECTED_MODEL}"
)

print(
    f"Feature set : "
    f"{SELECTED_FEATURE_SET}"
)

print(
    f"Alpha       : "
    f"{SELECTED_ALPHA}"
)

print(
    f"2022 MAE    : "
    f"{selected_mae:.3f}"
)

print(
    f"2022 RMSE   : "
    f"{selected_rmse:.3f}"
)

print(
    f"2022 R2     : "
    f"{selected_r2:.3f}"
)


print(
    "\nPersistence benchmark:"
)

print(
    f"MAE  = "
    f"{persistence_mae:.3f}"
)

print(
    f"RMSE = "
    f"{persistence_rmse:.3f}"
)

print(
    f"R2   = "
    f"{persistence_r2:.3f}"
)


print(
    "\nValidation MAE gain:"
)

print(
    f"{mae_gain:.3f} "
    f"({mae_gain_percent:.2f}%)"
)


# --------------------------------------------------
# 19. Print robustness warning
# --------------------------------------------------

print(
    "\nIMPORTANT ROBUSTNESS CAVEAT:"
)

print(
    f"2021 persistence MAE = "
    f"{float(rolling_2021['persistence_mae']):.3f}"
)

print(
    f"2021 augmented MAE   = "
    f"{float(rolling_2021['augmented_mae']):.3f}"
)


print(
    "\nThe selected Ridge model therefore "
    "has the best designated 2022 validation "
    "performance, but its incremental benefit "
    "over persistence is not temporally robust."
)


# --------------------------------------------------
# 20. Final-test lock statement
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "MODEL LOCK"
)

print(
    "========================================"
)


print(
    """
The predictive specification is now frozen
before evaluation of the 2023 final test.

Locked model:
Ridge regression

Locked alpha:
1.0

Locked predictors:
1. Previous-year CKD rate
2. Urine albumin care-process completion
3. Serum creatinine care-process completion
4. HbA1c <=58 mmol/mol
5. Blood pressure <=140/80
6. Combined statin prevention measure

Final refit:
Target years 2020 + 2021 + 2022
126 ICB-year observations

Final test:
Target year 2023
42 ICB observations

After 2023 performance is viewed, the
algorithm, features and alpha must not be
changed in response to the result.
""".strip()
)


# --------------------------------------------------
# 21. Final validation
# --------------------------------------------------

if len(
    summary_df.loc[
        summary_df[
            "selected"
        ]
    ]
) != 1:

    raise RuntimeError(
        "Exactly one final model must "
        "be selected."
    )


if len(lock_df) != 1:

    raise RuntimeError(
        "Expected one model-lock row."
    )


if lock_df[
    "final_test_target_year"
].iloc[0] != 2023:

    raise RuntimeError(
        "Final test year must be 2023."
    )


print(
    "\nModel-selection summary: PASSED"
)

print(
    "2023 final-test model remains unopened."
)


# --------------------------------------------------
# 22. Save outputs
# --------------------------------------------------

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


lock_df.to_csv(
    LOCK_OUTPUT,
    index=False
)


print(
    f"\nSaved model-selection summary to: "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Saved final model lock to: "
    f"{LOCK_OUTPUT}"
)