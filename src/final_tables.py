from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Input files
# --------------------------------------------------

MODEL_SELECTION_FILE = (
    TABLE_DIR
    / "model_selection_summary.csv"
)

FINAL_TEST_FILE = (
    TABLE_DIR
    / "final_test_metrics.csv"
)

ROLLING_FILE = (
    TABLE_DIR
    / "rolling_origin_icb_error_comparison.csv"
)

SENSITIVITY_FILE = (
    TABLE_DIR
    / "final_sensitivity_metrics.csv"
)

BOOTSTRAP_FILE = (
    TABLE_DIR
    / "final_sensitivity_bootstrap_summary.csv"
)

SHAP_FILE = (
    TABLE_DIR
    / "final_model_shap_global.csv"
)

VALIDATION_FILE = (
    TABLE_DIR
    / "validation_split_manifest.csv"
)

MODEL_LOCK_FILE = (
    TABLE_DIR
    / "final_model_lock.csv"
)


# --------------------------------------------------
# 3. Output files
# --------------------------------------------------

TABLE_01 = (
    TABLE_DIR
    / "dissertation_table_01_validation_design.csv"
)

TABLE_02 = (
    TABLE_DIR
    / "dissertation_table_02_model_selection.csv"
)

TABLE_03 = (
    TABLE_DIR
    / "dissertation_table_03_temporal_performance.csv"
)

TABLE_04 = (
    TABLE_DIR
    / "dissertation_table_04_final_test.csv"
)

TABLE_05 = (
    TABLE_DIR
    / "dissertation_table_05_sensitivity.csv"
)

TABLE_06 = (
    TABLE_DIR
    / "dissertation_table_06_shap_importance.csv"
)

MANIFEST_OUTPUT = (
    TABLE_DIR
    / "final_table_manifest.csv"
)


# --------------------------------------------------
# 4. Check required files
# --------------------------------------------------

required_files = [
    MODEL_SELECTION_FILE,
    FINAL_TEST_FILE,
    ROLLING_FILE,
    SENSITIVITY_FILE,
    BOOTSTRAP_FILE,
    SHAP_FILE,
    VALIDATION_FILE,
    MODEL_LOCK_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path}"
        )


# --------------------------------------------------
# 5. Load evidence
# --------------------------------------------------

model_selection = pd.read_csv(
    MODEL_SELECTION_FILE
)

final_test = pd.read_csv(
    FINAL_TEST_FILE
)

rolling = pd.read_csv(
    ROLLING_FILE
)

sensitivity = pd.read_csv(
    SENSITIVITY_FILE
)

bootstrap = pd.read_csv(
    BOOTSTRAP_FILE
)

shap_global = pd.read_csv(
    SHAP_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
)

model_lock = pd.read_csv(
    MODEL_LOCK_FILE
)


print(
    "All final table evidence files loaded."
)


# --------------------------------------------------
# 6. Integrity checks
# --------------------------------------------------

if len(model_lock) != 1:

    raise RuntimeError(
        "Expected one final model-lock row."
    )


if (
    model_lock[
        "status"
    ]
    .iloc[0]
    != "LOCKED_BEFORE_FINAL_TEST"
):

    raise RuntimeError(
        "Final model lock is invalid."
    )


if len(final_test) != 2:

    raise RuntimeError(
        "Expected two final-test metric rows."
    )


if len(shap_global) != 6:

    raise RuntimeError(
        "Expected six SHAP features."
    )


print(
    "Evidence integrity checks: PASSED"
)


# ==================================================
# TABLE 1
# Temporal validation design
# ==================================================

validation_rows = []


split_order = [
    "development_train",
    "validation",
    "final_test",
]


split_labels = {
    "development_train":
        "Development training",

    "validation":
        "Model validation",

    "final_test":
        "Final untouched test",
}


for split in split_order:

    subset = (
        validation.loc[
            validation[
                "split"
            ]
            == split
        ]
    )


    if len(subset) == 0:

        raise RuntimeError(
            f"Missing validation split: "
            f"{split}"
        )


    years = sorted(
        subset[
            "target_year"
        ]
        .unique()
        .tolist()
    )


    year_text = ", ".join(
        str(int(year))
        for year in years
    )


    validation_rows.append({
        "Stage":
            split_labels[
                split
            ],

        "Target year(s)":
            year_text,

        "ICB-year rows":
            len(subset),

        "Unique ICBs":
            subset[
                "icb_code"
            ]
            .nunique(),

        "Purpose":
            {
                "development_train":
                    (
                        "Initial model fitting "
                        "and development"
                    ),

                "validation":
                    (
                        "Model comparison and "
                        "hyperparameter selection"
                    ),

                "final_test":
                    (
                        "Single independent "
                        "out-of-sample evaluation"
                    ),
            }[
                split
            ],
    })


table_01 = pd.DataFrame(
    validation_rows
)


# Add final refit as a separate methodological row.

table_01 = pd.concat(
    [
        table_01,

        pd.DataFrame([
            {
                "Stage":
                    "Final model refit",

                "Target year(s)":
                    "2020, 2021, 2022",

                "ICB-year rows":
                    126,

                "Unique ICBs":
                    42,

                "Purpose":
                    (
                        "Refit locked model "
                        "before 2023 evaluation"
                    ),
            }
        ]),
    ],
    ignore_index=True,
)


table_01.to_csv(
    TABLE_01,
    index=False,
)


print(
    "\nTable 1 created: "
    "Temporal validation design"
)


# ==================================================
# TABLE 2
# Pre-test model selection
# ==================================================

table_02 = (
    model_selection[
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
    .copy()
)


table_02 = table_02.rename(
    columns={
        "candidate":
            "Candidate",

        "model":
            "Model",

        "feature_set":
            "Feature set",

        "alpha":
            "Alpha",

        "mae":
            "MAE",

        "rmse":
            "RMSE",

        "r2":
            "R²",

        "selected":
            "Selected for final test",
    }
)


for column in [
    "MAE",
    "RMSE",
    "R²",
]:

    table_02[
        column
    ] = (
        table_02[
            column
        ]
        .round(3)
    )


table_02.to_csv(
    TABLE_02,
    index=False,
)


print(
    "Table 2 created: "
    "Pre-test model selection"
)


# ==================================================
# TABLE 3
# Temporal out-of-sample performance
#
# 2021 and 2022:
# rolling-origin evaluation
#
# 2023:
# final untouched test
# ==================================================

required_rolling_columns = [
    "evaluation_year",
    "persistence_mae",
    "augmented_mae",
]


for column in required_rolling_columns:

    if column not in rolling.columns:

        raise RuntimeError(
            f"Missing rolling column: "
            f"{column}"
        )


temporal_rows = []


for _, row in (
    rolling
    .sort_values(
        "evaluation_year"
    )
    .iterrows()
):

    year = int(
        row[
            "evaluation_year"
        ]
    )


    persistence_mae = float(
        row[
            "persistence_mae"
        ]
    )


    ridge_mae = float(
        row[
            "augmented_mae"
        ]
    )


    temporal_rows.append({
        "Evaluation year":
            year,

        "Evaluation type":
            (
                "Retrospective "
                "rolling-origin audit"
                if year == 2021
                else
                "Designated validation"
            ),

        "Persistence MAE":
            persistence_mae,

        "Locked Ridge MAE":
            ridge_mae,

        "Persistence - Ridge MAE":
            (
                persistence_mae
                - ridge_mae
            ),

        "Better model":
            (
                "Ridge"
                if ridge_mae
                < persistence_mae
                else
                "Persistence"
            ),
    })


final_persistence = (
    final_test.loc[
        final_test[
            "model"
        ]
        == "persistence"
    ]
)


final_ridge = (
    final_test.loc[
        final_test[
            "model"
        ]
        == "ridge_regression"
    ]
)


if (
    len(final_persistence) != 1
    or
    len(final_ridge) != 1
):

    raise RuntimeError(
        "Could not recover unique "
        "2023 final-test rows."
    )


persistence_2023 = float(
    final_persistence[
        "mae"
    ]
    .iloc[0]
)


ridge_2023 = float(
    final_ridge[
        "mae"
    ]
    .iloc[0]
)


temporal_rows.append({
    "Evaluation year":
        2023,

    "Evaluation type":
        "Untouched final test",

    "Persistence MAE":
        persistence_2023,

    "Locked Ridge MAE":
        ridge_2023,

    "Persistence - Ridge MAE":
        (
            persistence_2023
            - ridge_2023
        ),

    "Better model":
        (
            "Ridge"
            if ridge_2023
            < persistence_2023
            else
            "Persistence"
        ),
})


table_03 = pd.DataFrame(
    temporal_rows
)


for column in [
    "Persistence MAE",
    "Locked Ridge MAE",
    "Persistence - Ridge MAE",
]:

    table_03[
        column
    ] = (
        table_03[
            column
        ]
        .round(3)
    )


table_03.to_csv(
    TABLE_03,
    index=False,
)


print(
    "Table 3 created: "
    "Temporal model performance"
)


# ==================================================
# TABLE 4
# Untouched 2023 final-test performance
# ==================================================

table_04 = (
    final_test[
        [
            "model",
            "feature_set",
            "alpha",
            "mae",
            "rmse",
            "r2",
            "mae_improvement_vs_persistence",
        ]
    ]
    .copy()
)


table_04[
    "Model"
] = table_04[
    "model"
].map({
    "persistence":
        "Persistence",

    "ridge_regression":
        "Locked Ridge",
})


table_04[
    "Feature specification"
] = table_04[
    "feature_set"
].map({
    "previous_year_ckd_only":
        "Previous-year CKD only",

    "lag_plus_compact_5":
        (
            "Previous-year CKD "
            "+ compact 5 NDA"
        ),
})


table_04 = table_04[
    [
        "Model",
        "Feature specification",
        "alpha",
        "mae",
        "rmse",
        "r2",
        "mae_improvement_vs_persistence",
    ]
]


table_04 = table_04.rename(
    columns={
        "alpha":
            "Alpha",

        "mae":
            "MAE",

        "rmse":
            "RMSE",

        "r2":
            "R²",

        "mae_improvement_vs_persistence":
            (
                "MAE improvement "
                "vs persistence"
            ),
    }
)


for column in [
    "MAE",
    "RMSE",
    "R²",
    "MAE improvement vs persistence",
]:

    table_04[
        column
    ] = (
        table_04[
            column
        ]
        .round(3)
    )


table_04.to_csv(
    TABLE_04,
    index=False,
)


print(
    "Table 4 created: "
    "Untouched 2023 final test"
)


# ==================================================
# TABLE 5
# Sensitivity and bootstrap analysis
# ==================================================

unweighted = (
    sensitivity.loc[
        sensitivity[
            "analysis"
        ]
        == "unweighted"
    ]
)


weighted = (
    sensitivity.loc[
        sensitivity[
            "analysis"
        ]
        == "population_weighted"
    ]
)


if (
    len(unweighted) != 1
    or
    len(weighted) != 1
):

    raise RuntimeError(
        "Could not uniquely recover "
        "sensitivity metric rows."
    )


bootstrap_unweighted = (
    bootstrap.loc[
        bootstrap[
            "analysis"
        ]
        == "unweighted_mae_difference"
    ]
)


bootstrap_weighted = (
    bootstrap.loc[
        bootstrap[
            "analysis"
        ]
        ==
        "population_weighted_mae_difference"
    ]
)


if (
    len(bootstrap_unweighted) != 1
    or
    len(bootstrap_weighted) != 1
):

    raise RuntimeError(
        "Could not uniquely recover "
        "bootstrap summary rows."
    )


table_05 = pd.DataFrame([
    {
        "Analysis":
            "Unweighted",

        "Persistence MAE":
            float(
                unweighted[
                    "persistence_mae"
                ]
                .iloc[0]
            ),

        "Ridge MAE":
            float(
                unweighted[
                    "ridge_mae"
                ]
                .iloc[0]
            ),

        "Persistence - Ridge MAE":
            float(
                unweighted[
                    "mae_difference_persistence_minus_ridge"
                ]
                .iloc[0]
            ),

        "Bootstrap 95% lower":
            float(
                bootstrap_unweighted[
                    "ci_2_5_percent"
                ]
                .iloc[0]
            ),

        "Bootstrap 95% upper":
            float(
                bootstrap_unweighted[
                    "ci_97_5_percent"
                ]
                .iloc[0]
            ),

        "Bootstrap P(persistence better)":
            float(
                bootstrap_unweighted[
                    "bootstrap_probability_persistence_better"
                ]
                .iloc[0]
            ),
    },

    {
        "Analysis":
            "Diabetes-population weighted",

        "Persistence MAE":
            float(
                weighted[
                    "persistence_mae"
                ]
                .iloc[0]
            ),

        "Ridge MAE":
            float(
                weighted[
                    "ridge_mae"
                ]
                .iloc[0]
            ),

        "Persistence - Ridge MAE":
            float(
                weighted[
                    "mae_difference_persistence_minus_ridge"
                ]
                .iloc[0]
            ),

        "Bootstrap 95% lower":
            float(
                bootstrap_weighted[
                    "ci_2_5_percent"
                ]
                .iloc[0]
            ),

        "Bootstrap 95% upper":
            float(
                bootstrap_weighted[
                    "ci_97_5_percent"
                ]
                .iloc[0]
            ),

        "Bootstrap P(persistence better)":
            float(
                bootstrap_weighted[
                    "bootstrap_probability_persistence_better"
                ]
                .iloc[0]
            ),
    },
])


for column in [
    "Persistence MAE",
    "Ridge MAE",
    "Persistence - Ridge MAE",
    "Bootstrap 95% lower",
    "Bootstrap 95% upper",
]:

    table_05[
        column
    ] = (
        table_05[
            column
        ]
        .round(3)
    )


table_05[
    "Bootstrap P(persistence better)"
] = (
    table_05[
        "Bootstrap P(persistence better)"
    ]
    .round(3)
)


table_05.to_csv(
    TABLE_05,
    index=False,
)


print(
    "Table 5 created: "
    "Sensitivity and bootstrap analysis"
)


# ==================================================
# TABLE 6
# SHAP global importance
# ==================================================

feature_labels = {

    "lagged_ckd_rate":
        "Previous-year CKD rate",

    "blood_pressure_le_140_80":
        "Blood pressure ≤140/80",

    "serum_creatinine":
        "Serum creatinine care process",

    "urine_albumin":
        "Urine albumin care process",

    "hba1c_le_58_mmol_mol_7_5pct":
        "HbA1c ≤58 mmol/mol",

    "combined_prevention_on_statins":
        "Combined statin prevention",
}


table_06 = (
    shap_global
    .sort_values(
        "global_rank"
    )
    .copy()
)


table_06[
    "Predictor"
] = (
    table_06[
        "feature"
    ]
    .map(
        feature_labels
    )
    .fillna(
        table_06[
            "feature"
        ]
    )
)


table_06 = table_06[
    [
        "global_rank",
        "Predictor",
        "mean_absolute_shap",
        "importance_share_percent",
        "standardized_coefficient",
        "coefficient_direction",
    ]
]


table_06 = table_06.rename(
    columns={
        "global_rank":
            "Rank",

        "mean_absolute_shap":
            "Mean absolute SHAP",

        "importance_share_percent":
            "Importance share (%)",

        "standardized_coefficient":
            "Standardized Ridge coefficient",

        "coefficient_direction":
            "Coefficient direction",
    }
)


table_06[
    "Mean absolute SHAP"
] = (
    table_06[
        "Mean absolute SHAP"
    ]
    .round(3)
)


table_06[
    "Importance share (%)"
] = (
    table_06[
        "Importance share (%)"
    ]
    .round(1)
)


table_06[
    "Standardized Ridge coefficient"
] = (
    table_06[
        "Standardized Ridge coefficient"
    ]
    .round(3)
)


table_06.to_csv(
    TABLE_06,
    index=False,
)


print(
    "Table 6 created: "
    "SHAP global importance"
)


# --------------------------------------------------
# 7. Build final table manifest
# --------------------------------------------------

manifest = pd.DataFrame([
    {
        "table":
            "Table 1",

        "title":
            "Temporal validation design",

        "file":
            TABLE_01.name,

        "recommended_section":
            "Methodology",
    },

    {
        "table":
            "Table 2",

        "title":
            "Pre-test model selection",

        "file":
            TABLE_02.name,

        "recommended_section":
            "Results - Model selection",
    },

    {
        "table":
            "Table 3",

        "title":
            (
                "Temporal out-of-sample "
                "model performance"
            ),

        "file":
            TABLE_03.name,

        "recommended_section":
            "Results - Robustness",
    },

    {
        "table":
            "Table 4",

        "title":
            (
                "Untouched 2023 final-test "
                "performance"
            ),

        "file":
            TABLE_04.name,

        "recommended_section":
            "Results - Final evaluation",
    },

    {
        "table":
            "Table 5",

        "title":
            (
                "Final sensitivity and "
                "bootstrap analysis"
            ),

        "file":
            TABLE_05.name,

        "recommended_section":
            "Results - Sensitivity analysis",
    },

    {
        "table":
            "Table 6",

        "title":
            "Global SHAP feature importance",

        "file":
            TABLE_06.name,

        "recommended_section":
            "Results - Explainability",
    },
])


manifest.to_csv(
    MANIFEST_OUTPUT,
    index=False,
)


# --------------------------------------------------
# 8. Display final tables
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL DISSERTATION TABLES"
)

print(
    "========================================"
)


print(
    "\nTABLE 1 - TEMPORAL VALIDATION DESIGN"
)

print(
    table_01.to_string(
        index=False
    )
)


print(
    "\nTABLE 2 - PRE-TEST MODEL SELECTION"
)

print(
    table_02.to_string(
        index=False
    )
)


print(
    "\nTABLE 3 - TEMPORAL PERFORMANCE"
)

print(
    table_03.to_string(
        index=False
    )
)


print(
    "\nTABLE 4 - FINAL 2023 TEST"
)

print(
    table_04.to_string(
        index=False
    )
)


print(
    "\nTABLE 5 - SENSITIVITY"
)

print(
    table_05.to_string(
        index=False
    )
)


print(
    "\nTABLE 6 - SHAP IMPORTANCE"
)

print(
    table_06.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    "TABLE MANIFEST"
)

print(
    "========================================"
)


print(
    manifest.to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Final checks
# --------------------------------------------------

expected_outputs = [
    TABLE_01,
    TABLE_02,
    TABLE_03,
    TABLE_04,
    TABLE_05,
    TABLE_06,
    MANIFEST_OUTPUT,
]


for file_path in expected_outputs:

    if not file_path.exists():

        raise RuntimeError(
            f"Output was not created: "
            f"{file_path}"
        )


if len(table_01) != 4:

    raise RuntimeError(
        "Expected four validation-design rows."
    )


if len(table_02) != 4:

    raise RuntimeError(
        "Expected four model-selection rows."
    )


if len(table_03) != 3:

    raise RuntimeError(
        "Expected three temporal "
        "evaluation rows."
    )


if len(table_04) != 2:

    raise RuntimeError(
        "Expected two final-test rows."
    )


if len(table_05) != 2:

    raise RuntimeError(
        "Expected two sensitivity rows."
    )


if len(table_06) != 6:

    raise RuntimeError(
        "Expected six SHAP rows."
    )


print(
    "\nFinal dissertation table audit: PASSED"
)


print(
    f"\nAll final tables saved in:"
)

print(
    TABLE_DIR
)