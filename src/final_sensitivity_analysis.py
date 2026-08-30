from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ckd_forecasting_master.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)


METRICS_OUTPUT = (
    OUTPUT_DIR
    / "final_sensitivity_metrics.csv"
)

PAIRED_OUTPUT = (
    OUTPUT_DIR
    / "final_sensitivity_paired_errors.csv"
)

BOOTSTRAP_OUTPUT = (
    OUTPUT_DIR
    / "final_sensitivity_bootstrap.csv"
)

BOOTSTRAP_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "final_sensitivity_bootstrap_summary.csv"
)


# --------------------------------------------------
# 2. Analysis settings
# --------------------------------------------------

FINAL_TEST_YEAR = 2023

N_BOOTSTRAP = 10000

RANDOM_STATE = 42


# --------------------------------------------------
# 3. Load files
# --------------------------------------------------

for file_path in [
    PANEL_FILE,
    PREDICTIONS_FILE,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


panel = pd.read_csv(
    PANEL_FILE
)

predictions = pd.read_csv(
    PREDICTIONS_FILE
)


print(
    f"Forecasting panel shape: {panel.shape}"
)

print(
    f"Final prediction shape: {predictions.shape}"
)


# --------------------------------------------------
# 4. Standardise ICB codes
# --------------------------------------------------

panel["icb_code"] = (
    panel["icb_code"]
    .astype("string")
    .str.strip()
)

predictions["icb_code"] = (
    predictions["icb_code"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 5. Recover 2023 diabetes population
#
# This is used ONLY as an evaluation weight.
# It is not added as a model predictor.
# --------------------------------------------------

population = (
    panel.loc[
        panel[
            "target_year"
        ]
        == FINAL_TEST_YEAR,
        [
            "icb_code",
            "icb_name",
            "diabetes_population",
        ],
    ]
    .copy()
)


if len(population) != 42:
    raise RuntimeError(
        f"Expected 42 population rows, "
        f"found {len(population)}."
    )


if population[
    "icb_code"
].nunique() != 42:
    raise RuntimeError(
        "Expected 42 unique ICBs."
    )


if population[
    "diabetes_population"
].isna().any():
    raise RuntimeError(
        "Missing diabetes population values."
    )


if (
    population[
        "diabetes_population"
    ]
    <= 0
).any():
    raise RuntimeError(
        "Non-positive diabetes population detected."
    )


# --------------------------------------------------
# 6. Merge final predictions with population
# --------------------------------------------------

analysis = predictions.merge(
    population[
        [
            "icb_code",
            "diabetes_population",
        ]
    ],
    on="icb_code",
    how="left",
    validate="one_to_one",
)


if len(analysis) != 42:
    raise RuntimeError(
        f"Expected 42 final-test rows, "
        f"found {len(analysis)}."
    )


required_columns = [
    "actual_ckd_rate",
    "persistence_prediction",
    "ridge_prediction",
    "diabetes_population",
]


if analysis[
    required_columns
].isna().any().any():
    raise RuntimeError(
        "Missing values detected in "
        "sensitivity-analysis data."
    )


print(
    "\n2023 final-test rows: 42"
)

print(
    "Population weighting data: PASSED"
)


# --------------------------------------------------
# 7. Error columns
# --------------------------------------------------

analysis[
    "persistence_error"
] = (
    analysis[
        "actual_ckd_rate"
    ]
    -
    analysis[
        "persistence_prediction"
    ]
)


analysis[
    "ridge_error"
] = (
    analysis[
        "actual_ckd_rate"
    ]
    -
    analysis[
        "ridge_prediction"
    ]
)


analysis[
    "persistence_absolute_error"
] = (
    analysis[
        "persistence_error"
    ]
    .abs()
)


analysis[
    "ridge_absolute_error"
] = (
    analysis[
        "ridge_error"
    ]
    .abs()
)


analysis[
    "absolute_error_difference"
] = (
    analysis[
        "persistence_absolute_error"
    ]
    -
    analysis[
        "ridge_absolute_error"
    ]
)


analysis[
    "ridge_better"
] = (
    analysis[
        "absolute_error_difference"
    ]
    > 0
)


# --------------------------------------------------
# 8. Metric helpers
# --------------------------------------------------

def mae(actual, predicted):

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (
                actual - predicted
            )
            ** 2
        )
    )


def weighted_mae(
    actual,
    predicted,
    weights,
):

    absolute_errors = np.abs(
        actual - predicted
    )

    return np.average(
        absolute_errors,
        weights=weights,
    )


def weighted_rmse(
    actual,
    predicted,
    weights,
):

    squared_errors = (
        actual - predicted
    ) ** 2

    return np.sqrt(
        np.average(
            squared_errors,
            weights=weights,
        )
    )


# --------------------------------------------------
# 9. Arrays
# --------------------------------------------------

actual = (
    analysis[
        "actual_ckd_rate"
    ]
    .to_numpy()
)


persistence_pred = (
    analysis[
        "persistence_prediction"
    ]
    .to_numpy()
)


ridge_pred = (
    analysis[
        "ridge_prediction"
    ]
    .to_numpy()
)


weights = (
    analysis[
        "diabetes_population"
    ]
    .to_numpy(
        dtype=float
    )
)


# --------------------------------------------------
# 10. Unweighted metrics
# --------------------------------------------------

persistence_mae = mae(
    actual,
    persistence_pred,
)

ridge_mae = mae(
    actual,
    ridge_pred,
)


persistence_rmse = rmse(
    actual,
    persistence_pred,
)

ridge_rmse = rmse(
    actual,
    ridge_pred,
)


unweighted_mae_difference = (
    persistence_mae
    -
    ridge_mae
)


# Positive difference means Ridge is better.
# Negative difference means persistence is better.


# --------------------------------------------------
# 11. Population-weighted metrics
# --------------------------------------------------

persistence_weighted_mae = (
    weighted_mae(
        actual,
        persistence_pred,
        weights,
    )
)


ridge_weighted_mae = (
    weighted_mae(
        actual,
        ridge_pred,
        weights,
    )
)


persistence_weighted_rmse = (
    weighted_rmse(
        actual,
        persistence_pred,
        weights,
    )
)


ridge_weighted_rmse = (
    weighted_rmse(
        actual,
        ridge_pred,
        weights,
    )
)


weighted_mae_difference = (
    persistence_weighted_mae
    -
    ridge_weighted_mae
)


# --------------------------------------------------
# 12. ICB-level paired comparison
# --------------------------------------------------

ridge_better_count = int(
    analysis[
        "ridge_better"
    ]
    .sum()
)


persistence_better_count = int(
    (
        analysis[
            "absolute_error_difference"
        ]
        < 0
    )
    .sum()
)


equal_count = int(
    (
        analysis[
            "absolute_error_difference"
        ]
        == 0
    )
    .sum()
)


median_error_difference = float(
    analysis[
        "absolute_error_difference"
    ]
    .median()
)


mean_error_difference = float(
    analysis[
        "absolute_error_difference"
    ]
    .mean()
)


# --------------------------------------------------
# 13. Bootstrap paired MAE differences
#
# Resample ICBs with replacement.
#
# Difference:
# persistence MAE - Ridge MAE
#
# > 0 means Ridge is better
# < 0 means persistence is better
#
# This is a sensitivity analysis, not proof
# of independence between ICBs.
# --------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)


n_icbs = len(
    analysis
)


bootstrap_rows = []


for bootstrap_id in range(
    1,
    N_BOOTSTRAP + 1,
):

    sampled_indices = (
        rng.integers(
            low=0,
            high=n_icbs,
            size=n_icbs,
        )
    )


    sampled_actual = (
        actual[
            sampled_indices
        ]
    )


    sampled_persistence = (
        persistence_pred[
            sampled_indices
        ]
    )


    sampled_ridge = (
        ridge_pred[
            sampled_indices
        ]
    )


    sampled_weights = (
        weights[
            sampled_indices
        ]
    )


    bootstrap_persistence_mae = (
        mae(
            sampled_actual,
            sampled_persistence,
        )
    )


    bootstrap_ridge_mae = (
        mae(
            sampled_actual,
            sampled_ridge,
        )
    )


    bootstrap_unweighted_difference = (
        bootstrap_persistence_mae
        -
        bootstrap_ridge_mae
    )


    bootstrap_persistence_weighted_mae = (
        weighted_mae(
            sampled_actual,
            sampled_persistence,
            sampled_weights,
        )
    )


    bootstrap_ridge_weighted_mae = (
        weighted_mae(
            sampled_actual,
            sampled_ridge,
            sampled_weights,
        )
    )


    bootstrap_weighted_difference = (
        bootstrap_persistence_weighted_mae
        -
        bootstrap_ridge_weighted_mae
    )


    bootstrap_rows.append({
        "bootstrap_id":
            bootstrap_id,

        "unweighted_mae_difference":
            bootstrap_unweighted_difference,

        "weighted_mae_difference":
            bootstrap_weighted_difference,
    })


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# --------------------------------------------------
# 14. Bootstrap uncertainty summary
# --------------------------------------------------

def bootstrap_summary(
    series,
    observed_difference,
):

    lower = float(
        np.percentile(
            series,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            series,
            97.5,
        )
    )

    median = float(
        np.median(
            series
        )
    )

    probability_ridge_better = float(
        np.mean(
            series > 0
        )
    )

    probability_persistence_better = float(
        np.mean(
            series < 0
        )
    )

    return {
        "observed_difference":
            observed_difference,

        "bootstrap_median_difference":
            median,

        "ci_2_5_percent":
            lower,

        "ci_97_5_percent":
            upper,

        "bootstrap_probability_ridge_better":
            probability_ridge_better,

        "bootstrap_probability_persistence_better":
            probability_persistence_better,
    }


unweighted_bootstrap_summary = (
    bootstrap_summary(
        bootstrap_df[
            "unweighted_mae_difference"
        ],
        unweighted_mae_difference,
    )
)


weighted_bootstrap_summary = (
    bootstrap_summary(
        bootstrap_df[
            "weighted_mae_difference"
        ],
        weighted_mae_difference,
    )
)


bootstrap_summary_df = pd.DataFrame([
    {
        "analysis":
            "unweighted_mae_difference",

        **unweighted_bootstrap_summary,
    },

    {
        "analysis":
            "population_weighted_mae_difference",

        **weighted_bootstrap_summary,
    },
])


# --------------------------------------------------
# 15. Main sensitivity metrics table
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "analysis":
            "unweighted",

        "persistence_mae":
            persistence_mae,

        "ridge_mae":
            ridge_mae,

        "persistence_rmse":
            persistence_rmse,

        "ridge_rmse":
            ridge_rmse,

        "mae_difference_persistence_minus_ridge":
            unweighted_mae_difference,
    },

    {
        "analysis":
            "population_weighted",

        "persistence_mae":
            persistence_weighted_mae,

        "ridge_mae":
            ridge_weighted_mae,

        "persistence_rmse":
            persistence_weighted_rmse,

        "ridge_rmse":
            ridge_weighted_rmse,

        "mae_difference_persistence_minus_ridge":
            weighted_mae_difference,
    },
])


# --------------------------------------------------
# 16. Print results
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL 2023 SENSITIVITY ANALYSIS"
)

print(
    "========================================"
)


print(
    "\nUnweighted metrics:"
)

print(
    f"Persistence MAE  = "
    f"{persistence_mae:.3f}"
)

print(
    f"Ridge MAE        = "
    f"{ridge_mae:.3f}"
)

print(
    f"MAE difference   = "
    f"{unweighted_mae_difference:.3f}"
)

print(
    f"Persistence RMSE = "
    f"{persistence_rmse:.3f}"
)

print(
    f"Ridge RMSE       = "
    f"{ridge_rmse:.3f}"
)


print(
    "\nPopulation-weighted metrics:"
)

print(
    f"Persistence weighted MAE  = "
    f"{persistence_weighted_mae:.3f}"
)

print(
    f"Ridge weighted MAE        = "
    f"{ridge_weighted_mae:.3f}"
)

print(
    f"Weighted MAE difference   = "
    f"{weighted_mae_difference:.3f}"
)

print(
    f"Persistence weighted RMSE = "
    f"{persistence_weighted_rmse:.3f}"
)

print(
    f"Ridge weighted RMSE       = "
    f"{ridge_weighted_rmse:.3f}"
)


print(
    "\nICB-level paired comparison:"
)

print(
    f"Ridge better       : "
    f"{ridge_better_count}/42"
)

print(
    f"Persistence better : "
    f"{persistence_better_count}/42"
)

print(
    f"Equal              : "
    f"{equal_count}/42"
)

print(
    f"Mean paired error difference   : "
    f"{mean_error_difference:.3f}"
)

print(
    f"Median paired error difference : "
    f"{median_error_difference:.3f}"
)


# --------------------------------------------------
# 17. Print bootstrap results
# --------------------------------------------------

print(
    "\nBootstrap uncertainty:"
)

for _, row in (
    bootstrap_summary_df
    .iterrows()
):

    print(
        "\n"
        f"{row['analysis']}"
    )

    print(
        f"Observed difference: "
        f"{row['observed_difference']:.3f}"
    )

    print(
        f"95% bootstrap interval: "
        f"[{row['ci_2_5_percent']:.3f}, "
        f"{row['ci_97_5_percent']:.3f}]"
    )

    print(
        f"P(Ridge better): "
        f"{row['bootstrap_probability_ridge_better']:.3f}"
    )

    print(
        f"P(Persistence better): "
        f"{row['bootstrap_probability_persistence_better']:.3f}"
    )


# --------------------------------------------------
# 18. Evidence-based conclusion
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "SENSITIVITY CONCLUSION"
)

print(
    "========================================"
)


if (
    weighted_mae_difference < 0
    and
    unweighted_mae_difference < 0
):

    print(
        "Persistence outperformed the locked "
        "Ridge model under both unweighted and "
        "diabetes-population-weighted MAE."
    )

else:

    print(
        "The ranking of persistence and Ridge "
        "changed under the population-weighted "
        "sensitivity analysis."
    )


unweighted_ci_lower = (
    unweighted_bootstrap_summary[
        "ci_2_5_percent"
    ]
)

unweighted_ci_upper = (
    unweighted_bootstrap_summary[
        "ci_97_5_percent"
    ]
)


if unweighted_ci_upper < 0:

    print(
        "The paired ICB bootstrap distribution "
        "also favoured persistence across the "
        "95% percentile interval."
    )

elif unweighted_ci_lower > 0:

    print(
        "The paired ICB bootstrap distribution "
        "favoured Ridge across the 95% "
        "percentile interval."
    )

else:

    print(
        "The 95% paired ICB bootstrap interval "
        "crossed zero, indicating uncertainty "
        "about the exact magnitude of the "
        "performance difference."
    )


print(
    "\nBootstrap resampling treats ICBs as the "
    "resampling units and is presented as a "
    "sensitivity analysis. It does not remove "
    "possible spatial dependence between ICBs."
)


# --------------------------------------------------
# 19. Round saved outputs
# --------------------------------------------------

for dataframe in [
    metrics_df,
    analysis,
    bootstrap_df,
    bootstrap_summary_df,
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
# 20. Integrity checks
# --------------------------------------------------

if len(
    analysis
) != 42:
    raise RuntimeError(
        "Expected 42 paired ICB rows."
    )


if (
    ridge_better_count
    +
    persistence_better_count
    +
    equal_count
) != 42:
    raise RuntimeError(
        "ICB comparison counts do not sum to 42."
    )


if len(
    bootstrap_df
) != N_BOOTSTRAP:
    raise RuntimeError(
        "Bootstrap iteration count mismatch."
    )


if bootstrap_df[
    [
        "unweighted_mae_difference",
        "weighted_mae_difference",
    ]
].isna().any().any():
    raise RuntimeError(
        "Missing bootstrap values detected."
    )


print(
    "\nFinal sensitivity audit: PASSED"
)


# --------------------------------------------------
# 21. Save outputs
# --------------------------------------------------

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False,
)


analysis.to_csv(
    PAIRED_OUTPUT,
    index=False,
)


bootstrap_df.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False,
)


bootstrap_summary_df.to_csv(
    BOOTSTRAP_SUMMARY_OUTPUT,
    index=False,
)


print(
    f"\nSaved sensitivity metrics to: "
    f"{METRICS_OUTPUT}"
)

print(
    f"Saved paired ICB errors to: "
    f"{PAIRED_OUTPUT}"
)

print(
    f"Saved bootstrap distribution to: "
    f"{BOOTSTRAP_OUTPUT}"
)

print(
    f"Saved bootstrap summary to: "
    f"{BOOTSTRAP_SUMMARY_OUTPUT}"
)