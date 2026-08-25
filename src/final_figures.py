from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_PREDICTIONS_FILE = (
    TABLE_DIR
    / "final_test_predictions.csv"
)

GLOBAL_SHAP_FILE = (
    TABLE_DIR
    / "final_model_shap_global.csv"
)

CKD_HISTORY_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)

ROLLING_FILE = (
    TABLE_DIR
    / "rolling_origin_icb_error_comparison.csv"
)

FINAL_METRICS_FILE = (
    TABLE_DIR
    / "final_test_metrics.csv"
)

MANIFEST_OUTPUT = (
    TABLE_DIR
    / "final_figure_manifest.csv"
)


# --------------------------------------------------
# 2. Plot settings
# --------------------------------------------------

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(
    fig,
    base_name,
):

    png_path = (
        FIGURE_DIR
        / f"{base_name}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{base_name}.pdf"
    )

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, pdf_path


# --------------------------------------------------
# 3. Check files
# --------------------------------------------------

required_files = [
    FINAL_PREDICTIONS_FILE,
    GLOBAL_SHAP_FILE,
    CKD_HISTORY_FILE,
    ROLLING_FILE,
    FINAL_METRICS_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path}"
        )


# --------------------------------------------------
# 4. Load results
# --------------------------------------------------

predictions = pd.read_csv(
    FINAL_PREDICTIONS_FILE
)

shap_global = pd.read_csv(
    GLOBAL_SHAP_FILE
)

ckd_history = pd.read_csv(
    CKD_HISTORY_FILE
)

rolling = pd.read_csv(
    ROLLING_FILE
)

final_metrics = pd.read_csv(
    FINAL_METRICS_FILE
)


print(
    f"Final predictions: "
    f"{predictions.shape}"
)

print(
    f"Global SHAP: "
    f"{shap_global.shape}"
)

print(
    f"CKD history: "
    f"{ckd_history.shape}"
)


# --------------------------------------------------
# 5. Basic integrity checks
# --------------------------------------------------

if len(predictions) != 42:

    raise RuntimeError(
        "Expected 42 final-test predictions."
    )


if len(shap_global) != 6:

    raise RuntimeError(
        "Expected six SHAP predictor rows."
    )


if set(
    predictions[
        "target_year"
    ]
) != {2023}:

    raise RuntimeError(
        "Final prediction file must contain "
        "only 2023."
    )


figure_manifest = []


# ==================================================
# FIGURE 1
# 2023 observed vs forecast CKD
# ==================================================

sorted_predictions = (
    predictions
    .sort_values(
        "actual_ckd_rate"
    )
    .reset_index(
        drop=True
    )
)


x = np.arange(
    1,
    len(sorted_predictions) + 1,
)


fig, ax = plt.subplots(
    figsize=(10, 5.8)
)


ax.plot(
    x,
    sorted_predictions[
        "actual_ckd_rate"
    ],
    marker="o",
    markersize=3.5,
    linewidth=1.8,
    label="Observed 2023",
)


ax.plot(
    x,
    sorted_predictions[
        "persistence_prediction"
    ],
    marker="s",
    markersize=3,
    linewidth=1.5,
    label="Persistence forecast",
)


ax.plot(
    x,
    sorted_predictions[
        "ridge_prediction"
    ],
    marker="^",
    markersize=3,
    linewidth=1.5,
    label="Locked Ridge forecast",
)


ax.set_title(
    "Observed and Forecast 2023 CKD Risk Across ICBs"
)

ax.set_xlabel(
    "ICBs ordered by observed 2023 CKD risk"
)

ax.set_ylabel(
    "CKD risk rate per 1,000"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.25
)

clean_axes(ax)


png, pdf = save_figure(
    fig,
    "fig_01_2023_observed_vs_forecasts",
)


figure_manifest.append({
    "figure":
        "Figure 1",

    "title":
        (
            "Observed and forecast "
            "2023 CKD risk"
        ),

    "png_file":
        png.name,

    "pdf_file":
        pdf.name,

    "purpose":
        (
            "Compares observed 2023 CKD "
            "rates with persistence and "
            "locked Ridge forecasts."
        ),
})


print(
    "\nFigure 1 saved."
)


# ==================================================
# FIGURE 2
# Paired absolute-error comparison
# ==================================================

persistence_error = (
    predictions[
        "persistence_absolute_error"
    ]
)

ridge_error = (
    predictions[
        "ridge_absolute_error"
    ]
)


limit = float(
    max(
        persistence_error.max(),
        ridge_error.max(),
    )
    * 1.08
)


fig, ax = plt.subplots(
    figsize=(7, 6.5)
)


ax.scatter(
    persistence_error,
    ridge_error,
    s=42,
    alpha=0.8,
)


ax.plot(
    [0, limit],
    [0, limit],
    linestyle="--",
    linewidth=1.2,
    label="Equal absolute error",
)


ax.set_xlim(
    0,
    limit,
)

ax.set_ylim(
    0,
    limit,
)


ax.set_title(
    "ICB-Level 2023 Forecast Error Comparison"
)

ax.set_xlabel(
    "Persistence absolute error"
)

ax.set_ylabel(
    "Locked Ridge absolute error"
)


ridge_better = int(
    predictions[
        "ridge_better_than_persistence"
    ]
    .sum()
)

persistence_better = (
    len(predictions)
    - ridge_better
)


ax.text(
    0.04,
    0.95,
    (
        f"Persistence better: "
        f"{persistence_better}/42\n"
        f"Ridge better: "
        f"{ridge_better}/42"
    ),
    transform=ax.transAxes,
    va="top",
)


ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

clean_axes(ax)


png, pdf = save_figure(
    fig,
    "fig_02_2023_paired_absolute_errors",
)


figure_manifest.append({
    "figure":
        "Figure 2",

    "title":
        "ICB-level absolute-error comparison",

    "png_file":
        png.name,

    "pdf_file":
        pdf.name,

    "purpose":
        (
            "Shows paired forecast errors "
            "for persistence and Ridge "
            "across all 42 ICBs."
        ),
})


print(
    "Figure 2 saved."
)


# ==================================================
# FIGURE 3
# Global SHAP feature importance
# ==================================================

feature_labels = {

    "lagged_ckd_rate":
        "Previous-year CKD rate",

    "blood_pressure_le_140_80":
        "BP ≤140/80",

    "serum_creatinine":
        "Serum creatinine care process",

    "urine_albumin":
        "Urine albumin care process",

    "hba1c_le_58_mmol_mol_7_5pct":
        "HbA1c ≤58 mmol/mol",

    "combined_prevention_on_statins":
        "Combined statin prevention",
}


shap_plot = (
    shap_global
    .sort_values(
        "mean_absolute_shap",
        ascending=False,
    )
    .copy()
)


shap_plot[
    "display_feature"
] = (
    shap_plot[
        "feature"
    ]
    .map(
        feature_labels
    )
    .fillna(
        shap_plot[
            "feature"
        ]
    )
)


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


bars = ax.barh(
    shap_plot[
        "display_feature"
    ],
    shap_plot[
        "mean_absolute_shap"
    ],
)


ax.invert_yaxis()


ax.set_title(
    "Global SHAP Importance for the Locked Ridge Model"
)

ax.set_xlabel(
    "Mean absolute SHAP value"
)

ax.set_ylabel(
    ""
)


for bar, percent in zip(
    bars,
    shap_plot[
        "importance_share_percent"
    ],
):

    ax.text(
        bar.get_width(),
        bar.get_y()
        + bar.get_height() / 2,
        f"  {percent:.1f}%",
        va="center",
        fontsize=9,
    )


clean_axes(ax)


png, pdf = save_figure(
    fig,
    "fig_03_global_shap_importance",
)


figure_manifest.append({
    "figure":
        "Figure 3",

    "title":
        "Global SHAP importance",

    "png_file":
        png.name,

    "pdf_file":
        pdf.name,

    "purpose":
        (
            "Summarises the contribution "
            "of each locked predictor to "
            "2023 Ridge forecasts."
        ),
})


print(
    "Figure 3 saved."
)


# ==================================================
# FIGURE 4
# CKD temporal dynamics
# ==================================================

ckd_recent = (
    ckd_history.loc[
        ckd_history[
            "year"
        ]
        .between(
            2019,
            2023,
        )
    ]
    .copy()
)


year_counts = (
    ckd_recent
    .groupby(
        "year"
    )
    .size()
)


if not (
    year_counts == 42
).all():

    raise RuntimeError(
        "Expected exactly 42 ICBs "
        "for each year 2019-2023."
    )


temporal_summary = (
    ckd_recent
    .groupby(
        "year"
    )[
        "ckd_risk_rate_per_1000"
    ]
    .agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


ax.fill_between(
    temporal_summary[
        "year"
    ],
    temporal_summary[
        "q25"
    ],
    temporal_summary[
        "q75"
    ],
    alpha=0.18,
    label="ICB interquartile range",
)


ax.plot(
    temporal_summary[
        "year"
    ],
    temporal_summary[
        "mean"
    ],
    marker="o",
    linewidth=2,
    label="Mean CKD risk",
)


for _, row in (
    temporal_summary
    .iterrows()
):

    ax.annotate(
        f"{row['mean']:.1f}",
        (
            row["year"],
            row["mean"],
        ),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
    )


ax.set_xticks(
    temporal_summary[
        "year"
    ]
)


ax.set_title(
    "Temporal Distribution of ICB-Level CKD Risk, 2019–2023"
)

ax.set_xlabel(
    "Target year"
)

ax.set_ylabel(
    "CKD risk rate per 1,000"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.25
)

clean_axes(ax)


png, pdf = save_figure(
    fig,
    "fig_04_ckd_temporal_dynamics",
)


figure_manifest.append({
    "figure":
        "Figure 4",

    "title":
        "CKD temporal dynamics",

    "png_file":
        png.name,

    "pdf_file":
        pdf.name,

    "purpose":
        (
            "Displays the systematic "
            "2019-2020 level shift and "
            "subsequent CKD distribution."
        ),
})


print(
    "Figure 4 saved."
)


# ==================================================
# FIGURE 5
# Out-of-sample MAE across forecast origins
# ==================================================

required_rolling_columns = [
    "evaluation_year",
    "persistence_mae",
    "augmented_mae",
]


for column in required_rolling_columns:

    if column not in rolling.columns:

        raise RuntimeError(
            f"Missing rolling-origin column: "
            f"{column}"
        )


rolling_plot = (
    rolling[
        required_rolling_columns
    ]
    .copy()
)


persistence_2023 = (
    final_metrics.loc[
        final_metrics[
            "model"
        ]
        == "persistence",
        "mae",
    ]
)


ridge_2023 = (
    final_metrics.loc[
        final_metrics[
            "model"
        ]
        == "ridge_regression",
        "mae",
    ]
)


if (
    len(persistence_2023) != 1
    or
    len(ridge_2023) != 1
):

    raise RuntimeError(
        "Could not uniquely recover "
        "2023 final-test metrics."
    )


year_2023_row = pd.DataFrame([
    {
        "evaluation_year":
            2023,

        "persistence_mae":
            float(
                persistence_2023.iloc[0]
            ),

        "augmented_mae":
            float(
                ridge_2023.iloc[0]
            ),
    }
])


performance_plot = pd.concat(
    [
        rolling_plot,
        year_2023_row,
    ],
    ignore_index=True,
)


performance_plot = (
    performance_plot
    .sort_values(
        "evaluation_year"
    )
    .reset_index(
        drop=True
    )
)


if list(
    performance_plot[
        "evaluation_year"
    ]
) != [
    2021,
    2022,
    2023,
]:

    raise RuntimeError(
        "Expected evaluation years "
        "2021, 2022 and 2023."
    )


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


ax.plot(
    performance_plot[
        "evaluation_year"
    ],
    performance_plot[
        "persistence_mae"
    ],
    marker="o",
    markersize=7,
    linewidth=2,
    label="Persistence",
)


ax.plot(
    performance_plot[
        "evaluation_year"
    ],
    performance_plot[
        "augmented_mae"
    ],
    marker="s",
    markersize=7,
    linewidth=2,
    label="Ridge + lag + compact NDA",
)


for _, row in (
    performance_plot
    .iterrows()
):

    ax.annotate(
        f"{row['persistence_mae']:.2f}",
        (
            row["evaluation_year"],
            row["persistence_mae"],
        ),
        textcoords="offset points",
        xytext=(-16, 8),
        fontsize=8.5,
    )

    ax.annotate(
        f"{row['augmented_mae']:.2f}",
        (
            row["evaluation_year"],
            row["augmented_mae"],
        ),
        textcoords="offset points",
        xytext=(7, 8),
        fontsize=8.5,
    )


ax.set_xticks(
    [
        2021,
        2022,
        2023,
    ]
)


ax.set_title(
    "Out-of-Sample MAE Across Temporal Forecast Origins"
)

ax.set_xlabel(
    "Evaluation target year"
)

ax.set_ylabel(
    "Mean absolute error"
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.25
)

clean_axes(ax)


png, pdf = save_figure(
    fig,
    "fig_05_temporal_model_performance",
)


figure_manifest.append({
    "figure":
        "Figure 5",

    "title":
        (
            "Temporal out-of-sample "
            "model performance"
        ),

    "png_file":
        png.name,

    "pdf_file":
        pdf.name,

    "purpose":
        (
            "Shows that the Ridge "
            "improvement seen in 2022 "
            "did not generalise across "
            "forecast origins."
        ),
})


print(
    "Figure 5 saved."
)


# --------------------------------------------------
# 6. Save figure manifest
# --------------------------------------------------

manifest_df = pd.DataFrame(
    figure_manifest
)


manifest_df.to_csv(
    MANIFEST_OUTPUT,
    index=False,
)


# --------------------------------------------------
# 7. Final output summary
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "FINAL FIGURE GENERATION"
)

print(
    "========================================"
)


print(
    "\nFigures generated:"
)


print(
    manifest_df[
        [
            "figure",
            "title",
            "png_file",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nAll figures saved as:"
)

print(
    "- 300 DPI PNG"
)

print(
    "- vector PDF"
)


print(
    f"\nFigure directory:"
)

print(
    FIGURE_DIR
)


print(
    f"\nFigure manifest:"
)

print(
    MANIFEST_OUTPUT
)


print(
    "\nFinal figure audit: PASSED"
)