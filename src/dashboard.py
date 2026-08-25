from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetic CKD Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


FINAL_PREDICTIONS_FILE = (
    TABLE_DIR
    / "final_test_predictions.csv"
)

FINAL_METRICS_FILE = (
    TABLE_DIR
    / "final_test_metrics.csv"
)

GLOBAL_SHAP_FILE = (
    TABLE_DIR
    / "final_model_shap_global.csv"
)

LOCAL_SHAP_FILE = (
    TABLE_DIR
    / "final_model_shap_2023.csv"
)

SENSITIVITY_FILE = (
    TABLE_DIR
    / "final_sensitivity_metrics.csv"
)

BOOTSTRAP_FILE = (
    TABLE_DIR
    / "final_sensitivity_bootstrap_summary.csv"
)

ROLLING_FILE = (
    TABLE_DIR
    / "rolling_origin_icb_error_comparison.csv"
)

CKD_HISTORY_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)

MODEL_LOCK_FILE = (
    TABLE_DIR
    / "final_model_lock.csv"
)


# --------------------------------------------------
# 3. Required file check
# --------------------------------------------------

REQUIRED_FILES = [
    FINAL_PREDICTIONS_FILE,
    FINAL_METRICS_FILE,
    GLOBAL_SHAP_FILE,
    LOCAL_SHAP_FILE,
    SENSITIVITY_FILE,
    BOOTSTRAP_FILE,
    ROLLING_FILE,
    CKD_HISTORY_FILE,
    MODEL_LOCK_FILE,
]


missing_files = [
    path
    for path in REQUIRED_FILES
    if not path.exists()
]


if missing_files:

    st.error(
        "Required dissertation output files "
        "are missing."
    )

    for path in missing_files:
        st.code(str(path))

    st.stop()


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

@st.cache_data
def load_data():

    predictions = pd.read_csv(
        FINAL_PREDICTIONS_FILE
    )

    metrics = pd.read_csv(
        FINAL_METRICS_FILE
    )

    shap_global = pd.read_csv(
        GLOBAL_SHAP_FILE
    )

    shap_local = pd.read_csv(
        LOCAL_SHAP_FILE
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILE
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP_FILE
    )

    rolling = pd.read_csv(
        ROLLING_FILE
    )

    ckd_history = pd.read_csv(
        CKD_HISTORY_FILE
    )

    model_lock = pd.read_csv(
        MODEL_LOCK_FILE
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "shap_global": shap_global,
        "shap_local": shap_local,
        "sensitivity": sensitivity,
        "bootstrap": bootstrap,
        "rolling": rolling,
        "ckd_history": ckd_history,
        "model_lock": model_lock,
    }


data = load_data()

predictions = data["predictions"]
metrics = data["metrics"]
shap_global = data["shap_global"]
shap_local = data["shap_local"]
sensitivity = data["sensitivity"]
bootstrap = data["bootstrap"]
rolling = data["rolling"]
ckd_history = data["ckd_history"]
model_lock = data["model_lock"]


# --------------------------------------------------
# 5. Standardise identifiers
# --------------------------------------------------

for dataframe in [
    predictions,
    shap_local,
    ckd_history,
]:

    if "icb_code" in dataframe.columns:

        dataframe["icb_code"] = (
            dataframe["icb_code"]
            .astype("string")
            .str.strip()
        )


# --------------------------------------------------
# 6. Feature labels
# --------------------------------------------------

FEATURE_LABELS = {

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


# --------------------------------------------------
# 7. Recover final metrics
# --------------------------------------------------

persistence_row = (
    metrics.loc[
        metrics["model"]
        == "persistence"
    ]
)


ridge_row = (
    metrics.loc[
        metrics["model"]
        == "ridge_regression"
    ]
)


if (
    len(persistence_row) != 1
    or
    len(ridge_row) != 1
):

    st.error(
        "Final metric rows could not "
        "be uniquely recovered."
    )

    st.stop()


persistence_row = persistence_row.iloc[0]
ridge_row = ridge_row.iloc[0]


persistence_mae = float(
    persistence_row["mae"]
)

persistence_rmse = float(
    persistence_row["rmse"]
)

persistence_r2 = float(
    persistence_row["r2"]
)


ridge_mae = float(
    ridge_row["mae"]
)

ridge_rmse = float(
    ridge_row["rmse"]
)

ridge_r2 = float(
    ridge_row["r2"]
)


ridge_mae_difference = (
    persistence_mae
    - ridge_mae
)


# --------------------------------------------------
# 8. Sidebar navigation
# --------------------------------------------------

st.sidebar.title(
    "Navigation"
)


page = st.sidebar.radio(
    "Select dashboard page",
    [
        "Overview",
        "ICB Explorer",
        "Explainability",
        "Validation & Robustness",
        "Methodology & Limitations",
    ],
)


st.sidebar.divider()


st.sidebar.caption(
    "MSc Data Science dissertation "
    "decision-support prototype."
)


st.sidebar.caption(
    "Population-level analysis only. "
    "Not a clinical decision tool."
)


# ==================================================
# PAGE 1
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title(
        "Population-Level Diabetic CKD "
        "Forecasting Dashboard"
    )

    st.caption(
        "Explainable forecasting of "
        "ICB-level diabetic chronic kidney "
        "disease risk using aggregated "
        "NHS open data."
    )


    st.info(
        "Final independent testing showed that "
        "the simple previous-year persistence "
        "forecast was more accurate than the "
        "locked Ridge machine-learning model "
        "for 2023. The dashboard therefore "
        "presents persistence as the primary "
        "forecasting benchmark and the Ridge "
        "model as an explainable ML comparator."
    )


    # ----------------------------------------------
    # Main metric cards
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Persistence MAE",
            f"{persistence_mae:.3f}",
        )


    with col2:

        st.metric(
            "Locked Ridge MAE",
            f"{ridge_mae:.3f}",
            delta=(
                f"{ridge_mae_difference:.3f} "
                "vs persistence"
            ),
        )


    with col3:

        st.metric(
            "Persistence R²",
            f"{persistence_r2:.3f}",
        )


    with col4:

        st.metric(
            "Locked Ridge R²",
            f"{ridge_r2:.3f}",
        )


    st.divider()


    # ----------------------------------------------
    # Final test comparison
    # ----------------------------------------------

    st.subheader(
        "2023 Final Out-of-Sample Evaluation"
    )


    comparison_df = pd.DataFrame({
        "Model": [
            "Persistence",
            "Locked Ridge",
        ],

        "MAE": [
            persistence_mae,
            ridge_mae,
        ],

        "RMSE": [
            persistence_rmse,
            ridge_rmse,
        ],

        "R²": [
            persistence_r2,
            ridge_r2,
        ],
    })


    st.dataframe(
        comparison_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R²": "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )


    chart_data = (
        comparison_df[
            [
                "Model",
                "MAE",
            ]
        ]
        .set_index(
            "Model"
        )
    )


    st.bar_chart(
        chart_data
    )


    st.caption(
        "Lower MAE indicates better forecast "
        "accuracy. The 2023 test was evaluated "
        "after the Ridge model specification "
        "had been formally locked."
    )


    st.divider()


    # ----------------------------------------------
    # ICB-level result
    # ----------------------------------------------

    ridge_better_count = int(
        predictions[
            "ridge_better_than_persistence"
        ]
        .sum()
    )


    persistence_better_count = (
        len(predictions)
        - ridge_better_count
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "ICBs where persistence was better",
            f"{persistence_better_count} / 42",
        )


    with col2:

        st.metric(
            "ICBs where Ridge was better",
            f"{ridge_better_count} / 42",
        )


    st.divider()


    # ----------------------------------------------
    # Temporal performance
    # ----------------------------------------------

    st.subheader(
        "Performance Across Forecast Origins"
    )


    temporal = (
        rolling[
            [
                "evaluation_year",
                "persistence_mae",
                "augmented_mae",
            ]
        ]
        .copy()
    )


    temporal = pd.concat(
        [
            temporal,

            pd.DataFrame([
                {
                    "evaluation_year":
                        2023,

                    "persistence_mae":
                        persistence_mae,

                    "augmented_mae":
                        ridge_mae,
                }
            ]),
        ],
        ignore_index=True,
    )


    temporal = (
        temporal
        .sort_values(
            "evaluation_year"
        )
        .rename(
            columns={
                "evaluation_year":
                    "Year",

                "persistence_mae":
                    "Persistence",

                "augmented_mae":
                    "Locked Ridge",
            }
        )
        .set_index(
            "Year"
        )
    )


    st.line_chart(
        temporal
    )


    st.caption(
        "The Ridge model improved on persistence "
        "in the designated 2022 validation year, "
        "but the improvement did not generalise "
        "to the earlier 2021 origin or the "
        "untouched 2023 final test."
    )


# ==================================================
# PAGE 2
# ICB EXPLORER
# ==================================================

elif page == "ICB Explorer":

    st.title(
        "ICB Forecast Explorer"
    )


    icb_lookup = (
        predictions[
            [
                "icb_code",
                "icb_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "icb_name"
        )
    )


    icb_options = {
        row["icb_name"]:
            row["icb_code"]

        for _, row
        in icb_lookup.iterrows()
    }


    selected_name = st.selectbox(
        "Select an Integrated Care Board",
        list(
            icb_options.keys()
        ),
    )


    selected_code = (
        icb_options[
            selected_name
        ]
    )


    selected = (
        predictions.loc[
            predictions[
                "icb_code"
            ]
            == selected_code
        ]
        .iloc[0]
    )


    st.subheader(
        selected_name
    )

    st.caption(
        f"ICB code: {selected_code}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Observed 2023 CKD rate",
            (
                f"{selected['actual_ckd_rate']:.2f}"
                " per 1,000"
            ),
        )


    with col2:

        st.metric(
            "Persistence forecast",
            (
                f"{selected['persistence_prediction']:.2f}"
                " per 1,000"
            ),
            delta=(
                f"Absolute error "
                f"{selected['persistence_absolute_error']:.2f}"
            ),
            delta_color="off",
        )


    with col3:

        st.metric(
            "Locked Ridge forecast",
            (
                f"{selected['ridge_prediction']:.2f}"
                " per 1,000"
            ),
            delta=(
                f"Absolute error "
                f"{selected['ridge_absolute_error']:.2f}"
            ),
            delta_color="off",
        )


    # ----------------------------------------------
    # Forecast chart
    # ----------------------------------------------

    st.subheader(
        "2023 Forecast Comparison"
    )


    local_forecast = pd.DataFrame({
        "Estimate": [
            selected[
                "actual_ckd_rate"
            ],

            selected[
                "persistence_prediction"
            ],

            selected[
                "ridge_prediction"
            ],
        ]
    }, index=[
        "Observed",
        "Persistence",
        "Locked Ridge",
    ])


    st.bar_chart(
        local_forecast
    )


    # ----------------------------------------------
    # Historical CKD trend
    # ----------------------------------------------

    st.subheader(
        "Historical CKD Risk"
    )


    icb_history = (
        ckd_history.loc[
            (
                ckd_history[
                    "icb_code"
                ]
                == selected_code
            )
            &
            (
                ckd_history[
                    "year"
                ]
                .between(
                    2019,
                    2023,
                )
            ),
            [
                "year",
                "ckd_risk_rate_per_1000",
            ],
        ]
        .sort_values(
            "year"
        )
        .rename(
            columns={
                "year":
                    "Year",

                "ckd_risk_rate_per_1000":
                    "CKD risk per 1,000",
            }
        )
        .set_index(
            "Year"
        )
    )


    st.line_chart(
        icb_history
    )


    # ----------------------------------------------
    # Local explanation
    # ----------------------------------------------

    st.subheader(
        "Ridge Prediction Explanation"
    )


    local_shap = (
        shap_local.loc[
            shap_local[
                "icb_code"
            ]
            == selected_code
        ]
    )


    if len(local_shap) == 1:

        local_shap = (
            local_shap.iloc[0]
        )


        shap_rows = []


        for feature in FEATURE_LABELS:

            shap_column = (
                f"{feature}_shap"
            )


            value_column = (
                f"{feature}_value"
            )


            if (
                shap_column
                in local_shap.index
            ):

                shap_rows.append({
                    "Predictor":
                        FEATURE_LABELS[
                            feature
                        ],

                    "Observed value":
                        local_shap[
                            value_column
                        ],

                    "SHAP contribution":
                        local_shap[
                            shap_column
                        ],
                })


        local_explanation = (
            pd.DataFrame(
                shap_rows
            )
            .sort_values(
                "SHAP contribution",
                key=lambda x: x.abs(),
                ascending=False,
            )
        )


        st.dataframe(
            local_explanation.style.format({
                "Observed value":
                    "{:.2f}",

                "SHAP contribution":
                    "{:+.3f}",
            }),
            width="stretch",
            hide_index=True,
        )


        st.caption(
            "Positive SHAP values push the "
            "Ridge prediction above its expected "
            "prediction; negative values push it "
            "below. These are model contributions, "
            "not causal effects."
        )


# ==================================================
# PAGE 3
# EXPLAINABILITY
# ==================================================

elif page == "Explainability":

    st.title(
        "Explainability"
    )


    st.info(
        "SHAP is used to describe how the locked "
        "Ridge model generated its predictions. "
        "The analysis explains the model rather "
        "than establishing causal clinical effects."
    )


    # ----------------------------------------------
    # Global importance
    # ----------------------------------------------

    st.subheader(
        "Global SHAP Importance"
    )


    global_display = (
        shap_global
        .sort_values(
            "global_rank"
        )
        .copy()
    )


    global_display[
        "Predictor"
    ] = (
        global_display[
            "feature"
        ]
        .map(
            FEATURE_LABELS
        )
        .fillna(
            global_display[
                "feature"
            ]
        )
    )


    chart = (
        global_display[
            [
                "Predictor",
                "mean_absolute_shap",
            ]
        ]
        .set_index(
            "Predictor"
        )
    )


    st.bar_chart(
        chart
    )


    table = (
        global_display[
            [
                "global_rank",
                "Predictor",
                "mean_absolute_shap",
                "importance_share_percent",
                "standardized_coefficient",
                "coefficient_direction",
            ]
        ]
        .rename(
            columns={
                "global_rank":
                    "Rank",

                "mean_absolute_shap":
                    "Mean |SHAP|",

                "importance_share_percent":
                    "Importance share (%)",

                "standardized_coefficient":
                    "Standardized coefficient",

                "coefficient_direction":
                    "Direction",
            }
        )
    )


    st.dataframe(
        table.style.format({
            "Mean |SHAP|":
                "{:.3f}",

            "Importance share (%)":
                "{:.1f}",

            "Standardized coefficient":
                "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )


    # ----------------------------------------------
    # Dominance statement
    # ----------------------------------------------

    lag_share = (
        global_display.loc[
            global_display[
                "feature"
            ]
            == "lagged_ckd_rate",
            "importance_share_percent",
        ]
    )


    if len(lag_share) == 1:

        st.metric(
            "Previous-year CKD share of "
            "global SHAP importance",
            f"{float(lag_share.iloc[0]):.1f}%",
        )


    driver_counts = (
        shap_local[
            "largest_absolute_shap_feature"
        ]
        .value_counts()
        .rename_axis(
            "feature"
        )
        .reset_index(
            name="ICBs"
        )
    )


    driver_counts[
        "Predictor"
    ] = (
        driver_counts[
            "feature"
        ]
        .map(
            FEATURE_LABELS
        )
        .fillna(
            driver_counts[
                "feature"
            ]
        )
    )


    st.subheader(
        "Largest Local Driver Across ICBs"
    )


    st.dataframe(
        driver_counts[
            [
                "Predictor",
                "ICBs",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


    st.warning(
        "Because the data are aggregated at "
        "ICB level, SHAP contributions must "
        "not be interpreted as individual-level "
        "risk factors or treatment effects."
    )


# ==================================================
# PAGE 4
# VALIDATION AND ROBUSTNESS
# ==================================================

elif page == "Validation & Robustness":

    st.title(
        "Validation & Robustness"
    )


    st.subheader(
        "Temporal Evaluation"
    )


    temporal_table = (
        temporal.reset_index()
        if "temporal" in locals()
        else None
    )


    rolling_table = (
        rolling[
            [
                "evaluation_year",
                "persistence_mae",
                "augmented_mae",
            ]
        ]
        .copy()
    )


    rolling_table = pd.concat(
        [
            rolling_table,

            pd.DataFrame([
                {
                    "evaluation_year":
                        2023,

                    "persistence_mae":
                        persistence_mae,

                    "augmented_mae":
                        ridge_mae,
                }
            ]),
        ],
        ignore_index=True,
    )


    rolling_table[
        "better_model"
    ] = np.where(
        rolling_table[
            "persistence_mae"
        ]
        <
        rolling_table[
            "augmented_mae"
        ],
        "Persistence",
        "Locked Ridge",
    )


    rolling_table = rolling_table.rename(
        columns={
            "evaluation_year":
                "Evaluation year",

            "persistence_mae":
                "Persistence MAE",

            "augmented_mae":
                "Locked Ridge MAE",

            "better_model":
                "Better model",
        }
    )


    st.dataframe(
        rolling_table.style.format({
            "Persistence MAE":
                "{:.3f}",

            "Locked Ridge MAE":
                "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )


    st.caption(
        "2021 is a retrospective rolling-origin "
        "robustness audit, 2022 is the designated "
        "validation year, and 2023 is the untouched "
        "final test."
    )


    st.divider()


    # ----------------------------------------------
    # Sensitivity metrics
    # ----------------------------------------------

    st.subheader(
        "Population-Weighted Sensitivity"
    )


    sensitivity_display = (
        sensitivity.copy()
    )


    sensitivity_display = (
        sensitivity_display.rename(
            columns={
                "analysis":
                    "Analysis",

                "persistence_mae":
                    "Persistence MAE",

                "ridge_mae":
                    "Ridge MAE",

                "persistence_rmse":
                    "Persistence RMSE",

                "ridge_rmse":
                    "Ridge RMSE",

                "mae_difference_persistence_minus_ridge":
                    (
                        "Persistence - Ridge MAE"
                    ),
            }
        )
    )


    st.dataframe(
        sensitivity_display.style.format({
            "Persistence MAE":
                "{:.3f}",

            "Ridge MAE":
                "{:.3f}",

            "Persistence RMSE":
                "{:.3f}",

            "Ridge RMSE":
                "{:.3f}",

            "Persistence - Ridge MAE":
                "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )


    st.caption(
        "A negative Persistence - Ridge MAE "
        "difference indicates that persistence "
        "has the lower MAE."
    )


    st.divider()


    # ----------------------------------------------
    # Bootstrap
    # ----------------------------------------------

    st.subheader(
        "Bootstrap Sensitivity"
    )


    bootstrap_display = (
        bootstrap[
            [
                "analysis",
                "observed_difference",
                "ci_2_5_percent",
                "ci_97_5_percent",
                "bootstrap_probability_persistence_better",
            ]
        ]
        .copy()
        .rename(
            columns={
                "analysis":
                    "Analysis",

                "observed_difference":
                    "Observed MAE difference",

                "ci_2_5_percent":
                    "95% lower",

                "ci_97_5_percent":
                    "95% upper",

                "bootstrap_probability_persistence_better":
                    "P(persistence better)",
            }
        )
    )


    st.dataframe(
        bootstrap_display.style.format({
            "Observed MAE difference":
                "{:.3f}",

            "95% lower":
                "{:.3f}",

            "95% upper":
                "{:.3f}",

            "P(persistence better)":
                "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )


    st.warning(
        "Bootstrap resampling uses ICBs as the "
        "resampling units. It does not eliminate "
        "potential spatial dependence between ICBs."
    )


# ==================================================
# PAGE 5
# METHODOLOGY AND LIMITATIONS
# ==================================================

elif page == "Methodology & Limitations":

    st.title(
        "Methodology & Limitations"
    )


    st.subheader(
        "Locked Machine-Learning Model"
    )


    lock = (
        model_lock.iloc[0]
    )


    st.write(
        "**Model:** Ridge regression"
    )

    st.write(
        f"**Alpha:** {float(lock['alpha']):.1f}"
    )

    st.write(
        "**Feature structure:** "
        "previous-year CKD rate + five "
        "pre-specified NDA indicators"
    )


    locked_feature_table = pd.DataFrame({
        "Predictor": [
            "Previous-year CKD rate",
            FEATURE_LABELS[
                lock["nda_feature_1"]
            ],
            FEATURE_LABELS[
                lock["nda_feature_2"]
            ],
            FEATURE_LABELS[
                lock["nda_feature_3"]
            ],
            FEATURE_LABELS[
                lock["nda_feature_4"]
            ],
            FEATURE_LABELS[
                lock["nda_feature_5"]
            ],
        ]
    })


    st.dataframe(
        locked_feature_table,
        width="stretch",
        hide_index=True,
    )


    st.subheader(
        "Temporal Design"
    )


    st.markdown(
        """
- Development training: target years **2020–2021**
- Designated validation: **2022**
- Model specification locked before the final test
- Final refit: **2020–2022**
- Untouched final test: **2023**
- Unit of analysis: **ICB-year**
"""
    )


    st.subheader(
        "Interpretation"
    )


    st.markdown(
        """
The final 2023 evaluation did not support
the hypothesis that the selected aggregate
NDA care indicators consistently improve
forecast accuracy beyond previous-year CKD
burden.

The persistence benchmark therefore provides
the stronger operational forecast within the
evaluated period. The Ridge model remains useful
for studying the incremental contribution of
the selected population-level indicators and
for demonstrating explainable machine learning.
"""
    )


    st.subheader(
        "Important Limitations"
    )


    st.markdown(
        """
- The analysis uses **aggregated population-level data**, not patient-level records.
- Results must not be interpreted as individual clinical risk predictions.
- Associations and SHAP values are **not causal effects**.
- Historical geographic reconstruction to current ICB boundaries introduces uncertainty.
- The number of temporal forecast origins is small.
- ICB observations may not be fully independent because of geographic and health-system similarities.
- Changes in healthcare access, recording and service delivery may contribute to temporal distribution shifts.
- The CKD outcome reflects recorded population-level burden and may partly depend on healthcare detection and recording processes.
"""
    )


    st.warning(
        "Research prototype only. "
        "Not intended for diagnosis, treatment, "
        "patient-level decision making or direct "
        "clinical deployment."
    )


# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Explainable Machine Learning for "
    "Population-Level Forecasting of "
    "Diabetic Complication Risk Proxies "
    "Using NHS Open Data."
)