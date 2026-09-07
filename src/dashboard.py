from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CKD Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


MASTER_FILE = (
    DATA_DIR
    / "ckd_forecasting_master.csv"
)

PREDICTIONS_FILE = (
    TABLE_DIR
    / "final_test_predictions.csv"
)

SHAP_GLOBAL_FILE = (
    TABLE_DIR
    / "final_model_shap_global.csv"
)

SHAP_LOCAL_FILE = (
    TABLE_DIR
    / "final_model_shap_2023.csv"
)

ROLLING_FILE = (
    TABLE_DIR
    / "rolling_origin_metrics.csv"
)


# =========================================================
# 3. DISPLAY LABELS
# =========================================================

FEATURE_LABELS = {
    "lagged_ckd_rate":
        "Previous-year CKD",

    "previous_year_ckd_rate":
        "Previous-year CKD",

    "urine_albumin":
        "Urine albumin monitoring",

    "serum_creatinine":
        "Serum creatinine monitoring",

    "hba1c_le_58_mmol_mol_7_5pct":
        "HbA1c ≤58 mmol/mol",

    "blood_pressure_le_140_80":
        "Blood pressure ≤140/80",

    "combined_prevention_on_statins":
        "Combined statin prevention",
}


# =========================================================
# 4. CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* Main page width and spacing */
    .block-container {
        max-width: 1280px;
        padding-top: 3.4rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(49, 51, 63, 0.12);
    }

    /* Remove unnecessary Streamlit controls for demo */
    [data-testid="stAppDeployButton"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    /* Page label */
    .project-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        opacity: 0.62;
        margin-bottom: 0.45rem;
    }

    /* Main title */
    .main-title {
        font-size: 2.55rem;
        font-weight: 760;
        line-height: 1.12;
        margin-bottom: 0.55rem;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.04rem;
        opacity: 0.72;
        max-width: 960px;
        margin-bottom: 1.6rem;
        line-height: 1.55;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 12px;
        padding: 1rem;
        min-height: 125px;
    }

    /* Slightly reduce metric value size */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    /* Footer */
    .footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(49, 51, 63, 0.12);
        opacity: 0.62;
        font-size: 0.82rem;
    }

    /* Small explanatory text */
    .small-note {
        opacity: 0.72;
        font-size: 0.92rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. HELPERS
# =========================================================

@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def first_existing(
    columns,
    candidates,
):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mae = np.mean(
        np.abs(
            actual
            - predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                actual
                - predicted
            ) ** 2
        )
    )

    denominator = np.sum(
        (
            actual
            - np.mean(actual)
        ) ** 2
    )

    if denominator == 0:

        r2 = np.nan

    else:

        r2 = (
            1
            -
            np.sum(
                (
                    actual
                    - predicted
                ) ** 2
            )
            / denominator
        )

    return (
        mae,
        rmse,
        r2,
    )


def weighted_mae(
    actual,
    predicted,
    weights,
):

    return np.average(
        np.abs(
            np.asarray(actual)
            - np.asarray(predicted)
        ),
        weights=np.asarray(weights),
    )


def page_header(
    title,
    subtitle,
):

    st.markdown(
        """
        <div class="project-label">
        MSc Data Science Dissertation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="main-title">
        {title}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="subtitle">
        {subtitle}
        </div>
        """,
        unsafe_allow_html=True,
    )


def friendly_feature_name(
    feature,
):

    feature = str(feature)

    return FEATURE_LABELS.get(
        feature,
        feature
        .replace("_", " ")
        .title(),
    )


# =========================================================
# 6. LOAD DATA
# =========================================================

master = load_csv(
    MASTER_FILE
)

predictions = load_csv(
    PREDICTIONS_FILE
)

shap_global = load_csv(
    SHAP_GLOBAL_FILE
)

shap_local = load_csv(
    SHAP_LOCAL_FILE
)

rolling = load_csv(
    ROLLING_FILE
)


# =========================================================
# 7. VALIDATE FINAL PREDICTIONS
# =========================================================

if predictions.empty:

    st.error(
        "Final prediction data were not found. "
        "Run `python src/final_test_evaluation.py` first."
    )

    st.stop()


actual_col = first_existing(
    predictions.columns,
    [
        "actual_ckd_rate",
        "observed_ckd_rate_per_1000",
        "target_ckd_rate_per_1000",
        "ckd_risk_rate_per_1000",
    ],
)


persistence_col = first_existing(
    predictions.columns,
    [
        "persistence_prediction",
        "persistence_pred",
    ],
)


ridge_col = first_existing(
    predictions.columns,
    [
        "ridge_prediction",
        "locked_ridge_prediction",
    ],
)


if (
    actual_col is None
    or persistence_col is None
    or ridge_col is None
):

    st.error(
        "The dashboard could not identify the expected "
        "final prediction columns."
    )

    st.write(
        predictions.columns.tolist()
    )

    st.stop()


# =========================================================
# 8. FINAL METRICS
# =========================================================

(
    persistence_mae,
    persistence_rmse,
    persistence_r2,
) = calculate_metrics(
    predictions[actual_col],
    predictions[persistence_col],
)


(
    ridge_mae,
    ridge_rmse,
    ridge_r2,
) = calculate_metrics(
    predictions[actual_col],
    predictions[ridge_col],
)


ridge_mae_gap = (
    ridge_mae
    - persistence_mae
)


ridge_worse_percent = (
    ridge_mae_gap
    / persistence_mae
    * 100
)


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## CKD Forecasting"
    )

    st.caption(
        "Population-level research prototype"
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "ICB Explorer",
            "Explainability",
            "Validation & Robustness",
            "Methodology & Limitations",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        "### Research scope"
    )

    st.markdown(
        """
        Forecasting recorded CKD burden among
        people with type 2 diabetes across
        NHS Integrated Care Boards.
        """
    )

    st.warning(
        "Research prototype only. "
        "Not intended for individual diagnosis "
        "or clinical decision-making."
    )


# =========================================================
# 10. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    page_header(
        "CKD Forecasting Dashboard",
        (
            "Population-level forecasting of chronic kidney disease "
            "burden among people with type 2 diabetes across "
            "42 NHS Integrated Care Boards."
        ),
    )


    st.info(
        "Final 2023 finding: the previous-year persistence "
        "benchmark outperformed the locked Ridge model. "
        "Persistence is therefore presented as the primary "
        "forecasting benchmark, while Ridge is retained as "
        "the explainable machine-learning comparator."
    )


    st.success(
        "**Best final 2023 forecasting approach: Persistence**"
    )


    col1, col2, col3 = st.columns(
        3
    )


    with col1:

        st.metric(
            "Persistence MAE",
            f"{persistence_mae:.3f}",
            help=(
                "Mean Absolute Error. "
                "Lower values indicate better forecasting accuracy."
            ),
        )


    with col2:

        st.metric(
            "Locked Ridge MAE",
            f"{ridge_mae:.3f}",
            delta=(
                f"{ridge_mae_gap:.3f} higher"
            ),
            delta_color="inverse",
        )


    with col3:

        st.metric(
            "ICBs Evaluated",
            f"{len(predictions)}",
        )


    st.caption(
        f"Ridge MAE was {ridge_worse_percent:.1f}% higher "
        "than persistence on the final 2023 test. "
        "Lower MAE is better."
    )


    st.divider()


    # -----------------------------------------------------
    # Final model comparison
    # -----------------------------------------------------

    st.subheader(
        "2023 Final Out-of-Sample Evaluation"
    )

    st.caption(
        "The model specification was locked before "
        "evaluating the 2023 outcome."
    )


    final_table = pd.DataFrame(
        {
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
        }
    )


    st.dataframe(
        final_table,
        column_config={
            "MAE":
                st.column_config.NumberColumn(
                    "MAE",
                    format="%.3f",
                ),

            "RMSE":
                st.column_config.NumberColumn(
                    "RMSE",
                    format="%.3f",
                ),

            "R²":
                st.column_config.NumberColumn(
                    "R²",
                    format="%.3f",
                ),
        },
        hide_index=True,
        width="stretch",
    )


    # Horizontal MAE comparison
    overview_chart_data = (
        final_table[
            [
                "Model",
                "MAE",
            ]
        ]
    )


    overview_chart = (
        alt.Chart(
            overview_chart_data
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "MAE:Q",
                title="Mean Absolute Error (lower is better)",
            ),

            y=alt.Y(
                "Model:N",
                sort=[
                    "Persistence",
                    "Locked Ridge",
                ],
                title=None,
            ),

            tooltip=[
                alt.Tooltip(
                    "Model:N"
                ),
                alt.Tooltip(
                    "MAE:Q",
                    format=".3f",
                ),
            ],
        )
        .properties(
            height=170,
        )
    )


    overview_labels = (
        alt.Chart(
            overview_chart_data
        )
        .mark_text(
            align="left",
            dx=5,
        )
        .encode(
            x="MAE:Q",
            y=alt.Y(
                "Model:N",
                sort=[
                    "Persistence",
                    "Locked Ridge",
                ],
            ),
            text=alt.Text(
                "MAE:Q",
                format=".3f",
            ),
        )
    )


    st.altair_chart(
        overview_chart
        + overview_labels,
        width="stretch",
    )


    st.divider()


    # -----------------------------------------------------
    # Main finding
    # -----------------------------------------------------

    st.subheader(
        "Main Finding"
    )


    left, right = st.columns(
        [
            1.6,
            1,
        ]
    )


    with left:

        st.markdown(
            """
            The Ridge model produced reasonable forecasts,
            but adding the selected diabetes-care indicators
            **did not consistently improve prediction**
            beyond the previous year's CKD burden.

            Ridge slightly outperformed persistence during the
            2022 validation year, but this advantage did not
            generalise to the final 2023 test.

            This demonstrates why machine-learning models should
            be compared against strong and realistic simple
            benchmarks.
            """
        )


    with right:

        st.metric(
            "Persistence R²",
            f"{persistence_r2:.3f}",
        )

        st.metric(
            "Locked Ridge R²",
            f"{ridge_r2:.3f}",
        )


# =========================================================
# 11. ICB EXPLORER
# =========================================================

elif page == "ICB Explorer":

    page_header(
        "ICB Explorer",
        (
            "Explore historical CKD burden, final 2023 forecasts "
            "and local Ridge explanations for individual NHS ICBs."
        ),
    )


    code_col = first_existing(
        predictions.columns,
        [
            "icb_code",
            "ICB code",
        ],
    )


    name_col = first_existing(
        predictions.columns,
        [
            "icb_name",
            "ICB name",
        ],
    )


    if code_col is None:

        st.error(
            "ICB code column was not found."
        )

        st.stop()


    if name_col is not None:

        icb_options = (
            predictions[
                [
                    code_col,
                    name_col,
                ]
            ]
            .drop_duplicates()
            .sort_values(
                name_col
            )
        )


        selected_name = st.selectbox(
            "Select an ICB",
            icb_options[
                name_col
            ].tolist(),
        )


        selected_code = (
            icb_options.loc[
                icb_options[
                    name_col
                ]
                == selected_name,
                code_col,
            ]
            .iloc[0]
        )


    else:

        selected_code = st.selectbox(
            "Select an ICB",
            sorted(
                predictions[
                    code_col
                ]
                .astype(str)
                .unique()
            ),
        )

        selected_name = selected_code


    selected = (
        predictions[
            predictions[
                code_col
            ].astype(str)
            ==
            str(
                selected_code
            )
        ]
        .iloc[0]
    )


    st.subheader(
        selected_name
    )

    st.caption(
        f"ICB code: {selected_code}"
    )


    # -----------------------------------------------------
    # Forecast metrics
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        st.metric(
            "Observed 2023 CKD",
            f"{selected[actual_col]:.1f}",
        )


    with c2:

        st.metric(
            "Persistence Forecast",
            f"{selected[persistence_col]:.1f}",
        )


    with c3:

        st.metric(
            "Locked Ridge Forecast",
            f"{selected[ridge_col]:.1f}",
        )


    st.caption(
        "All three values are CKD rates per 1,000 "
        "people with type 2 diabetes."
    )


    st.divider()


    # -----------------------------------------------------
    # Historical CKD burden
    # -----------------------------------------------------

    st.subheader(
        "Observed CKD Burden Over Time"
    )


    if (
        not master.empty
        and "icb_code" in master.columns
        and "target_year" in master.columns
    ):

        target_col = first_existing(
            master.columns,
            [
                "target_ckd_rate_per_1000",
                "ckd_risk_rate_per_1000",
            ],
        )


        if target_col is not None:

            history = (
                master[
                    master[
                        "icb_code"
                    ].astype(str)
                    ==
                    str(
                        selected_code
                    )
                ][
                    [
                        "target_year",
                        target_col,
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "target_year"
                )
                .rename(
                    columns={
                        "target_year":
                            "Year",

                        target_col:
                            "CKD rate per 1,000",
                    }
                )
            )


            history[
                "Year"
            ] = history[
                "Year"
            ].astype(int)


            history_chart = (
                alt.Chart(
                    history
                )
                .mark_line(
                    point=True,
                )
                .encode(
                    x=alt.X(
                        "Year:O",
                        title="Target year",
                    ),

                    y=alt.Y(
                        "CKD rate per 1,000:Q",
                        title="Recorded CKD rate per 1,000",
                        scale=alt.Scale(
                            zero=False
                        ),
                    ),

                    tooltip=[
                        alt.Tooltip(
                            "Year:O"
                        ),

                        alt.Tooltip(
                            "CKD rate per 1,000:Q",
                            format=".1f",
                        ),
                    ],
                )
                .properties(
                    height=300,
                )
            )


            st.altair_chart(
                history_chart,
                width="stretch",
            )


    st.divider()


    # -----------------------------------------------------
    # Forecast comparison
    # -----------------------------------------------------

    st.subheader(
        "2023 Forecast Comparison"
    )


    comparison = pd.DataFrame(
        {
            "Series": [
                "Observed",
                "Persistence",
                "Locked Ridge",
            ],

            "CKD rate": [
                float(
                    selected[
                        actual_col
                    ]
                ),

                float(
                    selected[
                        persistence_col
                    ]
                ),

                float(
                    selected[
                        ridge_col
                    ]
                ),
            ],
        }
    )


    comparison_chart = (
        alt.Chart(
            comparison
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "CKD rate:Q",
                title="CKD rate per 1,000",
                scale=alt.Scale(
                    zero=True
                ),
            ),

            y=alt.Y(
                "Series:N",
                sort=[
                    "Observed",
                    "Persistence",
                    "Locked Ridge",
                ],
                title=None,
            ),

            tooltip=[
                alt.Tooltip(
                    "Series:N"
                ),

                alt.Tooltip(
                    "CKD rate:Q",
                    format=".1f",
                ),
            ],
        )
        .properties(
            height=220,
        )
    )


    comparison_labels = (
        alt.Chart(
            comparison
        )
        .mark_text(
            align="left",
            dx=5,
        )
        .encode(
            x="CKD rate:Q",

            y=alt.Y(
                "Series:N",
                sort=[
                    "Observed",
                    "Persistence",
                    "Locked Ridge",
                ],
            ),

            text=alt.Text(
                "CKD rate:Q",
                format=".1f",
            ),
        )
    )


    st.altair_chart(
        comparison_chart
        + comparison_labels,
        width="stretch",
    )


    st.divider()


    # -----------------------------------------------------
    # Local SHAP explanation
    # -----------------------------------------------------

    st.subheader(
        "Local Ridge Explanation"
    )

    st.caption(
        "Shows how each predictor pushed the selected "
        "ICB's Ridge forecast above or below the model baseline."
    )


    if (
        not shap_local.empty
        and "icb_code"
        in shap_local.columns
    ):

        local_match = (
            shap_local[
                shap_local[
                    "icb_code"
                ].astype(str)
                ==
                str(
                    selected_code
                )
            ]
        )


        if not local_match.empty:

            local_row = (
                local_match.iloc[0]
            )


            base_col, prediction_col = st.columns(
                2
            )


            with base_col:

                st.metric(
                    "Ridge Baseline Prediction",
                    f"{local_row['shap_base_value']:.1f}",
                    help=(
                        "Expected Ridge prediction before "
                        "the selected ICB's feature contributions "
                        "are added."
                    ),
                )


            with prediction_col:

                st.metric(
                    "Final Ridge Forecast",
                    f"{local_row['ridge_prediction']:.1f}",
                )


            local_feature_map = [
                (
                    "Previous-year CKD",
                    "lagged_ckd_rate_value",
                    "lagged_ckd_rate_shap",
                ),

                (
                    "Urine albumin monitoring",
                    "urine_albumin_value",
                    "urine_albumin_shap",
                ),

                (
                    "Serum creatinine monitoring",
                    "serum_creatinine_value",
                    "serum_creatinine_shap",
                ),

                (
                    "HbA1c ≤58 mmol/mol",
                    "hba1c_le_58_mmol_mol_7_5pct_value",
                    "hba1c_le_58_mmol_mol_7_5pct_shap",
                ),

                (
                    "Blood pressure ≤140/80",
                    "blood_pressure_le_140_80_value",
                    "blood_pressure_le_140_80_shap",
                ),

                (
                    "Combined statin prevention",
                    "combined_prevention_on_statins_value",
                    "combined_prevention_on_statins_shap",
                ),
            ]


            local_rows = []


            for (
                display_name,
                value_column,
                shap_column,
            ) in local_feature_map:

                if (
                    value_column
                    in shap_local.columns
                    and shap_column
                    in shap_local.columns
                ):

                    local_rows.append(
                        {
                            "Feature":
                                display_name,

                            "Feature value":
                                float(
                                    local_row[
                                        value_column
                                    ]
                                ),

                            "SHAP contribution":
                                float(
                                    local_row[
                                        shap_column
                                    ]
                                ),
                        }
                    )


            local_df = pd.DataFrame(
                local_rows
            )


            local_df[
                "Absolute contribution"
            ] = (
                local_df[
                    "SHAP contribution"
                ].abs()
            )


            local_df = (
                local_df
                .sort_values(
                    "Absolute contribution",
                    ascending=False,
                )
                .reset_index(drop=True)
            )


            local_chart = (
                alt.Chart(
                    local_df
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "SHAP contribution:Q",
                        title="SHAP contribution to Ridge forecast",
                    ),

                    y=alt.Y(
                        "Feature:N",
                        sort="-x",
                        title=None,
                    ),

                    tooltip=[
                        alt.Tooltip(
                            "Feature:N"
                        ),

                        alt.Tooltip(
                            "Feature value:Q",
                            format=".2f",
                        ),

                        alt.Tooltip(
                            "SHAP contribution:Q",
                            format="+.3f",
                        ),
                    ],
                )
                .properties(
                    height=300,
                )
            )


            st.altair_chart(
                local_chart,
                width="stretch",
            )


            st.dataframe(
                local_df[
                    [
                        "Feature",
                        "Feature value",
                        "SHAP contribution",
                    ]
                ],
                column_config={
                    "Feature value":
                        st.column_config.NumberColumn(
                            "Feature value",
                            format="%.2f",
                        ),

                    "SHAP contribution":
                        st.column_config.NumberColumn(
                            "SHAP contribution",
                            format="%.3f",
                        ),
                },
                hide_index=True,
                width="stretch",
            )


            dominant_feature = (
                local_row[
                    "largest_absolute_shap_feature"
                ]
            )


            dominant_value = float(
                local_row[
                    "largest_absolute_shap_value"
                ]
            )


            dominant_display = (
                friendly_feature_name(
                    dominant_feature
                )
            )


            st.info(
                f"**Largest local driver:** "
                f"{dominant_display} "
                f"({dominant_value:+.3f} SHAP)."
            )


            st.caption(
                "Positive SHAP values push the Ridge forecast "
                "upward relative to its baseline prediction; "
                "negative values push it downward. "
                "SHAP explains model behaviour and does not "
                "establish causal clinical effects."
            )


        else:

            st.warning(
                "No local SHAP explanation was found "
                "for this ICB."
            )


    else:

        st.warning(
            "Local SHAP output was not found. "
            "Run `python src/final_model_explainability.py`."
        )


# =========================================================
# 12. EXPLAINABILITY PAGE
# =========================================================

elif page == "Explainability":

    page_header(
        "Explainability",
        (
            "Understand which variables contributed most strongly "
            "to the locked Ridge forecasting model."
        ),
    )


    st.warning(
        "SHAP explains the fitted model. "
        "It does not prove causal relationships, "
        "treatment effects or individual clinical risk."
    )


    if shap_global.empty:

        st.error(
            "Global SHAP output was not found."
        )

        st.stop()


    feature_col = first_existing(
        shap_global.columns,
        [
            "feature",
            "Feature",
        ],
    )


    shap_col = first_existing(
        shap_global.columns,
        [
            "mean_absolute_shap",
            "mean_abs_shap",
        ],
    )


    share_col = first_existing(
        shap_global.columns,
        [
            "importance_share_percent",
            "importance_percent",
        ],
    )


    if (
        feature_col is None
        or shap_col is None
    ):

        st.dataframe(
            shap_global,
            width="stretch",
        )

    else:

        global_view = (
            shap_global.copy()
        )


        global_view[
            "Feature"
        ] = global_view[
            feature_col
        ].apply(
            friendly_feature_name
        )


        global_view = (
            global_view
            .sort_values(
                shap_col,
                ascending=False,
            )
            .reset_index(drop=True)
        )


        st.subheader(
            "Global Feature Importance"
        )


        global_chart = (
            alt.Chart(
                global_view
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{shap_col}:Q",
                    title="Mean absolute SHAP value",
                ),

                y=alt.Y(
                    "Feature:N",
                    sort="-x",
                    title=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "Feature:N"
                    ),

                    alt.Tooltip(
                        f"{shap_col}:Q",
                        title="Mean |SHAP|",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=330,
            )
        )


        st.altair_chart(
            global_chart,
            width="stretch",
        )


        display_columns = [
            "Feature",
            shap_col,
        ]


        if share_col is not None:

            display_columns.append(
                share_col
            )


        global_table = (
            global_view[
                display_columns
            ]
            .copy()
        )


        rename_map = {
            shap_col:
                "Mean |SHAP|",
        }


        if share_col is not None:

            rename_map[
                share_col
            ] = "Importance share (%)"


        global_table = (
            global_table
            .rename(
                columns=rename_map
            )
        )


        column_config = {
            "Mean |SHAP|":
                st.column_config.NumberColumn(
                    "Mean |SHAP|",
                    format="%.3f",
                ),
        }


        if (
            "Importance share (%)"
            in global_table.columns
        ):

            column_config[
                "Importance share (%)"
            ] = (
                st.column_config.NumberColumn(
                    "Importance share (%)",
                    format="%.1f",
                )
            )


        st.dataframe(
            global_table,
            column_config=column_config,
            hide_index=True,
            width="stretch",
        )


        top_feature = (
            global_view.iloc[0]
        )


        if share_col is not None:

            st.info(
                f"The dominant global predictor was "
                f"**{top_feature['Feature']}**, "
                f"accounting for approximately "
                f"**{top_feature[share_col]:.1f}%** "
                "of total mean absolute SHAP importance."
            )

        else:

            st.info(
                f"The dominant global predictor was "
                f"**{top_feature['Feature']}**."
            )


        st.markdown(
            """
            **Interpretation**

            The dominance of previous-year CKD indicates that
            historical disease burden contributed much more to
            the Ridge forecasts than any individual diabetes-care
            indicator.

            This supports the final finding that adding the selected
            care indicators did not provide consistent incremental
            forecasting value beyond previous CKD burden.
            """
        )


# =========================================================
# 13. VALIDATION & ROBUSTNESS PAGE
# =========================================================

elif page == "Validation & Robustness":

    page_header(
        "Validation & Robustness",
        (
            "Assess whether model performance remained stable "
            "across forecast years and sensitivity analyses."
        ),
    )


    # -----------------------------------------------------
    # Chronological design
    # -----------------------------------------------------

    st.subheader(
        "Chronological Validation"
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    with col1:

        st.metric(
            "Development",
            "2020–2021",
        )


    with col2:

        st.metric(
            "Validation",
            "2022",
        )


    with col3:

        st.metric(
            "Final Refit",
            "2020–2022",
        )


    with col4:

        st.metric(
            "Final Test",
            "2023",
        )


    st.caption(
        "Chronological validation prevents future observations "
        "from being randomly mixed into earlier model development."
    )


    st.divider()


    # -----------------------------------------------------
    # Rolling-origin performance
    # -----------------------------------------------------

    st.subheader(
        "Temporal Performance"
    )


    rolling_required = {
        "evaluation_year",
        "model",
        "mae",
    }


    if (
        not rolling.empty
        and rolling_required.issubset(
            set(
                rolling.columns
            )
        )
    ):

        temporal = (
            rolling[
                [
                    "evaluation_year",
                    "model",
                    "mae",
                ]
            ]
            .pivot(
                index="evaluation_year",
                columns="model",
                values="mae",
            )
            .reset_index()
        )


        temporal = temporal.rename(
            columns={
                "evaluation_year":
                    "Year",

                "persistence":
                    "Persistence",

                "ridge_lag_plus_compact_5":
                    "Locked Ridge",
            }
        )


        temporal[
            "Stage"
        ] = temporal[
            "Year"
        ].map(
            {
                2021:
                    "Retrospective robustness",

                2022:
                    "Validation",
            }
        )


        final_temporal = pd.DataFrame(
            {
                "Year": [
                    2023
                ],

                "Persistence": [
                    persistence_mae
                ],

                "Locked Ridge": [
                    ridge_mae
                ],

                "Stage": [
                    "Final test"
                ],
            }
        )


        temporal = pd.concat(
            [
                temporal,
                final_temporal,
            ],
            ignore_index=True,
        )


        temporal = (
            temporal
            .sort_values(
                "Year"
            )
            .reset_index(drop=True)
        )


        st.dataframe(
            temporal,
            column_config={
                "Persistence":
                    st.column_config.NumberColumn(
                        "Persistence MAE",
                        format="%.3f",
                    ),

                "Locked Ridge":
                    st.column_config.NumberColumn(
                        "Locked Ridge MAE",
                        format="%.3f",
                    ),
            },
            hide_index=True,
            width="stretch",
        )


        temporal_long = (
            temporal[
                [
                    "Year",
                    "Persistence",
                    "Locked Ridge",
                ]
            ]
            .melt(
                id_vars=[
                    "Year"
                ],

                var_name="Model",

                value_name="MAE",
            )
        )


        temporal_chart = (
            alt.Chart(
                temporal_long
            )
            .mark_line(
                point=True,
            )
            .encode(
                x=alt.X(
                    "Year:O",
                    title="Evaluation year",
                ),

                y=alt.Y(
                    "MAE:Q",
                    title="Mean Absolute Error",
                    scale=alt.Scale(
                        zero=False
                    ),
                ),

                strokeDash=alt.StrokeDash(
                    "Model:N",
                    title="Model",
                ),

                tooltip=[
                    alt.Tooltip(
                        "Year:O"
                    ),

                    alt.Tooltip(
                        "Model:N"
                    ),

                    alt.Tooltip(
                        "MAE:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=300,
            )
        )


        st.altair_chart(
            temporal_chart,
            width="stretch",
        )


        st.info(
            "Ridge performed substantially worse than persistence "
            "in the 2021 robustness check, slightly better in the "
            "2022 validation year, and worse again on the final "
            "2023 test. Incremental ML value was therefore not "
            "temporally consistent."
        )


    else:

        st.warning(
            "Rolling-origin results could not be loaded from "
            "`outputs/tables/rolling_origin_metrics.csv`."
        )

        st.caption(
            "If needed, regenerate them with "
            "`python src/rolling_origin_audit.py`."
        )


    st.divider()


    # -----------------------------------------------------
    # Final paired comparison
    # -----------------------------------------------------

    st.subheader(
        "Final 2023 Robustness"
    )


    persistence_error = np.abs(
        predictions[
            actual_col
        ]
        -
        predictions[
            persistence_col
        ]
    )


    ridge_error = np.abs(
        predictions[
            actual_col
        ]
        -
        predictions[
            ridge_col
        ]
    )


    ridge_better = int(
        (
            ridge_error
            <
            persistence_error
        ).sum()
    )


    persistence_better = int(
        (
            persistence_error
            <
            ridge_error
        ).sum()
    )


    c1, c2 = st.columns(
        2
    )


    with c1:

        st.metric(
            "Persistence Better",
            f"{persistence_better}/42 ICBs",
        )


    with c2:

        st.metric(
            "Ridge Better",
            f"{ridge_better}/42 ICBs",
        )


    # -----------------------------------------------------
    # Population weighted MAE
    # -----------------------------------------------------

    if (
        not master.empty
        and "diabetes_population"
        in master.columns
        and "target_year"
        in master.columns
        and "icb_code"
        in master.columns
        and "icb_code"
        in predictions.columns
    ):

        weights = (
            master[
                master[
                    "target_year"
                ]
                == 2023
            ][
                [
                    "icb_code",
                    "diabetes_population",
                ]
            ]
            .drop_duplicates()
        )


        weighted = (
            predictions.merge(
                weights,
                on="icb_code",
                how="left",
            )
        )


        if (
            weighted[
                "diabetes_population"
            ]
            .notna()
            .all()
        ):

            persistence_weighted = (
                weighted_mae(
                    weighted[
                        actual_col
                    ],

                    weighted[
                        persistence_col
                    ],

                    weighted[
                        "diabetes_population"
                    ],
                )
            )


            ridge_weighted = (
                weighted_mae(
                    weighted[
                        actual_col
                    ],

                    weighted[
                        ridge_col
                    ],

                    weighted[
                        "diabetes_population"
                    ],
                )
            )


            st.markdown(
                "### Population-Weighted MAE"
            )


            c1, c2 = st.columns(
                2
            )


            with c1:

                st.metric(
                    "Persistence",
                    f"{persistence_weighted:.3f}",
                )


            with c2:

                st.metric(
                    "Locked Ridge",
                    f"{ridge_weighted:.3f}",
                    delta=(
                        f"{ridge_weighted - persistence_weighted:.3f} "
                        "higher"
                    ),
                    delta_color="inverse",
                )


    st.success(
        "The sensitivity analyses supported the same overall "
        "conclusion: persistence was the more reliable final "
        "forecasting approach."
    )


    st.caption(
        "The paired ICB bootstrap sensitivity analysis also "
        "favoured persistence, while recognising that spatial "
        "dependence between ICBs may remain."
    )


# =========================================================
# 14. METHODOLOGY & LIMITATIONS PAGE
# =========================================================

elif page == "Methodology & Limitations":

    page_header(
        "Methodology & Limitations",
        (
            "How the longitudinal forecasting dataset was "
            "constructed, evaluated and interpreted."
        ),
    )


    # -----------------------------------------------------
    # Dataset at a glance
    # -----------------------------------------------------

    st.subheader(
        "Dataset at a Glance"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    with c1:

        st.metric(
            "ICBs",
            "42",
        )


    with c2:

        st.metric(
            "Forecast Periods",
            "4",
        )


    with c3:

        st.metric(
            "ICB-Year Rows",
            "168",
        )


    with c4:

        st.metric(
            "Common NDA Indicators",
            "19",
        )


    st.divider()


    # -----------------------------------------------------
    # Pipeline
    # -----------------------------------------------------

    st.subheader(
        "Forecasting Pipeline"
    )


    st.markdown(
        """
        **1. Harmonise annual NDA releases**  
        Historical National Diabetes Audit datasets were
        standardised to common variable definitions and
        a consistent 42-ICB geography.

        **2. Combine years into one longitudinal dataset**  
        The 2018/19, 2019/20, 2020/21 and 2021/22
        predictor datasets were concatenated vertically.

        **3. Align predictors with future CKD burden**

        - 2018/19 indicators → **2020 CKD**
        - 2019/20 indicators → **2021 CKD**
        - 2020/21 indicators → **2022 CKD**
        - 2021/22 indicators → **2023 CKD**

        **4. Add previous-year CKD**  
        Previous-year CKD burden was used both as the
        persistence benchmark and as a predictor in the
        augmented Ridge model.

        **5. Audit and select predictors**  
        Temporal stability, feature redundancy, clinical
        relevance and parsimony were assessed before defining
        the compact five-indicator NDA feature set.

        **6. Compare forecasting approaches**  
        Persistence, Linear Regression, Ridge Regression,
        Random Forest and Gradient Boosting were evaluated.

        **7. Lock the final ML specification**  
        Ridge Regression with **alpha = 1.0** and the compact
        indicator set was locked before final 2023 testing.

        **8. Explain and stress-test the result**  
        SHAP, population-weighted evaluation and paired
        bootstrap sensitivity analysis were used.
        """
    )


    st.divider()


    # -----------------------------------------------------
    # Final predictor set
    # -----------------------------------------------------

    st.subheader(
        "Final Ridge Predictors"
    )


    predictor_table = pd.DataFrame(
        {
            "Predictor": [
                "Previous-year CKD rate",
                "Urine albumin monitoring",
                "Serum creatinine monitoring",
                "HbA1c ≤58 mmol/mol",
                "Blood pressure ≤140/80",
                "Combined statin prevention",
            ],

            "Role": [
                "Previous disease burden",
                "Renal monitoring",
                "Renal monitoring",
                "Glycaemic control",
                "Blood-pressure control",
                "Cardiovascular risk management",
            ],
        }
    )


    st.dataframe(
        predictor_table,
        hide_index=True,
        width="stretch",
    )


    st.caption(
        "Urine albumin and serum creatinine represent "
        "care-process completion percentages, not laboratory values."
    )


    st.divider()


    # -----------------------------------------------------
    # Limitations
    # -----------------------------------------------------

    st.subheader(
        "Important Limitations"
    )


    st.markdown(
        """
        - **Aggregate ICB-level data:** the model does not predict individual patient outcomes.
        - **Ecological fallacy:** area-level relationships cannot be assumed to apply to individual patients.
        - **Limited temporal depth:** only four final forecast periods were available.
        - **Historical geography:** earlier CCG data required harmonisation to ICB geography.
        - **Temporal distribution shift:** NHS service delivery and recording changed across years.
        - **Spatial dependence:** neighbouring ICBs may not be fully independent.
        - **No causal interpretation:** coefficients and SHAP values explain model behaviour, not treatment effects.
        """
    )


    st.info(
        "This dashboard is a population-level research "
        "and service-planning prototype, not a clinical "
        "prediction system."
    )


    st.divider()


    # -----------------------------------------------------
    # Future work
    # -----------------------------------------------------

    st.subheader(
        "Future Development"
    )


    st.markdown(
        """
        Future work could extend the prototype through:

        - additional years of longitudinal NHS data,
        - broader population-health and socioeconomic predictors,
        - explicit spatial and spatiotemporal modelling,
        - prospective validation on newly released future data,
        - and, where governance permits, comparison with
          appropriately protected patient-level datasets.
        """
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    MSc Data Science Dissertation ·
    Public aggregate NHS data ·
    Population-level analysis only
    </div>
    """,
    unsafe_allow_html=True,
)