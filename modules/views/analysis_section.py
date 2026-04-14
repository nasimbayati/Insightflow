import streamlit as st

from modules.analysis import get_categorical_columns, get_numeric_columns
from modules.reporting import format_outlier_summary
from modules.views.layout import render_section_header
from modules.views.shared import (
    render_bullet_list,
    render_categorical_summary_tabs,
    render_compact_dataframe,
)


def render_analysis_section(analysis_df, analysis_report, chart_recommendations):
    st.divider()
    render_section_header(
        "Data Analysis",
        "Explore the current dataset view through overview metrics, category summaries, numeric stats, and relationship scans.",
        step=8,
    )

    rows, cols = analysis_report["shape"]
    analysis_metrics = st.columns(4)
    analysis_metrics[0].metric("Rows in View", rows)
    analysis_metrics[1].metric("Columns in View", cols)
    analysis_metrics[2].metric("Numeric Columns", len(get_numeric_columns(analysis_df, exclude_id_like=False)))
    analysis_metrics[3].metric("Categorical Columns", len(get_categorical_columns(analysis_df)))
    if rows == 0:
        st.info(
            "The active analysis view has 0 rows. The structure remains visible for transparency, but meaningful analysis and charting resume only after rows are restored."
        )

    analysis_tabs = st.tabs(["Overview", "Numeric", "Categories", "Relationships"])

    with analysis_tabs[0]:
        overview_left, overview_right = st.columns([1, 1], gap="large")

        with overview_left:
            st.write("### Key Findings")
            strongest_correlation = analysis_report["strongest_correlation"]
            if strongest_correlation:
                col_a, col_b = strongest_correlation["columns"]
                st.write(
                    f"- Strongest relationship: **{col_a}** vs **{col_b}** "
                    f"with correlation **{strongest_correlation['value']:.2f}**"
                )
            else:
                st.write("- No strong correlation signal is available yet.")

            if analysis_report["categorical_summary"]:
                first_category, first_summary = next(iter(analysis_report["categorical_summary"].items()))
                st.write(
                    f"- Most informative category summary is currently **{first_category}**, "
                    f"with **{first_summary.index[0]}** as the top value."
                )
            else:
                st.write("- No compact categorical summary was generated for this view.")

        with overview_right:
            outlier_df = format_outlier_summary(analysis_report["outlier_summary"])
            if outlier_df.empty:
                st.success("No strong numeric outliers were detected.")
            else:
                render_compact_dataframe(
                    outlier_df,
                    height=230,
                    title="Outlier snapshot",
                    caption="Columns with the strongest concentration of outliers.",
                )

    with analysis_tabs[1]:
        numeric_left, numeric_right = st.columns([1.1, 0.9], gap="large")

        with numeric_left:
            if analysis_report["numeric_summary"] is not None:
                render_compact_dataframe(
                    analysis_report["numeric_summary"],
                    height=320,
                    title="Numeric summary",
                    caption="Core descriptive statistics for numeric columns.",
                )
            else:
                st.info("No numeric columns are available for summary statistics.")

        with numeric_right:
            outlier_df = format_outlier_summary(analysis_report["outlier_summary"])
            if outlier_df.empty:
                st.success("No strong numeric outliers were detected.")
            else:
                render_compact_dataframe(
                    outlier_df,
                    height=260,
                    title="Outlier scan",
                    caption="IQR-based outlier scan across numeric columns.",
                )

    with analysis_tabs[2]:
        render_categorical_summary_tabs(analysis_report["categorical_summary"])

    with analysis_tabs[3]:
        relationships_left, relationships_right = st.columns([1, 1], gap="large")

        with relationships_left:
            if analysis_report["correlation_matrix"] is not None:
                render_compact_dataframe(
                    analysis_report["correlation_matrix"],
                    height=260,
                    title="Correlation matrix",
                    caption="Pairwise numeric correlations in the active dataset view.",
                )
            else:
                st.info("Not enough numeric columns are available for correlation analysis.")

        with relationships_right:
            st.write("### Recommended Relationship Views")
            relationship_views = [
                recommendation["title"]
                for recommendation in chart_recommendations
                if recommendation["key"] in {"scatter_relationship", "correlation_heatmap", "trend_analysis"}
            ]
            if relationship_views:
                render_bullet_list(relationship_views)
            else:
                st.info("No relationship-focused chart recommendations are available.")
