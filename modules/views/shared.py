import io
from html import escape

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from modules.analysis import (
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    select_categorical_chart_column,
    select_numeric_chart_column,
    select_scatter_columns,
    select_time_series_columns,
)
from modules.reporting import format_outlier_summary
from modules.visualization import (
    build_grouped_metric_summary,
    plot_boxplot,
    plot_boxplot_by_group,
    plot_categorical_bar,
    plot_correlation_heatmap,
    plot_grouped_bar,
    plot_grouped_line,
    plot_numeric_histogram,
    plot_scatter,
    plot_time_series,
)


def render_block_header(title=None, caption=None):
    if not title and not caption:
        return

    parts = ['<div class="if-block-head">']
    if title:
        parts.append(f'<div class="if-block-title">{escape(str(title))}</div>')
    if caption:
        parts.append(f'<div class="if-block-caption">{escape(str(caption))}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_badge_row(items):
    if not items:
        return

    badges = "".join(f'<span class="if-badge">{escape(str(item))}</span>' for item in items if str(item).strip())
    if badges:
        st.markdown(f'<div class="if-badge-row">{badges}</div>', unsafe_allow_html=True)


def render_assumptions_bar(chart_view_mode, pipeline_preferences, applied_filters=None):
    roles = getattr(pipeline_preferences, "column_roles", None)
    items = [f"View: {chart_view_mode}"]

    duplicate_mode = pipeline_preferences.get("duplicate_rule_mode", "exact")
    duplicate_subset = pipeline_preferences.get("duplicate_subset")
    if duplicate_subset:
        items.append(f"Duplicate key: {', '.join(duplicate_subset)}")
    else:
        items.append("Duplicate key: exact rows")

    if roles is not None:
        if roles.id_columns:
            items.append(f"ID: {', '.join(roles.id_columns)}")
        if roles.time_column:
            items.append(f"Time: {roles.time_column}")
        if roles.metric_column:
            items.append(f"Metric: {roles.metric_column}")
        if roles.segment_column:
            items.append(f"Segment: {roles.segment_column}")
        if roles.outcome_column:
            items.append(f"Outcome: {roles.outcome_column}")

    if applied_filters:
        items.append(f"Filters: {len(applied_filters)} active")
    else:
        items.append("Filters: none")

    if duplicate_mode == "exact" and (roles is None or not roles.has_user_roles()):
        caption = "InsightFlow is using conservative defaults. Add column roles when you know the business meaning of the fields."
    else:
        caption = "Current interpretation assumptions. Change these controls if the file should be read differently."

    with st.container(border=True):
        render_block_header("Current Interpretation", caption)
        render_badge_row(items)


def render_score_breakdown(title, breakdown_items, caption=None):
    with st.container(border=True):
        render_block_header(title, caption)
        if not breakdown_items:
            st.info("No score details are available.")
            return

        for item in breakdown_items:
            penalty = item.get("penalty", 0)
            prefix = f"-{penalty}" if penalty else "0"
            st.write(f"- {item['label']}: {prefix} points")
            st.caption(item.get("detail", ""))


def render_review_mode_selector():
    st.markdown(
        """
        <div class="if-review-switch">
            <div class="if-review-title">Review mode</div>
            <div class="if-review-copy">Choose how much detail you want. Start with Boardroom Brief for the fastest judging path, then switch to Evidence or Full Audit when you need more proof.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.radio(
        "Review mode",
        ["Boardroom Brief", "Evidence", "Full Audit"],
        horizontal=True,
        label_visibility="collapsed",
    )


def prepare_display_data(data, index_label="Value"):
    if isinstance(data, pd.Series):
        value_label = data.name or "Value"
        display_data = data.reset_index(name=value_label)
        first_column = display_data.columns[0]
        return display_data.rename(columns={first_column: index_label})

    if isinstance(data, pd.DataFrame):
        display_data = data.copy()
        numeric_columns = display_data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            display_data[numeric_columns] = display_data[numeric_columns].round(3)
        return display_data

    return data


def render_compact_dataframe(data, height=None, title=None, caption=None, index_label="Value"):
    data = prepare_display_data(data, index_label=index_label)

    if data is None:
        return

    if isinstance(data, pd.DataFrame) and data.empty:
        st.info("No rows to display.")
        return

    row_count = len(data) if hasattr(data, "__len__") else 0

    with st.container(border=True):
        render_block_header(title, caption)
        if isinstance(data, pd.DataFrame) and row_count <= 8 and height is None:
            st.table(data.style.hide(axis="index"))
        else:
            computed_height = height or min(44 + row_count * 34, 310)
            st.dataframe(
                data,
                width="stretch",
                hide_index=True,
                height=computed_height,
            )


def slugify_filename_fragment(value):
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return collapsed or "asset"


def render_centered_chart(fig, title=None, caption=None, detail=None, export_file_name=None):
    if fig is None:
        return

    png_bytes = None
    if export_file_name:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        png_bytes = buffer.getvalue()

    with st.container(border=True):
        render_block_header(title, caption)
        st.pyplot(fig, width="stretch")
        if detail:
            st.markdown(
                f'<div class="if-chart-readout">{escape(str(detail))}</div>',
                unsafe_allow_html=True,
            )
        if png_bytes:
            st.download_button(
                "Download chart PNG",
                data=png_bytes,
                file_name=export_file_name,
                mime="image/png",
                width="stretch",
            )
    plt.close(fig)


def render_histogram(
    df,
    analysis_report,
    preferred_column=None,
    title=None,
    caption=None,
    detail=None,
    export_file_name=None,
):
    numeric_col = select_numeric_chart_column(df, preferred_column)
    if numeric_col is None:
        st.info("No numeric columns are available for distribution analysis.")
        return

    clip_percentiles = None
    outlier_info = analysis_report["outlier_summary"].get(numeric_col)
    if outlier_info:
        clip_percentiles = (0.01, 0.99)
        st.warning(
            f"'{numeric_col}' has noticeable outliers. The histogram below trims the top and bottom 1% for display only."
        )

    fig = plot_numeric_histogram(df, numeric_col, clip_percentiles=clip_percentiles)
    render_centered_chart(
        fig,
        title=title or f"Distribution of {numeric_col}",
        caption=caption,
        detail=detail,
        export_file_name=export_file_name,
    )


def render_bullet_list(items):
    for item in items:
        st.write(f"- {item}")


def render_question_shortcuts(questions, key_prefix):
    if not questions:
        return

    st.caption("Smart questions you can run without an API key.")
    columns = st.columns(2, gap="small")
    for index, question in enumerate(questions):
        with columns[index % 2]:
            if st.button(question, key=f"{key_prefix}_{index}", width="stretch"):
                st.session_state["llm_custom_input"] = question
                st.rerun()


def render_insight_card(title, items, caption=None, tone="sky", note=None):
    normalized_items = [str(item) for item in items if str(item).strip()]
    if normalized_items:
        list_html = "".join(f"<li>{escape(item)}</li>" for item in normalized_items)
        body_html = f'<ul class="if-insight-list">{list_html}</ul>'
    else:
        body_html = '<div class="if-insight-note">No insight text is available for this section yet.</div>'

    note_html = f'<div class="if-insight-note">{escape(str(note))}</div>' if note else ""

    st.markdown(
        f"""
        <div class="if-insight-card if-insight-card-{escape(tone)}">
            <div class="if-insight-card-title">{escape(title)}</div>
            {f'<div class="if-insight-card-caption">{escape(caption)}</div>' if caption else ''}
            {body_html}
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_micro_card(title, value, caption=None, tone="sky"):
    st.markdown(
        f"""
        <div class="if-micro-card if-micro-card-{escape(tone)}">
            <div class="if-micro-card-title">{escape(title)}</div>
            <div class="if-micro-card-value">{escape(str(value))}</div>
            {f'<div class="if-micro-card-caption">{escape(str(caption))}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_summary(recommendations, title=None, caption=None):
    with st.container(border=True):
        render_block_header(title, caption)
        if not recommendations:
            st.info("No chart recommendations are available for the current view.")
            return

        for recommendation in recommendations:
            description = escape(recommendation["description"])
            observation = recommendation.get("observation")
            observation_html = (
                f'<div class="if-insight-note">{escape(str(observation))}</div>' if observation else ""
            )
            st.markdown(
                (
                    '<div class="if-recommendation-card">'
                    f'<div class="if-recommendation-title">{escape(recommendation["title"])}</div>'
                    f'<div class="if-recommendation-copy">{description}</div>'
                    f"{observation_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_centered_copy(title=None, caption=None, body=None):
    with st.container(border=True):
        render_block_header(title, caption)
        if body:
            st.markdown(body)


def render_chart_recommendation_item(df, analysis_report, recommendation, export_prefix=None):
    observation = recommendation.get("observation")
    export_file_name = None
    if export_prefix:
        export_file_name = f"{slugify_filename_fragment(export_prefix)}_{slugify_filename_fragment(recommendation['title'])}.png"

    if recommendation["key"] == "category_bar":
        fig = plot_categorical_bar(df, recommendation["columns"][0])
        render_centered_chart(
            fig,
            title=recommendation["title"],
            caption=recommendation["description"],
            detail=observation,
            export_file_name=export_file_name,
        )
    elif recommendation["key"] == "numeric_hist":
        render_histogram(
            df,
            analysis_report,
            preferred_column=recommendation["columns"][0],
            title=recommendation["title"],
            caption=recommendation["description"],
            detail=observation,
            export_file_name=export_file_name,
        )
    elif recommendation["key"] == "scatter_relationship":
        fig = plot_scatter(df, recommendation["columns"][0], recommendation["columns"][1])
        render_centered_chart(
            fig,
            title=recommendation["title"],
            caption=recommendation["description"],
            detail=observation,
            export_file_name=export_file_name,
        )
    elif recommendation["key"] == "correlation_heatmap":
        corr_matrix = analysis_report["correlation_matrix"]
        if corr_matrix is not None:
            fig = plot_correlation_heatmap(corr_matrix)
            render_centered_chart(
                fig,
                title=recommendation["title"],
                caption=recommendation["description"],
                detail=observation,
                export_file_name=export_file_name,
            )
    elif recommendation["key"] == "trend_analysis":
        fig = plot_time_series(df, recommendation["columns"][0], recommendation["columns"][1])
        render_centered_chart(
            fig,
            title=recommendation["title"],
            caption=recommendation["description"],
            detail=observation,
            export_file_name=export_file_name,
        )


def render_categorical_summary_tabs(categorical_summary):
    if not categorical_summary:
        st.info("No categorical columns are suitable for category summary.")
        return

    labels = [column if len(column) <= 18 else f"{column[:18]}..." for column in categorical_summary.keys()]
    tabs = st.tabs(labels)

    for tab, (column, summary) in zip(tabs, categorical_summary.items()):
        with tab:
            render_compact_dataframe(
                summary.rename("Count"),
                height=240,
                title=f"{column} breakdown",
                caption=f"Top values for '{column}' in the current view.",
                index_label="Value",
            )


def render_chart_recommendations(df, analysis_report, chart_recommendations, export_prefix=None):
    if not chart_recommendations:
        st.info("No automatic chart recommendations are available for this dataset yet.")
        return

    labels = [
        recommendation["title"] if len(recommendation["title"]) <= 22 else f"{recommendation['title'][:22]}..."
        for recommendation in chart_recommendations
    ]
    tabs = st.tabs(labels)

    for tab, recommendation in zip(tabs, chart_recommendations):
        with tab:
            render_chart_recommendation_item(df, analysis_report, recommendation, export_prefix=export_prefix)


def render_custom_chart_builder(df, analysis_report, export_prefix=None):
    if df.empty:
        st.info("No rows are available in the current view, so a custom chart cannot be built yet.")
        return

    numeric_columns = get_numeric_columns(df, exclude_id_like=False)
    categorical_columns = get_categorical_columns(df)
    datetime_columns = get_datetime_columns(df)
    groupable_columns = list(df.columns)
    ordered_columns = list(dict.fromkeys(datetime_columns + numeric_columns + categorical_columns))

    chart_mode = st.selectbox(
        "Custom chart type",
        [
            "Bar: counts or aggregated metric",
            "Line: metric over column",
            "Histogram: numeric distribution",
            "Scatter: numeric relationship",
            "Box plot: numeric by group",
        ],
        help="Choose the chart pattern first, then choose the exact columns that should drive it.",
    )

    if chart_mode == "Bar: counts or aggregated metric":
        control_left, control_right = st.columns(2, gap="large")
        with control_left:
            group_col = st.selectbox("Group by column", groupable_columns)
            metric_choice = st.selectbox("Metric", ["Row count"] + numeric_columns)
        with control_right:
            aggregation = (
                "count"
                if metric_choice == "Row count"
                else st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"], index=0)
            )
            sort_order = st.selectbox("Sort order", ["descending", "ascending", "natural"], index=0)

        unique_values = int(df[group_col].nunique(dropna=True))
        slider_max = min(max(unique_values, 3), 40)
        if slider_max <= 3:
            top_n = slider_max
            st.caption(f"'{group_col}' only has {unique_values or 0} unique values in the current view, so all groups will be shown.")
        else:
            top_n = st.slider(
                "How many groups to display",
                min_value=3,
                max_value=slider_max,
                value=min(slider_max, 12),
            )
        if unique_values > top_n:
            st.caption(
                f"'{group_col}' has {unique_values} unique values. The chart shows the top {top_n} groups based on the selected sort order."
            )

        value_col = None if metric_choice == "Row count" else metric_choice
        summary_df, value_label = build_grouped_metric_summary(
            df,
            group_col=group_col,
            value_col=value_col,
            aggregation=aggregation,
            top_n=top_n,
            sort_order=sort_order,
        )
        if summary_df is None:
            st.info("The selected combination does not produce a usable grouped summary.")
            return

        render_compact_dataframe(
            summary_df,
            height=260,
            title=f"{value_label} by {group_col}",
            caption="This is the grouped summary behind the bar chart.",
        )
        fig = plot_grouped_bar(summary_df, group_col, value_label)
        render_centered_chart(
            fig,
            title=f"{value_label} by {group_col}",
            caption=f"Custom grouped bar chart based on '{group_col}' and '{metric_choice}'.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'custom')}_{slugify_filename_fragment(group_col)}_bar.png",
        )

    elif chart_mode == "Line: metric over column":
        if not ordered_columns or not numeric_columns:
            st.info("A line chart needs one ordered column and one numeric metric.")
            return

        control_left, control_right = st.columns(2, gap="large")
        with control_left:
            x_col = st.selectbox(
                "Sequence / x-axis column",
                ordered_columns,
                help="Dates and numeric columns work best. Year, quarter, or ordered counters are also good choices.",
            )
        with control_right:
            value_col = st.selectbox("Metric column", numeric_columns)
            aggregation = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"], index=0)

        summary_df, value_label = build_grouped_metric_summary(
            df,
            group_col=x_col,
            value_col=value_col,
            aggregation=aggregation,
            top_n=max(df[x_col].nunique(dropna=True), 3),
            sort_order="natural",
        )
        if summary_df is None:
            st.info("The selected line chart combination does not produce a usable summary.")
            return

        render_compact_dataframe(
            summary_df,
            height=260,
            title=f"{value_label} over {x_col}",
            caption="This is the ordered summary behind the line chart.",
        )
        fig = plot_grouped_line(summary_df, x_col, value_label)
        render_centered_chart(
            fig,
            title=f"{value_label} over {x_col}",
            caption=f"Custom line chart showing how '{value_col}' changes across '{x_col}'.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'custom')}_{slugify_filename_fragment(x_col)}_line.png",
        )

    elif chart_mode == "Histogram: numeric distribution":
        if not numeric_columns:
            st.info("A histogram needs at least one numeric column.")
            return

        numeric_col = st.selectbox("Numeric column", numeric_columns)
        render_histogram(
            df,
            analysis_report,
            preferred_column=numeric_col,
            title=f"Distribution of {numeric_col}",
            caption=f"Custom histogram based on the numeric values in '{numeric_col}'.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'custom')}_{slugify_filename_fragment(numeric_col)}_hist.png",
        )

    elif chart_mode == "Scatter: numeric relationship":
        if len(numeric_columns) < 2:
            st.info("A scatter plot needs at least two numeric columns.")
            return

        control_left, control_right = st.columns(2, gap="large")
        with control_left:
            x_col = st.selectbox("X-axis metric", numeric_columns, index=0)
        with control_right:
            y_options = [column for column in numeric_columns if column != x_col] or numeric_columns
            y_col = st.selectbox("Y-axis metric", y_options, index=0)

        fig = plot_scatter(df, x_col, y_col)
        render_centered_chart(
            fig,
            title=f"{y_col} vs {x_col}",
            caption="Custom scatter plot based on the exact numeric pair you selected.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'custom')}_{slugify_filename_fragment(x_col)}_vs_{slugify_filename_fragment(y_col)}.png",
        )

    else:
        if not categorical_columns or not numeric_columns:
            st.info("A grouped box plot needs one grouping column and one numeric metric.")
            return

        control_left, control_right = st.columns(2, gap="large")
        with control_left:
            group_col = st.selectbox("Group column", categorical_columns)
        with control_right:
            value_col = st.selectbox("Numeric metric", numeric_columns)

        group_count = int(df[group_col].nunique(dropna=True))
        slider_max = min(max(group_count, 3), 20)
        if slider_max <= 3:
            max_groups = slider_max
            st.caption(f"'{group_col}' only has {group_count or 0} unique values in the current view, so all groups will be shown.")
        else:
            max_groups = st.slider(
                "How many groups to include",
                min_value=3,
                max_value=slider_max,
                value=min(slider_max, 10),
            )
        if group_count > max_groups:
            st.caption(
                f"'{group_col}' has {group_count} unique values. The box plot includes the {max_groups} most common groups so it stays readable."
            )

        fig = plot_boxplot_by_group(df, group_col, value_col, max_groups=max_groups)
        render_centered_chart(
            fig,
            title=f"{value_col} distribution by {group_col}",
            caption="Custom grouped box plot showing the spread of a numeric metric across categories.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'custom')}_{slugify_filename_fragment(group_col)}_{slugify_filename_fragment(value_col)}_box.png",
        )


def render_guided_analysis(
    option_key,
    df,
    validation_report,
    analysis_report,
    chart_recommendations,
    get_categorical_columns_fn,
    focus_columns=None,
    export_prefix=None,
    analysis_preferences=None,
):
    focus_columns = focus_columns or []
    analysis_preferences = analysis_preferences or {}

    if option_key == "quality_overview":
        score = validation_report["quality_score"]
        label = validation_report["quality_label"]
        st.metric("Readiness Score", f"{score}/100", label)
        st.write(
            f"InsightFlow estimates this file as **{label.lower()}** based on missing values, duplicates, and invalid values detected in the raw upload."
        )
        st.caption(
            "This is a heuristic readiness estimate. It is useful for triage, but it does not prove that the file is analytically or statistically correct."
        )

        if not validation_report["missing"].empty:
            render_compact_dataframe(
                validation_report["missing"].rename("Missing Values"),
                title="Missing values overview",
                caption="Columns with blank or null values in the uploaded file.",
                index_label="Column",
            )
        if validation_report["duplicate_count"] > 0:
            st.write(
                f"Duplicate rows detected: **{validation_report['duplicate_count']}** using "
                f"**{', '.join(validation_report['duplicate_subset']) if validation_report['duplicate_subset'] else 'exact row matching'}**."
            )
    elif option_key == "missing_breakdown":
        if validation_report["missing"].empty:
            st.success("No missing values were detected in the uploaded CSV.")
        else:
            render_compact_dataframe(
                validation_report["missing"].rename("Missing Values"),
                title="Missing values breakdown",
                caption="Missing-value counts by column.",
                index_label="Column",
            )
    elif option_key == "duplicate_review":
        if validation_report["duplicate_count"] == 0:
            st.success("No duplicate rows were detected with the current duplicate strategy.")
        else:
            basis = ", ".join(validation_report["duplicate_subset"]) if validation_report["duplicate_subset"] else "exact row matching"
            render_compact_dataframe(
                validation_report["duplicate_rows"],
                height=260,
                title="Duplicate preview",
                caption=f"Rows flagged as duplicates using {basis}.",
            )
    elif option_key == "summary_statistics":
        numeric_summary = analysis_report["numeric_summary"]
        if numeric_summary is None:
            st.info("No numeric columns are available for summary statistics.")
        else:
            render_compact_dataframe(
                numeric_summary,
                height=320,
                title="Summary statistics",
                caption="Numeric distribution metrics for the current view.",
            )
    elif option_key == "top_categories":
        preferred_column = next((col for col in focus_columns if col in get_categorical_columns_fn(df)), None)
        categorical_col = select_categorical_chart_column(df, preferred_column)
        if categorical_col is None:
            st.info("No categorical columns are available for category comparison.")
        else:
            category_summary = df[categorical_col].value_counts(dropna=False).head(10)
            render_compact_dataframe(
                category_summary.rename("Count"),
                height=260,
                title=f"{categorical_col} comparison",
                caption=f"Top values for '{categorical_col}'.",
                index_label="Value",
            )
            fig = plot_categorical_bar(df, categorical_col)
            render_centered_chart(
                fig,
                title=f"{categorical_col} bar chart",
                caption=f"Category frequency view for '{categorical_col}'.",
                export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_{slugify_filename_fragment(categorical_col)}_bar_chart.png",
            )
    elif option_key == "numeric_distribution":
        preferred_column = next(
            (col for col in focus_columns if col in get_numeric_columns(df, exclude_id_like=False)),
            None,
        )
        render_histogram(
            df,
            analysis_report,
            preferred_column=preferred_column,
            title="Numeric distribution",
            caption="Distribution of the selected numeric metric.",
            export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_numeric_distribution.png",
        )
    elif option_key == "outlier_check":
        outlier_df = format_outlier_summary(analysis_report["outlier_summary"])
        if outlier_df.empty:
            st.info("No strong numeric outliers were detected with the current IQR-based scan.")
        else:
            render_compact_dataframe(
                outlier_df,
                height=260,
                title="Outlier summary",
                caption="Columns with notable IQR-based outliers.",
            )
            preferred_column = next(
                (col for col in focus_columns if col in analysis_report["outlier_summary"]),
                outlier_df.iloc[0]["Column"],
            )
            fig = plot_boxplot(df, preferred_column)
            render_centered_chart(
                fig,
                title=f"{preferred_column} outlier check",
                caption="Box plot for the selected outlier-heavy column.",
                export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_{slugify_filename_fragment(preferred_column)}_outlier_check.png",
            )
    elif option_key == "correlation_analysis":
        corr_matrix = analysis_report["correlation_matrix"]
        if corr_matrix is None:
            st.info("Not enough numeric columns are available for correlation analysis.")
        else:
            render_compact_dataframe(
                corr_matrix,
                height=260,
                title="Correlation matrix",
                caption="Pairwise correlation values across numeric columns.",
            )
            fig = plot_correlation_heatmap(corr_matrix)
            render_centered_chart(
                fig,
                title="Correlation heatmap",
                caption="Visual view of numeric relationships.",
                export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_correlation_heatmap.png",
            )
    elif option_key == "scatter_relationship":
        numeric_focus = [col for col in focus_columns if col in get_numeric_columns(df, exclude_id_like=False)]
        preferred_pair = numeric_focus[:2] if len(numeric_focus) >= 2 else None
        scatter_cols = select_scatter_columns(df, preferred_pair)
        if scatter_cols is None:
            st.info("At least two numeric columns are required for scatter analysis.")
        else:
            fig = plot_scatter(df, scatter_cols[0], scatter_cols[1])
            render_centered_chart(
                fig,
                title=f"{scatter_cols[0]} vs {scatter_cols[1]}",
                caption="Scatter plot for the strongest numeric pair.",
                export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_{slugify_filename_fragment(scatter_cols[0])}_vs_{slugify_filename_fragment(scatter_cols[1])}.png",
            )
    elif option_key == "trend_analysis":
        preferred_numeric = next(
            (col for col in focus_columns if col in get_numeric_columns(df, exclude_id_like=False)),
            analysis_preferences.get("trend_value_column"),
        )
        time_series_cols = select_time_series_columns(
            df,
            preferred_value_column=preferred_numeric,
            preferred_date_column=analysis_preferences.get("trend_date_column"),
        )
        if time_series_cols is None:
            st.info("Trend analysis needs at least one date column and one numeric column.")
        else:
            fig = plot_time_series(df, time_series_cols[0], time_series_cols[1])
            render_centered_chart(
                fig,
                title=f"{time_series_cols[1]} over time",
                caption=f"Trend view using '{time_series_cols[0]}' as the time axis.",
                export_file_name=f"{slugify_filename_fragment(export_prefix or 'guided')}_{slugify_filename_fragment(time_series_cols[1])}_over_time.png",
            )
    elif option_key == "recommended_charts":
        render_chart_recommendations(df, analysis_report, chart_recommendations, export_prefix=export_prefix)
