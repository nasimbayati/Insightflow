import pandas as pd
import streamlit as st

from modules.analysis import (
    apply_dataset_filters,
    get_datetime_columns,
    get_filterable_categorical_columns,
    get_groupable_columns,
    get_numeric_columns,
)
from modules.cleaning import CleaningConfig
from modules.llm_insights import (
    DEFAULT_LLM_MODEL,
    SUPPORTED_LLM_MODELS,
    is_llm_configured,
    resolve_api_key,
)
from modules.pipeline_config import ColumnRoles, PipelinePreferences, UNSPECIFIED_OPTION
from modules.validation import is_probable_id_column


def get_narrative_mode(api_key):
    if is_llm_configured(api_key):
        return {
            "label": "Hybrid",
            "caption": "Rule-based insights are always available, and LLM generation can be enabled on demand.",
        }

    return {
        "label": "Fallback ready",
        "caption": "The app is running in competition-safe mode with built-in insights only. No API access is required.",
    }


def render_llm_settings():
    with st.sidebar:
        st.markdown("### LLM Settings")
        st.caption("Use OpenAI to generate a richer narrative and answer custom analysis requests.")
        model = st.selectbox("LLM model", SUPPORTED_LLM_MODELS, index=SUPPORTED_LLM_MODELS.index(DEFAULT_LLM_MODEL))
        api_key_input = st.text_input(
            "OpenAI API key",
            value="",
            type="password",
            placeholder="sk-...",
            help="Optional if OPENAI_API_KEY is already set in your environment or Streamlit secrets.",
        )

        resolved_key = resolve_api_key(api_key_input, getattr(st, "secrets", None))
        mode_info = get_narrative_mode(resolved_key)
        if is_llm_configured(resolved_key):
            st.success("LLM narrative is available.")
        else:
            st.info("No API key detected. The app will stay on rule-based insights until a key is provided.")

        st.caption(f"Narrative mode: {mode_info['label']}. {mode_info['caption']}")

        return model, resolved_key


def get_cleaning_config_from_controls(df, column_types, suggested_duplicate_subset):
    numeric_strategy_options = {
        "Median fill": "median",
        "Mean fill": "mean",
        "Leave as missing": "leave",
    }
    categorical_strategy_options = {
        "Fill with Unknown": "unknown",
        "Fill with mode": "mode",
        "Leave as missing": "leave",
    }
    duplicate_options = {
        "Remove duplicates": "remove",
        "Keep duplicates": "keep",
    }
    text_strategy_options = {
        "Trim only": "strip",
        "Lowercase": "lower",
        "Title case": "title",
    }
    audience_options = {
        "Executive brief": "executive",
        "Analyst detail": "analyst",
        "Operational review": "operator",
    }
    duplicate_rule_options = {"Exact row matching": "exact"}
    if suggested_duplicate_subset:
        duplicate_rule_options[f"Suggested business key ({', '.join(suggested_duplicate_subset)})"] = "suggested"
    duplicate_rule_options["Custom columns"] = "custom"

    numeric_columns = [column for column, detected_type in column_types.items() if detected_type == "Numeric"]
    preferred_trend_numeric_columns = [
        column
        for column in numeric_columns
        if not is_probable_id_column(df[column], column, column_types.get(column))
    ] or numeric_columns
    categorical_columns = [column for column, detected_type in column_types.items() if detected_type == "Categorical"]
    date_columns = [column for column, detected_type in column_types.items() if detected_type == "Date"]
    all_columns = list(df.columns)
    probable_id_columns = [
        column
        for column in all_columns
        if is_probable_id_column(df[column], column, column_types.get(column))
    ]
    groupable_columns = [
        column
        for column in get_groupable_columns(df)
        if column not in probable_id_columns
    ] or [column for column in all_columns if column not in probable_id_columns] or all_columns
    time_axis_candidates = list(dict.fromkeys(date_columns + groupable_columns))

    controls_left, controls_right = st.columns([0.9, 1.1], gap="large")

    with controls_left:
        chart_view_mode = st.radio(
            "Active analysis view",
            ["Cleaned Data", "Raw Data"],
            horizontal=True,
            help="Cleaned Data applies your cleaning strategy. Raw Data keeps all uploaded rows while still standardizing numeric and date types for analysis.",
        )
        audience_label = st.selectbox(
            "Insight audience",
            list(audience_options.keys()),
            index=0,
            help="Choose how InsightFlow should frame the narrative summary and recommended actions.",
        )
        with st.container(border=True):
            st.markdown("#### Column roles")
            st.caption(
                "Optional but recommended: tell InsightFlow what the columns mean. This keeps IDs, time axes, metrics, segments, and outcomes from being confused with each other."
            )
            id_columns = st.multiselect(
                "Identifier column(s)",
                all_columns,
                default=probable_id_columns[:3],
                help="Identifiers are treated as record keys, not performance metrics.",
            )
            time_choice = st.selectbox(
                "Time / order column",
                [UNSPECIFIED_OPTION] + time_axis_candidates,
                index=0,
                help="Use dates, years, quarters, periods, or another ordered column when the file has one.",
            )
            metric_choice = st.selectbox(
                "Primary metric",
                [UNSPECIFIED_OPTION] + preferred_trend_numeric_columns,
                index=0,
                help="The main numeric value to summarize and chart. Examples could be score, revenue, quantity, cost, duration, or rating.",
            )
            segment_choice = st.selectbox(
                "Primary segment / group",
                [UNSPECIFIED_OPTION] + groupable_columns,
                index=0,
                help="The main grouping field for comparisons. Examples could be subject, region, product, department, year, quarter, status, or category.",
            )
            outcome_choice = st.selectbox(
                "Outcome / target column",
                [UNSPECIFIED_OPTION] + all_columns,
                index=0,
                help="Optional. Use this when one column represents the result you care about most, such as grade, churn, status, return flag, or pass/fail.",
            )

        selected_trend_date = None if time_choice == UNSPECIFIED_OPTION else time_choice
        selected_trend_value = None if metric_choice == UNSPECIFIED_OPTION else metric_choice

    with controls_right:
        with st.expander("Pipeline settings", expanded=False):
            numeric_label = st.selectbox(
                "Numeric missing values",
                list(numeric_strategy_options.keys()),
                index=0,
            )
            categorical_label = st.selectbox(
                "Categorical missing values",
                list(categorical_strategy_options.keys()),
                index=0,
            )
            duplicate_label = st.selectbox(
                "Duplicate handling",
                list(duplicate_options.keys()),
                index=0,
            )
            duplicate_rule_label = st.selectbox(
                "Duplicate detection rule",
                list(duplicate_rule_options.keys()),
                index=0,
                help="Exact row matching is safest. Column-based duplicate rules are useful, but they are only trustworthy once they match the business meaning of a repeated record.",
            )
            text_label = st.selectbox(
                "Categorical text normalization",
                list(text_strategy_options.keys()),
                index=0,
                help="Trim only preserves user-entered casing. Lowercase and Title case are optional normalization modes.",
            )

            duplicate_rule_mode = duplicate_rule_options[duplicate_rule_label]
            if duplicate_rule_mode == "custom":
                duplicate_subset = st.multiselect(
                    "Custom duplicate key columns",
                    list(df.columns),
                    default=suggested_duplicate_subset or [],
                    help="Choose the columns that together define a repeated business record.",
                )
            elif duplicate_rule_mode == "suggested":
                duplicate_subset = suggested_duplicate_subset
                st.caption(
                    f"Using the suggested business key: {', '.join(suggested_duplicate_subset)}. Confirm this only if these columns truly define a duplicate in your domain."
                )
            else:
                duplicate_subset = None
                st.caption("Using exact row matching. This is the most conservative duplicate rule.")

        with st.expander("Column-specific cleaning overrides", expanded=False):
            numeric_column_strategies = {}
            categorical_column_strategies = {}

            if numeric_columns:
                selected_numeric_override_columns = st.multiselect(
                    "Numeric columns with custom missing-value handling",
                    numeric_columns,
                    help="Use this when one numeric field should be handled differently from the global numeric strategy.",
                )
                for column in selected_numeric_override_columns:
                    override_label = st.selectbox(
                        f"{column} numeric strategy",
                        list(numeric_strategy_options.keys()),
                        index=list(numeric_strategy_options.values()).index(numeric_strategy_options[numeric_label]),
                        key=f"numeric_override_{column}",
                    )
                    numeric_column_strategies[column] = numeric_strategy_options[override_label]

            if categorical_columns:
                selected_categorical_override_columns = st.multiselect(
                    "Categorical columns with custom missing-value handling",
                    categorical_columns,
                    help="Use this when one category field should keep blanks, while others use Unknown or mode.",
                )
                for column in selected_categorical_override_columns:
                    override_label = st.selectbox(
                        f"{column} categorical strategy",
                        list(categorical_strategy_options.keys()),
                        index=list(categorical_strategy_options.values()).index(
                            categorical_strategy_options[categorical_label]
                        ),
                        key=f"categorical_override_{column}",
                    )
                    categorical_column_strategies[column] = categorical_strategy_options[override_label]

    cleaning_config = CleaningConfig(
        numeric_missing_strategy=numeric_strategy_options[numeric_label],
        categorical_missing_strategy=categorical_strategy_options[categorical_label],
        duplicate_action=duplicate_options[duplicate_label],
        categorical_text_strategy=text_strategy_options[text_label],
        numeric_column_strategies=numeric_column_strategies,
        categorical_column_strategies=categorical_column_strategies,
    )

    resolved_duplicate_rule_mode = duplicate_rule_options[duplicate_rule_label]
    if resolved_duplicate_rule_mode == "custom" and not duplicate_subset:
        st.caption("No custom duplicate key columns were selected, so InsightFlow is falling back to exact row matching.")
        resolved_duplicate_rule_mode = "exact"

    column_roles = ColumnRoles(
        id_columns=tuple(id_columns),
        time_column=selected_trend_date,
        metric_column=selected_trend_value,
        segment_column=None if segment_choice == UNSPECIFIED_OPTION else segment_choice,
        outcome_column=None if outcome_choice == UNSPECIFIED_OPTION else outcome_choice,
    )
    pipeline_preferences = PipelinePreferences(
        duplicate_subset=tuple(duplicate_subset) if duplicate_subset else None,
        duplicate_rule_mode=resolved_duplicate_rule_mode,
        trend_date_column=selected_trend_date,
        trend_value_column=selected_trend_value,
        column_roles=column_roles,
    )

    return chart_view_mode, cleaning_config, audience_options[audience_label], pipeline_preferences


def build_ingestion_notes(ingestion_metadata):
    notes = []

    if ingestion_metadata["repaired_row_count"] > 0:
        notes.append(
            f"Repaired {ingestion_metadata['repaired_row_count']} short row(s) by padding missing trailing cells. Sample lines: {ingestion_metadata['repaired_row_numbers']}."
        )
    if ingestion_metadata["skipped_row_count"] > 0:
        notes.append(
            f"Skipped {ingestion_metadata['skipped_row_count']} malformed row(s) with too many fields. Sample lines: {ingestion_metadata['skipped_row_numbers']}."
        )
    if ingestion_metadata["blank_row_count"] > 0:
        notes.append(f"Ignored {ingestion_metadata['blank_row_count']} blank row(s) during ingestion.")
    if ingestion_metadata["empty_dataset"]:
        notes.append("The CSV contains headers but no data rows.")

    return notes


def render_filter_controls(df, key_prefix):
    if df.empty:
        st.info("No rows are available for filtering in the current view.")
        return df.copy(), []

    categorical_filters = {}
    numeric_filters = {}
    date_filters = {}

    categorical_columns = get_filterable_categorical_columns(df)
    numeric_columns = [
        col
        for col in get_numeric_columns(df, exclude_id_like=True)
        if pd.to_numeric(df[col], errors="coerce").dropna().nunique() > 1
    ]
    date_columns = get_datetime_columns(df)

    filter_columns = st.columns(3, gap="large")

    with filter_columns[0]:
        selected_categorical_columns = st.multiselect(
            "Categorical filters",
            categorical_columns,
            key=f"{key_prefix}_categorical_columns",
        )
        for column in selected_categorical_columns:
            options = df[column].dropna().astype(str).drop_duplicates().sort_values().tolist()
            selected_values = st.multiselect(
                f"{column} values",
                options,
                default=options,
                key=f"{key_prefix}_categorical_{column}",
            )
            if selected_values and len(selected_values) < len(options):
                categorical_filters[column] = selected_values

    with filter_columns[1]:
        selected_numeric_columns = st.multiselect(
            "Numeric filters",
            numeric_columns,
            key=f"{key_prefix}_numeric_columns",
        )
        for column in selected_numeric_columns:
            numeric_series = pd.to_numeric(df[column], errors="coerce").dropna()
            if numeric_series.empty:
                continue

            min_value = float(numeric_series.min())
            max_value = float(numeric_series.max())
            if min_value == max_value:
                st.caption(f"{column} has a constant value and cannot be sliced further.")
                continue

            selected_range = st.slider(
                f"{column} range",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                key=f"{key_prefix}_numeric_{column}",
            )
            if selected_range != (min_value, max_value):
                numeric_filters[column] = selected_range

    with filter_columns[2]:
        selected_date_columns = st.multiselect(
            "Date filters",
            date_columns,
            key=f"{key_prefix}_date_columns",
        )
        for column in selected_date_columns:
            date_series = pd.to_datetime(df[column], errors="coerce").dropna()
            if date_series.empty:
                continue

            min_date = date_series.min().date()
            max_date = date_series.max().date()
            selected_dates = st.date_input(
                f"{column} range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"{key_prefix}_date_{column}",
            )

            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
            elif isinstance(selected_dates, list) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
            else:
                continue

            if (start_date, end_date) != (min_date, max_date):
                date_filters[column] = (
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                )

    filtered_df, applied_filters = apply_dataset_filters(
        df,
        categorical_filters=categorical_filters,
        numeric_filters=numeric_filters,
        date_filters=date_filters,
    )
    return filtered_df, applied_filters
