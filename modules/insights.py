import pandas as pd

from modules.analysis import (
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    select_categorical_chart_column,
    select_numeric_chart_column,
    select_scatter_columns,
    select_time_series_columns,
)


ANALYSIS_KEYWORDS = {
    "missing_breakdown": ["missing", "null", "empty", "na"],
    "duplicate_review": ["duplicate", "duplicates", "dedupe", "repeat"],
    "summary_statistics": ["summary", "stats", "statistics", "metric", "metrics"],
    "top_categories": ["category", "categories", "segment", "segments", "group", "groups", "compare"],
    "correlation_analysis": ["correlation", "correlations", "heatmap", "relationship"],
    "numeric_distribution": ["distribution", "histogram", "spread"],
    "outlier_check": ["outlier", "outliers", "anomaly", "anomalies", "extreme"],
    "scatter_relationship": ["scatter", "pair", "relationship between"],
    "trend_analysis": ["trend", "time", "date", "daily", "weekly", "monthly", "yearly"],
    "recommended_charts": ["chart", "charts", "plot", "visual", "visualize", "graph"],
}

AUDIENCE_MODES = {
    "executive": {
        "label": "Executive brief",
        "summary_line": "This narrative is tuned for fast decision-making, so it emphasizes the largest risk, the clearest driver, and the first action to take.",
        "profile_phrase": "decision-ready comparison",
        "next_step_line": "Use the first recommended chart to explain the headline signal quickly before sharing the file upward.",
        "risk_line": "Treat the current narrative as directional until the highest-risk quality issue is resolved or explicitly accepted.",
    },
    "analyst": {
        "label": "Analyst detail",
        "summary_line": "This narrative is tuned for diagnostic review, so treat the strongest patterns as hypotheses to validate with slicing, chart evidence, and alternative assumptions.",
        "profile_phrase": "diagnostic comparison",
        "next_step_line": "Start with the strongest chart, then test whether the same pattern survives filtering and segment comparison.",
        "risk_line": "Validate the flagged fields before treating the strongest relationships as stable enough for downstream modeling or formal reporting.",
    },
    "operator": {
        "label": "Operational review",
        "summary_line": "This narrative is tuned for operational follow-up, so it emphasizes where process, workload, or field-level cleanup should be addressed first.",
        "profile_phrase": "operational comparison",
        "next_step_line": "Use the first recommended chart to locate where action or process review is needed before you escalate the result.",
        "risk_line": "Resolve the highest-friction data issue first so execution teams do not act on a noisy signal.",
    },
}


def recommend_charts(df, preferred_value_column=None, preferred_date_column=None, preferred_category_column=None):
    recommendations = []

    categorical_col = select_categorical_chart_column(df, preferred_category_column)
    if categorical_col:
        recommendations.append(
            {
                "key": "category_bar",
                "title": f"Category distribution: '{categorical_col}'",
                "description": f"Start here to see which groups dominate '{categorical_col}' and whether the category mix is balanced enough for a clean comparison.",
                "columns": [categorical_col],
            }
        )

    numeric_col = select_numeric_chart_column(df, preferred_value_column)
    if numeric_col:
        recommendations.append(
            {
                "key": "numeric_hist",
                "title": f"Distribution check: '{numeric_col}'",
                "description": f"Use this first when you need to judge spread, skew, and whether extreme values in '{numeric_col}' could distort the rest of the analysis.",
                "columns": [numeric_col],
            }
        )

    scatter_cols = select_scatter_columns(df)
    if scatter_cols:
        recommendations.append(
            {
                "key": "scatter_relationship",
                "title": f"Relationship check: '{scatter_cols[0]}' vs '{scatter_cols[1]}'",
                "description": "Use this to test whether two important numeric measures move together, split into separate clusters, or reveal unusual records worth investigating.",
                "columns": list(scatter_cols),
            }
        )
        recommendations.append(
            {
                "key": "correlation_heatmap",
                "title": "Correlation heatmap",
                "description": "Use this for a fast scan of the full numeric field set before choosing one relationship to explain in detail.",
                "columns": list(scatter_cols),
            }
        )

    time_series_cols = select_time_series_columns(
        df,
        preferred_value_column=preferred_value_column or numeric_col,
        preferred_date_column=preferred_date_column,
    )
    if time_series_cols:
        recommendations.append(
            {
                "key": "trend_analysis",
                "title": f"Trend view: '{time_series_cols[1]}' over '{time_series_cols[0]}'",
                "description": f"Use this to see whether '{time_series_cols[1]}' is improving, weakening, or shifting unexpectedly across time.",
                "columns": list(time_series_cols),
            }
        )

    return recommendations


def _describe_relationship_strength(value):
    absolute_value = abs(value)
    if absolute_value >= 0.8:
        return "very strong"
    if absolute_value >= 0.6:
        return "strong"
    if absolute_value >= 0.4:
        return "moderate"
    if absolute_value >= 0.2:
        return "light"
    return "weak"


def _format_metric_value(value):
    numeric_value = float(value)
    if abs(numeric_value) >= 1000:
        return f"{numeric_value:,.0f}"
    if numeric_value.is_integer():
        return f"{numeric_value:.0f}"
    return f"{numeric_value:.1f}"


def _build_segment_signal(df, preferred_category_column=None, preferred_value_column=None):
    category_col = select_categorical_chart_column(df, preferred_category_column)
    numeric_col = select_numeric_chart_column(df, preferred_value_column)

    if category_col is None or numeric_col is None:
        return None

    working_df = df[[category_col, numeric_col]].copy()
    working_df[numeric_col] = pd.to_numeric(working_df[numeric_col], errors="coerce")
    working_df[category_col] = working_df[category_col].astype(str)
    working_df = working_df.dropna()

    if working_df.empty:
        return None

    grouped = working_df.groupby(category_col)[numeric_col].mean().sort_values(ascending=False)
    if len(grouped) < 2:
        return None

    top_label = str(grouped.index[0])
    top_value = float(grouped.iloc[0])
    bottom_label = str(grouped.index[-1])
    bottom_value = float(grouped.iloc[-1])

    return {
        "category_column": category_col,
        "numeric_column": numeric_col,
        "top_label": top_label,
        "top_value": top_value,
        "bottom_label": bottom_label,
        "bottom_value": bottom_value,
    }


def _build_trend_signal(df, preferred_date_column=None, preferred_value_column=None):
    from modules.analysis import assess_trend_signal

    return assess_trend_signal(
        df,
        preferred_value_column=preferred_value_column,
        preferred_date_column=preferred_date_column,
    )


def _calculate_insight_confidence(active_df, validation_report, analysis_report):
    rows, cols = active_df.shape
    total_cells = max(rows * max(cols, 1), 1)
    active_missing_ratio = (active_df.isna().sum().sum() / total_cells) if cols > 0 else 0
    invalid_count = validation_report["invalid_numeric_count"] + validation_report["invalid_date_count"]
    score = 100
    reasons = []
    breakdown = []

    if rows == 0:
        score = 0
        reasons.append("The active view has zero rows, so there is no evidence base for a trustworthy narrative.")
        breakdown.append({"label": "Empty active view", "penalty": 100, "detail": "No rows remain in the active view."})
    elif rows < 8:
        score -= 30
        reasons.append("The active view is very small, so a few rows can change the story disproportionately.")
        breakdown.append({"label": "Small sample", "penalty": 30, "detail": f"Only {rows} row(s) are in the active view."})
    elif rows < 20:
        score -= 15
        reasons.append("The active view is relatively small, so the strongest signals should be treated as directional.")
        breakdown.append({"label": "Limited sample", "penalty": 15, "detail": f"The active view has {rows} row(s), which is enough for direction but not high confidence."})

    if active_missing_ratio >= 0.1:
        score -= 20
        reasons.append("Missing values still occupy a meaningful share of the active view.")
        breakdown.append({"label": "Missing-value pressure", "penalty": 20, "detail": f"About {active_missing_ratio:.0%} of cells are still blank or null."})
    elif active_missing_ratio >= 0.03:
        score -= 10
        reasons.append("Some missing values remain, which lowers confidence in summary totals and averages.")
        breakdown.append({"label": "Residual missing values", "penalty": 10, "detail": f"About {active_missing_ratio:.0%} of cells are still blank or null."})

    if validation_report["duplicate_count"] > 0:
        score -= 10
        reasons.append("Duplicate rows were detected in the raw upload and may still affect interpretation if the rule is not business-correct.")
        breakdown.append({"label": "Duplicate risk", "penalty": 10, "detail": f"{validation_report['duplicate_count']} duplicate row(s) were detected in the raw upload."})

    if invalid_count > 0:
        score -= 15
        reasons.append("Parsing problems were found in numeric or date fields, which weakens downstream confidence.")
        breakdown.append({"label": "Parsing issues", "penalty": 15, "detail": f"{invalid_count} invalid numeric/date value(s) were detected."})

    if analysis_report["outlier_summary"]:
        max_outlier_share = max(item["share"] for item in analysis_report["outlier_summary"].values())
        if max_outlier_share >= 0.15:
            score -= 12
            reasons.append("At least one numeric field is materially influenced by outliers.")
            breakdown.append({"label": "High outlier pressure", "penalty": 12, "detail": f"The heaviest outlier column affects about {max_outlier_share:.0%} of non-null values."})
        else:
            score -= 6
            reasons.append("Outliers are present, so some averages may be less trustworthy than medians.")
            breakdown.append({"label": "Moderate outlier pressure", "penalty": 6, "detail": f"The heaviest outlier column affects about {max_outlier_share:.0%} of non-null values."})

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "High"
    elif score >= 55:
        label = "Medium"
    else:
        label = "Low"

    if not reasons:
        reasons.append(
            "The active view has enough rows and limited validation noise, so the main signals are relatively trustworthy."
        )
        breakdown.append({"label": "Stable evidence base", "penalty": 0, "detail": "No major confidence penalties were triggered."})

    breakdown.append(
        {
            "label": "Heuristic scoring model",
            "penalty": 0,
            "detail": "Confidence is a rule-based trust estimate, not a calibrated probability or significance test.",
        }
    )

    return score, label, reasons, breakdown


def _build_next_questions(validation_report, analysis_report, segment_signal, trend_signal, strongest_correlation):
    questions = []

    if segment_signal:
        questions.append(
            f"Why is '{segment_signal['top_label']}' outperforming '{segment_signal['bottom_label']}' in '{segment_signal['category_column']}'?"
        )

    if strongest_correlation:
        col_a, col_b = strongest_correlation["columns"]
        questions.append(f"What is driving the relationship between '{col_a}' and '{col_b}'?")

    if trend_signal:
        questions.append(
            f"Is the {trend_signal['direction']} movement in '{trend_signal['value_column']}' structural, seasonal, or driven by a small set of periods?"
        )

    if analysis_report["outlier_summary"]:
        top_outlier_col = max(
            analysis_report["outlier_summary"].items(),
            key=lambda item: item[1]["share"],
        )[0]
        questions.append(f"Are the extreme values in '{top_outlier_col}' valid business events or data-quality issues?")

    if not validation_report["missing"].empty:
        top_missing_col = validation_report["missing"].index[0]
        questions.append(f"Would the story change if missing values in '{top_missing_col}' were cleaned or backfilled differently?")

    deduped = []
    for question in questions:
        if question not in deduped:
            deduped.append(question)

    return deduped[:4]


def annotate_chart_recommendations(df, analysis_report, chart_recommendations):
    annotated = []

    for recommendation in chart_recommendations:
        observation = None

        if recommendation["key"] == "category_bar":
            column = recommendation["columns"][0]
            value_counts = df[column].dropna().astype(str).value_counts().head(3)
            if not value_counts.empty:
                top_label = value_counts.index[0]
                top_count = int(value_counts.iloc[0])
                observation = f"Why start here: '{top_label}' currently leads '{column}' with {top_count} record(s), so this chart will show quickly whether one group is dominating the file."
                if len(value_counts) > 1:
                    second_label = value_counts.index[1]
                    second_count = int(value_counts.iloc[1])
                    observation += f" The next most common value is '{second_label}' with {second_count}."

        elif recommendation["key"] == "numeric_hist":
            column = recommendation["columns"][0]
            numeric_series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not numeric_series.empty:
                observation = (
                    f"Why start here: '{column}' ranges from {_format_metric_value(numeric_series.min())} to {_format_metric_value(numeric_series.max())} "
                    f"with a median of {_format_metric_value(numeric_series.median())}, so you can judge spread before trusting averages."
                )
                outlier_info = analysis_report["outlier_summary"].get(column)
                if outlier_info:
                    observation += f" About {outlier_info['share']:.0%} of values are flagged as outliers."

        elif recommendation["key"] == "scatter_relationship":
            x_col, y_col = recommendation["columns"]
            strongest = analysis_report["strongest_correlation"]
            if strongest and tuple(strongest["columns"]) == (x_col, y_col):
                corr_value = float(strongest["value"])
                strength = _describe_relationship_strength(corr_value)
                direction = "positive" if corr_value >= 0 else "negative"
                observation = (
                    f"Why start here: '{x_col}' and '{y_col}' show a {strength} {direction} relationship "
                    f"(correlation {corr_value:.2f}), so this plot can confirm whether the pattern is clean or driven by a few unusual records."
                )
                if strongest.get("support_reason"):
                    observation += f" Guardrail: {strongest['support_reason']}"

        elif recommendation["key"] == "correlation_heatmap":
            strongest = analysis_report["strongest_correlation"]
            if strongest:
                col_a, col_b = strongest["columns"]
                strength = _describe_relationship_strength(strongest["value"])
                observation = (
                    f"Why start here: the strongest pair in the file is '{col_a}' and '{col_b}' with a {strength} "
                    f"correlation of {strongest['value']:.2f}, so the heatmap is a fast way to see what deserves deeper explanation."
                )
                if strongest.get("support_reason"):
                    observation += f" Guardrail: {strongest['support_reason']}"

        elif recommendation["key"] == "trend_analysis":
            date_col, value_col = recommendation["columns"]
            trend_signal = analysis_report.get("trend_signal")
            if trend_signal and trend_signal["date_column"] == date_col and trend_signal["value_column"] == value_col:
                observation = (
                    f"Why start here: '{value_col}' moves {trend_signal['direction']} from {_format_metric_value(trend_signal['first_value'])} to {_format_metric_value(trend_signal['last_value'])} "
                    f"across the available time range, so this view is the quickest way to spot momentum or deterioration."
                )
                if trend_signal.get("support_reason"):
                    observation += f" Guardrail: {trend_signal['support_reason']}"

        enriched = recommendation.copy()
        enriched["observation"] = observation
        annotated.append(enriched)

    return annotated


def calculate_data_quality_score(raw_df, validation_report):
    rows, cols = raw_df.shape
    total_cells = max(rows * max(cols, 1), 1)

    missing_count = int(validation_report["missing"].sum()) if not validation_report["missing"].empty else 0
    duplicate_count = int(validation_report["duplicate_count"])
    invalid_numeric_count = int(validation_report["invalid_numeric_count"])
    invalid_date_count = int(validation_report["invalid_date_count"])

    missing_penalty = min(35, (missing_count / total_cells) * 200)
    duplicate_penalty = min(25, (duplicate_count / max(rows, 1)) * 120)
    invalid_penalty = min(20, ((invalid_numeric_count + invalid_date_count) / total_cells) * 250)

    score = max(0, round(100 - missing_penalty - duplicate_penalty - invalid_penalty))

    if score >= 90:
        label = "Excellent"
    elif score >= 75:
        label = "Good"
    elif score >= 60:
        label = "Fair"
    else:
        label = "Needs attention"

    breakdown = [
        {
            "label": "Missing-value penalty",
            "penalty": round(missing_penalty),
            "detail": f"{missing_count} missing cell(s) across {total_cells} total cells.",
        },
        {
            "label": "Duplicate penalty",
            "penalty": round(duplicate_penalty),
            "detail": f"{duplicate_count} duplicate row(s) relative to {max(rows, 1)} total row(s).",
        },
        {
            "label": "Invalid-value penalty",
            "penalty": round(invalid_penalty),
            "detail": f"{invalid_numeric_count + invalid_date_count} invalid numeric/date value(s) were detected.",
        },
        {
            "label": "Heuristic readiness model",
            "penalty": 0,
            "detail": "This is a rule-based readiness score for structural data quality, not a calibrated measure of business truth.",
        },
    ]

    return score, label, breakdown


def build_suggested_analyses(df, validation_report, chart_recommendations):
    options = [
        {
            "key": "quality_overview",
            "label": "Data quality overview",
            "description": "Review overall readiness, missing-value pressure, duplicates, and parsing issues in one place.",
        },
        {
            "key": "summary_statistics",
            "label": "Summary statistics",
            "description": "Review the main numeric metrics and scale of the active dataset view.",
        },
    ]

    if not validation_report["missing"].empty:
        options.append(
            {
                "key": "missing_breakdown",
                "label": "Missing values breakdown",
                "description": "Inspect which columns are incomplete and where cleanup effort is concentrated.",
            }
        )

    if validation_report["duplicate_count"] > 0:
        options.append(
            {
                "key": "duplicate_review",
                "label": "Duplicate record review",
                "description": "Review the duplicate strategy, the columns used, and the rows that were flagged.",
            }
        )

    if any(item["key"] == "category_bar" for item in chart_recommendations):
        options.append(
            {
                "key": "top_categories",
                "label": "Category comparison",
                "description": "Compare the most common categories and see which values dominate the dataset.",
            }
        )

    if any(item["key"] == "numeric_hist" for item in chart_recommendations):
        options.append(
            {
                "key": "numeric_distribution",
                "label": "Numeric distribution",
                "description": "Inspect the shape, spread, and skew of the most informative numeric measure.",
            }
        )
        options.append(
            {
                "key": "outlier_check",
                "label": "Outlier analysis",
                "description": "Check for extreme numeric values that may distort charts or need business review.",
            }
        )

    if any(item["key"] == "correlation_heatmap" for item in chart_recommendations):
        options.append(
            {
                "key": "correlation_analysis",
                "label": "Correlation analysis",
                "description": "Scan the strongest numeric relationships before drilling into a single pair.",
            }
        )
        options.append(
            {
                "key": "scatter_relationship",
                "label": "Scatter relationship",
                "description": "Inspect the strongest numeric pair in a focused relationship view.",
            }
        )

    if any(item["key"] == "trend_analysis" for item in chart_recommendations):
        options.append(
            {
                "key": "trend_analysis",
                "label": "Trend analysis",
                "description": "Look for movement over time when a usable date column is available.",
            }
        )

    options.append(
        {
            "key": "recommended_charts",
            "label": "Recommended charts",
            "description": "See which charts the app recommends for this dataset and the rationale behind each one.",
        }
    )

    return options


def interpret_custom_request(request, df, suggested_analyses):
    normalized_request = request.lower().strip()
    focus_columns = [col for col in df.columns if col.lower() in normalized_request]
    matched_keys = []

    for key, keywords in ANALYSIS_KEYWORDS.items():
        if any(keyword in normalized_request for keyword in keywords):
            matched_keys.append(key)

    if not matched_keys and focus_columns:
        focus_column = focus_columns[0]
        series = df[focus_column]

        if str(series.dtype).startswith("datetime"):
            matched_keys = ["trend_analysis"]
        elif str(series.dtype) in {"object", "string"}:
            matched_keys = ["top_categories"]
        else:
            matched_keys = ["numeric_distribution"]

    if not matched_keys:
        valid_keys = {item["key"] for item in suggested_analyses}
        fallback_key = "quality_overview" if "quality_overview" in valid_keys else suggested_analyses[0]["key"]
        matched_keys = [fallback_key]

    valid_keys = {item["key"] for item in suggested_analyses}
    primary_key = next((key for key in matched_keys if key in valid_keys), suggested_analyses[0]["key"])

    return {
        "primary_key": primary_key,
        "matched_keys": [key for key in matched_keys if key in valid_keys],
        "focus_columns": focus_columns,
    }


def generate_ai_insights(
    raw_df,
    active_df,
    validation_report,
    analysis_report,
    chart_recommendations,
    view_label,
    audience_mode="executive",
    preferred_date_column=None,
    preferred_value_column=None,
    preferred_category_column=None,
    column_roles=None,
):
    audience = AUDIENCE_MODES.get(audience_mode, AUDIENCE_MODES["executive"])
    score, quality_label, quality_score_breakdown = calculate_data_quality_score(raw_df, validation_report)
    rows, cols = active_df.shape
    numeric_column_count = len(get_numeric_columns(active_df, exclude_id_like=False))
    categorical_column_count = len(get_categorical_columns(active_df))
    date_column_count = len(get_datetime_columns(active_df))
    total_active_cells = max(rows * max(cols, 1), 1)
    active_missing_count = int(active_df.isna().sum().sum()) if cols > 0 else 0
    completeness_ratio = max(0.0, 1 - (active_missing_count / total_active_cells))
    strongest_correlation = analysis_report["strongest_correlation"]
    numeric_summary = analysis_report["numeric_summary"]
    segment_signal = analysis_report.get("segment_signal")
    trend_signal = analysis_report.get("trend_signal")
    confidence_score, confidence_label, confidence_reasons, confidence_breakdown = _calculate_insight_confidence(
        active_df,
        validation_report,
        analysis_report,
    )

    executive_summary = [
        f"The {view_label.lower()} gives you {rows} row(s) across {cols} column(s) to work with.",
        f"Overall quality is {quality_label.lower()} at {score}/100, which means the file is usable for guided analysis but still worth checking before high-stakes decisions.",
        audience["summary_line"],
    ]
    if segment_signal:
        segment_line = (
            f"Average '{segment_signal['numeric_column']}' is strongest in '{segment_signal['top_label']}' and weakest in '{segment_signal['bottom_label']}'."
            if segment_signal["support_label"] == "Supported"
            else f"A directional segment gap appears in '{segment_signal['category_column']}', where '{segment_signal['top_label']}' currently leads '{segment_signal['bottom_label']}'."
        )
        executive_summary.append(f"{segment_line} Guardrail: {segment_signal['support_reason']}")
    elif strongest_correlation:
        col_a, col_b = strongest_correlation["columns"]
        executive_summary.append(
            f"The strongest numeric relationship is between {col_a} and {col_b} at correlation {strongest_correlation['value']:.2f}. Guardrail: {strongest_correlation['support_reason']}"
        )
    if trend_signal:
        executive_summary.append(
            f"'{trend_signal['value_column']}' moves {trend_signal['direction']} across {trend_signal['date_column']}, from {_format_metric_value(trend_signal['first_value'])} to {_format_metric_value(trend_signal['last_value'])}. Guardrail: {trend_signal['support_reason']}"
        )
    elif analysis_report["categorical_summary"]:
        first_category, first_summary = next(iter(analysis_report["categorical_summary"].items()))
        executive_summary.append(
            f"The most common value in '{first_category}' is '{first_summary.index[0]}' with {int(first_summary.iloc[0])} records, which gives you a quick first segmentation view."
        )

    dataset_profile = [
        f"This view is structurally ready for {audience['profile_phrase']} because it contains {numeric_column_count} numeric column(s), {categorical_column_count} categorical column(s), and {date_column_count} date column(s).",
        f"Observed completeness is {completeness_ratio:.0%}, which tells you how much of the active view is still populated after cleaning and slicing decisions.",
    ]
    if column_roles is not None and column_roles.has_user_roles():
        dataset_profile.append(
            "User-defined column roles are active, so the narrative uses business meaning before falling back to automatic data-type inference."
        )
    if chart_recommendations:
        dataset_profile.append(
            f"InsightFlow found {len(chart_recommendations)} recommended chart path(s), so the dataset is rich enough for guided visual explanation instead of one generic graph."
        )
    if rows == 0:
        dataset_profile.append("The current view has zero rows, so the app can still describe structure but cannot make trustworthy claims about behavior.")

    quality_highlights = []
    if not validation_report["missing"].empty:
        top_missing_col = validation_report["missing"].index[0]
        quality_highlights.append(
            f"The heaviest cleanup burden sits in '{top_missing_col}', which has {int(validation_report['missing'].iloc[0])} missing value(s)."
        )
        quality_highlights.append(f"Across the active view, about {1 - completeness_ratio:.0%} of cells are still blank or null, so totals and averages may still hide gaps.")
    if validation_report["duplicate_count"] > 0:
        duplicate_basis = ", ".join(validation_report["duplicate_subset"]) if validation_report["duplicate_subset"] else "all columns"
        quality_highlights.append(
            f"{validation_report['duplicate_count']} duplicate row(s) were detected using {duplicate_basis}, so counts may be overstated until that rule is confirmed."
        )
    if validation_report["invalid_numeric"]:
        quality_highlights.append("At least one numeric field still contains values that could not be parsed cleanly, which can weaken comparisons and charts.")
    if validation_report["invalid_dates"]:
        quality_highlights.append("At least one date field contains invalid date strings, so trend reads should be checked before being presented.")
    if not quality_highlights:
        quality_highlights.append("No major validation problems were detected, so the file is already in strong working shape.")

    key_drivers = []
    if segment_signal:
        key_drivers.append(
            f"The fastest business read comes from '{segment_signal['category_column']}' versus average '{segment_signal['numeric_column']}': '{segment_signal['top_label']}' leads at {_format_metric_value(segment_signal['top_value'])}, while '{segment_signal['bottom_label']}' trails at {_format_metric_value(segment_signal['bottom_value'])}. Guardrail: {segment_signal['support_reason']}"
        )
    if strongest_correlation:
        col_a, col_b = strongest_correlation["columns"]
        key_drivers.append(
            f"'{col_a}' and '{col_b}' move together at correlation {strongest_correlation['value']:.2f}. Guardrail: {strongest_correlation['support_reason']}"
        )
    if trend_signal:
        key_drivers.append(
            f"The time signal is also usable: '{trend_signal['value_column']}' trends {trend_signal['direction']} from {_format_metric_value(trend_signal['first_value'])} to {_format_metric_value(trend_signal['last_value'])}. Guardrail: {trend_signal['support_reason']}"
        )
    if analysis_report["categorical_summary"]:
        first_category, first_summary = next(iter(analysis_report["categorical_summary"].items()))
        key_drivers.append(
            f"'{first_category}' is a usable segmentation field because its leading value accounts for about {int(first_summary.iloc[0]) / max(int(first_summary.sum()), 1):.0%} of the top-category sample."
        )
    if numeric_summary is not None and "std" in numeric_summary.columns:
        spread_series = numeric_summary["std"].fillna(0)
        if not spread_series.empty and float(spread_series.max()) > 0:
            key_drivers.append(f"'{spread_series.idxmax()}' has the widest numeric spread, so it is likely to drive the biggest visual differences across charts and summaries.")
    if chart_recommendations:
        key_drivers.append("The current structure is rich enough to support category, distribution, relationship, and trend views without any hardcoded demo-column assumptions.")
    if not key_drivers:
        key_drivers.append("The dataset is relatively simple, so the strongest move is likely targeted filtering plus one focused chart rather than a broad dashboard sweep.")

    analysis_highlights = []
    if analysis_report["outlier_summary"]:
        top_outlier_col, top_outlier_info = max(analysis_report["outlier_summary"].items(), key=lambda item: item[1]["share"])
        analysis_highlights.append(f"'{top_outlier_col}' contains the heaviest concentration of outliers at {top_outlier_info['share']:.0%} of non-null values, so raw averages there deserve caution.")
    if numeric_summary is not None and "max" in numeric_summary.columns and "min" in numeric_summary.columns:
        numeric_ranges = (numeric_summary["max"] - numeric_summary["min"]).fillna(0)
        if not numeric_ranges.empty and float(numeric_ranges.max()) > 0:
            analysis_highlights.append(f"'{numeric_ranges.idxmax()}' has the widest observed range, which makes it the best first candidate for distribution and anomaly review.")
    if segment_signal:
        analysis_highlights.append(
            f"The segment gap between '{segment_signal['top_label']}' and '{segment_signal['bottom_label']}' should be read as {segment_signal['support_label'].lower()} evidence."
        )
    if trend_signal:
        analysis_highlights.append(
            f"The time story from {trend_signal['start_date']} to {trend_signal['end_date']} is currently rated {trend_signal['support_label'].lower()} support."
        )
    if chart_recommendations:
        analysis_highlights.append(f"{len(chart_recommendations)} chart type(s) are recommended automatically, so the next visual step is already narrowed down for the user.")
    if not analysis_highlights:
        analysis_highlights.append("The dataset is structurally simple, so focused exploration will likely outperform a broad multi-chart review.")

    risk_flags = []
    if rows == 0:
        risk_flags.append("The active view has no rows, so any narrative claim about performance or patterns should be treated as unavailable.")
    if not validation_report["missing"].empty:
        risk_flags.append("Missing values may bias category totals, trend lines, or summary statistics if they are concentrated in the columns that drive the story.")
    if validation_report["duplicate_count"] > 0:
        risk_flags.append("Duplicate records may overstate counts or totals unless the current duplicate rule matches the business definition of a repeated record.")
    if analysis_report["outlier_summary"]:
        risk_flags.append("Outliers may compress histograms and pull averages away from the typical case, so medians and trimmed views may be more trustworthy.")
    if validation_report["invalid_numeric"] or validation_report["invalid_dates"]:
        risk_flags.append("Parsing issues in numeric or date columns can quietly weaken downstream summaries, especially if the affected fields feed charts or trend lines.")
    if strongest_correlation and strongest_correlation["support_label"] != "Supported":
        risk_flags.append("The strongest numeric relationship is still only directional, so it should not be treated as a stable driver without more evidence.")
    if trend_signal and trend_signal["support_label"] != "Supported":
        risk_flags.append("The current trend read is based on limited time coverage, so treat it as directional rather than forecast-like evidence.")
    if segment_signal and segment_signal["support_label"] != "Supported":
        risk_flags.append("The current segment gap is real enough to inspect, but not strong enough to frame as a settled performance difference yet.")
    if risk_flags:
        risk_flags.append(audience["risk_line"])
    else:
        risk_flags.append("No major analytic risk signals were detected in the current view.")

    next_steps = [audience["next_step_line"]]
    if chart_recommendations:
        next_steps.append(f"Start with {chart_recommendations[0]['title'].lower()} so the first visual matches the strongest structure in the file.")
    if not validation_report["missing"].empty:
        next_steps.append("Review the columns with the highest missing values before presenting any business conclusion as final.")
    if validation_report["duplicate_count"] > 0:
        next_steps.append("Confirm that the duplicate-removal strategy matches the business meaning of a repeated record.")
    if analysis_report["outlier_summary"]:
        next_steps.append("Inspect the outlier-heavy numeric fields to decide whether extreme values are valid events or data issues.")
    if segment_signal:
        next_steps.append(f"Compare '{segment_signal['top_label']}' against '{segment_signal['bottom_label']}' in '{segment_signal['category_column']}' to explain the largest segment gap.")
    if trend_signal:
        next_steps.append(f"Use the time view on '{trend_signal['value_column']}' to validate whether the observed {trend_signal['direction']} movement is operationally meaningful.")

    next_questions = _build_next_questions(validation_report, analysis_report, segment_signal, trend_signal, strongest_correlation)

    assumption_badges = ["Heuristic readiness score", "Heuristic narrative confidence"]
    if column_roles is not None and column_roles.has_user_roles():
        assumption_badges.append("User-defined column roles")
    assumption_badges.append("User-confirmed duplicate rule" if validation_report["duplicate_subset"] else "Conservative duplicate rule")
    if preferred_date_column:
        assumption_badges.append(f"User-selected trend date: {preferred_date_column}")
    elif trend_signal:
        assumption_badges.append("Auto-selected trend date")
    if preferred_value_column:
        assumption_badges.append(f"User-selected trend metric: {preferred_value_column}")
    elif trend_signal:
        assumption_badges.append("Auto-selected trend metric")
    if preferred_category_column:
        assumption_badges.append(f"User-selected segment: {preferred_category_column}")
    if strongest_correlation:
        assumption_badges.append(f"Correlation support: {strongest_correlation['support_label']}")
    if trend_signal:
        assumption_badges.append(f"Trend support: {trend_signal['support_label']}")
    if segment_signal:
        assumption_badges.append(f"Segment support: {segment_signal['support_label']}")

    signal_guardrails = []
    if strongest_correlation:
        col_a, col_b = strongest_correlation["columns"]
        signal_guardrails.append(
            f"Correlation guardrail for '{col_a}' vs '{col_b}': {strongest_correlation['support_label']} support based on {strongest_correlation['sample_size']} paired row(s). {strongest_correlation['support_reason']}"
        )
    if trend_signal:
        signal_guardrails.append(
            f"Trend guardrail for '{trend_signal['value_column']}' over '{trend_signal['date_column']}': {trend_signal['support_label']} support across {trend_signal['distinct_periods']} distinct period(s). {trend_signal['support_reason']}"
        )
    if segment_signal:
        signal_guardrails.append(
            f"Segment guardrail for '{segment_signal['category_column']}' on '{segment_signal['numeric_column']}': {segment_signal['support_label']} support with a minimum group size of {segment_signal['min_group_size']}. {segment_signal['support_reason']}"
        )

    return {
        "quality_score": score,
        "quality_label": quality_label,
        "quality_score_breakdown": quality_score_breakdown,
        "audience_label": audience["label"],
        "column_roles": column_roles.to_dict() if column_roles is not None and column_roles.has_user_roles() else {},
        "insight_confidence_score": confidence_score,
        "insight_confidence_label": confidence_label,
        "insight_confidence_reasons": confidence_reasons,
        "insight_confidence_breakdown": confidence_breakdown,
        "executive_summary": executive_summary,
        "dataset_profile": dataset_profile,
        "quality_highlights": quality_highlights,
        "key_drivers": key_drivers,
        "analysis_highlights": analysis_highlights,
        "risk_flags": risk_flags,
        "next_steps": next_steps,
        "next_questions": next_questions,
        "assumption_badges": assumption_badges,
        "signal_guardrails": signal_guardrails,
    }
