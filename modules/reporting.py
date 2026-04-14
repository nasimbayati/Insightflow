import pandas as pd

from modules.validation import (
    check_duplicates,
    check_missing_values,
    detect_column_types,
    find_invalid_date_values,
    find_invalid_numeric_values,
)


def format_column_types(column_types):
    return pd.DataFrame(
        [{"Column": column, "Detected Type": detected_type} for column, detected_type in column_types.items()]
    )


def format_issue_dict(issue_dict, header):
    rows = []

    for column, values in issue_dict.items():
        preview = ", ".join(str(value) for value in values[:5])
        if len(values) > 5:
            preview = f"{preview}, ..."

        rows.append(
            {
                "Column": column,
                header: preview,
                "Count": len(values),
            }
        )

    return pd.DataFrame(rows)


def format_outlier_summary(outlier_summary):
    rows = []

    for column, details in outlier_summary.items():
        rows.append(
            {
                "Column": column,
                "Outlier Count": details["count"],
                "Share": f"{details['share']:.0%}",
                "Lower Bound": round(details["lower_bound"], 2),
                "Upper Bound": round(details["upper_bound"], 2),
            }
        )

    return pd.DataFrame(rows).sort_values(by="Outlier Count", ascending=False) if rows else pd.DataFrame()


def build_validation_snapshot(df, duplicate_subset=None):
    column_types = detect_column_types(df)
    missing = check_missing_values(df)
    invalid_numeric = find_invalid_numeric_values(df, column_types)
    invalid_dates = find_invalid_date_values(df, column_types)
    return {
        "missing_total": int(missing.sum()) if not missing.empty else 0,
        "duplicate_count": check_duplicates(df, subset=duplicate_subset),
        "invalid_numeric_count": sum(len(values) for values in invalid_numeric.values()),
        "invalid_date_count": sum(len(values) for values in invalid_dates.values()),
    }


def build_cleaning_impact_items(raw_df, cleaned_df, duplicate_subset=None):
    raw_snapshot = build_validation_snapshot(raw_df, duplicate_subset=duplicate_subset)
    cleaned_snapshot = build_validation_snapshot(cleaned_df, duplicate_subset=duplicate_subset)
    return [
        f"Missing cells: {raw_snapshot['missing_total']} -> {cleaned_snapshot['missing_total']}",
        f"Duplicate rows: {raw_snapshot['duplicate_count']} -> {cleaned_snapshot['duplicate_count']}",
        f"Invalid numeric values: {raw_snapshot['invalid_numeric_count']} -> {cleaned_snapshot['invalid_numeric_count']}",
        f"Invalid date values: {raw_snapshot['invalid_date_count']} -> {cleaned_snapshot['invalid_date_count']}",
    ]


def build_decision_brief_markdown(
    file_name,
    view_label,
    ai_report,
    chart_recommendations,
    cleaning_impact_items,
    applied_filters=None,
):
    lines = [
        "# InsightFlow Decision Brief",
        "",
        f"- File: {file_name}",
        f"- View: {view_label}",
        f"- Insight audience: {ai_report.get('audience_label', 'Executive brief')}",
        f"- Heuristic readiness score: {ai_report['quality_score']}/100 ({ai_report['quality_label']})",
        f"- Heuristic narrative confidence: {ai_report['insight_confidence_label']}",
        "",
        "## Executive Summary",
    ]
    lines.extend(f"- {item}" for item in ai_report["executive_summary"])
    lines.extend(
        [
            "",
            "## Top Risk",
            f"- {ai_report['risk_flags'][0] if ai_report['risk_flags'] else 'No major risks detected.'}",
            "",
            "## Key Driver",
            f"- {ai_report['key_drivers'][0] if ai_report['key_drivers'] else 'No major drivers detected.'}",
            "",
            "## Recommended Actions",
        ]
    )
    lines.extend(f"- {item}" for item in ai_report["next_steps"])
    lines.extend(["", "## Cleaning Impact"])
    lines.extend(f"- {item}" for item in cleaning_impact_items)
    lines.extend(["", "## Confidence Notes"])
    lines.extend(f"- {item}" for item in ai_report["insight_confidence_reasons"])
    lines.extend(
        [
            "",
            "## Scoring Note",
            "- InsightFlow readiness and confidence signals are heuristic summaries of missing values, duplicates, parsing issues, row count, and outlier pressure. They are not statistical guarantees.",
        ]
    )
    if ai_report["next_questions"]:
        lines.extend(["", "## Next Questions"])
        lines.extend(f"- {item}" for item in ai_report["next_questions"])
    if chart_recommendations:
        lines.extend(["", "## Recommended Charts"])
        for recommendation in chart_recommendations[:3]:
            lines.append(f"- {recommendation['title']}: {recommendation['description']}")
    if applied_filters:
        lines.extend(["", "## Active Filters"])
        lines.extend(f"- {item}" for item in applied_filters)
    column_roles = ai_report.get("column_roles")
    if column_roles:
        lines.extend(["", "## Column Roles"])
        for label, value in column_roles.items():
            if value:
                lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _serialize_pipeline_preferences(pipeline_preferences):
    if hasattr(pipeline_preferences, "to_dict"):
        return pipeline_preferences.to_dict()
    return pipeline_preferences


def build_run_report_payload(
    file_name,
    view_label,
    ingestion_metadata,
    validation_report,
    analysis_report,
    ai_report,
    cleaning_config,
    pipeline_preferences,
    cleaning_impact_items,
    transformation_log,
    chart_recommendations,
    applied_filters=None,
    duplicate_diagnostics=None,
):
    return {
        "file_name": file_name,
        "view_label": view_label,
        "ingestion_metadata": ingestion_metadata,
        "validation_summary": {
            "missing_cells": int(validation_report["missing"].sum()) if not validation_report["missing"].empty else 0,
            "duplicate_count": int(validation_report["duplicate_count"]),
            "duplicate_subset": validation_report.get("duplicate_subset"),
            "invalid_numeric_count": int(validation_report["invalid_numeric_count"]),
            "invalid_date_count": int(validation_report["invalid_date_count"]),
        },
        "duplicate_diagnostics": duplicate_diagnostics,
        "analysis_summary": {
            "shape": analysis_report["shape"],
            "strongest_correlation": analysis_report.get("strongest_correlation"),
            "trend_signal": analysis_report.get("trend_signal"),
            "segment_signal": analysis_report.get("segment_signal"),
            "outlier_summary": analysis_report.get("outlier_summary"),
        },
        "ai_summary": {
            "quality_score": ai_report["quality_score"],
            "quality_label": ai_report["quality_label"],
            "quality_score_breakdown": ai_report.get("quality_score_breakdown", []),
            "insight_confidence_label": ai_report["insight_confidence_label"],
            "insight_confidence_breakdown": ai_report.get("insight_confidence_breakdown", []),
            "executive_summary": ai_report.get("executive_summary", []),
            "risk_flags": ai_report.get("risk_flags", []),
            "next_steps": ai_report.get("next_steps", []),
            "next_questions": ai_report.get("next_questions", []),
            "assumption_badges": ai_report.get("assumption_badges", []),
        },
        "cleaning_config": {
            "numeric_missing_strategy": cleaning_config.numeric_missing_strategy,
            "categorical_missing_strategy": cleaning_config.categorical_missing_strategy,
            "duplicate_action": cleaning_config.duplicate_action,
            "categorical_text_strategy": cleaning_config.categorical_text_strategy,
            "numeric_column_strategies": cleaning_config.numeric_column_strategies,
            "categorical_column_strategies": cleaning_config.categorical_column_strategies,
        },
        "pipeline_preferences": _serialize_pipeline_preferences(pipeline_preferences),
        "applied_filters": applied_filters or [],
        "cleaning_impact": cleaning_impact_items,
        "transformation_log": transformation_log,
        "chart_recommendations": chart_recommendations,
    }
