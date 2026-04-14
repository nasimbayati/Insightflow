from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modules.analysis import build_analysis_report
from modules.cleaning import clean_data, standardize_data
from modules.ingestion import ingest_csv
from modules.insights import annotate_chart_recommendations, build_suggested_analyses, generate_ai_insights, recommend_charts
from modules.llm_insights import build_llm_context
from modules.pipeline_config import coerce_pipeline_preferences
from modules.reporting import build_cleaning_impact_items, build_decision_brief_markdown, build_run_report_payload
from modules.validation import (
    check_duplicates,
    check_missing_values,
    detect_column_types,
    evaluate_duplicate_rule,
    find_invalid_date_values,
    find_invalid_numeric_values,
    get_duplicate_rows,
    suggest_duplicate_subset,
)


@dataclass
class UploadRunContext:
    raw_df: pd.DataFrame
    ingestion_metadata: dict
    raw_column_types: dict
    suggested_duplicate_subset: list | None
    missing: pd.Series
    invalid_numeric: dict
    invalid_dates: dict
    invalid_numeric_count: int
    invalid_date_count: int
    source_label: str
    file_stem: str


@dataclass
class BaseRunContext:
    upload: UploadRunContext
    chart_view_mode: str
    cleaning_config: object
    audience_mode: str
    pipeline_preferences: dict
    duplicate_subset: list | None
    duplicate_count: int
    duplicate_rows: pd.DataFrame
    duplicate_diagnostics: dict
    validation_report: dict
    raw_analysis_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    transformation_log: list[str]
    base_df_to_use: pd.DataFrame
    base_view_label: str
    cleaning_impact_items: list[str]
    analysis_report: dict
    chart_recommendations: list[dict]
    ai_report: dict
    run_report_payload: dict
    decision_brief_markdown: str


@dataclass
class AnalysisRunContext:
    base: BaseRunContext
    analysis_df: pd.DataFrame
    applied_filters: list[str]
    view_label: str
    validation_report: dict
    analysis_report: dict
    chart_recommendations: list[dict]
    ai_report: dict
    run_report_payload: dict
    decision_brief_markdown: str
    suggested_analyses: list[dict]
    llm_context: dict


def _build_validation_report(upload_context, duplicate_subset, duplicate_count, duplicate_rows, duplicate_diagnostics):
    return {
        "missing": upload_context.missing,
        "duplicate_count": duplicate_count,
        "duplicate_subset": duplicate_subset,
        "duplicate_rows": duplicate_rows,
        "invalid_numeric": upload_context.invalid_numeric,
        "invalid_dates": upload_context.invalid_dates,
        "invalid_numeric_count": upload_context.invalid_numeric_count,
        "invalid_date_count": upload_context.invalid_date_count,
        "duplicate_diagnostics": duplicate_diagnostics,
    }


def create_upload_run_context(uploaded_file, max_size_mb, source_label):
    raw_df, ingestion_metadata = ingest_csv(uploaded_file, max_size_mb=max_size_mb)
    raw_column_types = detect_column_types(raw_df)
    suggested_duplicate_subset = suggest_duplicate_subset(raw_df, raw_column_types)
    missing = check_missing_values(raw_df)
    invalid_numeric = find_invalid_numeric_values(raw_df, raw_column_types)
    invalid_dates = find_invalid_date_values(raw_df, raw_column_types)

    return UploadRunContext(
        raw_df=raw_df,
        ingestion_metadata=ingestion_metadata,
        raw_column_types=raw_column_types,
        suggested_duplicate_subset=suggested_duplicate_subset,
        missing=missing,
        invalid_numeric=invalid_numeric,
        invalid_dates=invalid_dates,
        invalid_numeric_count=sum(len(values) for values in invalid_numeric.values()),
        invalid_date_count=sum(len(values) for values in invalid_dates.values()),
        source_label=source_label,
        file_stem=Path(ingestion_metadata["filename"]).stem,
    )


def build_base_run_context(upload_context, chart_view_mode, cleaning_config, audience_mode, pipeline_preferences):
    pipeline_preferences = coerce_pipeline_preferences(pipeline_preferences)
    duplicate_subset = pipeline_preferences["duplicate_subset"]
    duplicate_count = check_duplicates(upload_context.raw_df, subset=duplicate_subset)
    duplicate_rows = get_duplicate_rows(upload_context.raw_df, subset=duplicate_subset).head(10)
    duplicate_diagnostics = evaluate_duplicate_rule(
        upload_context.raw_df,
        subset=duplicate_subset,
        column_types=upload_context.raw_column_types,
    )

    validation_report = _build_validation_report(
        upload_context,
        duplicate_subset,
        duplicate_count,
        duplicate_rows,
        duplicate_diagnostics,
    )

    raw_analysis_df = standardize_data(
        upload_context.raw_df,
        upload_context.raw_column_types,
        cleaning_config=cleaning_config,
    )
    cleaned_df, transformation_log = clean_data(
        df=upload_context.raw_df,
        column_types=upload_context.raw_column_types,
        duplicate_subset=duplicate_subset,
        cleaning_config=cleaning_config,
    )

    base_df_to_use = cleaned_df if chart_view_mode == "Cleaned Data" else raw_analysis_df
    base_view_label = "cleaned dataset" if chart_view_mode == "Cleaned Data" else "raw dataset"
    cleaning_impact_items = build_cleaning_impact_items(
        upload_context.raw_df,
        cleaned_df,
        duplicate_subset=duplicate_subset,
    )
    analysis_report = build_analysis_report(
        base_df_to_use,
        preferred_value_column=pipeline_preferences.preferred_metric_column(),
        preferred_date_column=pipeline_preferences.preferred_time_column(),
        preferred_category_column=pipeline_preferences.preferred_segment_column(),
    )
    chart_recommendations = annotate_chart_recommendations(
        base_df_to_use,
        analysis_report,
        recommend_charts(
            base_df_to_use,
            preferred_value_column=pipeline_preferences.preferred_metric_column(),
            preferred_date_column=pipeline_preferences.preferred_time_column(),
            preferred_category_column=pipeline_preferences.preferred_segment_column(),
        ),
    )
    ai_report = generate_ai_insights(
        raw_df=upload_context.raw_df,
        active_df=base_df_to_use,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        view_label=base_view_label,
        audience_mode=audience_mode,
        preferred_date_column=pipeline_preferences.preferred_time_column(),
        preferred_value_column=pipeline_preferences.preferred_metric_column(),
        preferred_category_column=pipeline_preferences.preferred_segment_column(),
        column_roles=pipeline_preferences.column_roles,
    )
    validation_report["quality_score"] = ai_report["quality_score"]
    validation_report["quality_label"] = ai_report["quality_label"]
    run_report_payload = build_run_report_payload(
        upload_context.ingestion_metadata["filename"],
        base_view_label,
        upload_context.ingestion_metadata,
        validation_report,
        analysis_report,
        ai_report,
        cleaning_config,
        pipeline_preferences,
        cleaning_impact_items,
        transformation_log,
        chart_recommendations,
        duplicate_diagnostics=duplicate_diagnostics,
    )
    decision_brief_markdown = build_decision_brief_markdown(
        upload_context.ingestion_metadata["filename"],
        base_view_label,
        ai_report,
        chart_recommendations,
        cleaning_impact_items,
    )

    return BaseRunContext(
        upload=upload_context,
        chart_view_mode=chart_view_mode,
        cleaning_config=cleaning_config,
        audience_mode=audience_mode,
        pipeline_preferences=pipeline_preferences,
        duplicate_subset=duplicate_subset,
        duplicate_count=duplicate_count,
        duplicate_rows=duplicate_rows,
        duplicate_diagnostics=duplicate_diagnostics,
        validation_report=validation_report,
        raw_analysis_df=raw_analysis_df,
        cleaned_df=cleaned_df,
        transformation_log=transformation_log,
        base_df_to_use=base_df_to_use,
        base_view_label=base_view_label,
        cleaning_impact_items=cleaning_impact_items,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        ai_report=ai_report,
        run_report_payload=run_report_payload,
        decision_brief_markdown=decision_brief_markdown,
    )


def build_analysis_run_context(base_context, analysis_df, applied_filters):
    view_label = f"filtered {base_context.base_view_label}" if applied_filters else base_context.base_view_label
    analysis_report = build_analysis_report(
        analysis_df,
        preferred_value_column=base_context.pipeline_preferences.preferred_metric_column(),
        preferred_date_column=base_context.pipeline_preferences.preferred_time_column(),
        preferred_category_column=base_context.pipeline_preferences.preferred_segment_column(),
    )
    chart_recommendations = annotate_chart_recommendations(
        analysis_df,
        analysis_report,
        recommend_charts(
            analysis_df,
            preferred_value_column=base_context.pipeline_preferences.preferred_metric_column(),
            preferred_date_column=base_context.pipeline_preferences.preferred_time_column(),
            preferred_category_column=base_context.pipeline_preferences.preferred_segment_column(),
        ),
    )
    validation_report = dict(base_context.validation_report)
    ai_report = generate_ai_insights(
        raw_df=base_context.upload.raw_df,
        active_df=analysis_df,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        view_label=view_label,
        audience_mode=base_context.audience_mode,
        preferred_date_column=base_context.pipeline_preferences.preferred_time_column(),
        preferred_value_column=base_context.pipeline_preferences.preferred_metric_column(),
        preferred_category_column=base_context.pipeline_preferences.preferred_segment_column(),
        column_roles=base_context.pipeline_preferences.column_roles,
    )
    validation_report["quality_score"] = ai_report["quality_score"]
    validation_report["quality_label"] = ai_report["quality_label"]
    run_report_payload = build_run_report_payload(
        base_context.upload.ingestion_metadata["filename"],
        view_label,
        base_context.upload.ingestion_metadata,
        validation_report,
        analysis_report,
        ai_report,
        base_context.cleaning_config,
        base_context.pipeline_preferences,
        base_context.cleaning_impact_items,
        base_context.transformation_log,
        chart_recommendations,
        applied_filters=applied_filters,
        duplicate_diagnostics=base_context.duplicate_diagnostics,
    )
    decision_brief_markdown = build_decision_brief_markdown(
        base_context.upload.ingestion_metadata["filename"],
        view_label,
        ai_report,
        chart_recommendations,
        base_context.cleaning_impact_items,
        applied_filters=applied_filters,
    )
    suggested_analyses = build_suggested_analyses(analysis_df, validation_report, chart_recommendations)
    llm_context = build_llm_context(
        raw_df=base_context.upload.raw_df,
        active_df=analysis_df,
        raw_column_types=base_context.upload.raw_column_types,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        view_label=view_label,
        active_filters=applied_filters,
    )

    return AnalysisRunContext(
        base=base_context,
        analysis_df=analysis_df,
        applied_filters=applied_filters,
        view_label=view_label,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        ai_report=ai_report,
        run_report_payload=run_report_payload,
        decision_brief_markdown=decision_brief_markdown,
        suggested_analyses=suggested_analyses,
        llm_context=llm_context,
    )


def build_boardroom_fingerprint(base_context, review_mode):
    return {
        "file_name": base_context.upload.ingestion_metadata["filename"],
        "view": base_context.base_view_label,
        "review_mode": review_mode,
        "duplicate_subset": base_context.duplicate_subset,
        "column_roles": base_context.pipeline_preferences.column_roles.to_dict(),
        "cleaning": {
            "numeric": base_context.cleaning_config.numeric_missing_strategy,
            "categorical": base_context.cleaning_config.categorical_missing_strategy,
            "duplicate_action": base_context.cleaning_config.duplicate_action,
            "text": base_context.cleaning_config.categorical_text_strategy,
        },
    }


def build_analysis_fingerprint(analysis_context, review_mode):
    return {
        "file_name": analysis_context.base.upload.ingestion_metadata["filename"],
        "view": analysis_context.view_label,
        "review_mode": review_mode,
        "filters": analysis_context.applied_filters,
        "duplicate_subset": analysis_context.base.duplicate_subset,
        "trend_date": analysis_context.base.pipeline_preferences["trend_date_column"],
        "trend_value": analysis_context.base.pipeline_preferences["trend_value_column"],
        "column_roles": analysis_context.base.pipeline_preferences.column_roles.to_dict(),
        "audience": analysis_context.base.audience_mode,
    }
