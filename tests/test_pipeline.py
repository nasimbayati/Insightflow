from io import BytesIO
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.analysis import apply_dataset_filters, build_analysis_report, select_time_series_columns
from modules.artifacts import load_recent_registry_entries, persist_run_artifacts
from modules.cleaning import clean_data, standardize_data
from modules.ingestion import ingest_csv
from modules.insights import (
    build_suggested_analyses,
    generate_ai_insights,
    interpret_custom_request,
    recommend_charts,
)
from modules.monitoring import build_monitoring_snapshot, log_monitoring_event
from modules.pipeline_config import ColumnRoles, PipelinePreferences, coerce_pipeline_preferences
from modules.validation import detect_column_types, evaluate_duplicate_rule, suggest_duplicate_subset
from modules.visualization import build_grouped_metric_summary


def test_clean_data_preserves_text_casing_and_removes_dynamic_duplicates():
    df = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "product": ["Apple", " apple ", "Orange"],
            "region": ["North", "north", "South"],
            "amount": ["10", "10", "15"],
        }
    )

    column_types = detect_column_types(df)
    duplicate_subset = suggest_duplicate_subset(df, column_types)
    cleaned_df, transformation_log = clean_data(df, column_types, duplicate_subset=duplicate_subset)

    assert duplicate_subset is not None
    assert len(cleaned_df) == 2
    assert cleaned_df.iloc[0]["product"] == "Apple"
    assert any("Removed 1 duplicate row(s)." in item for item in transformation_log)


def test_standardize_data_converts_numeric_and_date_types_without_removing_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "sales": ["1,200", "850"],
            "event_date": ["2024-01-01", "2024-01-02"],
        }
    )

    column_types = detect_column_types(df)
    standardized_df = standardize_data(df, column_types)

    assert len(standardized_df) == 2
    assert standardized_df["sales"].dtype.kind in {"i", "f"}
    assert str(standardized_df["event_date"].dtype).startswith("datetime64")


def test_chart_recommendations_and_custom_request_mapping():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "West", "North", "South"],
            "sales": [100, 120, 90, 130, 110],
            "profit": [20, 25, 15, 28, 23],
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )

    chart_recommendations = recommend_charts(df)
    suggested_analyses = build_suggested_analyses(
        df,
        validation_report={
            "missing": pd.Series(dtype="int64"),
            "duplicate_count": 0,
        },
        chart_recommendations=chart_recommendations,
    )
    interpreted_request = interpret_custom_request("show sales trend", df, suggested_analyses)

    recommended_keys = {item["key"] for item in chart_recommendations}

    assert "category_bar" in recommended_keys
    assert "numeric_hist" in recommended_keys
    assert "correlation_heatmap" in recommended_keys
    assert "trend_analysis" in recommended_keys
    assert interpreted_request["primary_key"] == "trend_analysis"
    assert "sales" in interpreted_request["focus_columns"]


def test_clean_data_supports_configurable_missing_value_strategies():
    df = pd.DataFrame(
        {
            "region": [" North ", None, "South"],
            "sales": ["10", None, "20"],
        }
    )

    column_types = detect_column_types(df)
    cleaned_df, transformation_log = clean_data(
        df,
        column_types,
        cleaning_config={
            "numeric_missing_strategy": "mean",
            "categorical_missing_strategy": "mode",
            "duplicate_action": "keep",
            "categorical_text_strategy": "title",
        },
    )

    assert cleaned_df.loc[1, "sales"] == 15
    assert cleaned_df.loc[1, "region"] == "North"
    assert any("Numeric missing strategy: mean." in item for item in transformation_log)
    assert any("Categorical missing strategy: mode." in item for item in transformation_log)


def test_clean_data_supports_column_specific_missing_value_overrides():
    df = pd.DataFrame(
        {
            "sales": ["10", None, "20"],
            "cost": ["3", None, "9"],
            "region": ["North", None, "South"],
            "status": ["Open", None, "Closed"],
        }
    )

    column_types = detect_column_types(df)
    cleaned_df, _ = clean_data(
        df,
        column_types,
        cleaning_config={
            "numeric_missing_strategy": "median",
            "categorical_missing_strategy": "unknown",
            "numeric_column_strategies": {"cost": "leave"},
            "categorical_column_strategies": {"status": "mode"},
        },
    )

    assert cleaned_df.loc[1, "sales"] == 15
    assert pd.isna(cleaned_df.loc[1, "cost"])
    assert cleaned_df.loc[1, "region"] == "Unknown"
    assert cleaned_df.loc[1, "status"] == "Closed"


def test_resolve_cleaning_config_accepts_config_like_objects():
    class LegacyCleaningConfig:
        numeric_missing_strategy = "mean"
        categorical_missing_strategy = "mode"
        duplicate_action = "keep"
        categorical_text_strategy = "title"
        numeric_column_strategies = {"sales": "leave"}
        categorical_column_strategies = {"region": "unknown"}

    from modules.cleaning import resolve_cleaning_config

    resolved = resolve_cleaning_config(LegacyCleaningConfig())

    assert resolved.numeric_missing_strategy == "mean"
    assert resolved.categorical_missing_strategy == "mode"
    assert resolved.duplicate_action == "keep"
    assert resolved.categorical_text_strategy == "title"
    assert resolved.numeric_column_strategies == {"sales": "leave"}
    assert resolved.categorical_column_strategies == {"region": "unknown"}


def test_clean_data_protects_identifier_and_high_cardinality_columns():
    df = pd.DataFrame(
        {
            "customer_id": ["1001", None, "1003", "1004", "1005"],
            "email": ["a@test.com", None, "c@test.com", "d@test.com", "e@test.com"],
            "sales": ["10", None, "20", "30", "40"],
        }
    )

    column_types = detect_column_types(df)
    cleaned_df, transformation_log = clean_data(df, column_types)

    assert pd.isna(cleaned_df.loc[1, "customer_id"])
    assert pd.isna(cleaned_df.loc[1, "email"])
    assert cleaned_df.loc[1, "sales"] == 25
    assert any("probable identifier" in item for item in transformation_log)
    assert any("high-cardinality categorical field" in item for item in transformation_log)


def test_build_analysis_report_detects_outliers():
    df = pd.DataFrame(
        {
            "sales": [10, 12, 11, 200, 13],
            "profit": [1, 1.5, 1.8, 2.0, 2.2],
        }
    )

    analysis_report = build_analysis_report(df)

    assert "sales" in analysis_report["outlier_summary"]
    assert analysis_report["outlier_summary"]["sales"]["count"] == 1


def test_ingest_csv_repairs_short_rows_and_skips_long_rows():
    class UploadedBytesIO(BytesIO):
        def __init__(self, payload, name):
            super().__init__(payload)
            self.name = name
            self.size = len(payload)

    payload = b"region,sales\nNorth,100\nSouth\nWest,150,extra\n"
    uploaded_file = UploadedBytesIO(payload, "sample.csv")

    df, metadata = ingest_csv(uploaded_file)

    assert df.shape == (2, 2)
    assert metadata["repaired_row_count"] == 1
    assert metadata["skipped_row_count"] == 1
    assert metadata["skipped_rows"][0]["reason"] == "Too many fields for the detected header"
    assert "West,150,extra" in metadata["skipped_rows"][0]["row_text"]


def test_apply_dataset_filters_handles_categorical_numeric_and_date_constraints():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "North", "West"],
            "sales": [100, 120, 130, 90],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        }
    )

    filtered_df, applied_filters = apply_dataset_filters(
        df,
        categorical_filters={"region": ["North"]},
        numeric_filters={"sales": (110, 140)},
        date_filters={"event_date": (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))},
    )

    assert filtered_df.shape == (1, 3)
    assert filtered_df.iloc[0]["sales"] == 130
    assert len(applied_filters) == 3


def test_select_time_series_columns_prefers_requested_or_better_date_axis():
    df = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "sales": [10, 12, 14, 16],
        }
    )

    auto_selected = select_time_series_columns(df)
    preferred_selected = select_time_series_columns(df, preferred_date_column="event_date")

    assert auto_selected[0] == "event_date"
    assert preferred_selected[0] == "event_date"


def test_build_analysis_report_includes_signal_guardrails():
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "revenue": [100, 105, 110, 80, 82, 85, 120, 125, 128],
            "margin": [30, 31, 33, 22, 23, 24, 35, 36, 37],
            "event_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
        }
    )

    analysis_report = build_analysis_report(df)

    assert analysis_report["strongest_correlation"]["support_label"] in {"Directional", "Supported"}
    assert analysis_report["strongest_correlation"]["sample_size"] >= 8
    assert analysis_report["trend_signal"]["distinct_periods"] >= 4
    assert analysis_report["segment_signal"]["min_group_size"] >= 2


def test_evaluate_duplicate_rule_flags_weak_single_column_key():
    df = pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "South"],
            "sales": [10, 11, 12, 13, 14],
        }
    )

    column_types = detect_column_types(df)
    diagnostics = evaluate_duplicate_rule(df, subset=["region"], column_types=column_types)

    assert diagnostics["status"] == "Weak"
    assert diagnostics["duplicate_row_count"] > 0
    assert diagnostics["unique_record_ratio"] < 1


def test_generate_ai_insights_respects_audience_mode():
    raw_df = pd.DataFrame(
        {
            "region": ["North", "South", "North", "South"],
            "sales": [100, 80, 120, 70],
            "returns": [3, 7, 2, 8],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        }
    )
    active_df = raw_df.copy()
    validation_report = {
        "missing": pd.Series(dtype="int64"),
        "duplicate_count": 0,
        "duplicate_subset": None,
        "duplicate_rows": pd.DataFrame(),
        "invalid_numeric": {},
        "invalid_dates": {},
        "invalid_numeric_count": 0,
        "invalid_date_count": 0,
    }
    analysis_report = build_analysis_report(active_df)
    chart_recommendations = recommend_charts(active_df)

    ai_report = generate_ai_insights(
        raw_df=raw_df,
        active_df=active_df,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        view_label="cleaned dataset",
        audience_mode="operator",
        preferred_date_column="event_date",
        preferred_value_column="sales",
    )

    combined_text = " ".join(ai_report["executive_summary"] + ai_report["next_steps"] + ai_report["risk_flags"]).lower()

    assert ai_report["audience_label"] == "Operational review"
    assert "process" in combined_text or "operational" in combined_text or "execution" in combined_text
    assert any(item["label"] == "Missing-value penalty" for item in ai_report["quality_score_breakdown"])
    assert any("Heuristic readiness score" == badge for badge in ai_report["assumption_badges"])
    assert ai_report["insight_confidence_breakdown"]
    assert ai_report["signal_guardrails"]


def test_artifact_registry_and_monitoring_are_persisted(tmp_path):
    report_payload = {"quality_score": 82, "summary": ["test"]}
    artifact_manifest = persist_run_artifacts(
        tmp_path,
        file_stem="demo",
        report_payload=report_payload,
        decision_brief_markdown="# Demo",
        cleaned_df=pd.DataFrame({"a": [1, 2]}),
        active_df=pd.DataFrame({"a": [1]}),
        rejected_rows_df=pd.DataFrame({"line_number": [3], "reason": ["extra fields"]}),
    )

    log_monitoring_event(tmp_path, "analysis_completed", payload={"quality_score": 82}, run_id=artifact_manifest["run_id"])

    recent_artifacts = load_recent_registry_entries(tmp_path, limit=2)
    monitoring_snapshot = build_monitoring_snapshot(tmp_path, limit=5)

    assert artifact_manifest["run_id"]
    assert Path(artifact_manifest["report_path"]).exists()
    assert recent_artifacts[0]["run_id"] == artifact_manifest["run_id"]
    assert monitoring_snapshot["latest_event"]["event_type"] == "analysis_completed"


def test_build_grouped_metric_summary_supports_user_selected_chart_columns():
    df = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 2, 3, 3],
            "subject": ["Math", "English", "Math", "English", "Math", "English"],
            "year": [2005, 2005, 2005, 2005, 2006, 2006],
            "score": [41, 55, 98, 63, 71, 53],
        }
    )

    subject_summary, subject_label = build_grouped_metric_summary(
        df,
        group_col="subject",
        value_col="score",
        aggregation="mean",
        top_n=10,
        sort_order="natural",
    )
    year_summary, year_label = build_grouped_metric_summary(
        df,
        group_col="year",
        value_col="score",
        aggregation="mean",
        top_n=10,
        sort_order="natural",
    )
    student_summary, student_label = build_grouped_metric_summary(
        df,
        group_col="student_id",
        aggregation="count",
        top_n=10,
        sort_order="natural",
    )

    assert subject_label == "Mean of score"
    assert year_label == "Mean of score"
    assert student_label == "Row count"
    assert subject_summary["subject"].tolist() == ["English", "Math"]
    assert year_summary["year"].tolist() == [2005, 2006]
    assert student_summary["student_id"].astype(str).tolist() == ["1", "2", "3"]


def test_pipeline_preferences_support_generic_column_roles():
    preferences = PipelinePreferences(
        duplicate_rule_mode="exact",
        column_roles=ColumnRoles(
            id_columns=("student_id",),
            time_column="year",
            metric_column="score",
            segment_column="subject",
            outcome_column="grade",
        ),
    )

    assert preferences.preferred_time_column() == "year"
    assert preferences.preferred_metric_column() == "score"
    assert preferences.preferred_segment_column() == "subject"
    assert "Primary metric: score" in preferences.column_roles.assumption_items()

    coerced = coerce_pipeline_preferences(preferences.to_dict())
    assert coerced.column_roles.id_columns == ("student_id",)
    assert coerced.preferred_time_column() == "year"


def test_numeric_time_axis_and_segment_roles_drive_analysis():
    df = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 2, 3, 3],
            "subject": ["Math", "English", "Math", "English", "Math", "English"],
            "year": [2005, 2005, 2006, 2006, 2007, 2007],
            "score": [41, 55, 98, 63, 71, 53],
            "grade": ["D", "C", "A+", "B", "B", "C"],
        }
    )

    report = build_analysis_report(
        df,
        preferred_value_column="score",
        preferred_date_column="year",
        preferred_category_column="subject",
    )
    chart_recommendations = recommend_charts(
        df,
        preferred_value_column="score",
        preferred_date_column="year",
        preferred_category_column="subject",
    )

    assert report["trend_signal"]["date_column"] == "year"
    assert report["trend_signal"]["value_column"] == "score"
    assert report["segment_signal"]["category_column"] == "subject"
    assert chart_recommendations[0]["columns"] == ["subject"]
    assert any(item["key"] == "trend_analysis" and item["columns"] == ["year", "score"] for item in chart_recommendations)
