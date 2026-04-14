from io import BytesIO
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.analysis import apply_dataset_filters, build_analysis_report
from modules.cleaning import CleaningConfig
from modules.pipeline_service import (
    build_analysis_fingerprint,
    build_analysis_run_context,
    build_base_run_context,
    build_boardroom_fingerprint,
    create_upload_run_context,
)
from modules.views import analysis_section, insights_section, workflow_sections


class UploadedBytesIO(BytesIO):
    def __init__(self, payload, name):
        super().__init__(payload)
        self.name = name
        self.size = len(payload)


class FakeBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None


class FakeStreamlit(FakeBlock):
    def __init__(self):
        self.session_state = {}

    def divider(self):
        return None

    def columns(self, spec, gap=None):
        count = spec if isinstance(spec, int) else len(spec)
        return [FakeBlock() for _ in range(count)]

    def tabs(self, labels):
        return [FakeBlock() for _ in labels]

    def expander(self, *args, **kwargs):
        return FakeBlock()

    def container(self, *args, **kwargs):
        return FakeBlock()

    def download_button(self, *args, **kwargs):
        return None

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index] if options else None

    def text_input(self, *args, **kwargs):
        return ""

    def button(self, *args, **kwargs):
        return False

    def spinner(self, *args, **kwargs):
        return FakeBlock()

    def rerun(self):
        return None


def test_pipeline_service_builds_run_contexts():
    payload = (
        b"region,sales,profit,event_date\n"
        b"North,100,20,2024-01-01\n"
        b"North,100,20,2024-01-01\n"
        b"South,80,12,2024-01-02\n"
        b"West,150,40,2024-01-03\n"
    )
    uploaded_file = UploadedBytesIO(payload, "service.csv")

    upload_context = create_upload_run_context(uploaded_file, max_size_mb=5, source_label="uploaded CSV")
    base_context = build_base_run_context(
        upload_context,
        chart_view_mode="Cleaned Data",
        cleaning_config=CleaningConfig(),
        audience_mode="executive",
        pipeline_preferences={
            "duplicate_subset": upload_context.suggested_duplicate_subset,
            "duplicate_rule_mode": "suggested" if upload_context.suggested_duplicate_subset else "exact",
            "trend_date_column": "event_date",
            "trend_value_column": "sales",
        },
    )

    filtered_df, applied_filters = apply_dataset_filters(
        base_context.base_df_to_use,
        categorical_filters={"region": ["North", "West"]},
    )
    analysis_context = build_analysis_run_context(base_context, filtered_df, applied_filters)

    assert upload_context.file_stem == "service"
    assert base_context.validation_report["quality_score"] == base_context.ai_report["quality_score"]
    assert base_context.cleaning_impact_items
    assert analysis_context.view_label.startswith("filtered")
    assert analysis_context.suggested_analyses
    assert analysis_context.llm_context["view_label"] == analysis_context.view_label
    assert build_boardroom_fingerprint(base_context, "Boardroom Brief")["view"] == base_context.base_view_label
    assert build_analysis_fingerprint(analysis_context, "Evidence")["filters"] == applied_filters


def test_workflow_sections_smoke(monkeypatch, tmp_path):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(workflow_sections, "st", fake_st)
    monkeypatch.setattr(workflow_sections, "render_section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_sections, "render_bullet_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_sections, "render_compact_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_sections, "render_centered_copy", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_sections, "render_block_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        workflow_sections,
        "build_monitoring_snapshot",
        lambda project_root, limit=8: {"latest_event": None, "event_counts": {}, "recent_events": []},
    )

    raw_df = pd.DataFrame({"region": ["North"], "sales": [100]})
    ingestion_metadata = {
        "file_size_mb": 0.01,
        "encoding": "utf-8",
        "filename": "demo.csv",
        "max_size_mb": 10,
        "skipped_row_count": 0,
        "skipped_rows": [],
        "skipped_rows_preview": [],
    }
    workflow_sections.render_ingestion_section(tmp_path, raw_df, ingestion_metadata, "demo", [])

    workflow_sections.render_full_audit_section(
        raw_df=raw_df,
        missing=pd.Series(dtype="int64"),
        invalid_numeric={},
        invalid_dates={},
        duplicate_subset=None,
        duplicate_count=0,
        duplicate_rows=pd.DataFrame(),
        duplicate_diagnostics={
            "status": "Conservative",
            "complete_rows": 1,
            "total_rows": 1,
            "detail": "Exact match.",
        },
        raw_column_types={"region": "Categorical", "sales": "Numeric"},
        base_df_to_use=raw_df,
        chart_view_mode="Cleaned Data",
        transformation_log=["No changes"],
    )

    filtered_df, applied_filters = workflow_sections.render_filtering_section(
        raw_df,
        "Cleaned Data",
        raw_df,
        lambda df, key_prefix: (df.copy(), []),
    )
    assert filtered_df.equals(raw_df)
    assert applied_filters == []


def test_analysis_and_insights_sections_smoke(monkeypatch):
    fake_st = FakeStreamlit()

    monkeypatch.setattr(analysis_section, "st", fake_st)
    monkeypatch.setattr(analysis_section, "render_section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_section, "render_compact_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_section, "render_categorical_summary_tabs", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis_section, "render_bullet_list", lambda *args, **kwargs: None)

    monkeypatch.setattr(insights_section, "st", fake_st)
    monkeypatch.setattr(insights_section, "render_section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_badge_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_score_breakdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_insight_card", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_recommendation_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_artifact_registry_panel", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_block_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_question_shortcuts", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_guided_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "render_chart_recommendations", lambda *args, **kwargs: None)
    monkeypatch.setattr(insights_section, "load_recent_registry_entries", lambda *args, **kwargs: [])
    monkeypatch.setattr(insights_section, "is_llm_configured", lambda api_key: False)

    df = pd.DataFrame(
        {
            "region": ["North", "South", "West"],
            "sales": [100, 80, 140],
            "profit": [20, 12, 35],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    analysis_report = build_analysis_report(df)
    chart_recommendations = [
        {"key": "category_bar", "title": "Region distribution", "description": "Compare categories."}
    ]
    ai_report = {
        "quality_score": 84,
        "quality_label": "Good",
        "insight_confidence_label": "Medium",
        "audience_label": "Executive brief",
        "assumption_badges": ["Heuristic readiness score"],
        "quality_score_breakdown": [],
        "insight_confidence_breakdown": [],
        "executive_summary": ["Summary"],
        "dataset_profile": ["Profile"],
        "quality_highlights": ["Quality"],
        "key_drivers": ["Drivers"],
        "analysis_highlights": ["Highlights"],
        "risk_flags": ["Risk"],
        "next_steps": ["Act"],
        "next_questions": ["What next?"],
        "signal_guardrails": ["Directional trend"],
    }

    analysis_section.render_analysis_section(df, analysis_report, chart_recommendations)
    insights_section.render_insights_section(
        project_root=PROJECT_ROOT,
        latest_run_artifact=None,
        ai_report=ai_report,
        chart_recommendations=chart_recommendations,
        narrative_mode={"label": "Fallback ready", "caption": "No API required."},
        llm_api_key="",
        llm_model="gpt-5-mini",
        llm_context={"view_label": "cleaned dataset"},
        view_label="cleaned dataset",
    )
    insights_section.render_guided_exploration_section(
        file_stem="demo",
        analysis_df=df,
        validation_report={
            "missing": pd.Series(dtype="int64"),
            "duplicate_count": 0,
            "duplicate_subset": None,
            "duplicate_rows": pd.DataFrame(),
            "invalid_numeric": {},
            "invalid_dates": {},
            "invalid_numeric_count": 0,
            "invalid_date_count": 0,
            "quality_score": 84,
            "quality_label": "Good",
        },
        analysis_report=analysis_report,
        chart_recommendations=chart_recommendations,
        ai_report=ai_report,
        llm_api_key="",
        llm_model="gpt-5-mini",
        llm_context={"view_label": "cleaned dataset"},
        view_label="cleaned dataset",
        suggested_analyses=[{"label": "Summary statistics", "key": "summary_statistics", "description": "Stats"}],
        pipeline_preferences={"trend_date_column": "event_date", "trend_value_column": "sales"},
    )
