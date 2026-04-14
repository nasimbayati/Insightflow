import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.analysis import build_analysis_report
from modules.insights import generate_ai_insights, recommend_charts
from modules.llm_insights import build_llm_context, get_llm_cache_key
from modules.validation import check_missing_values, detect_column_types, suggest_duplicate_subset


def test_build_llm_context_contains_expected_sections():
    raw_df = pd.DataFrame(
        {
            "region": ["North", "South", "West"],
            "sales": [100, 120, 90],
            "profit": [20, 25, 15],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )

    column_types = detect_column_types(raw_df)
    analysis_report = build_analysis_report(raw_df)
    validation_report = {
        "missing": check_missing_values(raw_df),
        "duplicate_count": 0,
        "duplicate_subset": suggest_duplicate_subset(raw_df, column_types),
        "invalid_numeric": {},
        "invalid_dates": {},
        "invalid_numeric_count": 0,
        "invalid_date_count": 0,
    }

    ai_report = generate_ai_insights(
        raw_df=raw_df,
        active_df=raw_df,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=recommend_charts(raw_df),
        view_label="cleaned dataset",
    )
    validation_report["quality_score"] = ai_report["quality_score"]
    validation_report["quality_label"] = ai_report["quality_label"]

    context = build_llm_context(
        raw_df=raw_df,
        active_df=raw_df,
        raw_column_types=column_types,
        validation_report=validation_report,
        analysis_report=analysis_report,
        chart_recommendations=recommend_charts(raw_df),
        view_label="cleaned dataset",
    )

    assert context["view_label"] == "cleaned dataset"
    assert context["raw_dataset_shape"]["rows"] == 3
    assert len(context["columns"]) == 4
    assert "analysis" in context
    assert "recommended_charts" in context
    assert len(context["preview_rows"]) == 3


def test_llm_cache_key_changes_with_request():
    context = {"a": 1, "b": [1, 2, 3]}
    key_one = get_llm_cache_key("custom", "gpt-5-mini", "cleaned dataset", context, request="show trend")
    key_two = get_llm_cache_key("custom", "gpt-5-mini", "cleaned dataset", context, request="show outliers")

    assert key_one != key_two
