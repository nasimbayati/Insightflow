import json
import io
import zipfile
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.reporting import build_decision_brief_markdown
from modules.views.shared import (
    render_badge_row,
    render_block_header,
    render_chart_recommendation_item,
    render_compact_dataframe,
    render_insight_card,
    render_micro_card,
    render_score_breakdown,
)


def render_decision_mode(
    file_name,
    ai_report,
    analysis_report,
    chart_recommendations,
    view_label,
    active_df,
    cleaned_export_df,
    cleaning_impact_items,
    applied_filters=None,
    run_report_payload=None,
):
    rows, cols = analysis_report["shape"]
    leading_chart = chart_recommendations[0] if chart_recommendations else None
    filter_note = "Manual filters are applied." if applied_filters else "No manual filters are applied yet."
    summary_items = ai_report.get("executive_summary") or ["No executive summary is available yet."]
    risk_items = ai_report.get("risk_flags") or ["No major risk has been detected in the current view."]
    driver_items = ai_report.get("key_drivers") or ["No clear driver is available in the current view."]
    next_steps = ai_report.get("next_steps") or ["Review the strongest validation issue before acting on the data."]
    confidence_reasons = ai_report.get("insight_confidence_reasons") or [
        "Confidence notes are not available for the current view."
    ]
    next_questions = ai_report.get("next_questions") or ["No follow-up questions are available yet."]
    decision_brief_markdown = build_decision_brief_markdown(
        file_name,
        view_label,
        ai_report,
        chart_recommendations,
        cleaning_impact_items,
        applied_filters=applied_filters,
    )
    boardroom_memo = (
        f"{summary_items[0]} Main risk: {risk_items[0]} "
        f"Key driver: {driver_items[0]} "
        f"Recommended move: {next_steps[0]}"
    )

    with st.container(border=True):
        render_block_header(
            "Boardroom Brief",
            f"If you only read one thing first, use this block. It turns the {view_label.lower()} into a short {ai_report.get('audience_label', 'executive brief').lower()} backed by evidence. {filter_note}",
        )
        st.markdown(f'<div class="if-brief-memo">{escape(boardroom_memo)}</div>', unsafe_allow_html=True)
        render_badge_row(ai_report.get("assumption_badges", []))
        metric_row = st.columns(4, gap="large")
        metric_row[0].metric("Readiness Score", f"{ai_report['quality_score']}/100", ai_report["quality_label"])
        metric_row[1].metric("Narrative Confidence", ai_report["insight_confidence_label"])
        metric_row[2].metric("Rows x Columns", f"{rows} x {cols}")
        metric_row[3].metric("Recommended Charts", len(chart_recommendations))
        st.caption(
            "Readiness score and narrative confidence are heuristic signals based on missing values, duplicates, parsing issues, row count, and outlier pressure. They are guidance, not statistical guarantees."
        )
        with st.expander("Why these scores?", expanded=False):
            breakdown_left, breakdown_right = st.columns(2, gap="large")
            with breakdown_left:
                render_score_breakdown(
                    "Readiness score breakdown",
                    ai_report.get("quality_score_breakdown", []),
                    caption="How file-quality penalties influenced the readiness score.",
                )
            with breakdown_right:
                render_score_breakdown(
                    "Narrative confidence breakdown",
                    ai_report.get("insight_confidence_breakdown", []),
                    caption="How evidence quality influenced the confidence label.",
                )

        summary_row = st.columns(4, gap="large")
        with summary_row[0]:
            render_micro_card(
                "Executive Summary",
                summary_items[0],
                summary_items[1] if len(summary_items) > 1 else None,
                tone="sky",
            )
        with summary_row[1]:
            render_micro_card(
                "Top Risk",
                risk_items[0],
                "The main reason to stay cautious before acting on this file.",
                tone="blush",
            )
        with summary_row[2]:
            render_micro_card(
                "Key Driver",
                driver_items[0],
                "The strongest current signal worth exploring further.",
                tone="apricot",
            )
        with summary_row[3]:
            start_here_value = next_steps[0]
            if leading_chart:
                start_here_value = f"{start_here_value} Start with {leading_chart['title'].lower()}."
            render_micro_card(
                "Start Here",
                start_here_value,
                "The fastest next move to turn this file into a usable decision.",
                tone="mint",
            )

        evidence_left, evidence_right = st.columns([1.15, 0.85], gap="large")

        with evidence_left:
            if chart_recommendations:
                evidence_recommendations = chart_recommendations[:2]
                if len(evidence_recommendations) == 1:
                    render_chart_recommendation_item(
                        active_df,
                        analysis_report,
                        evidence_recommendations[0],
                        export_prefix=f"{Path(file_name).stem}_decision_1",
                    )
                else:
                    evidence_tabs = st.tabs(["Primary Evidence", "Secondary Evidence"])
                    for index, (tab, recommendation) in enumerate(zip(evidence_tabs, evidence_recommendations), start=1):
                        with tab:
                            render_chart_recommendation_item(
                                active_df,
                                analysis_report,
                                recommendation,
                                export_prefix=f"{Path(file_name).stem}_decision_{index}",
                            )
            else:
                st.info("No strong evidence charts are available for the current dataset view.")

        with evidence_right:
            render_insight_card(
                "Narrative Confidence",
                confidence_reasons,
                caption="This tells you how much trust to place in the narrative, given the active evidence base.",
                tone="slate",
                note=f"Current confidence: {ai_report['insight_confidence_label']}",
            )
            render_insight_card(
                "Cleaning Impact",
                cleaning_impact_items,
                caption="A measurable before-and-after view of what the cleaning layer improved.",
                tone="mint",
            )
            render_insight_card(
                "Signal Guardrails",
                ai_report.get("signal_guardrails", []),
                caption="Why the strongest segment, trend, and correlation signals should be treated as supported, directional, or insufficient.",
                tone="blush",
            )
            render_insight_card(
                "Next Questions to Explore",
                next_questions,
                caption="Use these follow-up questions when you want to move from summary to investigation.",
                tone="sand",
            )

        package_buffer = io.BytesIO()
        with zipfile.ZipFile(package_buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr(
                f"insightflow_cleaned_{Path(file_name).stem}.csv",
                cleaned_export_df.to_csv(index=False),
            )
            package.writestr(
                f"insightflow_decision_brief_{Path(file_name).stem}.md",
                decision_brief_markdown,
            )
            if run_report_payload is not None:
                package.writestr(
                    f"insightflow_run_report_{Path(file_name).stem}.json",
                    json.dumps(run_report_payload, indent=2, default=str),
                )
        package_buffer.seek(0)

        export_left, export_mid, export_right, export_package = st.columns(4, gap="large")
        with export_left:
            st.download_button(
                "Download cleaned CSV",
                data=cleaned_export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"insightflow_cleaned_{Path(file_name).stem}.csv",
                mime="text/csv",
                width="stretch",
            )
        with export_mid:
            st.download_button(
                "Download decision brief",
                data=decision_brief_markdown.encode("utf-8"),
                file_name=f"insightflow_decision_brief_{Path(file_name).stem}.md",
                mime="text/markdown",
                width="stretch",
            )
        with export_right:
            if run_report_payload is not None:
                st.download_button(
                    "Download run report JSON",
                    data=json.dumps(run_report_payload, indent=2, default=str).encode("utf-8"),
                    file_name=f"insightflow_run_report_{Path(file_name).stem}.json",
                    mime="application/json",
                    width="stretch",
                )
        with export_package:
            st.download_button(
                "Download analysis package",
                data=package_buffer.getvalue(),
                file_name=f"insightflow_analysis_package_{Path(file_name).stem}.zip",
                mime="application/zip",
                width="stretch",
            )


def render_artifact_registry_panel(latest_artifact, recent_entries):
    with st.expander("Run artifacts and reproducibility", expanded=False):
        if latest_artifact:
            st.caption(f"Latest saved run: {latest_artifact['run_id']}")
            st.write(f"- Report JSON: `{latest_artifact['report_path']}`")
            st.write(f"- Decision brief: `{latest_artifact['brief_path']}`")
            st.write(f"- Cleaned dataset: `{latest_artifact['cleaned_csv_path']}`")
            st.write(f"- Active view: `{latest_artifact['active_csv_path']}`")
        if recent_entries:
            recent_df = pd.DataFrame(recent_entries)
            columns = [column for column in ["created_at", "run_id", "rows_active", "rows_cleaned"] if column in recent_df.columns]
            render_compact_dataframe(
                recent_df[columns],
                height=220,
                title="Recent run registry",
                caption="Persisted run metadata for reproducible reviews and export traceability.",
            )
