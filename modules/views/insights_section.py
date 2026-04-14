import streamlit as st

from modules.analysis import get_categorical_columns
from modules.artifacts import load_recent_registry_entries
from modules.insights import interpret_custom_request
from modules.llm_insights import (
    format_llm_error,
    generate_llm_custom_response,
    generate_llm_overview,
    get_llm_cache_key,
    is_llm_configured,
)
from modules.views.decision import render_artifact_registry_panel
from modules.views.layout import render_section_header
from modules.views.shared import (
    render_badge_row,
    render_block_header,
    render_chart_recommendations,
    render_custom_chart_builder,
    render_guided_analysis,
    render_insight_card,
    render_question_shortcuts,
    render_recommendation_summary,
    render_score_breakdown,
)


def render_insights_section(
    project_root,
    latest_run_artifact,
    ai_report,
    chart_recommendations,
    narrative_mode,
    llm_api_key,
    llm_model,
    llm_context,
    view_label,
):
    st.divider()
    render_section_header(
        "AI Insights",
        "Read the automatic narrative summary, data quality evaluation, and recommended next actions generated from the analysis results.",
        step=9,
    )

    insight_metrics = st.columns(4)
    insight_metrics[0].metric("Readiness Score", f"{ai_report['quality_score']}/100", ai_report["quality_label"])
    insight_metrics[1].metric("Narrative Confidence", ai_report["insight_confidence_label"])
    insight_metrics[2].metric("Recommended Charts", len(chart_recommendations))
    insight_metrics[3].metric("Narrative Mode", narrative_mode["label"])
    st.caption(f"Insight audience: {ai_report['audience_label']}. {narrative_mode['caption']}")
    st.caption(
        "These two scores are heuristic guidance layers. They summarize structural data quality and interpretation risk, but they do not validate business truth or statistical significance."
    )
    render_badge_row(ai_report.get("assumption_badges", []))
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

    insights_row_one = st.columns(2, gap="large")
    with insights_row_one[0]:
        render_insight_card(
            "Executive Summary",
            ai_report["executive_summary"],
            caption="High-level narrative for the current active dataset view.",
            tone="sky",
        )
    with insights_row_one[1]:
        render_insight_card(
            "Dataset Profile",
            ai_report["dataset_profile"],
            caption="How the active view is shaped and what kinds of fields it contains.",
            tone="mint",
        )

    insights_row_two = st.columns(2, gap="large")
    with insights_row_two[0]:
        render_insight_card(
            "Data Quality Highlights",
            ai_report["quality_highlights"],
            caption="The main quality signals that could affect trust in the uploaded file.",
            tone="blush",
        )
    with insights_row_two[1]:
        render_insight_card(
            "Key Drivers",
            ai_report["key_drivers"],
            caption="Columns or relationships that appear most useful for interpretation.",
            tone="apricot",
        )

    insights_row_three = st.columns(2, gap="large")
    with insights_row_three[0]:
        render_insight_card(
            "Analysis Highlights",
            ai_report["analysis_highlights"],
            caption="Signals extracted from summary statistics, distributions, and relationships.",
            tone="slate",
        )
    with insights_row_three[1]:
        render_insight_card(
            "Risk Flags",
            ai_report["risk_flags"],
            caption="Reasons to stay cautious before making business or operational decisions.",
            tone="sand",
        )

    render_insight_card(
        "Recommended Next Steps",
        ai_report["next_steps"],
        caption="The most defensible follow-up actions for this dataset.",
        tone="mint",
    )

    render_insight_card(
        "Next Questions to Explore",
        ai_report["next_questions"],
        caption="Suggested follow-up questions that keep the analysis moving toward a decision.",
        tone="sand",
    )

    render_insight_card(
        "Signal Guardrails",
        ai_report.get("signal_guardrails", []),
        caption="Why the strongest segment, trend, and correlation reads are treated as supported, directional, or insufficient.",
        tone="blush",
    )

    render_recommendation_summary(
        chart_recommendations,
        title="Recommended Charts and Why",
        caption="Each chart is chosen from the current data types, then explained using the actual values in this view.",
    )

    render_artifact_registry_panel(
        latest_run_artifact,
        load_recent_registry_entries(project_root, limit=5),
    )

    with st.container(border=True):
        render_block_header(
            "Model-Generated Narrative",
            "Optional LLM layer for a richer executive summary. The built-in cards above remain the default competition-safe path.",
        )
        if is_llm_configured(llm_api_key):
            overview_cache_key = get_llm_cache_key("overview", llm_model, view_label, llm_context)
            if st.button("Generate LLM narrative", key=f"generate_overview_{overview_cache_key}"):
                with st.spinner("Generating LLM narrative..."):
                    try:
                        overview_text = generate_llm_overview(llm_api_key, llm_model, llm_context)
                        st.session_state["llm_cache"][overview_cache_key] = {
                            "text": overview_text,
                            "error": None,
                        }
                    except Exception as exc:
                        st.session_state["llm_cache"][overview_cache_key] = {
                            "text": None,
                            "error": format_llm_error(exc),
                        }

            cached_overview = st.session_state["llm_cache"].get(overview_cache_key)
            st.caption(
                "The model receives a structured summary of the dataset, validation results, analysis metrics, chart recommendations, active filters, and a small preview sample."
            )
            if cached_overview and cached_overview.get("text"):
                st.markdown(cached_overview["text"])
            elif cached_overview and cached_overview.get("error"):
                st.error(cached_overview["error"])
            else:
                st.info("Generate a richer narrative here when you want an LLM-written executive summary.")
        else:
            st.info(
                "LLM mode is not configured. The built-in rule-based summary cards above remain fully usable without any API access."
            )


def render_guided_exploration_section(
    file_stem,
    analysis_df,
    validation_report,
    analysis_report,
    chart_recommendations,
    ai_report,
    llm_api_key,
    llm_model,
    llm_context,
    view_label,
    suggested_analyses,
    pipeline_preferences,
):
    st.divider()
    render_section_header(
        "Guided Exploration",
        "Choose a suggested analysis, ask for a custom view, or review the automatically recommended charts.",
        step=10,
    )

    explore_tabs = st.tabs(["Suggested Analysis", "Custom Request", "Automatic Charts", "Chart Builder"])

    with explore_tabs[0]:
        selected_label = st.selectbox(
            "Choose a guided analysis:",
            [item["label"] for item in suggested_analyses],
        )
        selected_analysis = next(item for item in suggested_analyses if item["label"] == selected_label)
        st.caption(selected_analysis["description"])
        render_guided_analysis(
            selected_analysis["key"],
            analysis_df,
            validation_report,
            analysis_report,
            chart_recommendations,
            get_categorical_columns,
            export_prefix=f"{file_stem}_guided",
            analysis_preferences=pipeline_preferences,
        )

    with explore_tabs[1]:
        render_question_shortcuts(ai_report["next_questions"][:4], "smart_question")
        custom_request = st.text_input(
            "Type what you want to analyze",
            placeholder="Examples: show correlations, analyze missing values, compare category, show price trend",
            key="llm_custom_input",
        )

        if custom_request.strip():
            request_left, request_right = st.columns([0.95, 1.05], gap="large")
            interpreted_request = interpret_custom_request(custom_request, analysis_df, suggested_analyses)
            focus_note = ", ".join(interpreted_request["focus_columns"]) if interpreted_request["focus_columns"] else "no specific column"

            with request_left:
                st.caption(
                    f"Rule-based interpretation: **{interpreted_request['primary_key'].replace('_', ' ')}** "
                    f"with focus on **{focus_note}**."
                )
                render_guided_analysis(
                    interpreted_request["primary_key"],
                    analysis_df,
                    validation_report,
                    analysis_report,
                    chart_recommendations,
                    get_categorical_columns,
                    focus_columns=interpreted_request["focus_columns"],
                    export_prefix=f"{file_stem}_custom",
                    analysis_preferences=pipeline_preferences,
                )

            with request_right:
                st.write("### LLM response")
                if is_llm_configured(llm_api_key):
                    custom_cache_key = get_llm_cache_key(
                        "custom",
                        llm_model,
                        view_label,
                        llm_context,
                        request=custom_request.strip(),
                    )

                    if st.button("Run LLM analysis", key=f"run_custom_{custom_cache_key}"):
                        with st.spinner("Generating model response..."):
                            try:
                                custom_text = generate_llm_custom_response(
                                    llm_api_key,
                                    llm_model,
                                    llm_context,
                                    custom_request.strip(),
                                )
                                st.session_state["llm_cache"][custom_cache_key] = {
                                    "text": custom_text,
                                    "error": None,
                                }
                            except Exception as exc:
                                st.session_state["llm_cache"][custom_cache_key] = {
                                    "text": None,
                                    "error": format_llm_error(exc),
                                }

                    cached_custom = st.session_state["llm_cache"].get(custom_cache_key)
                    st.caption("The model response is grounded in the structured summary shown in this app.")
                    if cached_custom and cached_custom.get("text"):
                        st.markdown(cached_custom["text"])
                    elif cached_custom and cached_custom.get("error"):
                        st.error(cached_custom["error"])
                    else:
                        st.info("Run the model when you want a narrative response to this specific question.")
                else:
                    st.info(
                        "No LLM key is configured. Judges and reviewers can still use the rule-based analysis on the left without any API access."
                    )
        else:
            st.info("Type a request to trigger a targeted analysis.")

    with explore_tabs[2]:
        render_chart_recommendations(
            analysis_df,
            analysis_report,
            chart_recommendations,
            export_prefix=f"{file_stem}_automatic",
        )

    with explore_tabs[3]:
        st.caption(
            "Choose the exact columns that should drive the chart. This is the non-hardcoded path for any CSV, whether the fields represent people, products, regions, years, categories, scores, revenue, tickets, or something else."
        )
        render_custom_chart_builder(
            analysis_df,
            analysis_report,
            export_prefix=f"{file_stem}_builder",
        )
