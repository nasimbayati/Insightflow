import pandas as pd
import streamlit as st

from modules.monitoring import build_monitoring_snapshot
from modules.reporting import format_column_types, format_issue_dict
from modules.views.layout import render_section_header
from modules.views.shared import (
    render_block_header,
    render_bullet_list,
    render_centered_copy,
    render_compact_dataframe,
)


def render_ingestion_section(project_root, raw_df, ingestion_metadata, file_stem, ingestion_notes):
    st.divider()
    render_section_header(
        "File Ingestion",
        "Review file-level checks, encoding detection, repaired rows, and malformed rows skipped during parsing.",
        step=2,
    )

    ingestion_metrics = st.columns(4)
    ingestion_metrics[0].metric("File Size", f"{ingestion_metadata['file_size_mb']:.2f} MB")
    ingestion_metrics[1].metric("Encoding", ingestion_metadata["encoding"])
    ingestion_metrics[2].metric("Rows Loaded", raw_df.shape[0])
    ingestion_metrics[3].metric("Rows Skipped", ingestion_metadata["skipped_row_count"])
    st.caption(f"File: {ingestion_metadata['filename']} | Upload limit: {ingestion_metadata['max_size_mb']} MB")

    rejected_rows_df = pd.DataFrame(ingestion_metadata["skipped_rows"])
    if ingestion_notes:
        render_bullet_list(ingestion_notes)
    else:
        st.success("The file passed ingestion checks without encoding, size, or malformed-row issues.")

    if ingestion_metadata["skipped_rows"]:
        rejected_preview_df = pd.DataFrame(ingestion_metadata["skipped_rows_preview"])
        render_compact_dataframe(
            rejected_preview_df,
            height=220,
            title="Rejected row preview",
            caption="Rows skipped during ingestion because they exceeded the detected header width.",
        )
        st.download_button(
            "Download rejected rows report",
            data=rejected_rows_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_stem}_rejected_rows.csv",
            mime="text/csv",
            width="stretch",
        )

    monitoring_snapshot = build_monitoring_snapshot(project_root, limit=8)
    with st.expander("Operational monitoring", expanded=False):
        latest_event = monitoring_snapshot["latest_event"]
        event_counts = monitoring_snapshot["event_counts"]
        metrics = st.columns(3)
        metrics[0].metric("Recent events", sum(event_counts.values()) if event_counts else 0)
        metrics[1].metric("Warnings", event_counts.get("WARNING", 0))
        metrics[2].metric("Errors", event_counts.get("ERROR", 0))
        if latest_event:
            st.caption(f"Latest event: {latest_event['event_type']} at {latest_event['timestamp']}")
        recent_events_df = pd.DataFrame(monitoring_snapshot["recent_events"])
        if not recent_events_df.empty:
            render_compact_dataframe(
                recent_events_df[["timestamp", "event_type", "status"]],
                height=220,
                title="Recent pipeline events",
                caption="Structured operational events captured for debugging and auditability.",
            )

    st.caption(
        "Operational monitoring: ingestion outcomes, repaired rows, skipped rows, and fatal processing errors are also logged server-side for debugging."
    )
    return rejected_rows_df


def render_full_audit_section(
    raw_df,
    missing,
    invalid_numeric,
    invalid_dates,
    duplicate_subset,
    duplicate_count,
    duplicate_rows,
    duplicate_diagnostics,
    raw_column_types,
    base_df_to_use,
    chart_view_mode,
    transformation_log,
):
    st.divider()
    render_section_header(
        "Data Validation",
        "Check missing values, duplicates, detected types, and invalid numeric or date entries before deeper analysis.",
        step=5,
    )

    validation_metrics = st.columns(4)
    validation_metrics[0].metric("Rows", raw_df.shape[0])
    validation_metrics[1].metric("Columns", raw_df.shape[1])
    validation_metrics[2].metric("Missing Cells", int(missing.sum()) if not missing.empty else 0)
    validation_metrics[3].metric("Duplicate Rows", duplicate_count)

    validation_tabs = st.tabs(["Missing", "Invalid Values", "Duplicates", "Column Types"])

    with validation_tabs[0]:
        if not missing.empty:
            render_compact_dataframe(
                missing.rename("Missing Values"),
                title="Missing values overview",
                caption="Columns with blank or null values in the uploaded file.",
                index_label="Column",
            )
        else:
            st.success("No missing values were found.")

    with validation_tabs[1]:
        invalid_left, invalid_right = st.columns([1, 1], gap="large")

        with invalid_left:
            invalid_numeric_df = format_issue_dict(invalid_numeric, "Invalid Values")
            if invalid_numeric_df.empty:
                st.success("No invalid numeric values were found.")
            else:
                render_compact_dataframe(
                    invalid_numeric_df,
                    height=240,
                    title="Invalid numeric values",
                    caption="Values that looked numeric but could not be parsed cleanly.",
                )

        with invalid_right:
            invalid_dates_df = format_issue_dict(invalid_dates, "Invalid Values")
            if invalid_dates_df.empty:
                st.success("No invalid date values were found.")
            else:
                render_compact_dataframe(
                    invalid_dates_df,
                    height=240,
                    title="Invalid date values",
                    caption="Values that looked like dates but failed date parsing.",
                )

    with validation_tabs[2]:
        duplicate_basis = ", ".join(duplicate_subset) if duplicate_subset else "exact row matching"
        st.caption(
            f"Diagnostic status: {duplicate_diagnostics['status']}. Comparable rows: {duplicate_diagnostics['complete_rows']} / {duplicate_diagnostics['total_rows']}. {duplicate_diagnostics['detail']}"
        )
        if duplicate_count > 0:
            render_compact_dataframe(
                duplicate_rows,
                height=260,
                title="Duplicate preview",
                caption=f"Rows flagged as duplicates using {duplicate_basis}.",
            )
        else:
            render_centered_copy(
                title="Duplicate review",
                caption=f"Current strategy: {duplicate_basis}.",
                body="No duplicate rows were found with the current strategy.",
            )

    with validation_tabs[3]:
        render_compact_dataframe(
            format_column_types(raw_column_types),
            height=260,
            title="Detected column types",
            caption="Automatic type inference used by the cleaning and analysis pipeline.",
        )

    st.divider()
    render_section_header(
        "Data Cleaning",
        "Review what the app changed and inspect the active dataset view that feeds the downstream analysis.",
        step=6,
    )
    cleaning_left, cleaning_right = st.columns([1.15, 0.85], gap="large")

    with cleaning_left:
        render_compact_dataframe(
            base_df_to_use.head(10),
            height=300,
            title=f"Preview of the selected view ({chart_view_mode})",
            caption="This is the selected dataset view before any analysis filters are applied.",
        )

    with cleaning_right:
        with st.expander("Show cleaning actions", expanded=True):
            render_block_header(
                "Transformation log",
                "Every cleaning action applied before the selected dataset view was produced.",
            )
            render_bullet_list(transformation_log)


def render_filtering_section(base_df_to_use, chart_view_mode, raw_df, filter_controls_fn):
    st.divider()
    render_section_header(
        "Filtering & Slicing",
        "Slice the active dataset view by category, numeric range, or date before running analysis and chart generation.",
        step=7,
    )
    analysis_df, applied_filters = filter_controls_fn(
        base_df_to_use,
        key_prefix=chart_view_mode.lower().replace(" ", "_"),
    )

    if applied_filters:
        render_bullet_list([f"Active filter: {item}" for item in applied_filters])
    else:
        st.info("No filters are currently applied. Analysis is using the full selected view.")

    if analysis_df.empty:
        if raw_df.empty:
            st.warning(
                "The uploaded file contains headers only. Validation and cleaning can still run, but analysis needs at least one data row."
            )
        elif applied_filters:
            st.warning("The current filter combination returns 0 rows. Clear or widen a filter to resume analysis.")
        else:
            st.warning(
                "The selected view currently contains 0 rows. Try switching to Raw Data or keeping duplicates to restore analysis."
            )

    return analysis_df, applied_filters
