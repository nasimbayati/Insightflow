import json
import logging
from pathlib import Path

import streamlit as st

from modules.artifacts import (
    load_recent_registry_entries,
    persist_run_artifacts,
)
from modules.ingestion import DEFAULT_MAX_FILE_SIZE_MB
from modules.monitoring import log_monitoring_event
from modules.pipeline_service import (
    build_analysis_fingerprint,
    build_analysis_run_context,
    build_base_run_context,
    build_boardroom_fingerprint,
    create_upload_run_context,
)
from modules.views.controls import (
    build_ingestion_notes as controls_build_ingestion_notes,
    get_cleaning_config_from_controls as controls_get_cleaning_config_from_controls,
    get_narrative_mode as controls_get_narrative_mode,
    render_filter_controls as controls_render_filter_controls,
    render_llm_settings as controls_render_llm_settings,
)
from modules.views.decision import (
    render_artifact_registry_panel as decision_render_artifact_registry_panel,
    render_decision_mode as decision_render_decision_mode,
)
from modules.views.layout import (
    load_demo_uploaded_file as layout_load_demo_uploaded_file,
    render_demo_dataset_card as layout_render_demo_dataset_card,
    render_empty_state_preview as layout_render_empty_state_preview,
    render_hero as layout_render_hero,
    render_section_header as layout_render_section_header,
)
from modules.views.analysis_section import (
    render_analysis_section as analysis_render_analysis_section,
)
from modules.views.insights_section import (
    render_guided_exploration_section as insights_render_guided_exploration_section,
    render_insights_section as insights_render_insights_section,
)
from modules.views.shared import (
    render_assumptions_bar as shared_render_assumptions_bar,
    render_bullet_list as shared_render_bullet_list,
    render_review_mode_selector as shared_render_review_mode_selector,
)
from modules.views.workflow_sections import (
    render_filtering_section as workflow_render_filtering_section,
    render_full_audit_section as workflow_render_full_audit_section,
    render_ingestion_section as workflow_render_ingestion_section,
)


LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SHOWCASE_DEMO_LABEL = "Revenue Operations Showcase"
DEMO_DATASETS = {
    SHOWCASE_DEMO_LABEL: {
        "file_name": "sample_revenue_ops_showcase.csv",
        "tagline": "Best for judges: one messy business file with obvious quality issues, a weak segment, a large outlier, and clear trend and relationship signals.",
        "highlights": [
            "duplicate records, missing values, and invalid numeric/date entries",
            "a standout enterprise deal that creates an outlier review moment",
            "a low-performing segment with higher returns and weaker margins",
            "enough structure for category, trend, distribution, and relationship charts",
        ],
    },
    "Retail Sales Dirty Sample": {
        "file_name": "sample_retail_sales_dirty.csv",
        "tagline": "A smaller retail sample with classic CSV cleanup issues for quick validation and cleaning demos.",
        "highlights": [
            "missing values and invalid entries",
            "duplicate review on retail-style records",
            "simple category and distribution charts",
        ],
    },
    "Support Tickets Sample": {
        "file_name": "sample_support_tickets.csv",
        "tagline": "Good for showing categorical breakdowns, workload distribution, and issue-triage insights.",
        "highlights": [
            "support-style categories and priorities",
            "clean category charts and operational comparisons",
            "useful guided analysis examples",
        ],
    },
    "Employee Survey Sample": {
        "file_name": "sample_employee_survey.csv",
        "tagline": "Good for HR-style segmentation, engagement analysis, and chart recommendation demos.",
        "highlights": [
            "engagement and training measures",
            "clear group comparisons across work modes or departments",
            "mixed data quality issues for realistic review",
        ],
    },
}


st.set_page_config(page_title="InsightFlow | Decision-Ready CSV Analytics", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --if-bg: #f5f7fb;
        --if-surface: rgba(255, 255, 255, 0.96);
        --if-surface-strong: rgba(255, 255, 255, 0.99);
        --if-surface-soft: rgba(251, 253, 255, 0.94);
        --if-border: rgba(78, 96, 122, 0.14);
        --if-border-strong: rgba(78, 96, 122, 0.22);
        --if-text: #213447;
        --if-muted: #647489;
        --if-sky: #e6eef9;
        --if-blush: #f5e8ee;
        --if-mint: #e8f3ee;
        --if-apricot: #faeddc;
        --if-accent: #6f8ea9;
        --if-accent-soft: #edf3f9;
        --if-shadow: 0 14px 34px rgba(41, 57, 81, 0.06);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(230, 238, 249, 0.65), transparent 28%),
            radial-gradient(circle at top right, rgba(245, 232, 238, 0.52), transparent 22%),
            radial-gradient(circle at bottom right, rgba(232, 243, 238, 0.54), transparent 24%),
            linear-gradient(180deg, rgba(247, 249, 252, 0.98) 0%, rgba(245, 248, 251, 0.97) 100%);
        color: var(--if-text);
    }
    .block-container {
        padding-top: 1.45rem;
        padding-bottom: 2.2rem;
    }
    .if-hero {
        position: relative;
        overflow: hidden;
        padding: 1.7rem 1.8rem 1.8rem;
        margin-bottom: 1.25rem;
        border-radius: 1.3rem;
        border: 1px solid var(--if-border-strong);
        box-shadow: var(--if-shadow);
        background:
            radial-gradient(circle at 12% 20%, rgba(230, 238, 249, 0.95), transparent 30%),
            radial-gradient(circle at 90% 18%, rgba(245, 232, 238, 0.78), transparent 24%),
            radial-gradient(circle at 78% 82%, rgba(232, 243, 238, 0.86), transparent 28%),
            radial-gradient(circle at 34% 88%, rgba(250, 237, 220, 0.72), transparent 24%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 251, 254, 0.98));
    }
    .if-hero-grid {
        position: relative;
        display: grid;
        gap: 0.48rem;
        max-width: 60rem;
        z-index: 1;
    }
    .if-logo-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.1rem;
    }
    .if-logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 3.2rem;
        height: 3.2rem;
        border-radius: 1rem;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #21415a;
        background:
            radial-gradient(circle at top left, rgba(255, 255, 255, 0.9), transparent 50%),
            linear-gradient(160deg, rgba(230, 238, 249, 1), rgba(232, 243, 238, 0.95));
        border: 1px solid rgba(86, 108, 132, 0.14);
        box-shadow: 0 12px 24px rgba(57, 76, 96, 0.08);
    }
    .if-logo-copy {
        display: grid;
        gap: 0.1rem;
    }
    .if-logo-label {
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 700;
        color: #5f758b;
    }
    .if-logo-title {
        font-size: 1.12rem;
        font-weight: 780;
        letter-spacing: -0.01em;
        color: #22384b;
    }
    .if-hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        width: fit-content;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #49677f;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(73, 103, 127, 0.12);
        backdrop-filter: blur(4px);
    }
    .if-brand-name {
        font-size: 2.7rem;
        line-height: 1.02;
        letter-spacing: -0.03em;
        font-weight: 800;
        color: #1f3447;
        margin-top: 0.12rem;
    }
    .if-hero-title {
        font-size: 1.82rem;
        line-height: 1.12;
        letter-spacing: -0.02em;
        font-weight: 760;
        color: #314a62;
        max-width: 50rem;
    }
    .if-hero-subtitle {
        font-size: 0.98rem;
        color: #61758b;
        max-width: 48rem;
    }
    .if-hero-proof {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.85rem;
    }
    .if-hero-proof-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.42rem 0.7rem;
        border-radius: 999px;
        font-size: 0.81rem;
        color: #3f5a72;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(84, 110, 135, 0.12);
        box-shadow: 0 8px 18px rgba(82, 96, 110, 0.04);
    }
    .if-section {
        display: flex;
        gap: 0.95rem;
        align-items: flex-start;
        margin: 0.45rem 0 0.95rem;
        padding: 1rem 1.05rem;
        border-radius: 1.05rem;
        border: 1px solid var(--if-border);
        box-shadow: 0 10px 24px rgba(67, 80, 96, 0.04);
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(249, 251, 253, 0.98));
    }
    .if-section-step-1 {
        background: linear-gradient(180deg, rgba(245, 249, 255, 0.99), rgba(240, 246, 253, 0.97));
    }
    .if-section-step-2 {
        background: linear-gradient(180deg, rgba(245, 247, 255, 0.99), rgba(241, 244, 253, 0.97));
    }
    .if-section-step-3 {
        background: linear-gradient(180deg, rgba(248, 244, 255, 0.99), rgba(244, 241, 252, 0.97));
    }
    .if-section-step-4 {
        background: linear-gradient(180deg, rgba(244, 250, 247, 0.99), rgba(239, 247, 243, 0.97));
    }
    .if-section-step-5 {
        background: linear-gradient(180deg, rgba(255, 248, 242, 0.99), rgba(252, 244, 238, 0.97));
    }
    .if-section-step-6 {
        background: linear-gradient(180deg, rgba(250, 247, 241, 0.99), rgba(247, 243, 236, 0.97));
    }
    .if-section-step-7 {
        background: linear-gradient(180deg, rgba(244, 248, 251, 0.99), rgba(239, 244, 248, 0.97));
    }
    .if-section-step-8 {
        background: linear-gradient(180deg, rgba(245, 247, 252, 0.99), rgba(240, 244, 249, 0.97));
    }
    .if-section-step-9 {
        background: linear-gradient(180deg, rgba(248, 249, 243, 0.99), rgba(244, 246, 238, 0.97));
    }
    .if-section-step-10 {
        background: linear-gradient(180deg, rgba(246, 247, 251, 0.99), rgba(241, 243, 248, 0.97));
    }
    .if-section-index {
        flex: 0 0 auto;
        width: 2.45rem;
        height: 2.45rem;
        border-radius: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.92rem;
        font-weight: 800;
        color: #33516a;
        background: linear-gradient(180deg, rgba(237, 243, 249, 0.96), rgba(230, 238, 249, 0.9));
        border: 1px solid rgba(111, 142, 169, 0.14);
    }
    .if-section-index-step-1 {
        background: linear-gradient(180deg, rgba(230, 238, 249, 0.98), rgba(220, 231, 245, 0.92));
    }
    .if-section-index-step-2 {
        background: linear-gradient(180deg, rgba(233, 236, 250, 0.98), rgba(226, 230, 246, 0.92));
    }
    .if-section-index-step-3 {
        background: linear-gradient(180deg, rgba(239, 233, 249, 0.98), rgba(233, 226, 244, 0.92));
    }
    .if-section-index-step-4 {
        background: linear-gradient(180deg, rgba(232, 243, 238, 0.98), rgba(224, 238, 231, 0.92));
    }
    .if-section-index-step-5 {
        background: linear-gradient(180deg, rgba(250, 237, 220, 0.98), rgba(247, 230, 208, 0.92));
    }
    .if-section-index-step-6 {
        background: linear-gradient(180deg, rgba(247, 239, 223, 0.98), rgba(243, 234, 213, 0.92));
    }
    .if-section-index-step-7 {
        background: linear-gradient(180deg, rgba(232, 239, 245, 0.98), rgba(223, 232, 239, 0.92));
    }
    .if-section-index-step-8 {
        background: linear-gradient(180deg, rgba(234, 238, 247, 0.98), rgba(226, 232, 243, 0.92));
    }
    .if-section-index-step-9 {
        background: linear-gradient(180deg, rgba(239, 242, 226, 0.98), rgba(233, 238, 217, 0.92));
    }
    .if-section-body {
        min-width: 0;
    }
    .if-step-label {
        font-size: 0.72rem;
        line-height: 1;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        font-weight: 700;
        color: var(--if-accent);
        margin-bottom: 0.24rem;
    }
    .if-section-title {
        font-size: 1.18rem;
        font-weight: 760;
        color: #22374b;
        margin-bottom: 0.2rem;
    }
    .if-section-subtitle {
        font-size: 0.91rem;
        color: var(--if-muted);
        line-height: 1.45;
    }
    .if-block-head {
        margin-bottom: 0.5rem;
    }
    .if-block-title {
        font-size: 1rem;
        font-weight: 700;
        color: #24374a;
        margin-bottom: 0.16rem;
    }
    .if-block-caption {
        font-size: 0.84rem;
        color: #6c7c8c;
        line-height: 1.4;
    }
    .if-demo-card {
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        border: 1px solid rgba(82, 97, 114, 0.12);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 251, 254, 0.95));
        box-shadow: 0 10px 24px rgba(67, 80, 96, 0.04);
        margin-bottom: 1rem;
    }
    .if-demo-title {
        font-size: 0.98rem;
        font-weight: 740;
        color: #24374a;
        margin-bottom: 0.22rem;
    }
    .if-demo-copy {
        font-size: 0.84rem;
        color: #61758b;
        line-height: 1.45;
        margin-bottom: 0.72rem;
    }
    .if-demo-list {
        margin: 0;
        padding-left: 1.05rem;
        color: #324960;
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .if-demo-list li {
        margin-bottom: 0.35rem;
    }
    .if-empty-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.8rem;
    }
    .if-empty-card {
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        border: 1px solid rgba(88, 108, 132, 0.12);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(249, 251, 253, 0.96));
        box-shadow: 0 10px 24px rgba(58, 76, 98, 0.05);
    }
    .if-empty-card-sky {
        background: linear-gradient(180deg, rgba(245, 249, 255, 0.98), rgba(239, 246, 253, 0.96));
    }
    .if-empty-card-mint {
        background: linear-gradient(180deg, rgba(246, 252, 248, 0.98), rgba(239, 247, 243, 0.96));
    }
    .if-empty-card-apricot {
        background: linear-gradient(180deg, rgba(255, 249, 243, 0.98), rgba(251, 244, 236, 0.96));
    }
    .if-empty-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        color: #637990;
        margin-bottom: 0.45rem;
    }
    .if-empty-title {
        font-size: 1.02rem;
        font-weight: 760;
        color: #23384c;
        margin-bottom: 0.35rem;
    }
    .if-empty-copy {
        font-size: 0.92rem;
        color: #62758a;
        line-height: 1.48;
    }
    .if-empty-list {
        margin: 0.55rem 0 0;
        padding-left: 1.05rem;
        color: #4f6478;
        font-size: 0.9rem;
    }
    .if-empty-list li {
        margin: 0.24rem 0;
    }
    .if-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.35rem 0 0.2rem;
    }
    .if-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.58rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #3d566d;
        background: rgba(242, 246, 251, 0.95);
        border: 1px solid rgba(93, 114, 136, 0.14);
    }
    .if-brief-memo {
        padding: 0.95rem 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(88, 108, 132, 0.12);
        background: linear-gradient(180deg, rgba(248, 251, 255, 0.98), rgba(243, 247, 252, 0.95));
        color: #30475d;
        line-height: 1.6;
        margin: 0.35rem 0 0.8rem;
    }
    .if-review-switch {
        padding: 0.95rem 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(88, 108, 132, 0.12);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(248, 250, 253, 0.96));
        box-shadow: 0 8px 18px rgba(60, 77, 97, 0.04);
        margin-bottom: 0.85rem;
    }
    .if-review-title {
        font-size: 1rem;
        font-weight: 760;
        color: #21374a;
        margin-bottom: 0.22rem;
    }
    .if-review-copy {
        font-size: 0.9rem;
        color: #62748a;
    }
    @media (max-width: 980px) {
        .if-empty-grid {
            grid-template-columns: 1fr;
        }
    }
    .if-micro-card {
        height: 100%;
        padding: 0.9rem 0.95rem;
        border-radius: 0.95rem;
        border: 1px solid rgba(82, 97, 114, 0.11);
        box-shadow: 0 8px 20px rgba(67, 80, 96, 0.04);
    }
    .if-micro-card-sky {
        background: linear-gradient(180deg, rgba(245, 249, 255, 0.98), rgba(239, 246, 253, 0.95));
    }
    .if-micro-card-mint {
        background: linear-gradient(180deg, rgba(244, 250, 247, 0.98), rgba(238, 247, 242, 0.95));
    }
    .if-micro-card-blush {
        background: linear-gradient(180deg, rgba(250, 245, 248, 0.98), rgba(245, 239, 244, 0.95));
    }
    .if-micro-card-apricot {
        background: linear-gradient(180deg, rgba(255, 248, 241, 0.98), rgba(251, 243, 235, 0.95));
    }
    .if-micro-card-title {
        font-size: 0.8rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #6d8194;
        margin-bottom: 0.42rem;
    }
    .if-micro-card-value {
        font-size: 0.98rem;
        font-weight: 730;
        color: #24374a;
        line-height: 1.38;
        margin-bottom: 0.3rem;
    }
    .if-micro-card-caption {
        font-size: 0.82rem;
        color: #607285;
        line-height: 1.42;
    }
    .if-insight-card {
        height: 100%;
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        border: 1px solid rgba(82, 97, 114, 0.12);
        box-shadow: 0 10px 24px rgba(67, 80, 96, 0.04);
        margin-bottom: 1rem;
    }
    .if-insight-card-sky {
        background: linear-gradient(180deg, rgba(245, 249, 255, 0.98), rgba(239, 246, 253, 0.95));
    }
    .if-insight-card-mint {
        background: linear-gradient(180deg, rgba(244, 250, 247, 0.98), rgba(238, 247, 242, 0.95));
    }
    .if-insight-card-blush {
        background: linear-gradient(180deg, rgba(250, 245, 248, 0.98), rgba(245, 239, 244, 0.95));
    }
    .if-insight-card-apricot {
        background: linear-gradient(180deg, rgba(255, 248, 241, 0.98), rgba(251, 243, 235, 0.95));
    }
    .if-insight-card-sand {
        background: linear-gradient(180deg, rgba(249, 246, 239, 0.98), rgba(245, 241, 233, 0.95));
    }
    .if-insight-card-slate {
        background: linear-gradient(180deg, rgba(245, 247, 251, 0.98), rgba(240, 244, 248, 0.95));
    }
    .if-insight-card-title {
        font-size: 1rem;
        font-weight: 760;
        color: #24374a;
        margin-bottom: 0.2rem;
    }
    .if-insight-card-caption {
        font-size: 0.84rem;
        color: #647489;
        line-height: 1.42;
        margin-bottom: 0.72rem;
    }
    .if-insight-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #31465d;
    }
    .if-insight-list li {
        margin-bottom: 0.48rem;
        line-height: 1.45;
    }
    .if-insight-note {
        margin-top: 0.72rem;
        padding: 0.7rem 0.82rem;
        border-radius: 0.82rem;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(82, 97, 114, 0.08);
        color: #506377;
        font-size: 0.84rem;
        line-height: 1.42;
    }
    .if-recommendation-stack {
        display: grid;
        gap: 0.8rem;
    }
    .if-recommendation-card {
        padding: 0.92rem 0.98rem;
        border-radius: 0.95rem;
        border: 1px solid rgba(82, 97, 114, 0.1);
        background: rgba(255, 255, 255, 0.82);
    }
    .if-recommendation-title {
        font-size: 0.96rem;
        font-weight: 720;
        color: #24374a;
        margin-bottom: 0.2rem;
    }
    .if-recommendation-copy {
        font-size: 0.85rem;
        color: #5d7084;
        line-height: 1.44;
    }
    .if-chart-readout {
        margin-top: 0.8rem;
        padding: 0.75rem 0.82rem;
        border-radius: 0.85rem;
        background: rgba(247, 250, 253, 0.92);
        border: 1px solid rgba(111, 142, 169, 0.12);
        color: #496074;
        font-size: 0.84rem;
        line-height: 1.42;
    }
    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(247, 250, 253, 0.96));
        border: 1px solid var(--if-border);
        box-shadow: 0 8px 18px rgba(82, 96, 110, 0.04);
        border-radius: 0.92rem;
        padding: 0.75rem 0.9rem;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.42rem;
        padding-bottom: 0.2rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        background: rgba(255, 255, 255, 0.84) !important;
        border: 1px solid var(--if-border) !important;
        color: #526579 !important;
        padding: 0.42rem 0.9rem !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(237, 243, 249, 0.96), rgba(232, 243, 238, 0.9)) !important;
        color: #29455d !important;
        border-color: rgba(84, 110, 135, 0.18) !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(247, 250, 253, 0.97));
        border: 1px dashed rgba(88, 112, 135, 0.22);
        border-radius: 1rem;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border-radius: 0.95rem;
        overflow: hidden;
        border: 1px solid rgba(82, 97, 114, 0.09);
        background: rgba(255, 255, 255, 0.98);
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(82, 97, 114, 0.09);
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.96);
    }
    div[data-testid="stAlert"] {
        border-radius: 0.95rem;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]),
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTable"]) {
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_session_state():
    st.session_state.setdefault("llm_cache", {})
    st.session_state.setdefault("llm_custom_input", "")
    st.session_state.setdefault("latest_run_artifact", None)
    st.session_state.setdefault("last_persisted_fingerprint", None)


def persist_current_run_artifact(file_stem, fingerprint, report_payload, decision_brief_markdown, cleaned_df, active_df, rejected_rows_df):
    if st.session_state.get("last_persisted_fingerprint") == fingerprint:
        return st.session_state.get("latest_run_artifact")

    artifact_manifest = persist_run_artifacts(
        PROJECT_ROOT,
        file_stem=file_stem,
        report_payload=report_payload,
        decision_brief_markdown=decision_brief_markdown,
        cleaned_df=cleaned_df,
        active_df=active_df,
        rejected_rows_df=rejected_rows_df,
    )
    st.session_state["latest_run_artifact"] = artifact_manifest
    st.session_state["last_persisted_fingerprint"] = fingerprint
    log_monitoring_event(
        PROJECT_ROOT,
        "run_artifact_persisted",
        payload={
            "run_id": artifact_manifest["run_id"],
            "rows_cleaned": artifact_manifest["rows_cleaned"],
            "rows_active": artifact_manifest["rows_active"],
            "rejected_rows": artifact_manifest["rejected_row_count"],
        },
        run_id=artifact_manifest["run_id"],
    )
    return artifact_manifest

initialize_session_state()
llm_model, llm_api_key = controls_render_llm_settings()

layout_render_hero()
layout_render_section_header(
    "Choose Data Source",
    "Upload your own CSV or open a bundled demo dataset. InsightFlow will validate the structure, clean issues, and surface the fastest path to a decision-ready summary.",
    step=1,
)

input_mode = st.radio(
    "How do you want to start?",
    ["Bundled demo dataset", "Upload CSV"],
    horizontal=True,
)

uploaded_file = None
selected_demo_label = None

if input_mode == "Bundled demo dataset":
    selected_demo_label = st.selectbox(
        "Choose a bundled demo dataset",
        list(DEMO_DATASETS.keys()),
        index=0,
    )
    demo_info = DEMO_DATASETS[selected_demo_label]
    layout_render_demo_dataset_card(demo_info)
    uploaded_file = layout_load_demo_uploaded_file(DATA_DIR, demo_info["file_name"])
    st.success(
        f"Loaded bundled demo dataset: {demo_info['file_name']}. This is the fastest path for reviewers: the Boardroom Brief appears below with no setup required."
    )
else:
    st.caption(
        f"Recommended showcase path: start with {SHOWCASE_DEMO_LABEL} once you want the strongest one-minute demo."
    )
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help=f"CSV only. InsightFlow currently accepts files up to {DEFAULT_MAX_FILE_SIZE_MB} MB.",
    )
    st.caption("Works with messy, real-world CSVs: missing values, duplicates, malformed rows, and invalid numeric/date formats.")

if uploaded_file is None:
    st.info("Upload a CSV file or open a bundled demo dataset to start the validation, cleaning, analysis, and insight pipeline.")
    layout_render_empty_state_preview()
else:
    try:
        source_label = selected_demo_label if selected_demo_label else "uploaded CSV"
        upload_context = create_upload_run_context(
            uploaded_file,
            DEFAULT_MAX_FILE_SIZE_MB,
            source_label=source_label,
        )
        st.success(f"Dataset loaded successfully from {upload_context.source_label}.")
        if upload_context.raw_df.empty:
            st.warning("The CSV was parsed successfully, but it contains headers only and no data rows.")
        st.caption(
            "Ingestion now validates file size, reports encoding, repairs short rows when safe, and skips malformed rows with too many fields."
        )

        ingestion_notes = controls_build_ingestion_notes(upload_context.ingestion_metadata)
        log_monitoring_event(
            PROJECT_ROOT,
            "ingestion_completed",
            payload={
                "file_name": upload_context.ingestion_metadata["filename"],
                "encoding": upload_context.ingestion_metadata["encoding"],
                "rows_loaded": int(upload_context.raw_df.shape[0]),
                "repaired_rows": upload_context.ingestion_metadata["repaired_row_count"],
                "skipped_rows": upload_context.ingestion_metadata["skipped_row_count"],
            },
        )
        rejected_rows_df = workflow_render_ingestion_section(
            PROJECT_ROOT,
            upload_context.raw_df,
            upload_context.ingestion_metadata,
            upload_context.file_stem,
            ingestion_notes,
        )

        st.divider()
        layout_render_section_header(
            "Pipeline Controls",
            "Choose the active dataset view and set configurable cleaning strategies before analysis begins.",
            step=3,
        )
        chart_view_mode, cleaning_config, audience_mode, pipeline_preferences = controls_get_cleaning_config_from_controls(
            upload_context.raw_df,
            upload_context.raw_column_types,
            upload_context.suggested_duplicate_subset,
        )
        base_context = build_base_run_context(
            upload_context,
            chart_view_mode,
            cleaning_config,
            audience_mode,
            pipeline_preferences,
        )
        shared_render_assumptions_bar(chart_view_mode, pipeline_preferences)

        duplicate_rule_caption = (
            "exact row matching"
            if base_context.duplicate_subset is None
            else ", ".join(base_context.duplicate_subset)
        )
        if pipeline_preferences["duplicate_rule_mode"] != "exact":
            st.caption(
                f"Current duplicate rule uses {duplicate_rule_caption}. This is a user-selected business key, so duplicate removal should be treated as domain-dependent rather than universally correct."
            )
        else:
            st.caption("Current duplicate rule uses exact row matching, which is the most conservative and least assumption-heavy option.")
        with st.expander("Duplicate-rule diagnostics", expanded=False):
            diag_left, diag_right = st.columns(2, gap="large")
            with diag_left:
                st.write(f"- Status: **{base_context.duplicate_diagnostics['status']}**")
                st.write(
                    f"- Comparable rows: **{base_context.duplicate_diagnostics['complete_rows']} / {base_context.duplicate_diagnostics['total_rows']}**"
                )
                st.write(f"- Duplicate rows matched: **{base_context.duplicate_diagnostics['duplicate_row_count']}**")
                st.caption(base_context.duplicate_diagnostics["detail"])
            with diag_right:
                st.write(f"- Completeness ratio: **{base_context.duplicate_diagnostics['completeness_ratio']:.0%}**")
                st.write(f"- Unique record ratio: **{base_context.duplicate_diagnostics['unique_record_ratio']:.0%}**")
                st.write(f"- Duplicate groups: **{base_context.duplicate_diagnostics['duplicate_group_count']}**")
                if base_context.duplicate_diagnostics["assumptions"]:
                    shared_render_bullet_list(base_context.duplicate_diagnostics["assumptions"])
        review_mode = shared_render_review_mode_selector()

        st.divider()
        layout_render_section_header(
            "Boardroom Brief",
            "Read the fastest decision-facing summary first, then switch to Evidence or Full Audit when you need proof, diagnostics, or full audit detail.",
            step=4,
        )
        decision_render_decision_mode(
            base_context.upload.ingestion_metadata["filename"],
            base_context.ai_report,
            base_context.analysis_report,
            base_context.chart_recommendations,
            base_context.base_view_label,
            base_context.base_df_to_use,
            base_context.cleaned_df,
            base_context.cleaning_impact_items,
            run_report_payload=base_context.run_report_payload,
        )

        if review_mode == "Boardroom Brief":
            boardroom_fingerprint = json.dumps(
                build_boardroom_fingerprint(base_context, review_mode),
                sort_keys=True,
                default=str,
            )
            persist_current_run_artifact(
                base_context.upload.file_stem,
                boardroom_fingerprint,
                base_context.run_report_payload,
                base_context.decision_brief_markdown,
                base_context.cleaned_df,
                base_context.base_df_to_use,
                rejected_rows_df,
            )
            log_monitoring_event(
                PROJECT_ROOT,
                "boardroom_brief_rendered",
                payload={
                    "file_name": base_context.upload.ingestion_metadata["filename"],
                    "view_label": base_context.base_view_label,
                    "rows_in_view": int(base_context.base_df_to_use.shape[0]),
                    "quality_score": base_context.ai_report["quality_score"],
                    "confidence_label": base_context.ai_report["insight_confidence_label"],
                },
                run_id=st.session_state.get("latest_run_artifact", {}).get("run_id"),
            )
            st.info("Switch to Evidence for the supporting charts and analysis, or to Full Audit for validation, cleaning, and traceability detail.")
            decision_render_artifact_registry_panel(
                st.session_state.get("latest_run_artifact"),
                load_recent_registry_entries(PROJECT_ROOT, limit=5),
            )
            st.stop()
        else:
            if review_mode == "Full Audit":
                workflow_render_full_audit_section(
                    base_context.upload.raw_df,
                    base_context.upload.missing,
                    base_context.upload.invalid_numeric,
                    base_context.upload.invalid_dates,
                    base_context.duplicate_subset,
                    base_context.duplicate_count,
                    base_context.duplicate_rows,
                    base_context.duplicate_diagnostics,
                    base_context.upload.raw_column_types,
                    base_context.base_df_to_use,
                    base_context.chart_view_mode,
                    base_context.transformation_log,
                )

            analysis_df, applied_filters = workflow_render_filtering_section(
                base_context.base_df_to_use,
                base_context.chart_view_mode,
                base_context.upload.raw_df,
                controls_render_filter_controls,
            )
            analysis_context = build_analysis_run_context(base_context, analysis_df, applied_filters)
            final_fingerprint = json.dumps(
                build_analysis_fingerprint(analysis_context, review_mode),
                sort_keys=True,
                default=str,
            )
            persist_current_run_artifact(
                base_context.upload.file_stem,
                final_fingerprint,
                analysis_context.run_report_payload,
                analysis_context.decision_brief_markdown,
                base_context.cleaned_df,
                analysis_context.analysis_df,
                rejected_rows_df,
            )
            log_monitoring_event(
                PROJECT_ROOT,
                "analysis_completed",
                payload={
                    "file_name": base_context.upload.ingestion_metadata["filename"],
                    "view_label": analysis_context.view_label,
                    "rows_in_view": int(analysis_context.analysis_df.shape[0]),
                    "quality_score": analysis_context.ai_report["quality_score"],
                    "confidence_label": analysis_context.ai_report["insight_confidence_label"],
                },
                run_id=st.session_state.get("latest_run_artifact", {}).get("run_id"),
            )
            narrative_mode = controls_get_narrative_mode(llm_api_key)

            analysis_render_analysis_section(
                analysis_context.analysis_df,
                analysis_context.analysis_report,
                analysis_context.chart_recommendations,
            )

        insights_render_insights_section(
            PROJECT_ROOT,
            st.session_state.get("latest_run_artifact"),
            analysis_context.ai_report,
            analysis_context.chart_recommendations,
            narrative_mode,
            llm_api_key,
            llm_model,
            analysis_context.llm_context,
            analysis_context.view_label,
        )

        insights_render_guided_exploration_section(
            base_context.upload.file_stem,
            analysis_context.analysis_df,
            analysis_context.validation_report,
            analysis_context.analysis_report,
            analysis_context.chart_recommendations,
            analysis_context.ai_report,
            llm_api_key,
            llm_model,
            analysis_context.llm_context,
            analysis_context.view_label,
            analysis_context.suggested_analyses,
            base_context.pipeline_preferences,
        )

    except ValueError as exc:
        LOGGER.warning("CSV processing warning: %s", exc)
        log_monitoring_event(
            PROJECT_ROOT,
            "pipeline_warning",
            status="WARNING",
            payload={"message": str(exc)},
        )
        st.error(str(exc))
    except Exception as exc:
        LOGGER.exception("Unexpected file processing error: %s", exc)
        log_monitoring_event(
            PROJECT_ROOT,
            "pipeline_failure",
            status="ERROR",
            payload={"message": str(exc)},
        )
        st.error("Error reading or processing file. Check the CSV structure and try again.")
