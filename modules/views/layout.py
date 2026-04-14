import io
from html import escape
from pathlib import Path

import streamlit as st


class UploadedBytesIO(io.BytesIO):
    def __init__(self, payload, name):
        super().__init__(payload)
        self.name = name
        self.size = len(payload)


def render_hero():
    st.markdown(
        """
        <div class="if-hero">
            <div class="if-hero-grid">
                <div class="if-logo-row">
                    <div class="if-logo-badge">IF</div>
                    <div class="if-logo-copy">
                        <div class="if-logo-label">InsightFlow</div>
                        <div class="if-logo-title">Decision-ready CSV analytics</div>
                    </div>
                </div>
                <div class="if-hero-kicker">From messy file to decision-ready story</div>
                <div class="if-brand-name">InsightFlow</div>
                <div class="if-hero-title">Decision-ready answers from messy CSVs</div>
                <div class="if-hero-subtitle">
                    Upload any CSV or open a bundled demo. InsightFlow validates data quality, cleans issues, surfaces the main risks and drivers, and explains what to do next.
                </div>
                <div class="if-hero-proof">
                    <div class="if-hero-proof-chip">Default demo opens a Boardroom Brief immediately</div>
                    <div class="if-hero-proof-chip">Structured validation before analysis</div>
                    <div class="if-hero-proof-chip">Role-aware charts, not hardcoded columns</div>
                    <div class="if-hero-proof-chip">Works without API access</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_demo_uploaded_file(data_dir, file_name):
    payload = (Path(data_dir) / file_name).read_bytes()
    return UploadedBytesIO(payload, file_name)


def render_demo_dataset_card(demo_info):
    highlights = "".join(f"<li>{escape(item)}</li>" for item in demo_info["highlights"])
    st.markdown(
        f"""
        <div class="if-demo-card">
            <div class="if-empty-kicker">Judge-ready default</div>
            <div class="if-demo-title">{escape(demo_info["file_name"])}</div>
            <div class="if-demo-copy">{escape(demo_info["tagline"])}</div>
            <ul class="if-demo-list">{highlights}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_preview():
    st.markdown(
        """
        <div class="if-empty-grid">
            <div class="if-empty-card if-empty-card-sky">
                <div class="if-empty-kicker">What You Get</div>
                <div class="if-empty-title">A decision brief, not a raw table dump</div>
                <div class="if-empty-copy">
                    InsightFlow validates messy CSVs, cleans the file, scores reliability, surfaces risks and drivers, and recommends the first chart to trust.
                </div>
                <ul class="if-empty-list">
                    <li>quality score and confidence</li>
                    <li>top risk, key driver, next action</li>
                    <li>downloadable charts and decision brief</li>
                </ul>
            </div>
            <div class="if-empty-card if-empty-card-mint">
                <div class="if-empty-kicker">Fastest Demo</div>
                <div class="if-empty-title">Use the bundled Revenue Operations Showcase</div>
                <div class="if-empty-copy">
                    This is the strongest judge path because it contains obvious quality issues, a weak segment, a large outlier, and enough structure for trend and relationship views.
                </div>
                <ul class="if-empty-list">
                    <li>duplicate records and invalid values</li>
                    <li>clear underperforming segment</li>
                    <li>chart-ready category, trend, and correlation signals</li>
                </ul>
            </div>
            <div class="if-empty-card if-empty-card-apricot">
                <div class="if-empty-kicker">What Reviewers Notice</div>
                <div class="if-empty-title">Reliable workflow before flashy AI</div>
                <div class="if-empty-copy">
                    The app still works without API access, which makes the analytics path defensible for competitions, demos, and enterprise review settings.
                </div>
                <ul class="if-empty-list">
                    <li>ingestion reliability and malformed-row handling</li>
                    <li>configurable cleaning rather than hardcoded rules</li>
                    <li>guided exploration with evidence-backed charts</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, subtitle, step=None):
    step_label = f"Step {step}" if step is not None else "Section"
    step_index = escape(str(step)) if step is not None else "•"
    section_class = f"if-section if-section-step-{step}" if step is not None else "if-section"
    index_class = f"if-section-index if-section-index-step-{step}" if step is not None else "if-section-index"
    st.markdown(
        f"""
        <div class="{section_class}">
            <div class="{index_class}">{step_index}</div>
            <div class="if-section-body">
                <div class="if-step-label">{escape(step_label)}</div>
                <div class="if-section-title">{escape(title)}</div>
                <div class="if-section-subtitle">{escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
