# InsightFlow — Decision-Ready Answers from Messy CSVs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-Pytest-green?logo=pytest&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-Optional%20LLM-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Upload any CSV. Clean the mess. Get a decision-ready explanation — not a raw table dump.**

InsightFlow is a production-style CSV analytics web app that handles the full pipeline: file ingestion with fault tolerance, schema validation, configurable cleaning, statistical analysis, smart chart recommendations, and narrative-driven insights — with an optional OpenAI layer on top. No API key required for core analytics.

---

## Table of Contents

- [Live Demo](#live-demo)
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Demo Datasets](#demo-datasets)
- [Design Decisions](#design-decisions)
- [Enterprise Scaling Path](#enterprise-scaling-path)
- [License](#license)

---

## Live Demo

> 🚀 *Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) — free in one click. See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions.*

**Quickest local demo:**
```bash
streamlit run app.py
```
Then select **"Revenue Operations Showcase"** from the built-in demo datasets. It surfaces duplicates, a bad date, an outlier enterprise deal, and a clear regional underperformer — all within seconds.

---

## What It Does

Most CSV tools assume clean data. Real business files aren't clean.

InsightFlow treats data quality as a first-class concern. Before you ever see a chart, it audits your file — flagging malformed rows, encoding issues, duplicate records, invalid values, and missing data — then lets you configure how each issue should be handled. The final output is a **decision brief**: a structured, exportable summary with confidence scoring, risk flags, key drivers, and recommended next steps.

The insight layer works entirely offline with rule-based logic. The optional OpenAI integration layers on top of structured, scoped context rather than replacing the analytics core.

---

## Architecture

```
┌─────────────┐
│  CSV Upload  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  INGESTION                                  │
│  • File type & size validation (25 MB max)  │
│  • Multi-encoding fallback (UTF-8, Latin-1) │
│  • Malformed row repair / skip              │
│  • Header normalization                     │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  VALIDATION                                 │
│  • Schema inference (Numeric/Date/Categorical│
│  • Missing value & null-variant detection   │
│  • Duplicate detection (multi-column)       │
│  • Invalid numeric & date extraction        │
│  • Quality score 0–100                      │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  CLEANING  (configurable per column)        │
│  • Numeric: median / mean / leave           │
│  • Categorical: Unknown / mode / leave      │
│  • Duplicates: remove / keep                │
│  • Text: strip / lowercase / title case     │
│  • Protected columns (ID-like auto-guarded) │
│  • Full transformation audit log            │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  ANALYSIS                                   │
│  • Summary stats, correlations, outliers    │
│  • Trend & segment signal detection         │
│  • Filter logic with row-count tracking     │
│  • Column role assignment (ID/metric/segment│
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  VISUALIZATION                              │
│  • 12+ auto-recommended chart types        │
│  • Custom chart builder (metric + grouping) │
│  • Pastel palette with semantic overrides   │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  INSIGHTS & DECISION MODE                   │
│  • Rule-based: drivers, risks, actions      │
│  • 3 narrative lenses: Exec / Analyst / Ops │
│  • Confidence scoring & quality flags       │
│  • Optional OpenAI narrative layer          │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  EXPORTS                                    │
│  • Cleaned CSV                              │
│  • Decision brief (Markdown)                │
│  • Chart PNGs                               │
│  • Full run report (JSON)                   │
│  • Bundled analysis package                 │
└─────────────────────────────────────────────┘
```

---

## Features

### 🔍 Ingestion & Fault Tolerance
- File-type and size validation (25 MB default limit)
- Multi-encoding fallback: UTF-8 → UTF-8-sig → Latin-1
- Automatic malformed-row repair (pads short rows, skips unparseable)
- Header normalization with duplicate-column handling
- Clear user-facing error messages for every failure mode

### ✅ Data Validation
- Automatic schema inference (Numeric, Date, Categorical, Unknown)
- Missing value normalization across 10+ null variants (`N/A`, `none`, `--`, etc.)
- Duplicate detection with configurable multi-column subsets
- Quality scoring (0–100) and confidence levels per column
- Invalid numeric/date value extraction with row-level tracing

### 🧹 Configurable Cleaning
- Per-column strategy selection — never forces one policy on every dataset
- Auto-protects ID-like and high-cardinality columns from mutation
- Tracks every transformation in a full audit log
- Dual data view: Raw (preserved) vs. Cleaned (processed) side by side

### 📊 Analysis
- Summary statistics, Pearson correlation, IQR-based outlier detection
- Trend signal detection (time series), segment signal detection (group patterns)
- Column role assignment separates business semantics from data type
- Active filter tracking with row-count transparency

### 📈 Visualization
- 12+ chart types: bar, histogram, boxplot, heatmap, scatter, line, grouped variants
- Type-aware auto-recommendations based on schema and column roles
- Interactive custom chart builder: user picks metric, grouping, and chart type
- All charts exportable as PNG

### 🧠 Decision Mode
- Boardroom-style brief: quality score, risk flags, key drivers, recommended actions
- Three narrative lenses: **Executive** (decision-ready), **Analyst** (diagnostic), **Operator** (workflow)
- Next-question suggestions for continued exploration
- Optional OpenAI integration for richer narrative generation — falls back gracefully without a key

### 📦 Exports
- Cleaned CSV download
- Decision brief as downloadable Markdown
- Chart PNGs per visualization
- Full run report (JSON with all metadata)
- Bundled analysis package (all artifacts in one download)

---

## Project Structure

```
InsightFlow/
├── app.py                          # Streamlit UI and pipeline orchestration
├── requirements.txt
├── tests/
│   ├── test_pipeline.py            # Schema, cleaning, chart rec, ingestion tests
│   ├── test_llm_insights.py        # LLM context structure and cache tests
│   └── test_pipeline_service_and_views.py
├── modules/
│   ├── ingestion.py                # File validation, encoding, malformed-row repair
│   ├── validation.py               # Schema inference, null detection, duplicate rules
│   ├── cleaning.py                 # Configurable cleaning strategies + audit log
│   ├── analysis.py                 # Stats, correlations, outliers, filters
│   ├── visualization.py            # Chart rendering (12+ types, matplotlib)
│   ├── insights.py                 # Rule-based insights, confidence scoring, narratives
│   ├── llm_insights.py             # Optional OpenAI integration layer
│   ├── pipeline_config.py          # ColumnRoles + PipelinePreferences dataclasses
│   ├── pipeline_service.py         # Pipeline orchestration context
│   ├── reporting.py                # Decision brief and run report generation
│   ├── artifacts.py                # Export artifact persistence
│   └── monitoring.py              # JSON Lines event logging
├── views/
│   ├── layout.py                   # Hero, demo cards, empty states
│   ├── controls.py                 # Filter, cleaning config, LLM controls
│   ├── decision.py                 # Decision mode rendering
│   ├── analysis_section.py         # Analysis view rendering
│   ├── insights_section.py         # Guided exploration, insights display
│   ├── workflow_sections.py        # Ingestion, validation, audit sections
│   └── shared.py                   # Shared UI components
└── data/
    ├── sample_revenue_ops_showcase.csv
    ├── sample_retail_sales_dirty.csv
    ├── sample_support_tickets.csv
    └── sample_employee_survey.csv
```

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Web Framework | Streamlit | Interactive UI with session state |
| Data Processing | Pandas, NumPy | DataFrame operations, schema coercion |
| Visualization | Matplotlib | Chart rendering and export |
| Testing | pytest | Unit and integration test suite |
| LLM (optional) | OpenAI API | Enhanced narrative generation |
| Language | Python 3.10+ | Core application language |

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/nasimbayati/insightflow.git
cd insightflow
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API key needed — select one of the bundled demo datasets to start immediately.

**4. Optional: enable LLM narratives**

Add your OpenAI key in one of three ways:
```bash
# Option A — environment variable
export OPENAI_API_KEY=sk-...

# Option B — Streamlit secrets file
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml

# Option C — sidebar input field in the app
```

---

## Running Tests

```bash
python -m pytest tests -q
```

**Test coverage includes:**
- Schema detection (numeric, date, categorical)
- Configurable missing-value strategies (median, mean, mode, leave)
- Duplicate removal and deduplication behavior
- IQR-based outlier detection
- Chart recommendation logic per column type
- Custom analysis request mapping
- Ingestion row-repair and row-skip behavior
- LLM context structure validation
- Cache key generation for LLM responses

---

## Demo Datasets

Four bundled datasets are included for instant demos without any file upload:

| Dataset | Best For |
|---|---|
| **Revenue Operations Showcase** | Full pipeline demo — has duplicates, bad date, outlier deal, regional underperformer |
| **Retail Sales (Dirty)** | Cleaning workflow — missing values, mixed formats, classic dirty data |
| **Support Tickets** | Categorical analysis — workload by category, priority, status |
| **Employee Survey** | Segment analysis — engagement scores by department and training type |

The **Revenue Operations Showcase** is the recommended starting point. It surfaces every major InsightFlow feature within a realistic business context.

---

## Design Decisions

**1. Separate pipeline logic from UI**
Each stage (ingestion, validation, cleaning, analysis, visualization, insights) lives in its own module. The Streamlit layer handles orchestration only — no business logic in the UI. This makes each stage independently testable and replaceable.

**2. Raw + Cleaned dual view**
InsightFlow keeps the original uploaded data untouched alongside the cleaned version. Users can compare both at any time, which builds trust and prevents silent data mutation.

**3. Configurable cleaning — not one fixed policy**
"Correct cleaning" is domain-specific. A finance file and a survey file should not be cleaned the same way. All cleaning decisions are explicit, user-configured strategies — never hardcoded defaults.

**4. Column roles separate business meaning from data type**
A column called `year` is numeric but should drive time-series grouping, not summary statistics. Column roles (ID, time axis, metric, segment, outcome) let users express that distinction explicitly.

**5. Rule-based insights first, LLM second**
The insight layer works completely offline with deterministic, rule-based logic. The OpenAI layer adds richer narrative on top of structured context — it never replaces the core analytics path. This means the app works reliably in any environment.

**6. Reliability at ingestion time**
Bad inputs fail early and clearly. Encoding issues, malformed rows, and header problems are caught at ingestion with user-friendly messages — not mid-pipeline exceptions.

**7. Guided exploration over open-ended agent behavior**
InsightFlow offers suggested analysis paths, keyword-based custom requests, and a chart builder rather than a free-form chat interface. This keeps the experience predictable and trustworthy while still feeling intelligent.

---

## Enterprise Scaling Path

InsightFlow is a single-app Streamlit implementation today. The modular architecture maps cleanly onto a distributed system:

| Stage | Enterprise Equivalent |
|---|---|
| File upload | Object storage (S3, GCS) + signed upload URLs |
| Ingestion + validation | AWS Lambda or containerized microservice |
| Cleaning pipeline | Async task queue (Celery, SQS + Lambda) |
| Analysis results | Cached by file hash in Redis / DynamoDB |
| Report generation | S3-hosted artifacts + pre-signed download links |
| LLM narratives | API Gateway → scoped context → provider abstraction |
| Monitoring events | CloudWatch Logs / structured JSON event stream |

---

## License

MIT © [Nasim Bayati](https://github.com/nasimbayati)
