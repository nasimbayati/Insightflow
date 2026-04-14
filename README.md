# InsightFlow: Decision-Ready Answers from Messy CSVs

InsightFlow is a production-style CSV analytics web app built with Streamlit. It is designed for unpredictable CSV files and walks users through ingestion, validation, cleaning, filtering, analysis, visualization, and insight generation.

One-line pitch:

`Upload any CSV, clean the mess, and get a decision-ready explanation instead of a raw table dump.`

## What It Does

- Accepts CSV uploads with file-type and file-size checks
- Detects encoding issues and handles malformed rows gracefully
- Infers schema for numeric, categorical, and date columns
- Flags missing values, duplicates, invalid numeric values, invalid dates, and outliers
- Applies configurable cleaning strategies
- Lets users assign generic column roles such as ID, time axis, metric, segment, and outcome
- Generates summary statistics, correlations, trend views, and anomaly cues
- Recommends relevant charts based on detected data types
- Produces decision-mode summaries with confidence scoring, cleaning impact, and next-question guidance
- Supports executive, analyst, and operational narrative lenses
- Exports cleaned CSVs, decision briefs, chart PNGs, run reports, and a bundled analysis package
- Produces rule-based executive insights with optional LLM enhancement

## Project Structure

- [app.py](/E:/Projects/InsightFlow/app.py): Streamlit UI and orchestration
- [ingestion.py](/E:/Projects/InsightFlow/modules/ingestion.py): file ingestion, encoding handling, malformed-row recovery
- [validation.py](/E:/Projects/InsightFlow/modules/validation.py): schema inference and validation checks
- [cleaning.py](/E:/Projects/InsightFlow/modules/cleaning.py): configurable cleaning strategies and transformation logging
- [analysis.py](/E:/Projects/InsightFlow/modules/analysis.py): summaries, correlations, outliers, and filter logic
- [visualization.py](/E:/Projects/InsightFlow/modules/visualization.py): charts and plot rendering
- [insights.py](/E:/Projects/InsightFlow/modules/insights.py): rule-based executive insights and guided analysis
- [llm_insights.py](/E:/Projects/InsightFlow/modules/llm_insights.py): optional OpenAI-backed narrative layer
- [tests](/E:/Projects/InsightFlow/tests): pipeline and LLM-context tests

## Running Locally

1. Install dependencies:

```powershell
pip install -r E:\Projects\InsightFlow\requirements.txt
```

2. Start the app:

```powershell
streamlit run E:\Projects\InsightFlow\app.py
```

3. Optional LLM setup:

- Provide an OpenAI API key in the sidebar
- or set `OPENAI_API_KEY`
- or create `.streamlit/secrets.toml`

The app still works without any API key. The built-in rule-based insights are the default fallback path.

## Best Demo Path

For the strongest showcase flow, use:

- [sample_revenue_ops_showcase.csv](/E:/Projects/InsightFlow/data/sample_revenue_ops_showcase.csv)

Why this demo file works well:

- duplicate records, missing values, and invalid numeric/date entries appear immediately
- one region clearly underperforms the others
- one large enterprise deal creates an obvious outlier review moment
- the file supports category, distribution, relationship, and trend charts without any special setup

## Architecture Decisions

### 1. Separate pipeline logic from UI

The app keeps ingestion, validation, cleaning, analysis, visualization, and insight generation in separate modules. This keeps the Streamlit layer focused on orchestration and presentation rather than embedding business logic directly in the interface.

Why it matters:

- easier to test
- easier to replace individual stages later
- safer to extend without breaking the whole app

### 2. Standardized raw view plus cleaned view

InsightFlow keeps two important modes:

- `Raw Data`: preserves uploaded rows while coercing types for analysis
- `Cleaned Data`: applies missing-value handling and duplicate strategy

This prevents the UI from forcing one interpretation of the dataset and gives the user a more trustworthy workflow.

### 3. Configurable cleaning instead of one fixed policy

Cleaning decisions are exposed as explicit strategies:

- numeric missing values: median, mean, leave
- categorical missing values: Unknown, mode, leave
- duplicate handling: remove, keep
- categorical text normalization: trim, lower, title

This is important because “correct cleaning” is domain-specific. A finance file, survey file, and support-ticket file should not all be cleaned the same way.

### 4. Column roles separate business meaning from data type

InsightFlow does not assume that a numeric column is always a metric. Users can mark fields as:

- ID columns
- time/order columns
- primary metrics
- primary segments/groups
- outcome or target columns

This matters for unpredictable files. A column like `year`, `quarter`, or `student_id` may be numeric, but it should often drive grouping or record identity rather than summary statistics.

### 5. Rule-based first, LLM second

The insight layer is reliable without any external model dependency. The optional LLM layer sits on top of structured context instead of replacing the core analytics path.

Why this was chosen:

- judges or recruiters may not have API access
- core analytics should still work offline or in restricted environments
- narratives stay grounded in explicit computed results

### 6. Reliability checks at ingestion time

Ingestion is handled as a first-class layer:

- file-size limits
- extension validation
- encoding fallback
- malformed-row recovery or skipping
- clear user-facing error messages

This keeps bad inputs from poisoning the downstream pipeline.

### 7. Guided exploration over unrestricted agent behavior

The app offers:

- automatic summary
- suggested analysis paths
- keyword-based custom requests
- user-driven chart builder
- optional LLM narrative

This keeps the experience understandable and dependable while still feeling intelligent.

## Enterprise-Scale Extension Path

InsightFlow is currently a single-app Streamlit implementation, but the architecture can scale into an enterprise analytics workflow.

### Short-term scaling

- persist upload metadata and transformation logs to a database
- move file storage to object storage
- cache analysis results by file hash
- run expensive profiling in background jobs
- add audit history for cleaning-strategy changes

### Mid-term scaling

- split ingestion, validation, cleaning, and reporting into separate services
- use a task queue for large-file processing
- support chunked CSV processing for very large datasets
- store profiling artifacts and chart metadata for reuse
- add user authentication, dataset ownership, and workspace isolation

### Enterprise workflow support

- role-based access control
- reusable data-quality rules by team or domain
- scheduled ingestion from shared storage or internal systems
- monitoring dashboards for failed ingestions and quality regressions
- signed report generation for stakeholder review

### LLM governance at scale

- keep deterministic analytics as the system of record
- pass only summarized, scoped context into models
- log prompts and model outputs for review
- support provider abstraction so teams can switch between built-in, OpenAI, or local model backends

## Reliability Coverage

The current app explicitly handles:

- malformed rows
- empty/header-only files
- single-column datasets
- missing values
- duplicates
- invalid numeric and date values
- outlier-heavy numeric columns
- filter combinations that reduce the active view to zero rows

## Tests

Current tests cover:

- schema detection
- configurable missing-value handling
- duplicate removal behavior
- outlier detection
- chart recommendation logic
- custom request mapping
- ingestion repair/skip behavior
- LLM context and caching behavior

Run tests with:

```powershell
python -m pytest E:\Projects\InsightFlow\tests -q
```
