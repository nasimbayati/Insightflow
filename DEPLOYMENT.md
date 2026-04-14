# InsightFlow Deployment Guide

Recommended host: Streamlit Community Cloud.

## Pre-Deployment Checks

Run locally:

```powershell
cd E:\Projects\InsightFlow
python -m pytest tests -q
python -m streamlit run app.py
```

Confirm:

- the bundled `Revenue Operations Showcase` dataset loads
- Boardroom Brief renders
- Column Roles and Current Interpretation appear
- Chart Builder works
- downloads work
- the app works without an OpenAI API key

## GitHub Setup

Create a new GitHub repository, then push this folder:

```powershell
cd E:\Projects\InsightFlow
git init -b main
git add .
git commit -m "Prepare InsightFlow for deployment"
git remote add origin https://github.com/YOUR_USERNAME/insightflow.git
git push -u origin main
```

Do not commit `.streamlit/secrets.toml`.

## Streamlit Community Cloud

1. Go to `https://share.streamlit.io`.
2. Sign in with GitHub.
3. Choose the `insightflow` repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

No secrets are required. The built-in rule-based insights are the default path. If you want optional LLM mode, add `OPENAI_API_KEY` in Streamlit Cloud secrets after deployment.

## Handshake Submission

Use the deployed app URL as the project link.

Recommended title:

```text
InsightFlow: Decision-Ready Answers from Messy CSVs
```

Submission description:

```text
InsightFlow turns messy CSVs into a decision brief instead of a raw table dump. I built it so students, analysts, and teams can upload unpredictable files and quickly understand quality risks, key drivers, and next actions. Built with Codex as a Streamlit pipeline with ingestion checks, schema validation, configurable cleaning, role-aware charts, rule-based insights, and downloadable reports.
```

Check the `Built with Codex` box.
