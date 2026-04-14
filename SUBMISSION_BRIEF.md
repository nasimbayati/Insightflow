# InsightFlow Submission Brief

Recommended title:

`InsightFlow: Decision-Ready Answers from Messy CSVs`

One-line pitch:

`A production-style CSV analytics app that cleans messy files and turns them into decision-ready insights.`

## Project Summary

InsightFlow is an AI-powered CSV analytics pipeline built for messy, unpredictable datasets. It accepts a CSV upload, validates structure and values, applies configurable cleaning strategies, generates analysis and charts, and produces both rule-based and optional LLM-backed insights.

## Prompt Alignment

This project addresses the `Data Science – CSV Insight Pipeline` prompt by implementing:

- ingestion checks for file type, size, encoding, and malformed rows
- schema inference and data-quality reporting
- configurable cleaning with transformation logging
- user-defined column roles for IDs, time axes, metrics, segments, and outcomes
- summary statistics, correlations, segmentation cues, trends, and outlier detection
- auto-generated charts based on detected data types
- a no-hardcoding chart builder where users choose the exact grouping, metric, and chart type
- executive summary, risk flags, and recommended next actions
- decision-mode briefing with confidence scoring and cleaning impact
- downloadable cleaned CSV, decision brief, chart assets, run report, and analysis package
- modular architecture with tests and reliability handling

## Key Product Decisions

- deterministic analytics first, optional LLM second
- configurable cleaning instead of hardcoded cleanup
- business column roles instead of assuming data type equals meaning
- raw and cleaned analysis modes
- guided exploration instead of brittle free-form automation
- server-side logging for ingestion and failure monitoring

## Why It Is Production-Oriented

- core analytics do not depend on API access
- malformed or low-quality input is handled explicitly
- cleaning behavior is visible and auditable
- charts are chosen based on dataset structure instead of demo-specific columns
- users can override chart and analysis meaning through column roles
- tests cover schema, missing-value behavior, outlier detection, and ingestion reliability

## Enterprise Scaling Direction

This app can evolve into an enterprise workflow by separating ingestion and analysis into services, using object storage and background jobs for large files, persisting validation artifacts and transformation logs, and introducing RBAC, governance, and provider-flexible LLM backends.
