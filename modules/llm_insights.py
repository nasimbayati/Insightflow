import json
import os
from hashlib import sha256

import pandas as pd
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError


DEFAULT_LLM_MODEL = "gpt-5-mini"
SUPPORTED_LLM_MODELS = ["gpt-5-mini", "gpt-5.2"]


def resolve_api_key(override_key=None, streamlit_secrets=None):
    if override_key and override_key.strip():
        return override_key.strip()

    if streamlit_secrets is not None:
        try:
            secret_value = streamlit_secrets.get("OPENAI_API_KEY")
        except Exception:
            secret_value = None

        if secret_value:
            return str(secret_value).strip()

    env_value = os.getenv("OPENAI_API_KEY")
    if env_value:
        return env_value.strip()

    return None


def is_llm_configured(api_key):
    return bool(api_key)


def _clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        return round(value, 4)

    return value


def _series_to_records(series, value_key):
    return [
        {
            "label": str(index),
            value_key: _clean_value(value),
        }
        for index, value in series.items()
    ]


def _df_to_records(df, limit=8):
    return [
        {column: _clean_value(value) for column, value in row.items()}
        for row in df.head(limit).to_dict(orient="records")
    ]


def _summarize_analysis_report(analysis_report):
    numeric_summary = analysis_report["numeric_summary"]
    categorical_summary = analysis_report["categorical_summary"]
    outlier_summary = analysis_report["outlier_summary"]

    return {
        "shape": {
            "rows": analysis_report["shape"][0],
            "columns": analysis_report["shape"][1],
        },
        "numeric_summary": _df_to_records(numeric_summary.reset_index(), limit=8) if numeric_summary is not None else [],
        "categorical_summary": {
            column: _series_to_records(summary, "count")
            for column, summary in list(categorical_summary.items())[:6]
        },
        "strongest_correlation": analysis_report["strongest_correlation"],
        "outlier_summary": {
            column: {
                key: _clean_value(value)
                for key, value in details.items()
            }
            for column, details in list(outlier_summary.items())[:6]
        },
    }


def build_llm_context(
    raw_df,
    active_df,
    raw_column_types,
    validation_report,
    analysis_report,
    chart_recommendations,
    view_label,
    active_filters=None,
):
    preview_columns = list(active_df.columns[:8])
    preview_df = active_df[preview_columns].head(6).copy() if preview_columns else active_df.head(6).copy()

    context = {
        "view_label": view_label,
        "raw_dataset_shape": {"rows": raw_df.shape[0], "columns": raw_df.shape[1]},
        "active_dataset_shape": {"rows": active_df.shape[0], "columns": active_df.shape[1]},
        "columns": [
            {
                "name": column,
                "detected_type": raw_column_types.get(column, "Unknown"),
            }
            for column in raw_df.columns
        ],
        "validation": {
            "missing_values": _series_to_records(validation_report["missing"], "count")
            if not validation_report["missing"].empty
            else [],
            "duplicate_count": int(validation_report["duplicate_count"]),
            "duplicate_subset": validation_report["duplicate_subset"] or [],
            "invalid_numeric": validation_report["invalid_numeric"],
            "invalid_dates": validation_report["invalid_dates"],
            "quality_score": validation_report.get("quality_score"),
            "quality_label": validation_report.get("quality_label"),
        },
        "analysis": _summarize_analysis_report(analysis_report),
        "recommended_charts": chart_recommendations,
        "active_filters": active_filters or [],
        "preview_rows": _df_to_records(preview_df, limit=6),
    }

    return context


def build_overview_messages(context):
    system_prompt = (
        "You are a senior data analyst writing for a non-technical user. "
        "Use only the provided dataset summary. Do not invent facts, unseen rows, or unsupported business claims. "
        "If the data is limited, say so clearly. Keep the tone clear, professional, and actionable. "
        "Return markdown with these headings exactly: "
        "## Executive Summary, ## Key Findings, ## Risks and Caveats, ## Recommended Next Actions."
    )

    user_prompt = (
        "Create a concise executive narrative for the uploaded CSV based on this structured context:\n\n"
        f"{json.dumps(context, indent=2, default=str)}"
    )

    return system_prompt, user_prompt


def build_custom_request_messages(context, request):
    system_prompt = (
        "You are a senior data analyst responding to a user's specific question about a CSV. "
        "Use only the provided structured context. Do not claim to have inspected rows or charts that are not included. "
        "If the request needs information not present in the context, explain that limitation and suggest the nearest valid analysis. "
        "Return markdown with these headings exactly: ## Answer, ## Evidence Used, ## Suggested Follow-up."
    )

    user_prompt = (
        f"User request: {request}\n\n"
        "Respond using this dataset context:\n\n"
        f"{json.dumps(context, indent=2, default=str)}"
    )

    return system_prompt, user_prompt


def _extract_text(response):
    if getattr(response, "output_text", None):
        return response.output_text.strip()

    output = getattr(response, "output", None) or []
    chunks = []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)

    return "\n".join(chunks).strip()


def call_openai_markdown(api_key, model, system_prompt, user_prompt):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _extract_text(response)


def generate_llm_overview(api_key, model, context):
    system_prompt, user_prompt = build_overview_messages(context)
    return call_openai_markdown(api_key, model, system_prompt, user_prompt)


def generate_llm_custom_response(api_key, model, context, request):
    system_prompt, user_prompt = build_custom_request_messages(context, request)
    return call_openai_markdown(api_key, model, system_prompt, user_prompt)


def get_llm_cache_key(prefix, model, view_label, context, request=None):
    payload = {
        "prefix": prefix,
        "model": model,
        "view_label": view_label,
        "request": request or "",
        "context": context,
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def format_llm_error(exc):
    if isinstance(exc, AuthenticationError):
        return "Authentication failed. Check that your OpenAI API key is valid."
    if isinstance(exc, RateLimitError):
        return "Rate limit reached. Try again in a moment or use a smaller model."
    if isinstance(exc, APIConnectionError):
        return "The app could not reach the OpenAI API. Check your network connection and try again."
    if isinstance(exc, APIError):
        return f"OpenAI API error: {exc}"
    return f"Unexpected LLM error: {exc}"
