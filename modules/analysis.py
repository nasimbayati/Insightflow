import itertools
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

from modules.validation import is_probable_id_column


MIN_CORRELATION_SAMPLE = 8
MIN_TREND_PERIODS = 4
MIN_SEGMENT_GROUP_SIZE = 2


def get_dataset_shape(df):
    return df.shape


def get_numeric_columns(df, exclude_id_like=False):
    numeric_cols = []

    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue

        if exclude_id_like and is_probable_id_column(df[col], col, "Numeric"):
            continue

        numeric_cols.append(col)

    return numeric_cols


def get_datetime_columns(df):
    return [col for col in df.columns if is_datetime64_any_dtype(df[col])]


def get_filterable_categorical_columns(df, max_unique=25):
    filterable_columns = []

    for col in get_categorical_columns(df):
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        unique_count = non_null.astype(str).nunique(dropna=True)
        if 2 <= unique_count <= max_unique:
            filterable_columns.append(col)

    return filterable_columns


def get_groupable_columns(df, max_unique=30):
    groupable_columns = []
    row_count = max(len(df), 1)
    dynamic_limit = min(max_unique, max(8, int(row_count * 0.6)))

    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        unique_count = non_null.astype(str).nunique(dropna=True)
        if 2 <= unique_count <= dynamic_limit or col in get_categorical_columns(df):
            groupable_columns.append(col)

    return groupable_columns


def get_categorical_columns(df):
    return [
        col
        for col in df.columns
        if is_object_dtype(df[col]) or is_string_dtype(df[col]) or str(df[col].dtype) == "category"
    ]


def get_numeric_summary(df):
    numeric_cols = get_numeric_columns(df, exclude_id_like=True)
    if not numeric_cols:
        numeric_cols = get_numeric_columns(df, exclude_id_like=False)

    if not numeric_cols:
        return None

    return df[numeric_cols].describe().T


def get_categorical_summary(df):
    summary = {}

    for col in get_categorical_columns(df):
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        unique_count = non_null.nunique(dropna=True)
        unique_ratio = unique_count / len(non_null)

        if unique_count <= 20 or unique_ratio <= 0.5:
            summary[col] = df[col].value_counts(dropna=False).head(5)

    return summary


def get_correlation_matrix(df):
    numeric_cols = [
        col
        for col in get_numeric_columns(df, exclude_id_like=True)
        if df[col].dropna().nunique() > 1
    ]

    if len(numeric_cols) < 2:
        return None

    return df[numeric_cols].corr(numeric_only=True)


def _label_correlation_support(sample_size, correlation_value):
    absolute_value = abs(correlation_value)

    if sample_size < 5:
        return "Insufficient", "Too few paired rows remain after null handling to treat this correlation as evidence."
    if sample_size < MIN_CORRELATION_SAMPLE:
        return "Directional", "The relationship is visible, but the paired sample is still small."
    if absolute_value < 0.2:
        return "Weak", "The correlation exists numerically, but the effect size is too small to lean on."
    if sample_size < 15 or absolute_value < 0.35:
        return "Directional", "The effect is real enough to inspect, but not strong enough to overstate."

    return "Supported", "The pair has enough coverage and effect size to treat as a meaningful signal."


def get_strongest_correlation_signal(df, corr_matrix=None):
    numeric_cols = [
        col
        for col in get_numeric_columns(df, exclude_id_like=True)
        if df[col].dropna().nunique() > 1
    ]

    if len(numeric_cols) < 2:
        return None

    strongest = None

    for col_a, col_b in itertools.combinations(numeric_cols, 2):
        pair_df = df[[col_a, col_b]].dropna()
        sample_size = len(pair_df)
        if sample_size < 3:
            continue

        correlation_value = pair_df[col_a].corr(pair_df[col_b])
        if pd.isna(correlation_value):
            continue

        candidate = {
            "columns": (col_a, col_b),
            "value": float(correlation_value),
            "sample_size": int(sample_size),
        }
        if strongest is None or (abs(candidate["value"]), candidate["sample_size"]) > (
            abs(strongest["value"]),
            strongest["sample_size"],
        ):
            strongest = candidate

    if strongest is None:
        return None

    support_label, support_reason = _label_correlation_support(
        strongest["sample_size"],
        strongest["value"],
    )
    strongest["support_label"] = support_label
    strongest["support_reason"] = support_reason
    strongest["is_supported"] = support_label == "Supported"

    if corr_matrix is not None and strongest["columns"][0] in corr_matrix.columns:
        strongest["matrix_value"] = float(corr_matrix.loc[strongest["columns"][0], strongest["columns"][1]])

    return strongest


def get_outlier_summary(df):
    outlier_summary = {}

    for col in get_numeric_columns(df, exclude_id_like=True):
        series = df[col].dropna()
        if len(series) < 4 or series.nunique() < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        outlier_summary[col] = {
            "count": outlier_count,
            "share": outlier_count / len(series),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "median": float(series.median()),
            "max": float(series.max()),
            "sample_size": int(len(series)),
        }

    return outlier_summary


def select_categorical_chart_column(df, preferred_column=None):
    categorical_cols = get_categorical_columns(df)

    if preferred_column in df.columns:
        preferred_non_null = df[preferred_column].dropna()
        unique_count = preferred_non_null.astype(str).nunique(dropna=True)
        if 2 <= unique_count <= max(30, min(len(preferred_non_null), 50)):
            return preferred_column

    candidates = []

    for col in categorical_cols:
        non_null = df[col].dropna()
        unique_count = non_null.nunique(dropna=True)
        if unique_count < 2:
            continue

        if unique_count <= 15:
            score = len(non_null) - unique_count
            candidates.append((score, col))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return None


def select_numeric_chart_column(df, preferred_column=None):
    numeric_cols = [
        col
        for col in get_numeric_columns(df, exclude_id_like=True)
        if df[col].dropna().nunique() > 1
    ]

    if preferred_column in numeric_cols:
        return preferred_column

    if numeric_cols:
        numeric_cols.sort(
            key=lambda col: (
                df[col].dropna().count(),
                df[col].std(skipna=True) if pd.notna(df[col].std(skipna=True)) else 0,
            ),
            reverse=True,
        )
        return numeric_cols[0]

    fallback_numeric_cols = [
        col for col in get_numeric_columns(df, exclude_id_like=False) if df[col].dropna().nunique() > 1
    ]
    return fallback_numeric_cols[0] if fallback_numeric_cols else None


def select_scatter_columns(df, preferred_columns=None):
    numeric_cols = [
        col
        for col in get_numeric_columns(df, exclude_id_like=True)
        if df[col].dropna().nunique() > 1
    ]

    if preferred_columns and len(preferred_columns) == 2 and all(col in numeric_cols for col in preferred_columns):
        return tuple(preferred_columns)

    if len(numeric_cols) < 2:
        return None

    strongest = get_strongest_correlation_signal(df)
    if strongest:
        return tuple(strongest["columns"])

    return tuple(numeric_cols[:2])


def _coerce_time_axis(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    if is_numeric_dtype(series) and numeric_series.dropna().nunique() > 1 and numeric_series.notna().mean() >= 0.5:
        return numeric_series, "numeric"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        datetime_series = pd.to_datetime(series, errors="coerce")
    if datetime_series.dropna().nunique() > 1 and datetime_series.notna().mean() >= 0.5:
        return datetime_series, "datetime"

    if numeric_series.dropna().nunique() > 1 and numeric_series.notna().mean() >= 0.5:
        return numeric_series, "numeric"

    return None, None


def select_time_axis_column(df, preferred_date_column=None, exclude_columns=None):
    exclude_columns = set(exclude_columns or [])
    date_cols = get_datetime_columns(df)

    if preferred_date_column in df.columns and preferred_date_column not in exclude_columns:
        preferred_series, _ = _coerce_time_axis(df[preferred_date_column])
        if preferred_series is not None:
            return preferred_date_column

    candidates = []
    total_rows = max(len(df), 1)
    candidate_columns = list(dict.fromkeys(date_cols + get_groupable_columns(df, max_unique=40)))
    for column in candidate_columns:
        if column in exclude_columns:
            continue
        if is_probable_id_column(df[column], column, None):
            continue

        axis_series, axis_kind = _coerce_time_axis(df[column])
        if axis_series is None:
            continue

        valid_axis = axis_series.dropna()
        distinct_periods = int(valid_axis.nunique())
        if distinct_periods < 2:
            continue

        coverage_ratio = valid_axis.count() / total_rows
        span_score = (
            (valid_axis.max() - valid_axis.min()).total_seconds()
            if axis_kind == "datetime"
            else float(valid_axis.max() - valid_axis.min())
        )
        candidates.append(
            (
                axis_kind == "datetime",
                distinct_periods >= MIN_TREND_PERIODS,
                round(coverage_ratio, 4),
                distinct_periods,
                span_score,
                column,
            )
        )

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][5]

    return next((column for column in date_cols if column not in exclude_columns), None)


def _label_trend_support(sample_size, distinct_periods, coverage_ratio):
    if sample_size < 4 or distinct_periods < 3:
        return "Insufficient", "There are not enough distinct time periods to treat this as a reliable trend."
    if distinct_periods < MIN_TREND_PERIODS or coverage_ratio < 0.45:
        return "Directional", "The time pattern is visible, but there are too few periods or too much missing time coverage."
    if distinct_periods < 7 or coverage_ratio < 0.65:
        return "Directional", "The trend is useful for direction, but not strong enough to overstate."

    return "Supported", "The trend uses enough periods and coverage to treat as a meaningful time signal."


def assess_trend_signal(df, preferred_value_column=None, preferred_date_column=None):
    value_col = select_numeric_chart_column(df, preferred_value_column)
    date_col = select_time_axis_column(df, preferred_date_column, exclude_columns={value_col} if value_col else None)
    if date_col is None or value_col is None:
        return None

    working_df = df[[date_col, value_col]].copy()
    axis_series, axis_kind = _coerce_time_axis(working_df[date_col])
    if axis_series is None:
        return None

    working_df["_time_axis"] = axis_series
    working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
    working_df = working_df.dropna(subset=["_time_axis", value_col]).sort_values("_time_axis")

    if working_df.empty:
        return None

    if working_df["_time_axis"].duplicated().any():
        grouped = working_df.groupby("_time_axis", as_index=False)[value_col].mean()
    else:
        grouped = working_df

    distinct_periods = int(grouped["_time_axis"].nunique())
    sample_size = int(len(working_df))
    coverage_ratio = sample_size / max(len(df), 1)
    support_label, support_reason = _label_trend_support(sample_size, distinct_periods, coverage_ratio)

    first_value = float(grouped[value_col].iloc[0])
    last_value = float(grouped[value_col].iloc[-1])
    direction = "upward" if last_value > first_value else "downward" if last_value < first_value else "flat"

    return {
        "date_column": date_col,
        "time_column": date_col,
        "axis_kind": axis_kind,
        "value_column": value_col,
        "first_value": first_value,
        "last_value": last_value,
        "direction": direction,
        "start_date": grouped["_time_axis"].iloc[0].date() if axis_kind == "datetime" else grouped["_time_axis"].iloc[0],
        "end_date": grouped["_time_axis"].iloc[-1].date() if axis_kind == "datetime" else grouped["_time_axis"].iloc[-1],
        "start_label": str(grouped["_time_axis"].iloc[0].date() if axis_kind == "datetime" else grouped["_time_axis"].iloc[0]),
        "end_label": str(grouped["_time_axis"].iloc[-1].date() if axis_kind == "datetime" else grouped["_time_axis"].iloc[-1]),
        "sample_size": sample_size,
        "distinct_periods": distinct_periods,
        "coverage_ratio": coverage_ratio,
        "support_label": support_label,
        "support_reason": support_reason,
        "is_supported": support_label == "Supported",
    }


def _calculate_effect_size(series_a, series_b):
    std_a = float(series_a.std(ddof=1)) if len(series_a) > 1 else 0.0
    std_b = float(series_b.std(ddof=1)) if len(series_b) > 1 else 0.0
    pooled_denominator = np.sqrt(((std_a ** 2) + (std_b ** 2)) / 2) if std_a or std_b else 0.0
    if pooled_denominator == 0:
        return 0.0
    return float((series_a.mean() - series_b.mean()) / pooled_denominator)


def _label_segment_support(min_group_size, effect_size, compared_groups):
    absolute_effect = abs(effect_size)

    if compared_groups < 2 or min_group_size < MIN_SEGMENT_GROUP_SIZE:
        return "Insufficient", "Not enough populated groups remain to trust the segment comparison."
    if min_group_size < 3 or absolute_effect < 0.35:
        return "Directional", "The segment gap is visible, but the sample size or effect size is still light."
    if min_group_size < 5 or absolute_effect < 0.6:
        return "Directional", "The segment difference is useful for direction, but not strong enough to overstate."

    return "Supported", "The segment comparison has enough group support and separation to treat as meaningful."


def assess_segment_signal(df, preferred_category_column=None, preferred_value_column=None):
    category_col = select_categorical_chart_column(df, preferred_category_column)
    value_col = select_numeric_chart_column(df, preferred_value_column)

    if category_col is None or value_col is None:
        return None

    working_df = df[[category_col, value_col]].copy()
    working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
    working_df[category_col] = working_df[category_col].astype("string").str.strip()
    working_df = working_df.dropna()

    if working_df.empty:
        return None

    group_counts = working_df.groupby(category_col)[value_col].count().sort_values(ascending=False)
    eligible_groups = group_counts[group_counts >= MIN_SEGMENT_GROUP_SIZE]
    if len(eligible_groups) < 2:
        return None

    grouped_means = (
        working_df[working_df[category_col].isin(eligible_groups.index)]
        .groupby(category_col)[value_col]
        .mean()
        .sort_values(ascending=False)
    )

    top_label = str(grouped_means.index[0])
    bottom_label = str(grouped_means.index[-1])
    top_series = working_df.loc[working_df[category_col] == top_label, value_col]
    bottom_series = working_df.loc[working_df[category_col] == bottom_label, value_col]
    effect_size = _calculate_effect_size(top_series, bottom_series)
    min_group_size = int(min(group_counts.loc[top_label], group_counts.loc[bottom_label]))
    support_label, support_reason = _label_segment_support(
        min_group_size,
        effect_size,
        int(len(eligible_groups)),
    )

    return {
        "category_column": category_col,
        "numeric_column": value_col,
        "top_label": top_label,
        "top_value": float(grouped_means.iloc[0]),
        "bottom_label": bottom_label,
        "bottom_value": float(grouped_means.iloc[-1]),
        "group_count": int(len(eligible_groups)),
        "min_group_size": min_group_size,
        "effect_size": float(effect_size),
        "support_label": support_label,
        "support_reason": support_reason,
        "is_supported": support_label == "Supported",
    }


def select_time_series_columns(df, preferred_value_column=None, preferred_date_column=None):
    trend_signal = assess_trend_signal(
        df,
        preferred_value_column=preferred_value_column,
        preferred_date_column=preferred_date_column,
    )
    if trend_signal is None:
        return None

    return trend_signal["date_column"], trend_signal["value_column"]


def apply_dataset_filters(df, categorical_filters=None, numeric_filters=None, date_filters=None):
    filtered_df = df.copy()
    applied_filters = []

    for column, selected_values in (categorical_filters or {}).items():
        if column not in filtered_df.columns or not selected_values:
            continue

        filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        applied_filters.append(f"{column}: {len(selected_values)} selected value(s)")

    for column, bounds in (numeric_filters or {}).items():
        if column not in filtered_df.columns or bounds is None:
            continue

        lower_bound, upper_bound = bounds
        numeric_series = pd.to_numeric(filtered_df[column], errors="coerce")
        filter_mask = numeric_series.between(lower_bound, upper_bound, inclusive="both")
        filtered_df = filtered_df[filter_mask.fillna(False)]
        applied_filters.append(f"{column}: between {lower_bound:.2f} and {upper_bound:.2f}")

    for column, bounds in (date_filters or {}).items():
        if column not in filtered_df.columns or bounds is None:
            continue

        start_date, end_date = bounds
        date_series = pd.to_datetime(filtered_df[column], errors="coerce")
        filter_mask = date_series.between(start_date, end_date, inclusive="both")
        filtered_df = filtered_df[filter_mask.fillna(False)]
        applied_filters.append(f"{column}: {start_date.date()} to {end_date.date()}")

    return filtered_df.copy(), applied_filters


def build_analysis_report(
    df,
    preferred_value_column=None,
    preferred_date_column=None,
    preferred_category_column=None,
):
    correlation_matrix = get_correlation_matrix(df)
    strongest_correlation = get_strongest_correlation_signal(df, correlation_matrix)
    trend_signal = assess_trend_signal(
        df,
        preferred_value_column=preferred_value_column,
        preferred_date_column=preferred_date_column,
    )
    segment_signal = assess_segment_signal(
        df,
        preferred_category_column=preferred_category_column,
        preferred_value_column=preferred_value_column,
    )

    return {
        "shape": get_dataset_shape(df),
        "numeric_summary": get_numeric_summary(df),
        "categorical_summary": get_categorical_summary(df),
        "correlation_matrix": correlation_matrix,
        "strongest_correlation": strongest_correlation,
        "outlier_summary": get_outlier_summary(df),
        "trend_signal": trend_signal,
        "segment_signal": segment_signal,
    }
