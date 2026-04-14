import warnings

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


NULL_LIKE_VALUES = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "nat",
    "missing",
}


def normalize_text_series(series):
    return series.map(lambda value: value.strip().lower() if isinstance(value, str) else value)


def normalize_missing_values(df):
    normalized_df = df.copy()

    for col in normalized_df.columns:
        series = normalized_df[col]

        if is_object_dtype(series) or is_string_dtype(series):
            stripped = series.map(lambda value: value.strip() if isinstance(value, str) else value)
            lower_values = stripped.map(lambda value: value.lower() if isinstance(value, str) else value)
            normalized_df[col] = stripped.mask(lower_values.isin(NULL_LIKE_VALUES), pd.NA)

    return normalized_df


def coerce_numeric_series(series):
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    working = series.astype("string")
    cleaned = (
        working.str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )

    return pd.to_numeric(cleaned, errors="coerce")


def coerce_datetime_series(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            return pd.to_datetime(series, errors="coerce")


def build_duplicate_comparison_frame(df):
    comparison_df = normalize_missing_values(df)

    for col in comparison_df.columns:
        series = comparison_df[col]

        if is_object_dtype(series) or is_string_dtype(series):
            comparison_df[col] = normalize_text_series(series)
        elif is_datetime64_any_dtype(series):
            comparison_df[col] = coerce_datetime_series(series)
        elif is_numeric_dtype(series):
            comparison_df[col] = pd.to_numeric(series, errors="coerce")

    return comparison_df


def check_missing_values(df):
    missing = normalize_missing_values(df).isna().sum()
    return missing[missing > 0].sort_values(ascending=False)


def is_probable_id_column(series, column_name, column_type=None):
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    column_name = column_name.lower()

    explicit_id_name = (
        column_name == "id"
        or column_name.endswith("_id")
        or column_name.endswith(" id")
        or column_name == "index"
        or column_name.endswith("_index")
    )

    if explicit_id_name and unique_ratio >= 0.8:
        return True

    if column_type == "Numeric" and unique_ratio >= 0.98 and len(non_null) >= 5:
        coerced = coerce_numeric_series(non_null)
        if not coerced.notna().all():
            return False

        is_integer_like = ((coerced % 1) == 0).all()
        is_monotonic = coerced.is_monotonic_increasing or coerced.is_monotonic_decreasing
        return is_integer_like and is_monotonic

    return False


def check_duplicates(df, subset=None):
    comparison_df = build_duplicate_comparison_frame(df)
    return int(comparison_df.duplicated(subset=subset).sum())


def get_duplicate_rows(df, subset=None):
    comparison_df = build_duplicate_comparison_frame(df)
    duplicate_mask = comparison_df.duplicated(subset=subset, keep=False)
    return df.loc[duplicate_mask].copy()


def detect_column_types(df):
    normalized_df = normalize_missing_values(df)
    column_types = {}

    for col in normalized_df.columns:
        series = normalized_df[col]
        non_null = series.dropna()

        if non_null.empty:
            column_types[col] = "Unknown"
            continue

        if is_numeric_dtype(series):
            column_types[col] = "Numeric"
            continue

        if is_datetime64_any_dtype(series):
            column_types[col] = "Date"
            continue

        numeric_conversion = coerce_numeric_series(non_null)
        numeric_success_rate = numeric_conversion.notna().mean()

        date_conversion = coerce_datetime_series(non_null)
        date_success_rate = date_conversion.notna().mean()

        if numeric_success_rate >= 0.8:
            column_types[col] = "Numeric"
        elif date_success_rate >= 0.8:
            column_types[col] = "Date"
        else:
            column_types[col] = "Categorical"

    return column_types


def suggest_duplicate_subset(df, column_types=None, max_columns=4):
    normalized_df = normalize_missing_values(df)
    column_types = column_types or detect_column_types(normalized_df)
    scored_columns = []

    for col in normalized_df.columns:
        series = normalized_df[col].dropna()
        if series.empty:
            continue

        if is_object_dtype(normalized_df[col]) or is_string_dtype(normalized_df[col]):
            series = normalize_text_series(series)

        unique_ratio = series.nunique(dropna=True) / len(series)
        repeated_ratio = 1 - unique_ratio
        if repeated_ratio <= 0:
            continue

        col_type = column_types.get(col, "Unknown")
        if is_probable_id_column(normalized_df[col], col, col_type):
            continue

        score = unique_ratio
        if col_type == "Date":
            score += 0.3
        elif col_type == "Numeric":
            score += 0.2
        elif col_type == "Categorical":
            score += 0.1

        if unique_ratio < 0.2:
            score -= 0.4
        elif unique_ratio < 0.4:
            score -= 0.2

        scored_columns.append((score, col))

    scored_columns.sort(reverse=True)
    subset = [col for score, col in scored_columns if score > 0][:max_columns]

    if len(subset) >= 2:
        return subset

    return None


def evaluate_duplicate_rule(df, subset=None, column_types=None):
    comparison_df = build_duplicate_comparison_frame(df)
    total_rows = len(comparison_df)

    if total_rows == 0:
        return {
            "mode": "exact" if subset is None else "column_subset",
            "status": "Unavailable",
            "detail": "The dataset has no rows, so duplicate diagnostics are unavailable.",
            "subset": subset or [],
            "total_rows": 0,
            "complete_rows": 0,
            "completeness_ratio": 0.0,
            "duplicate_row_count": 0,
            "duplicate_group_count": 0,
            "unique_record_ratio": 0.0,
            "assumptions": [],
        }

    if subset:
        available_subset = [column for column in subset if column in comparison_df.columns]
        if not available_subset:
            return {
                "mode": "column_subset",
                "status": "Unavailable",
                "detail": "None of the requested duplicate-key columns are present in the dataset.",
                "subset": subset,
                "total_rows": total_rows,
                "complete_rows": 0,
                "completeness_ratio": 0.0,
                "duplicate_row_count": 0,
                "duplicate_group_count": 0,
                "unique_record_ratio": 0.0,
                "assumptions": ["Requested duplicate rule could not be evaluated because the columns were missing."],
            }

        duplicate_frame = comparison_df[available_subset]
        complete_mask = duplicate_frame.notna().all(axis=1)
        complete_rows = int(complete_mask.sum())
        comparable_frame = duplicate_frame.loc[complete_mask]
        mode = "column_subset"
    else:
        available_subset = []
        comparable_frame = comparison_df
        complete_rows = total_rows
        mode = "exact"

    if comparable_frame.empty:
        return {
            "mode": mode,
            "status": "Weak" if subset else "Unavailable",
            "detail": "The current duplicate rule leaves no fully comparable rows after null handling.",
            "subset": subset or [],
            "total_rows": total_rows,
            "complete_rows": complete_rows,
            "completeness_ratio": complete_rows / max(total_rows, 1),
            "duplicate_row_count": 0,
            "duplicate_group_count": 0,
            "unique_record_ratio": 0.0,
            "assumptions": ["Rows with blanks in the duplicate key are excluded from duplicate matching."],
        }

    duplicate_mask = comparable_frame.duplicated(keep=False)
    duplicate_row_count = int(duplicate_mask.sum())
    duplicate_group_count = int(comparable_frame.loc[duplicate_mask].drop_duplicates().shape[0]) if duplicate_row_count else 0
    unique_record_ratio = comparable_frame.drop_duplicates().shape[0] / max(len(comparable_frame), 1)
    completeness_ratio = complete_rows / max(total_rows, 1)

    assumptions = []
    if subset:
        assumptions.append("Rows with blanks in any duplicate-key column are excluded from column-subset matching.")
        if column_types:
            id_like_columns = [
                column
                for column in available_subset
                if is_probable_id_column(df[column], column, column_types.get(column))
            ]
            if id_like_columns:
                assumptions.append(
                    f"Identifier-like columns are included in the duplicate rule: {', '.join(id_like_columns)}."
                )

    if mode == "exact":
        status = "Conservative"
        detail = "Exact row matching avoids business-key assumptions, but it only catches rows that repeat across every column."
    else:
        if completeness_ratio < 0.65:
            status = "Weak"
            detail = "A large share of rows are missing at least one duplicate-key field, so this rule is incomplete."
        elif len(available_subset) == 1:
            status = "Weak"
            detail = "A single-column duplicate key is usually too coarse to represent a business entity safely."
        elif unique_record_ratio < 0.2:
            status = "Weak"
            detail = "The chosen key is too coarse because too many rows collapse into the same duplicate bucket."
        elif duplicate_row_count == 0:
            status = "Sparse"
            detail = "The rule is structurally usable, but it does not currently detect any repeated records."
        else:
            status = "Usable"
            detail = "The current duplicate key is complete enough to use, but it should still match the real business definition of a repeated record."

    return {
        "mode": mode,
        "status": status,
        "detail": detail,
        "subset": available_subset if subset else [],
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "completeness_ratio": completeness_ratio,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_group_count": duplicate_group_count,
        "unique_record_ratio": unique_record_ratio,
        "assumptions": assumptions,
    }


def find_invalid_numeric_values(df, column_types):
    normalized_df = normalize_missing_values(df)
    invalid_numeric = {}

    for col, col_type in column_types.items():
        if col_type != "Numeric":
            continue

        non_null = normalized_df[col].dropna()
        converted = coerce_numeric_series(non_null)
        invalid_values = non_null[converted.isna()].astype(str).unique().tolist()

        if invalid_values:
            invalid_numeric[col] = invalid_values

    return invalid_numeric


def find_invalid_date_values(df, column_types):
    normalized_df = normalize_missing_values(df)
    invalid_dates = {}

    for col, col_type in column_types.items():
        if col_type != "Date":
            continue

        non_null = normalized_df[col].dropna()
        converted = coerce_datetime_series(non_null)
        invalid_values = non_null[converted.isna()].astype(str).unique().tolist()

        if invalid_values:
            invalid_dates[col] = invalid_values

    return invalid_dates
