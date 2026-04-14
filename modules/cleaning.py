from dataclasses import asdict, dataclass, field

import pandas as pd

from modules.validation import (
    build_duplicate_comparison_frame,
    coerce_datetime_series,
    coerce_numeric_series,
    is_probable_id_column,
    normalize_missing_values,
)


@dataclass(frozen=True)
class CleaningConfig:
    numeric_missing_strategy: str = "median"
    categorical_missing_strategy: str = "unknown"
    duplicate_action: str = "remove"
    categorical_text_strategy: str = "strip"
    numeric_column_strategies: dict = field(default_factory=dict)
    categorical_column_strategies: dict = field(default_factory=dict)


def resolve_cleaning_config(cleaning_config=None):
    if cleaning_config is None:
        return CleaningConfig()

    if isinstance(cleaning_config, CleaningConfig):
        return cleaning_config

    if hasattr(cleaning_config, "numeric_missing_strategy") and hasattr(cleaning_config, "categorical_missing_strategy"):
        return CleaningConfig(
            numeric_missing_strategy=getattr(cleaning_config, "numeric_missing_strategy", "median"),
            categorical_missing_strategy=getattr(cleaning_config, "categorical_missing_strategy", "unknown"),
            duplicate_action=getattr(cleaning_config, "duplicate_action", "remove"),
            categorical_text_strategy=getattr(cleaning_config, "categorical_text_strategy", "strip"),
            numeric_column_strategies=dict(getattr(cleaning_config, "numeric_column_strategies", {}) or {}),
            categorical_column_strategies=dict(getattr(cleaning_config, "categorical_column_strategies", {}) or {}),
        )

    merged_config = asdict(CleaningConfig())
    merged_config.update(cleaning_config)
    return CleaningConfig(**merged_config)


def _normalize_categorical_value(value, text_strategy):
    if not isinstance(value, str):
        return value

    stripped_value = value.strip()
    if text_strategy == "lower":
        return stripped_value.lower()
    if text_strategy == "title":
        return stripped_value.title()

    return stripped_value


def standardize_data(df, column_types, cleaning_config=None):
    config = resolve_cleaning_config(cleaning_config)
    standardized_df = normalize_missing_values(df)

    for col, col_type in column_types.items():
        if col_type == "Numeric":
            standardized_df[col] = coerce_numeric_series(standardized_df[col])
        elif col_type == "Date":
            standardized_df[col] = coerce_datetime_series(standardized_df[col])
        else:
            standardized_df[col] = standardized_df[col].map(
                lambda value: _normalize_categorical_value(value, config.categorical_text_strategy)
            )

    return standardized_df


def _fill_numeric_missing_values(series, strategy):
    missing_before = int(series.isna().sum())
    if missing_before == 0:
        return series, None

    if strategy == "leave":
        return series, f"Left {missing_before} missing/invalid numeric values unchanged."

    if strategy == "mean":
        fill_value = series.mean()
        strategy_label = "mean"
    else:
        fill_value = series.median()
        strategy_label = "median"

    if pd.notna(fill_value):
        filled_series = series.fillna(fill_value)
        return filled_series, f"Filled {missing_before} missing/invalid numeric values with {strategy_label} ({fill_value})."

    return series, f"Left {missing_before} missing/invalid numeric values unchanged because no valid {strategy_label} was available."


def _fill_categorical_missing_values(series, strategy):
    missing_before = int(series.isna().sum())
    if missing_before == 0:
        return series, None

    if strategy == "leave":
        return series, f"Left {missing_before} missing categorical values unchanged."

    if strategy == "mode":
        mode_values = series.dropna().mode()
        if not mode_values.empty:
            fill_value = mode_values.iloc[0]
            filled_series = series.fillna(fill_value)
            return filled_series, f"Filled {missing_before} missing categorical values with mode ('{fill_value}')."

        return series, f"Left {missing_before} missing categorical values unchanged because no valid mode was available."

    filled_series = series.fillna("Unknown")
    return filled_series, f"Filled {missing_before} missing categorical values with 'Unknown'."


def _is_high_cardinality_categorical(series):
    non_null = series.dropna()
    if len(non_null) < 4:
        return False

    unique_ratio = non_null.astype(str).nunique(dropna=True) / len(non_null)
    return unique_ratio >= 0.8


def clean_data(df, column_types, duplicate_subset=None, cleaning_config=None):
    config = resolve_cleaning_config(cleaning_config)
    cleaned_df = standardize_data(df, column_types, cleaning_config=config)
    transformation_log = [
        f"Numeric missing strategy: {config.numeric_missing_strategy}.",
        f"Categorical missing strategy: {config.categorical_missing_strategy}.",
        f"Duplicate handling: {config.duplicate_action}.",
        f"Categorical text normalization: {config.categorical_text_strategy}.",
    ]

    for col, col_type in column_types.items():
        if col_type == "Numeric":
            column_strategy = config.numeric_column_strategies.get(col, config.numeric_missing_strategy)
            if col not in config.numeric_column_strategies and is_probable_id_column(cleaned_df[col], col, col_type):
                column_strategy = "leave"
                transformation_log.append(
                    f"'{col}': Protected as a probable identifier, so missing numeric values were left unchanged."
                )
            cleaned_df[col], message = _fill_numeric_missing_values(
                cleaned_df[col],
                column_strategy,
            )
            if message:
                transformation_log.append(f"'{col}': {message}")

        elif col_type == "Date":
            missing_before = int(cleaned_df[col].isna().sum())
            if missing_before > 0:
                transformation_log.append(
                    f"'{col}': Standardized as datetime and left {missing_before} missing/invalid value(s) blank to avoid inventing dates."
                )

        else:
            column_strategy = config.categorical_column_strategies.get(col, config.categorical_missing_strategy)
            if col not in config.categorical_column_strategies and _is_high_cardinality_categorical(cleaned_df[col]):
                column_strategy = "leave"
                transformation_log.append(
                    f"'{col}': Protected as a high-cardinality categorical field, so missing values were left unchanged."
                )
            cleaned_df[col], message = _fill_categorical_missing_values(
                cleaned_df[col],
                column_strategy,
            )
            if message:
                transformation_log.append(f"'{col}': {message}")

    rows_before = len(cleaned_df)
    comparison_df = build_duplicate_comparison_frame(cleaned_df)
    duplicate_mask = comparison_df.duplicated(subset=duplicate_subset, keep="first")
    duplicates_found = int(duplicate_mask.sum())

    if duplicate_subset:
        transformation_log.append(
            f"Checked duplicates using suggested subset columns: {duplicate_subset}."
        )
    else:
        transformation_log.append("Checked exact duplicate rows across the full dataset.")

    if config.duplicate_action == "remove":
        cleaned_df = cleaned_df.loc[~duplicate_mask].copy()
        duplicates_removed = rows_before - len(cleaned_df)
        if duplicates_removed > 0:
            transformation_log.append(f"Removed {duplicates_removed} duplicate row(s).")
        else:
            transformation_log.append("No duplicate rows were removed.")
    else:
        transformation_log.append(f"Kept {duplicates_found} detected duplicate row(s) in the dataset.")

    return cleaned_df, transformation_log
