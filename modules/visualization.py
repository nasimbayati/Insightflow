import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


PASTEL_COLORS = [
    "#A8DADC",
    "#F4A6A6",
    "#FFD6A5",
    "#CDB4DB",
    "#BDE0FE",
    "#CAFFBF",
    "#FFF1B6",
    "#D8E2DC",
]

COLUMN_COLORS = {
    "id": "#BDE0FE",
    "price": "#FFD6A5",
    "quantity": "#CAFFBF",
}


def plot_categorical_bar(df, column, max_categories=10):
    value_counts = df[column].dropna().astype(str).value_counts().head(max_categories)
    colors = PASTEL_COLORS[: len(value_counts)]

    fig, ax = plt.subplots(figsize=(5, 3))
    value_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="gray")
    ax.set_title(f"Top Categories in {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()

    return fig


def plot_numeric_histogram(df, column, clip_percentiles=None, bins=12):
    series = df[column].dropna()
    if series.empty:
        return None

    numeric_series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if numeric_series.empty:
        return None

    display_series = numeric_series
    title_suffix = ""

    if clip_percentiles:
        lower_q, upper_q = clip_percentiles
        lower_bound = numeric_series.quantile(lower_q)
        upper_bound = numeric_series.quantile(upper_q)
        display_series = numeric_series.clip(lower=lower_bound, upper=upper_bound)
        title_suffix = " (trimmed for display)"

    color = COLUMN_COLORS.get(column.lower(), "#CDB4DB")

    fig, ax = plt.subplots(figsize=(5, 3))
    display_series.plot(kind="hist", bins=bins, ax=ax, color=color, edgecolor="gray")
    ax.axvline(numeric_series.median(), color="#3D405B", linestyle="--", linewidth=1.5, label="Median")
    ax.set_title(f"Distribution of {column}{title_suffix}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()

    return fig


def plot_boxplot(df, column):
    series = pd.to_numeric(df[column], errors="coerce").dropna().astype(float)
    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.boxplot(series, vert=True, patch_artist=True, boxprops={"facecolor": "#BDE0FE"})
    ax.set_title(f"Outlier Check for {column}")
    ax.set_ylabel(column)
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(corr_matrix):
    fig, ax = plt.subplots(figsize=(5, 3))

    cax = ax.matshow(corr_matrix, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index)
    ax.set_title("Correlation Heatmap", pad=20)

    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    plot_df = df[[x_col, y_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(plot_df[x_col], plot_df[y_col], color="#A8DADC", edgecolors="#3D405B", alpha=0.75)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()

    return fig


def plot_time_series(df, date_col, value_col):
    plot_df = df[[date_col, value_col]].copy()
    datetime_axis = pd.to_datetime(plot_df[date_col], errors="coerce")
    numeric_axis = pd.to_numeric(plot_df[date_col], errors="coerce")
    if datetime_axis.notna().mean() >= 0.5 and datetime_axis.dropna().nunique() > 1:
        plot_df[date_col] = datetime_axis
    elif numeric_axis.notna().mean() >= 0.5 and numeric_axis.dropna().nunique() > 1:
        plot_df[date_col] = numeric_axis
    else:
        plot_df[date_col] = plot_df[date_col].astype(str)

    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(date_col)
    if plot_df.empty:
        return None

    if plot_df[date_col].duplicated().any():
        plot_df = plot_df.groupby(date_col, as_index=False)[value_col].mean()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(plot_df[date_col], plot_df[value_col], color="#F4A6A6", linewidth=2, marker="o", markersize=4)
    ax.set_title(f"{value_col} Over Time")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()

    return fig


def build_grouped_metric_summary(df, group_col, value_col=None, aggregation="count", top_n=15, sort_order="descending"):
    working_df = df.copy()

    if group_col not in working_df.columns:
        return None, None

    group_series = working_df[group_col]
    group_is_datetime = is_datetime64_any_dtype(group_series)
    group_is_numeric = is_numeric_dtype(group_series)

    if value_col is None or aggregation == "count":
        summary = (
            working_df[group_col]
            .fillna("Missing")
            .astype(str)
            .value_counts(dropna=False)
            .rename("Row count")
            .reset_index()
        )
        summary.columns = [group_col, "Row count"]
        value_label = "Row count"
    else:
        if value_col not in working_df.columns:
            return None, None

        working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
        working_df = working_df.dropna(subset=[value_col]).copy()
        if working_df.empty:
            return None, None

        if group_is_datetime:
            working_df[group_col] = pd.to_datetime(working_df[group_col], errors="coerce")
            working_df = working_df.dropna(subset=[group_col])
        else:
            working_df[group_col] = working_df[group_col].fillna("Missing")

        grouped = working_df.groupby(group_col, dropna=False)[value_col]
        if aggregation == "sum":
            summary = grouped.sum().reset_index()
        elif aggregation == "median":
            summary = grouped.median().reset_index()
        elif aggregation == "min":
            summary = grouped.min().reset_index()
        elif aggregation == "max":
            summary = grouped.max().reset_index()
        elif aggregation == "count":
            summary = grouped.count().reset_index()
        else:
            summary = grouped.mean().reset_index()

        value_label = f"{aggregation.title()} of {value_col}"
        summary.columns = [group_col, value_label]

    if summary.empty:
        return None, None

    if sort_order == "ascending":
        summary = summary.sort_values(by=value_label, ascending=True)
    elif sort_order == "natural":
        if group_is_datetime:
            summary = summary.sort_values(by=group_col, ascending=True)
        elif group_is_numeric:
            numeric_order = pd.to_numeric(summary[group_col], errors="coerce")
            summary = summary.assign(_sort_value=numeric_order).sort_values(by=["_sort_value", group_col], ascending=True)
            summary = summary.drop(columns="_sort_value")
        else:
            summary = summary.sort_values(by=group_col, key=lambda series: series.astype(str).str.lower())
    else:
        summary = summary.sort_values(by=value_label, ascending=False)

    return summary.head(top_n).reset_index(drop=True), value_label


def plot_grouped_bar(summary_df, group_col, value_label):
    if summary_df is None or summary_df.empty:
        return None

    colors = [PASTEL_COLORS[index % len(PASTEL_COLORS)] for index in range(len(summary_df))]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(summary_df[group_col].astype(str), summary_df[value_label], color=colors, edgecolor="gray")
    ax.set_title(f"{value_label} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_label)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return fig


def plot_grouped_line(summary_df, group_col, value_label):
    if summary_df is None or summary_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x_values = summary_df[group_col]
    if not is_datetime64_any_dtype(x_values):
        x_values = summary_df[group_col].astype(str)
    ax.plot(x_values, summary_df[value_label], color="#F4A6A6", linewidth=2, marker="o", markersize=4)
    ax.set_title(f"{value_label} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_label)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return fig


def plot_boxplot_by_group(df, group_col, value_col, max_groups=12):
    working_df = df[[group_col, value_col]].copy()
    working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
    working_df = working_df.dropna(subset=[group_col, value_col])
    if working_df.empty:
        return None

    group_counts = working_df[group_col].astype(str).value_counts().head(max_groups)
    working_df[group_col] = working_df[group_col].astype(str)
    working_df = working_df[working_df[group_col].isin(group_counts.index)]
    if working_df.empty:
        return None

    grouped_values = [group[value_col].tolist() for _, group in working_df.groupby(group_col)]
    labels = list(working_df.groupby(group_col).groups.keys())

    fig, ax = plt.subplots(figsize=(6, 3.6))
    boxplot = ax.boxplot(grouped_values, patch_artist=True, labels=labels)
    for patch, color in zip(boxplot["boxes"], [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(labels))]):
        patch.set_facecolor(color)
    ax.set_title(f"{value_col} distribution by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return fig
