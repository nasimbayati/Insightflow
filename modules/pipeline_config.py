from dataclasses import asdict, dataclass, field


UNSPECIFIED_OPTION = "Auto / not specified"


@dataclass(frozen=True)
class ColumnRoles:
    id_columns: tuple[str, ...] = field(default_factory=tuple)
    time_column: str | None = None
    metric_column: str | None = None
    segment_column: str | None = None
    outcome_column: str | None = None

    def to_dict(self):
        return asdict(self)

    def has_user_roles(self):
        return any(
            [
                self.id_columns,
                self.time_column,
                self.metric_column,
                self.segment_column,
                self.outcome_column,
            ]
        )

    def assumption_items(self):
        items = []
        if self.id_columns:
            items.append(f"ID column(s): {', '.join(self.id_columns)}")
        if self.time_column:
            items.append(f"Time axis: {self.time_column}")
        if self.metric_column:
            items.append(f"Primary metric: {self.metric_column}")
        if self.segment_column:
            items.append(f"Primary segment: {self.segment_column}")
        if self.outcome_column:
            items.append(f"Outcome/target: {self.outcome_column}")
        if not items:
            items.append("Column roles: auto-selected from data types and safeguards")
        return items


@dataclass(frozen=True)
class PipelinePreferences:
    duplicate_subset: tuple[str, ...] | None = None
    duplicate_rule_mode: str = "exact"
    trend_date_column: str | None = None
    trend_value_column: str | None = None
    column_roles: ColumnRoles = field(default_factory=ColumnRoles)

    def preferred_time_column(self):
        return self.column_roles.time_column or self.trend_date_column

    def preferred_metric_column(self):
        return self.column_roles.metric_column or self.trend_value_column

    def preferred_segment_column(self):
        return self.column_roles.segment_column

    def to_dict(self):
        return {
            "duplicate_subset": list(self.duplicate_subset) if self.duplicate_subset else None,
            "duplicate_rule_mode": self.duplicate_rule_mode,
            "trend_date_column": self.trend_date_column,
            "trend_value_column": self.trend_value_column,
            "column_roles": self.column_roles.to_dict(),
        }

    def __getitem__(self, key):
        return self.to_dict()[key]

    def get(self, key, default=None):
        return self.to_dict().get(key, default)


def coerce_pipeline_preferences(value):
    if isinstance(value, PipelinePreferences):
        return value

    if value is None:
        return PipelinePreferences()

    roles_value = value.get("column_roles", {}) if isinstance(value, dict) else {}
    if isinstance(roles_value, ColumnRoles):
        column_roles = roles_value
    else:
        column_roles = ColumnRoles(
            id_columns=tuple(roles_value.get("id_columns", ()) or ()),
            time_column=roles_value.get("time_column"),
            metric_column=roles_value.get("metric_column"),
            segment_column=roles_value.get("segment_column"),
            outcome_column=roles_value.get("outcome_column"),
        )

    duplicate_subset = value.get("duplicate_subset") if isinstance(value, dict) else None
    return PipelinePreferences(
        duplicate_subset=tuple(duplicate_subset) if duplicate_subset else None,
        duplicate_rule_mode=value.get("duplicate_rule_mode", "exact") if isinstance(value, dict) else "exact",
        trend_date_column=value.get("trend_date_column") if isinstance(value, dict) else None,
        trend_value_column=value.get("trend_value_column") if isinstance(value, dict) else None,
        column_roles=column_roles,
    )
