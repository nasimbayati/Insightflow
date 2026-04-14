import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ARTIFACTS_DIRNAME = "artifacts"
RUNS_DIRNAME = "runs"
REGISTRY_FILENAME = "run_registry.jsonl"


def get_artifacts_root(project_root):
    return Path(project_root) / ARTIFACTS_DIRNAME


def get_registry_path(project_root):
    return get_artifacts_root(project_root) / REGISTRY_FILENAME


def ensure_artifact_directories(project_root):
    artifacts_root = get_artifacts_root(project_root)
    runs_root = artifacts_root / RUNS_DIRNAME
    artifacts_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    return artifacts_root, runs_root


def _utc_now():
    return datetime.now(timezone.utc)


def build_run_id(file_stem):
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{file_stem}_{suffix}"


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.head(25).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(25).to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def append_registry_entry(project_root, entry):
    registry_path = get_registry_path(project_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(entry), ensure_ascii=True) + "\n")
    return registry_path


def load_recent_registry_entries(project_root, limit=5):
    registry_path = get_registry_path(project_root)
    if not registry_path.exists():
        return []

    rows = []
    with registry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows[-limit:][::-1]


def persist_run_artifacts(
    project_root,
    file_stem,
    report_payload,
    decision_brief_markdown,
    cleaned_df,
    active_df,
    rejected_rows_df=None,
):
    _, runs_root = ensure_artifact_directories(project_root)
    run_id = build_run_id(file_stem)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "report.json"
    brief_path = run_dir / "decision_brief.md"
    cleaned_csv_path = run_dir / "cleaned_dataset.csv"
    active_csv_path = run_dir / "active_view.csv"
    rejected_csv_path = run_dir / "rejected_rows.csv"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report_payload), handle, indent=2, ensure_ascii=True)

    brief_path.write_text(decision_brief_markdown, encoding="utf-8")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    active_df.to_csv(active_csv_path, index=False)

    rejected_count = 0
    if rejected_rows_df is not None and not rejected_rows_df.empty:
        rejected_rows_df.to_csv(rejected_csv_path, index=False)
        rejected_count = int(len(rejected_rows_df))
    else:
        rejected_csv_path = None

    manifest = {
        "run_id": run_id,
        "created_at": _utc_now().isoformat(),
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "brief_path": str(brief_path),
        "cleaned_csv_path": str(cleaned_csv_path),
        "active_csv_path": str(active_csv_path),
        "rejected_csv_path": str(rejected_csv_path) if rejected_csv_path else None,
        "rows_cleaned": int(len(cleaned_df)),
        "rows_active": int(len(active_df)),
        "rejected_row_count": rejected_count,
    }

    append_registry_entry(project_root, manifest)
    return manifest
