import json
from datetime import datetime, timezone
from pathlib import Path

from modules.artifacts import get_artifacts_root


MONITORING_DIRNAME = "monitoring"
EVENT_LOG_FILENAME = "pipeline_events.jsonl"


def get_monitoring_log_path(project_root):
    monitoring_root = get_artifacts_root(project_root) / MONITORING_DIRNAME
    monitoring_root.mkdir(parents=True, exist_ok=True)
    return monitoring_root / EVENT_LOG_FILENAME


def _event_timestamp():
    return datetime.now(timezone.utc).isoformat()


def log_monitoring_event(project_root, event_type, status="INFO", payload=None, run_id=None):
    log_path = get_monitoring_log_path(project_root)
    event = {
        "timestamp": _event_timestamp(),
        "event_type": event_type,
        "status": status,
        "run_id": run_id,
        "payload": payload or {},
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    return event


def load_recent_monitoring_events(project_root, limit=20):
    log_path = get_monitoring_log_path(project_root)
    if not log_path.exists():
        return []

    rows = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows[-limit:][::-1]


def build_monitoring_snapshot(project_root, limit=20):
    events = load_recent_monitoring_events(project_root, limit=limit)
    counts = {}
    for event in events:
        counts[event["status"]] = counts.get(event["status"], 0) + 1

    return {
        "recent_events": events,
        "event_counts": counts,
        "latest_event": events[0] if events else None,
    }
