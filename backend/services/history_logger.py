"""Append-only JSON history under backend/resources/history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _resources_dir() -> Path:
    d = _backend_dir() / "resources"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history_dir() -> Path:
    h = _resources_dir() / "history"
    h.mkdir(parents=True, exist_ok=True)
    return h


def _load_history(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    try:
        with file_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_history(file_path: Path, entries: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(entries, fp, ensure_ascii=False, indent=2)


def append_history_record(
    filename: str,
    record: Dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> Path:
    file_path = _history_dir() / filename
    record = dict(record)
    if "timestamp" not in record or not record["timestamp"]:
        record["timestamp"] = (timestamp or datetime.utcnow()).isoformat()
    entries = _load_history(file_path)
    entries.append(record)
    _write_history(file_path, entries)
    return file_path


def load_history_entries(filename: str) -> List[Dict[str, Any]]:
    return _load_history(_history_dir() / filename)
