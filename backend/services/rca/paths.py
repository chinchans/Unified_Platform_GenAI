"""Filesystem roots for RCA / error-fixing pipeline and persisted artifacts."""

from __future__ import annotations

from pathlib import Path

# backend/services/rca/paths.py -> backend/
BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR: Path = BACKEND_DIR / "resources"
RCA_LOGS_DIR: Path = RESOURCES_DIR / "rca_logs"
BUG_HISTORY_DIR: Path = RESOURCES_DIR / "bug_history"
CODE_REVIEW_HISTORY_DIR: Path = RESOURCES_DIR / "code_review_history"
PIPELINE_DIR: Path = Path(__file__).resolve().parent / "Error_fixing_pipelin"


def ensure_resource_dirs() -> None:
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    RCA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    BUG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CODE_REVIEW_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
