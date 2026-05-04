"""Thin helpers for RCA-related flows (kept for callers that import `run_rca`)."""

from __future__ import annotations

from services.rca.bug_analysis import is_pipeline_available, run_error_fixing_analyze
from services.rca.schemas import ErrorAnalysisRequest


def run_rca(log_text: str) -> dict:
    """Backward-compatible single-string analyze (maps to error-fixing pipeline)."""
    if not is_pipeline_available():
        return {
            "framework": "rca",
            "status": "unavailable",
            "detail": "Error fixing pipeline not importable",
            "log_preview": log_text[:200],
        }
    try:
        return run_error_fixing_analyze(
            ErrorAnalysisRequest(error_message=log_text, log_file_path=None)
        )
    except Exception as e:
        return {
            "framework": "rca",
            "status": "error",
            "detail": str(e),
            "log_preview": log_text[:200],
        }
