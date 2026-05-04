"""Code review API routes (Layer 1-4 orchestration)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.code_review_runner import run_code_review
from services.rca.schemas import RunCodeReviewRequest

router = APIRouter()


@router.post("/run")
async def run_code_review_layers(request: RunCodeReviewRequest):
    try:
        return run_code_review(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run code review: {exc}") from exc
