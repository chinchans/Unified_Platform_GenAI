"""Legacy-compatible `/api/error-fixing` entrypoints."""

from fastapi import APIRouter, HTTPException

from services.rca.bug_analysis import is_pipeline_available, run_error_fixing_analyze
from services.rca.schemas import ErrorAnalysisRequest

router = APIRouter()


@router.post("/analyze")
async def analyze_error(request: ErrorAnalysisRequest):
    if not is_pipeline_available():
        raise HTTPException(status_code=503, detail="Error fixing pipeline not available")
    try:
        return run_error_fixing_analyze(request)
    except RuntimeError as e:
        msg = str(e)
        code = 503 if "not available" in msg.lower() else 500
        raise HTTPException(status_code=code, detail=msg) from e
