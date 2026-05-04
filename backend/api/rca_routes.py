import json
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.rca.bug_analysis import is_pipeline_available, run_error_fixing_analyze
from services.rca.paths import RCA_LOGS_DIR, ensure_resource_dirs
from services.rca.save_bug_analysis import save_bug_analysis_record
from services.rca.schemas import ErrorAnalysisRequest, SaveAnalysisRequest

router = APIRouter()


class RcaAnalyzeBody(BaseModel):
    """Bug analysis: legacy `log_text` and/or uploaded log / full pipeline fields."""

    log_text: Optional[str] = None
    log_file_name: Optional[str] = None
    error_message: Optional[str] = None
    log_file_path: Optional[str] = None
    openair_codebase_name: str = "openairinterface5g-develop"
    custom_deployment_context: Optional[Dict[str, Any]] = None
    crash_analysis: bool = False


@router.post("/upload-logs")
async def upload_rca_logs(file: UploadFile = File(...)):
    try:
        ensure_resource_dirs()
        upload_dir = RCA_LOGS_DIR
        metadata_file = upload_dir / "file_metadata.json"
        file_metadata: Dict[str, Any] = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                file_metadata = json.load(f)
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = upload_dir / safe_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_metadata[file.filename] = {
            "file_id": file_id,
            "saved_name": safe_filename,
            "file_path": str(file_path),
            "upload_date": datetime.now().isoformat(),
        }
        uploaded_file = {
            "file_id": file_id,
            "original_name": file.filename,
            "saved_name": safe_filename,
            "file_path": str(file_path),
            "size": file_path.stat().st_size,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(file_metadata, f, indent=2)
        return {
            "success": True,
            "file": uploaded_file,
            "files": [uploaded_file],
            "message": "Successfully uploaded 1 log file",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload log files: {str(e)}")


@router.post("/analyze")
async def analyze_rca(payload: RcaAnalyzeBody):
    if not is_pipeline_available():
        raise HTTPException(status_code=503, detail="Error fixing pipeline not available")

    ensure_resource_dirs()
    log_file_path = payload.log_file_path
    if payload.log_file_name and not log_file_path:
        metadata_file = RCA_LOGS_DIR / "file_metadata.json"
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="No log files have been uploaded yet")
        with open(metadata_file, "r", encoding="utf-8") as f:
            file_metadata = json.load(f)
        if payload.log_file_name not in file_metadata:
            raise HTTPException(status_code=404, detail=f"Log file not found: {payload.log_file_name}")
        log_file_path = file_metadata[payload.log_file_name]["file_path"]

    error_message = payload.error_message or payload.log_text
    req = ErrorAnalysisRequest(
        error_message=error_message,
        log_file_path=log_file_path,
        openair_codebase_name=payload.openair_codebase_name,
        custom_deployment_context=payload.custom_deployment_context,
        crash_analysis=payload.crash_analysis,
    )
    if not req.crash_analysis and not req.error_message and not req.log_file_path:
        raise HTTPException(
            status_code=400,
            detail="Provide error_message, log_text, log_file_path, or log_file_name (after upload).",
        )

    try:
        return run_error_fixing_analyze(req)
    except RuntimeError as e:
        msg = str(e)
        code = 503 if "not available" in msg.lower() else 500
        raise HTTPException(status_code=code, detail=msg) from e


@router.post("/save-analysis")
async def save_rca_analysis(request: SaveAnalysisRequest):
    try:
        return save_bug_analysis_record(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save bug analysis: {str(e)}") from e
