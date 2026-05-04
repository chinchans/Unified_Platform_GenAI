"""Pydantic models shared by RCA, error-fixing, and code-assistant routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorAnalysisRequest(BaseModel):
    error_message: Optional[str] = None
    log_file_path: Optional[str] = None
    openair_codebase_name: str = "openairinterface5g-develop"
    custom_deployment_context: Optional[Dict[str, Any]] = None
    crash_analysis: bool = False


class SaveAnalysisRequest(BaseModel):
    error_message: str
    log_file: str
    log_path: Optional[str] = ""
    code_dir: Optional[str] = ""
    results: Optional[Dict[str, Any]] = {}
    fix_suggestions: Optional[Dict[str, Any]] = {}


class ApplyPatchesRequest(BaseModel):
    analysis_filename: str
    selected_code_patches: List[str] = Field(default_factory=list)
    selected_config_patches: List[str] = Field(default_factory=list)
    code_dir: Optional[str] = None


class RunInvestigationRequest(BaseModel):
    analysis_filename: str


class GitCommitPushRequest(BaseModel):
    commit_message: str
    should_push: bool = False
    code_dir: Optional[str] = None
    selected_code_patches: Optional[List[str]] = Field(default_factory=list)
    selected_config_patches: Optional[List[str]] = Field(default_factory=list)
    analysis_filename: Optional[str] = None


class RunCodeReviewRequest(BaseModel):
    analysis_filename: str
    selected_layers: List[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    selected_code_patches: List[str] = Field(default_factory=list)
    selected_config_patches: List[str] = Field(default_factory=list)
    code_dir: Optional[str] = None
    continue_on_error: bool = True


class GitHistorySearchRequest(BaseModel):
    error_message: str
    top_k: int = 10
    openair_codebase_name: str = "openairinterface5g-develop"


class GitHistorySelectRequest(BaseModel):
    commit_hash: str
    error_message: Optional[str] = None
    openair_codebase_name: str = "openairinterface5g-develop"
    code_dir: Optional[str] = None
