"""Code assistant API (bug history, patches, investigation) — legacy URL compatible."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from services.history_logger import append_history_record, load_history_entries
from services.rca.paths import BUG_HISTORY_DIR, PIPELINE_DIR
from services.rca.schemas import (
    ApplyPatchesRequest,
    GitCommitPushRequest,
    GitHistorySearchRequest,
    GitHistorySelectRequest,
    RunInvestigationRequest,
)

router = APIRouter()


def _history_dir() -> Any:
    d = BUG_HISTORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_oai_git_repo(code_dir: str | None, openair_codebase_name: str) -> str | None:
    error_pipeline_path = PIPELINE_DIR.resolve()
    possible_paths: List[str] = []

    if code_dir and os.path.exists(code_dir):
        possible_paths.append(code_dir)

    pyqt_style = os.path.join("Error_fixing_pipelin", openair_codebase_name)
    if os.path.exists(pyqt_style):
        possible_paths.append(pyqt_style)

    cwd = os.getcwd()
    possible_paths.extend(
        [
            os.path.join(cwd, openair_codebase_name),
            os.path.join(cwd, "Error_fixing_pipelin", openair_codebase_name),
            str(error_pipeline_path / openair_codebase_name),
            str(error_pipeline_path / "openairinterface5g-develop"),
            str(error_pipeline_path / "openairinterface5g-test"),
        ]
    )

    for path in possible_paths:
        abs_path = os.path.abspath(os.path.normpath(path))
        if not (os.path.exists(abs_path) and os.path.isdir(abs_path)):
            continue
        if not os.path.exists(os.path.join(abs_path, ".git")):
            continue
        try:
            check_result = subprocess.run(
                ["git", "status"],
                cwd=abs_path,
                capture_output=True,
                timeout=3,
            )
            if check_result.returncode == 0:
                return abs_path
        except Exception:
            continue
    return None


def _extract_patch_payload_from_commit(commit: Dict[str, Any]) -> Dict[str, Any]:
    code_patches = commit.get("code_patches", []) or []
    config_patches = commit.get("config_patches", []) or []

    normalized_code_patches: List[Dict[str, Any]] = []
    for patch in code_patches:
        normalized_code_patches.append(
            {
                "function_name": patch.get("function") or patch.get("function_name", "unknown_function"),
                "file_path": patch.get("file") or patch.get("file_path", ""),
                "description": patch.get("description", "Patch from git history"),
                "patch_type": patch.get("patch_type", "modification"),
                "line_number": patch.get("line_number", ""),
                "original_code": patch.get("original_code", ""),
                "patched_code": patch.get("patched_code") or patch.get("suggested_code", ""),
                "suggested_code": patch.get("suggested_code") or patch.get("patched_code", ""),
            }
        )

    normalized_config_patches: List[Dict[str, Any]] = []
    for patch in config_patches:
        normalized_config_patches.append(
            {
                "config_name": patch.get("parameter") or patch.get("config_name", "unknown_parameter"),
                "file_path": patch.get("file") or patch.get("file_path", ""),
                "description": patch.get("description", "Config patch from git history"),
                "current_value": patch.get("current_value", ""),
                "new_value": patch.get("new_value") or patch.get("suggested_value", ""),
                "suggested_value": patch.get("suggested_value") or patch.get("new_value", ""),
                "patch_type": patch.get("patch_type", "set_value"),
                "line_number": patch.get("line_number", ""),
            }
        )

    return {
        "code_patches": normalized_code_patches,
        "config_patches": normalized_config_patches,
    }


@router.get("/bug-history")
async def get_bug_history():
    try:
        history_dir = _history_dir()
        history_files = [f for f in history_dir.iterdir() if f.is_file() and f.suffix == ".json"]
        history_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        history_list: List[Dict[str, Any]] = []
        for file_path in history_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                error_message = data.get("error_message", "Unknown error") or "Unknown error"
                timestamp = data.get("timestamp", "")
                display_text = error_message[:100] + "..." if len(error_message) > 100 else error_message
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        display_text = f"[{dt.strftime('%Y-%m-%d %H:%M')}] {display_text}"
                    except Exception:
                        pass
                results = data.get("results", {})
                phase3_fixes = results.get("phase3_fixes", {})
                fix_suggestion = phase3_fixes.get("fix_suggestion", {})
                code_count = len(fix_suggestion.get("code_patches", []))
                config_count = len(fix_suggestion.get("config_patches", []))
                deployment_context = data.get("deployment_context", {})
                if not isinstance(deployment_context, dict):
                    deployment_context = {}
                log_error_kind = (
                    data.get("log_error_kind")
                    or results.get("log_error_kind")
                    or deployment_context.get("log_error_kind")
                    or "other"
                )
                if log_error_kind not in {"runtime", "build", "cmake", "dependency", "other"}:
                    log_error_kind = "other"
                display_text += f" [Code:{code_count}, Config:{config_count}]"
                history_list.append(
                    {
                        "filename": file_path.name,
                        "display_text": display_text,
                        "error_message": error_message,
                        "timestamp": timestamp,
                        "log_error_kind": log_error_kind,
                        "code_patches_count": code_count,
                        "config_patches_count": config_count,
                    }
                )
            except Exception:
                continue
        return {"success": True, "history": history_list, "count": len(history_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bug history: {str(e)}")


@router.get("/load-analysis/{filename}")
async def load_bug_analysis(filename: str):
    try:
        history_dir = _history_dir()
        file_path = history_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        phase3_fixes = results.get("phase3_fixes", {})
        fix_suggestion = phase3_fixes.get("fix_suggestion", {})
        code_patches = fix_suggestion.get("code_patches", [])
        config_patches = fix_suggestion.get("config_patches", [])
        formatted_code_patches = []
        for patch in code_patches:
            function_name = patch.get("function_name", "Unknown")
            file_path_str = patch.get("file_path", "Unknown")
            file_name = os.path.basename(file_path_str)
            formatted_code_patches.append(
                {
                    "function_name": function_name,
                    "file_path": file_path_str,
                    "file_name": file_name,
                    "display_text": f"{function_name} ({file_name})",
                    "description": patch.get("description", "No description"),
                    "patch_type": patch.get("patch_type", "modification"),
                    "line_number": patch.get("line_number") or patch.get("line_numbers", ""),
                    "original_code": patch.get("original_code", ""),
                    "suggested_code": patch.get("suggested_code")
                    or patch.get("patched_code")
                    or patch.get("new_code", ""),
                    "patch_data": patch,
                }
            )
        formatted_config_patches = []
        for patch in config_patches:
            param_name = patch.get("config_name", patch.get("parameter_name", "Unknown"))
            file_path_str = patch.get("file_path", "Unknown")
            file_name = os.path.basename(file_path_str)
            formatted_config_patches.append(
                {
                    "config_name": param_name,
                    "file_path": file_path_str,
                    "file_name": file_name,
                    "display_text": f"{param_name} ({file_name})",
                    "description": patch.get("description", "No description"),
                    "current_value": patch.get("current_value", ""),
                    "suggested_value": patch.get("suggested_value") or patch.get("new_value", ""),
                    "patch_data": patch,
                }
            )
        error_message = data.get("error_message", "Unknown error") or "Unknown error"
        log_file = data.get("log_file", "N/A")
        log_path = data.get("log_path", "")
        timestamp = data.get("timestamp", "N/A")
        code_dir = data.get("code_dir", "")
        results_data = data.get("results", {})
        terminal_commands: List[Dict[str, Any]] = []
        phase4_commands = results.get("phase4_commands", {})
        if phase4_commands:
            terminal_commands_data = phase4_commands.get("terminal_commands", [])
            if terminal_commands_data:
                for cmd in terminal_commands_data:
                    if isinstance(cmd, dict):
                        terminal_commands.append(cmd)
                    else:
                        terminal_commands.append(
                            {"command": str(cmd), "explanation": "Investigation command"}
                        )
        git_metadata = data.get("git_metadata", {})
        from_git_history = data.get("from_git_history", False)
        return {
            "success": True,
            "analysis": {
                "error_message": error_message,
                "log_file": log_file,
                "log_path": log_path,
                "timestamp": timestamp,
                "code_dir": code_dir,
                "results": results_data,
                "log_error_kind": (
                    data.get("log_error_kind")
                    or results_data.get("log_error_kind")
                    or data.get("deployment_context", {}).get("log_error_kind")
                    or "other"
                ),
                "error_display": f"Error Message:\n{error_message}\n\nLog File: {log_file}\nTimestamp: {timestamp}\n",
                "code_patches": formatted_code_patches,
                "config_patches": formatted_config_patches,
                "terminal_commands": terminal_commands,
                "raw_data": data,
                "git_metadata": git_metadata if git_metadata else None,
                "from_git_history": from_git_history,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load bug analysis: {str(e)}")


@router.post("/apply-patches")
async def apply_patches(request: ApplyPatchesRequest):
    try:
        history_dir = _history_dir()
        file_path = history_dir / request.analysis_filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {request.analysis_filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        code_dir = request.code_dir or analysis_data.get("code_dir", "")
        if not code_dir:
            raise HTTPException(
                status_code=400,
                detail="Code directory is required. Please ensure the analysis includes a code directory.",
            )
        if not request.selected_code_patches and not request.selected_config_patches:
            raise HTTPException(status_code=400, detail="No patches selected for application")

        pipeline_path = PIPELINE_DIR.resolve()
        resources_dir = pipeline_path / "resources"
        resources_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_suggestions_file = resources_dir / f"fix_suggestions_{ts}.json"

        results = analysis_data.get("results", {})
        phase3_fixes = results.get("phase3_fixes", {})
        fix_suggestion = phase3_fixes.get("fix_suggestion", {})
        code_patches = fix_suggestion.get("code_patches", [])
        config_patches = fix_suggestion.get("config_patches", [])

        selected_code_patch_data = []
        for idx, patch in enumerate(code_patches):
            function_name = patch.get("function_name", "Unknown")
            file_path_str = patch.get("file_path", "Unknown")
            file_name = os.path.basename(file_path_str)
            display_text = f"{function_name} ({file_name})"
            if display_text in request.selected_code_patches:
                selected_code_patch_data.append(patch)

        selected_config_patch_data = []
        for idx, patch in enumerate(config_patches):
            param_name = patch.get("config_name", patch.get("parameter_name", "Unknown"))
            file_path_str = patch.get("file_path", "Unknown")
            file_name = os.path.basename(file_path_str)
            display_text = f"{param_name} ({file_name})"
            if display_text in request.selected_config_patches:
                selected_config_patch_data.append(patch)

        filtered_fix_suggestion = {
            "code_dir": code_dir,
            "fix_suggestion": {
                "code_dir": code_dir,
                "code_patches": selected_code_patch_data,
                "config_patches": selected_config_patch_data,
            },
        }
        with open(fix_suggestions_file, "w", encoding="utf-8") as f:
            json.dump(filtered_fix_suggestion, f, indent=2)

        absolute_file_path = str(fix_suggestions_file.resolve())
        if not fix_suggestions_file.exists():
            raise HTTPException(
                status_code=500, detail=f"Failed to create fix_suggestions file at {absolute_file_path}"
            )

        original_cwd = os.getcwd()
        try:
            os.chdir(pipeline_path)
            if not os.path.exists(absolute_file_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Fix suggestions file not found after directory change: {absolute_file_path}",
                )
            from services.rca.Error_fixing_pipelin.unified_patch_applicator import (
                UnifiedPatchApplicator,
            )

            applicator = UnifiedPatchApplicator(absolute_file_path)
            result = applicator.apply_all_patches(dry_run=False, backup=True)
        finally:
            os.chdir(original_cwd)

        return {
            "success": result.get("success", False),
            "total_applied": result.get("total_applied", 0),
            "total_failed": result.get("total_failed", 0),
            "applied_code_patches": result.get("applied_code_patches", []),
            "applied_config_patches": result.get("applied_config_patches", []),
            "failed_patches": result.get("failed_patches", []),
            "backup_location": result.get("backup_location", ""),
            "message": f"Applied {result.get('total_applied', 0)} patches successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply patches: {str(e)}")


@router.post("/run-investigation")
async def run_investigation_commands(request: RunInvestigationRequest):
    try:
        history_dir = _history_dir()
        file_path = history_dir / request.analysis_filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {request.analysis_filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        results = analysis_data.get("results", {})
        terminal_commands: List[Dict[str, Any]] = []
        phase4_commands = results.get("phase4_commands", {})
        if phase4_commands:
            terminal_commands_data = phase4_commands.get("terminal_commands", [])
            if isinstance(terminal_commands_data, dict):
                terminal_commands_data = terminal_commands_data.get("terminal_commands", [])
            if terminal_commands_data:
                for cmd in terminal_commands_data:
                    if isinstance(cmd, dict):
                        terminal_commands.append(cmd)
                    else:
                        terminal_commands.append(
                            {"command": str(cmd), "explanation": "Investigation command"}
                        )
        if not terminal_commands:
            return {
                "success": False,
                "message": "No investigation commands available for this analysis.",
                "commands": [],
                "results": [],
            }
        command_results = []
        for i, cmd_info in enumerate(terminal_commands, 1):
            command = cmd_info["command"]
            explanation = cmd_info.get("explanation", "No explanation provided")
            result_entry: Dict[str, Any] = {
                "command_number": i,
                "total_commands": len(terminal_commands),
                "command": command,
                "explanation": explanation,
                "status": "pending",
                "output": "",
                "stderr": "",
                "return_code": None,
                "error": None,
            }
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result_entry["return_code"] = result.returncode
                result_entry["output"] = result.stdout
                result_entry["stderr"] = result.stderr
                result_entry["status"] = "success" if result.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                result_entry["status"] = "timeout"
                result_entry["error"] = "Command timed out after 30 seconds"
            except Exception as e:
                result_entry["status"] = "error"
                result_entry["error"] = str(e)
            command_results.append(result_entry)
        return {
            "success": True,
            "message": f"Executed {len(terminal_commands)} commands",
            "commands": terminal_commands,
            "results": command_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run investigation commands: {str(e)}")


@router.post("/git-history/search")
async def search_git_history_fixes(request: GitHistorySearchRequest):
    try:
        from services.rca.Error_fixing_pipelin.smart_commit_selector import CommitSearcher, SmartSelector

        embeddings_dir = PIPELINE_DIR / "resources" / "embeddings" / "embeddings"
        if not embeddings_dir.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Git history embeddings directory not found: {embeddings_dir}",
            )

        original_cwd = os.getcwd()
        try:
            os.chdir(str(PIPELINE_DIR))
            searcher = CommitSearcher(
                embeddings_dir=str(embeddings_dir),
                validate_commits=False,
                openair_codebase_file_name=request.openair_codebase_name,
            )
            results = searcher.search(request.error_message, top_k=request.top_k)
            selector = SmartSelector(use_llm=False)
            selection_result = selector.select_best_fix(request.error_message, results)
        finally:
            os.chdir(original_cwd)

        formatted_results = []
        for item in results:
            formatted_results.append(
                {
                    "commit_hash": item.get("commit_hash"),
                    "commit_hash_short": item.get("commit_hash_short"),
                    "subject": item.get("subject"),
                    "author_name": item.get("author_name"),
                    "date_iso": item.get("date_iso"),
                    "similarity": item.get("similarity"),
                    "is_rca_commit": bool(item.get("is_rca_commit", False)),
                    "keywords": item.get("keywords", []),
                    "files_changed": item.get("files_changed", []),
                }
            )

        selected_commit = selection_result.get("commit")
        return {
            "success": True,
            "error_message": request.error_message,
            "selection_status": selection_result.get("status"),
            "confidence": selection_result.get("confidence"),
            "reasoning": selection_result.get("reasoning"),
            "selected_commit": {
                "commit_hash": selected_commit.get("commit_hash"),
                "commit_hash_short": selected_commit.get("commit_hash_short"),
                "subject": selected_commit.get("subject"),
                "is_rca_commit": bool(selected_commit.get("is_rca_commit", False)),
                "similarity": selected_commit.get("similarity"),
            }
            if isinstance(selected_commit, dict)
            else None,
            "results": formatted_results,
            "count": len(formatted_results),
            "selection_result": selection_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search git history: {str(e)}")


@router.post("/git-history/select")
async def select_git_history_fix(request: GitHistorySelectRequest):
    try:
        metadata_path = PIPELINE_DIR / "resources" / "embeddings" / "embeddings" / "git_commit_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=500, detail=f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        commit = None
        requested_hash = request.commit_hash.strip()
        requested_hash_lower = requested_hash.lower()
        for item in metadata:
            full_hash = str(item.get("commit_hash", "")).lower()
            short_hash = str(item.get("commit_hash_short", "")).lower()
            if (
                full_hash == requested_hash_lower
                or short_hash == requested_hash_lower
                or full_hash.startswith(requested_hash_lower)
            ):
                commit = item
                break

        if not commit:
            raise HTTPException(status_code=404, detail=f"Commit not found in metadata: {request.commit_hash}")

        git_repo = _resolve_oai_git_repo(request.code_dir, request.openair_codebase_name)
        git_diff = None
        if git_repo:
            hash_to_use = commit.get("commit_hash") or commit.get("commit_hash_short") or request.commit_hash
            try:
                show_result = subprocess.run(
                    ["git", "show", hash_to_use],
                    capture_output=True,
                    text=True,
                    cwd=git_repo,
                    timeout=30,
                )
                if show_result.returncode == 0:
                    git_diff = show_result.stdout
            except Exception:
                git_diff = None

        patch_payload = _extract_patch_payload_from_commit(commit)
        fix_suggestions_payload = {
            "from_git_history": True,
            "selection_result": {
                "status": "selected",
                "confidence": "manual",
                "commit": commit,
                "reasoning": "Commit explicitly selected by user",
            },
            "git_diff": git_diff,
            "current_branch": None,
            "files_changed_summary": commit.get("files_changed", []),
            "confidence": "manual",
            "score": commit.get("similarity"),
            "reasoning": "Commit explicitly selected by user",
            "full_commit": commit,
            "fix_suggestion": {
                "code_patches": patch_payload.get("code_patches", []),
                "config_patches": patch_payload.get("config_patches", []),
                "reason": f"Selected from git history commit {commit.get('commit_hash_short')}",
            },
        }

        save_analysis_payload = {
            "error_message": request.error_message or "Git history selected fix",
            "log_file": "git_history_selection",
            "log_path": "",
            "code_dir": request.code_dir or "",
            "results": {
                "phase3_fixes": {
                    "fix_suggestion": fix_suggestions_payload["fix_suggestion"],
                }
            },
            "fix_suggestions": fix_suggestions_payload,
        }

        return {
            "success": True,
            "message": "Git history fix prepared",
            "commit": {
                "commit_hash": commit.get("commit_hash"),
                "commit_hash_short": commit.get("commit_hash_short"),
                "subject": commit.get("subject"),
                "author_name": commit.get("author_name"),
                "date_iso": commit.get("date_iso"),
                "is_rca_commit": bool(commit.get("is_rca_commit", False)),
            },
            "git_diff_available": bool(git_diff),
            "fix_suggestions": fix_suggestions_payload,
            "save_analysis_payload": save_analysis_payload,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select git history fix: {str(e)}")


@router.post("/git-commit-push")
async def git_commit_and_push(request: GitCommitPushRequest):
    result: Dict[str, Any] = {
        "success": False,
        "committed": False,
        "pushed": False,
        "commit_hash": None,
        "error": None,
    }
    try:
        code_dir = request.code_dir
        if not code_dir:
            resources_dir = PIPELINE_DIR / "resources"
            if resources_dir.exists():
                json_files = [
                    f
                    for f in os.listdir(resources_dir)
                    if f.startswith("fix_suggestions_") and f.endswith(".json")
                ]
                if json_files:
                    json_files.sort(reverse=True)
                    fix_suggestions_file = resources_dir / json_files[0]
                    with open(fix_suggestions_file, "r", encoding="utf-8") as f:
                        suggestions_data = json.load(f)
                    code_dir = suggestions_data.get("code_dir", None)
        openair_codebase_file_name = (
            os.path.basename(code_dir.rstrip(os.sep)) if code_dir else "openairinterface5g-develop"
        )
        error_pipeline_path = PIPELINE_DIR.resolve()
        possible_paths: List[str] = []
        if code_dir and os.path.exists(code_dir):
            possible_paths.append(code_dir)
        pyqt_style = os.path.join("Error_fixing_pipelin", openair_codebase_file_name)
        if os.path.exists(pyqt_style):
            possible_paths.append(pyqt_style)
        cwd = os.getcwd()
        possible_paths.extend(
            [
                os.path.join(cwd, openair_codebase_file_name),
                os.path.join(cwd, "Error_fixing_pipelin", openair_codebase_file_name),
            ]
        )
        possible_paths.extend(
            [
                str(error_pipeline_path / openair_codebase_file_name),
                str(error_pipeline_path / "openairinterface5g-develop"),
                str(error_pipeline_path / "openairinterface5g-test"),
            ]
        )
        oai_dir = None
        for path in possible_paths:
            abs_path = os.path.abspath(os.path.normpath(path))
            if os.path.exists(abs_path) and os.path.isdir(abs_path) and os.path.exists(
                os.path.join(abs_path, ".git")
            ):
                try:
                    check_result = subprocess.run(
                        ["git", "status"], cwd=abs_path, capture_output=True, timeout=2
                    )
                    if check_result.returncode == 0:
                        oai_dir = abs_path
                        break
                except Exception:
                    continue
        if not oai_dir:
            result["error"] = (
                "Git repository not found. Please ensure the source code directory is a valid Git repository.\n\n"
                "Tried paths:\n" + "\n".join([f"  - {p}" for p in possible_paths])
            )
            return result

        git_check = subprocess.run(
            ["git", "status"], capture_output=True, text=True, cwd=oai_dir, timeout=10
        )
        if git_check.returncode != 0:
            result["error"] = "Not a Git repository or Git not available"
            return result

        add_result = subprocess.run(
            ["git", "add", "."], capture_output=True, text=True, cwd=oai_dir, timeout=10
        )
        if add_result.returncode != 0:
            result["error"] = f"Failed to add changes: {add_result.stderr}"
            return result

        diff_check = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], capture_output=True, text=True, cwd=oai_dir, timeout=10
        )
        if diff_check.returncode == 0:
            result["error"] = (
                "No changes to commit - patches may have been applied outside the Git repository "
                "or files are unchanged"
            )
            return result

        commit_result = subprocess.run(
            ["git", "commit", "-m", request.commit_message],
            capture_output=True,
            text=True,
            cwd=oai_dir,
            timeout=10,
        )
        if commit_result.returncode != 0:
            result["error"] = f"Failed to commit: {commit_result.stderr}"
            return result
        result["committed"] = True

        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=oai_dir, timeout=5
        )
        full_hash = None
        if hash_result.returncode == 0:
            full_hash = hash_result.stdout.strip()
            result["commit_hash"] = full_hash[:8]
            result["full_commit_hash"] = full_hash

        try:
            commit_record = {
                "activity_type": "git-commit",
                "activity_label": "Git Commit",
                "title": request.analysis_filename or "Git Commit",
                "timestamp": datetime.utcnow().isoformat(),
                "record": {
                    "commit_hash": result.get("commit_hash"),
                    "full_commit_hash": full_hash,
                    "commit_message": request.commit_message,
                    "code_dir": code_dir or str(oai_dir) if oai_dir else None,
                    "should_push": request.should_push,
                    "pushed": False,
                    "analysis_filename": request.analysis_filename,
                    "selected_code_patches": request.selected_code_patches or [],
                    "selected_config_patches": request.selected_config_patches or [],
                },
            }
            append_history_record("code_assistant_history.json", commit_record)
        except Exception:
            pass

        if request.should_push and result["committed"]:
            push_result = subprocess.run(
                ["git", "push"], capture_output=True, text=True, cwd=oai_dir, timeout=30
            )
            if push_result.returncode == 0:
                result["pushed"] = True
                try:
                    history_file = (
                        Path(__file__).resolve().parent.parent / "resources" / "history" / "code_assistant_history.json"
                    )
                    if history_file.exists() and full_hash:
                        entries = load_history_entries("code_assistant_history.json")
                        if entries and entries[-1].get("record", {}).get("full_commit_hash") == full_hash:
                            entries[-1]["record"]["pushed"] = True
                            entries[-1]["record"]["push_output"] = (
                                push_result.stdout.strip() if push_result.stdout else ""
                            )
                            history_file.parent.mkdir(parents=True, exist_ok=True)
                            with history_file.open("w", encoding="utf-8") as fp:
                                json.dump(entries, fp, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            else:
                result["error"] = f"Commit successful but push failed: {push_result.stderr}"
                result["success"] = True
                return result

        if result.get("committed"):
            result["embedding_updated"] = False
        result["success"] = True
        return result
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "committed": False,
            "pushed": False,
            "commit_hash": None,
            "error": "Git operation timed out. Please try again.",
        }
    except Exception as e:
        return {
            "success": False,
            "committed": False,
            "pushed": False,
            "commit_hash": None,
            "error": str(e),
        }
