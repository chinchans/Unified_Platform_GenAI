"""Persist bug analysis JSON to backend/resources/bug_history (legacy-compatible)."""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from services.history_logger import append_history_record
from services.rca.paths import BUG_HISTORY_DIR
from services.rca.schemas import SaveAnalysisRequest


def _extract_code_from_git_diff(
    git_diff_data: str, file_path: str, function_name: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    if not git_diff_data:
        return "", "", None
    try:
        file_basename = os.path.basename(file_path)
        file_pattern = r"diff --git a/[^\s]+ b/([^\s]+)"
        diff_sections = re.split(file_pattern, git_diff_data)
        target_diff = None
        full_file_path = None
        for i in range(1, len(diff_sections), 2):
            if i + 1 < len(diff_sections):
                current_file = diff_sections[i]
                if file_basename in current_file:
                    target_diff = diff_sections[i + 1]
                    full_file_path = current_file
                    break
        if not target_diff:
            return "", "", None
        original_code, patched_code = _extract_code_with_more_context(target_diff)
        return original_code, patched_code, full_file_path
    except Exception as e:
        print(f"Error extracting code from git diff: {e}")
        return "", "", None


def _extract_code_with_more_context(diff_section: str) -> Tuple[str, str]:
    lines = diff_section.split("\n")
    original_lines: list = []
    patched_lines: list = []
    for line in lines:
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            original_lines.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            patched_lines.append(line[1:])
        elif line.startswith(" "):
            context_line = line[1:]
            original_lines.append(context_line)
            patched_lines.append(context_line)
    return "\n".join(original_lines), "\n".join(patched_lines)


def _extract_config_from_git_diff(git_diff_data: str, param_name: str) -> Tuple[str, str]:
    if not git_diff_data:
        return "N/A", "N/A"
    try:
        lines = git_diff_data.split("\n")
        old_value = None
        new_value = None
        i = 0
        while i < len(lines):
            line = lines[i]
            if param_name in line:
                if line.startswith("-") and not line.startswith("---"):
                    old_line = line[1:].strip()
                    old_value = _extract_value_from_config_line(old_line)
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if next_line.strip() == "":
                            j += 1
                            continue
                        if (
                            param_name in next_line
                            and next_line.startswith("+")
                            and not next_line.startswith("+++")
                        ):
                            new_line = next_line[1:].strip()
                            new_value = _extract_value_from_config_line(new_line)
                            i = j
                            break
                        j += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    if new_value is None:
                        new_line = line[1:].strip()
                        new_value = _extract_value_from_config_line(new_line)
            i += 1
        return old_value or "N/A", new_value or "N/A"
    except Exception as e:
        print(f"Error extracting config from git diff: {e}")
        return "N/A", "N/A"


def _extract_value_from_config_line(line: str) -> str:
    match = re.search(r'=\s*"([^"]+)"', line)
    if match:
        return match.group(1)
    match = re.search(r"=\s*([^;]+);", line)
    if match:
        return match.group(1).strip()
    match = re.search(r'ipv4\s*=\s*"([^"]+)"', line)
    if match:
        return match.group(1)
    return line.strip()


def save_bug_analysis_record(request: SaveAnalysisRequest) -> Dict[str, Any]:
    history_dir = BUG_HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = history_dir / f"bug_analysis_{timestamp}.json"

    existing_rca_results: Dict[str, Any] = {}
    fs = request.fix_suggestions or {}
    if fs.get("from_git_history", False):
        try:
            json_files = list(history_dir.glob("bug_analysis_*.json"))
            matching_analyses = []
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    existing_log_file = existing_data.get("log_file", "")
                    if existing_log_file.lower() == request.log_file.lower():
                        is_git_fix = existing_data.get("from_git_history", False) or existing_data.get(
                            "source"
                        ) == "existing_fix"
                        if not is_git_fix:
                            matching_analyses.append(
                                {
                                    "file": json_file,
                                    "timestamp": existing_data.get("timestamp", ""),
                                    "data": existing_data,
                                }
                            )
                except (json.JSONDecodeError, KeyError, Exception):
                    continue
            if matching_analyses:
                matching_analyses.sort(key=lambda x: x["timestamp"], reverse=True)
                latest_rca = matching_analyses[0]["data"]
                existing_rca_results = latest_rca.get("results", {})
        except Exception as e:
            print(f"⚠️ Error while searching for existing RCA analysis: {e}")

    if existing_rca_results:
        results = copy.deepcopy(existing_rca_results)
    else:
        results = (request.results or {}).copy()

    from_git_history = False
    git_metadata: Dict[str, Any] = {}
    git_diff = None

    if request.fix_suggestions:
        from_git_history = bool(request.fix_suggestions.get("from_git_history", False))
        git_diff = request.fix_suggestions.get("git_diff")

        if from_git_history and request.code_dir:
            git_codebase_name = (
                os.path.basename(request.code_dir.rstrip(os.sep))
                if request.code_dir
                else "openairinterface5g-develop"
            )

            def update_deployment_context_paths(deployment_context_dict: Any) -> None:
                if not isinstance(deployment_context_dict, dict):
                    return
                ac = deployment_context_dict.get("active_configs")
                if "active_configs" in deployment_context_dict and isinstance(ac, list):
                    for config_entry in ac:
                        if isinstance(config_entry, dict) and "used" in config_entry:
                            old_path = config_entry.get("used", "")
                            if "openairinterface5g-develop" in old_path:
                                new_path = old_path.replace(
                                    "openairinterface5g-develop", git_codebase_name
                                )
                                config_entry["used"] = new_path

            if "deployment_context" in results and isinstance(results.get("deployment_context"), dict):
                update_deployment_context_paths(results["deployment_context"])
            if "phase2_analysis" in results and isinstance(results.get("phase2_analysis"), dict):
                phase2_analysis = results["phase2_analysis"]
                if "deployment_context" in phase2_analysis and isinstance(
                    phase2_analysis.get("deployment_context"), dict
                ):
                    update_deployment_context_paths(phase2_analysis["deployment_context"])

        if from_git_history:
            git_metadata = {
                "git_diff": git_diff,
                "current_branch": request.fix_suggestions.get("current_branch"),
                "files_changed_summary": request.fix_suggestions.get("files_changed_summary"),
                "selection_result": request.fix_suggestions.get("selection_result"),
                "confidence": request.fix_suggestions.get("confidence"),
                "score": request.fix_suggestions.get("score"),
                "reasoning": request.fix_suggestions.get("reasoning"),
                "full_commit": request.fix_suggestions.get("full_commit"),
            }
            git_metadata = {k: v for k, v in git_metadata.items() if v is not None}

    if request.fix_suggestions:
        if "phase3_fixes" not in results:
            results["phase3_fixes"] = {}
        fix_suggestion = request.fix_suggestions.get("fix_suggestion", {})
        if fix_suggestion:
            if from_git_history and git_diff:
                code_patches = fix_suggestion.get("code_patches", [])
                for patch in code_patches:
                    file_path_from_patch = patch.get("file_path", "")
                    function_name = patch.get("function_name", "")
                    original_code, patched_code, full_file_path = _extract_code_from_git_diff(
                        git_diff, file_path_from_patch, function_name
                    )
                    if original_code:
                        patch["original_code"] = original_code
                    if patched_code:
                        patch["patched_code"] = patched_code
                        patch["suggested_code"] = patched_code
                    if full_file_path and request.code_dir:
                        openair_codebase_file_name = (
                            os.path.basename(request.code_dir.rstrip(os.sep))
                            if request.code_dir
                            else "openairinterface5g-develop"
                        )
                        patch["file_path"] = (
                            f"Error_fixing_pipelin/{openair_codebase_file_name}/{full_file_path}"
                        )
                config_patches = fix_suggestion.get("config_patches", [])
                for patch in config_patches:
                    param_name = patch.get("parameter_name") or patch.get("config_name", "")
                    current_value, new_value = _extract_config_from_git_diff(git_diff, param_name)
                    if current_value != "N/A":
                        patch["current_value"] = current_value
                    if new_value != "N/A":
                        patch["new_value"] = new_value
                        patch["suggested_value"] = new_value
                    file_path_from_patch = patch.get("file_path", "")
                    if file_path_from_patch and request.code_dir:
                        openair_codebase_file_name = (
                            os.path.basename(request.code_dir.rstrip(os.sep))
                            if request.code_dir
                            else "openairinterface5g-develop"
                        )
                        file_basename = os.path.basename(file_path_from_patch)
                        if file_basename.endswith(".conf") or file_basename.endswith(".cfg"):
                            if "gnb" in file_basename.lower():
                                patch["file_path"] = (
                                    f"Error_fixing_pipelin/{openair_codebase_file_name}/"
                                    f"targets/PROJECTS/GENERIC-NR-5GC/CONF/{file_basename}"
                                )
                            elif "ue" in file_basename.lower():
                                patch["file_path"] = (
                                    f"Error_fixing_pipelin/{openair_codebase_file_name}/"
                                    f"openair3/NAS/TOOLS/{file_basename}"
                                )
                            else:
                                patch["file_path"] = (
                                    f"Error_fixing_pipelin/{openair_codebase_file_name}/{file_path_from_patch}"
                                )
                        else:
                            patch["file_path"] = (
                                f"Error_fixing_pipelin/{openair_codebase_file_name}/{file_path_from_patch}"
                            )

            existing_fix_suggestion = results["phase3_fixes"].get("fix_suggestion", {})
            merged_fix_suggestion = {**existing_fix_suggestion, **fix_suggestion}
            if (
                "root_cause_analysis" in existing_fix_suggestion
                and "root_cause_analysis" not in fix_suggestion
            ):
                merged_fix_suggestion["root_cause_analysis"] = existing_fix_suggestion[
                    "root_cause_analysis"
                ]
            if (
                "investigation_steps" in existing_fix_suggestion
                and "investigation_steps" not in fix_suggestion
            ):
                merged_fix_suggestion["investigation_steps"] = existing_fix_suggestion[
                    "investigation_steps"
                ]
            if (
                "specification_context" in existing_fix_suggestion
                and "specification_context" not in fix_suggestion
            ):
                merged_fix_suggestion["specification_context"] = existing_fix_suggestion[
                    "specification_context"
                ]
            results["phase3_fixes"]["fix_suggestion"] = merged_fix_suggestion

        terminal_commands = request.fix_suggestions.get("terminal_commands", [])
        if not terminal_commands and "phase4_commands" in request.fix_suggestions:
            terminal_commands = request.fix_suggestions["phase4_commands"].get("terminal_commands", [])
        if terminal_commands:
            if "phase4_commands" not in results:
                results["phase4_commands"] = {}
            results["phase4_commands"]["terminal_commands"] = terminal_commands

    history_data = {
        "error_message": request.error_message,
        "log_file": request.log_file,
        "log_path": request.log_path,
        "code_dir": request.code_dir,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "history_file": str(history_file),
        "source": "existing_fix" if from_git_history else "rca_analysis",
        "from_git_history": from_git_history,
        "git_metadata": git_metadata if git_metadata else None,
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    try:
        bug_history_record = {
            "log_file": request.log_file,
            "output": request.fix_suggestions or results,
            "metadata": {
                "history_file": str(history_file),
                "log_path": request.log_path,
                "code_dir": request.code_dir,
            },
        }
        append_history_record("bug_discovery_history.json", bug_history_record)
    except Exception as log_error:
        print(f"⚠️ Failed to append bug discovery history record: {log_error}")

    phase3_fixes = results.get("phase3_fixes", {})
    fix_sg = phase3_fixes.get("fix_suggestion", {})
    print(
        f"   Code patches: {len(fix_sg.get('code_patches', []))}, "
        f"Config patches: {len(fix_sg.get('config_patches', []))}"
    )

    return {
        "success": True,
        "message": "Analysis saved to history",
        "history_file": str(history_file),
        "filename": history_file.name,
    }
