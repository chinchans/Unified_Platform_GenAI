import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import state as shared_state

from log_capture import capture_stdio, merge_server_logs
from streaming import ndjson_stream_from_callable

router = APIRouter()


class CodeGenerationRequest(BaseModel):
    prompt: str = ""
    user_message: str = ""
    model: str = ""
    branch: str = "new_feature"


class CodeGenerationReviewRequest(BaseModel):
    action: str


TARGET_WORKSPACE = Path("/home/tcs/Phani/OAI-CU")
CODE_GEN_PROMPT_DIR = Path(
    "/home/tcs/Phani/Code_Gen/Backend/CodeGenerationFramework-main/Code_Gen/outputs/code_generation_prompts"
)
RUNTIME_PROMPT_FILENAME = ".cursor_codegen_prompt.txt"


def _merge_state(extra: Dict[str, Any]) -> None:
    current = shared_state.get_state() or {}
    if not isinstance(current, dict):
        current = {}
    merged = dict(current)
    merged.update(extra)
    shared_state.set_state(merged)


def _get_git_status(workspace: Path) -> List[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git status failed").strip())
    return [line.rstrip("\n") for line in (proc.stdout or "").splitlines() if line.strip()]


def _parse_status_paths(lines: List[str]) -> Dict[str, List[str]]:
    tracked: List[str] = []
    untracked: List[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].strip()
        if path == RUNTIME_PROMPT_FILENAME:
            continue
        if status == "??":
            untracked.append(path)
        else:
            tracked.append(path)
    return {
        "tracked_paths": sorted(set(tracked)),
        "untracked_paths": sorted(set(untracked)),
    }


def _subtract_paths(current: List[str], baseline: List[str]) -> List[str]:
    baseline_set = set(baseline)
    return sorted([p for p in current if p not in baseline_set])


def _diff_numstat(workspace: Path, paths: List[str]) -> Dict[str, Dict[str, int]]:
    if not paths:
        return {}
    proc = subprocess.run(
        ["git", "diff", "--numstat", "--"] + paths,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins_raw, del_raw, file_path = parts[0].strip(), parts[1].strip(), parts[2].strip()
        ins = int(ins_raw) if ins_raw.isdigit() else 0
        dels = int(del_raw) if del_raw.isdigit() else 0
        out[file_path] = {"insertions": ins, "deletions": dels}
    return out


def _diff_patch_by_file(workspace: Path, paths: List[str]) -> Dict[str, str]:
    if not paths:
        return {}
    out: Dict[str, str] = {}
    for p in paths:
        proc = subprocess.run(
            ["git", "diff", "--", p],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            out[p] = (proc.stdout or "").strip()
    return out


def _latest_prompt_path() -> Optional[Path]:
    if not CODE_GEN_PROMPT_DIR.is_dir():
        return None
    files = [p for p in CODE_GEN_PROMPT_DIR.glob("code_prompt_*.txt") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _latest_prompt_path_from_dir(prompt_dir: Path) -> Optional[Path]:
    if not prompt_dir.is_dir():
        return None
    files = [p for p in prompt_dir.glob("code_prompt_*.txt") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _resolve_prompt(req: CodeGenerationRequest) -> tuple[str, str]:
    prompt = str(req.prompt or "").strip()
    if prompt:
        return prompt, ""
    orch = shared_state.get_state() or {}
    stored_prompt_path = str(orch.get("code_generation_prompt_path") or "").strip()
    if stored_prompt_path:
        stored_path_obj = Path(stored_prompt_path).expanduser()
        prompt_dir = stored_path_obj.parent if stored_path_obj.parent else CODE_GEN_PROMPT_DIR
        latest_from_same_dir = _latest_prompt_path_from_dir(prompt_dir)
        if latest_from_same_dir and latest_from_same_dir.is_file():
            return latest_from_same_dir.read_text(encoding="utf-8").strip(), str(latest_from_same_dir)

    prompt_path = _latest_prompt_path()
    if prompt_path and prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip(), str(prompt_path)
    return "", ""


def _compose_codegen_prompt(base_prompt: str, user_message: str) -> str:
    base = str(base_prompt or "").strip()
    msg = str(user_message or "").strip()
    if base and msg:
        return f"{msg}\n\n{base}"
    return msg or base


def _code_generation_core(req: CodeGenerationRequest):
    orch = shared_state.get_state() or {}
    workspace_raw = (
        orch.get("repo_path")
        or orch.get("repository_path")
        or orch.get("code_repo_path")
        or os.getenv("CODE_REPO_PATH")
        or str(TARGET_WORKSPACE)
    )
    workspace = Path(str(workspace_raw)).expanduser()
    branch = str(req.branch or "new_feature").strip() or "new_feature"
    if not workspace.is_dir():
        return {"success": False, "error": f"Workspace path does not exist: {workspace}"}
    if not (workspace / ".git").is_dir():
        return {"success": False, "error": f"Not a git repository: {workspace}"}

    prompt, saved_prompt_path = _resolve_prompt(req)
    combined_prompt = _compose_codegen_prompt(prompt, req.user_message)
    if not combined_prompt:
        return {"success": False, "error": "No generated prompt available to run."}

    cursor_bin = shutil.which("cursor")
    if not cursor_bin:
        return {"success": False, "error": "Cursor CLI not found in PATH."}

    runtime_prompt = workspace / RUNTIME_PROMPT_FILENAME
    try:
        if runtime_prompt.exists():
            runtime_prompt.unlink()
    except Exception:
        pass

    try:
        pre_status_lines = _get_git_status(workspace)
    except Exception as exc:
        return {"success": False, "error": f"Failed to read git status: {exc}"}
    pre_paths = _parse_status_paths(pre_status_lines)
    preexisting_tracked = pre_paths["tracked_paths"]

    checkout_proc = subprocess.run(
        ["git", "checkout", branch],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if checkout_proc.returncode != 0:
        return {
            "success": False,
            "error": f"Failed to checkout branch '{branch}': {(checkout_proc.stderr or checkout_proc.stdout).strip()}",
            "workspace_path": str(workspace),
            "branch": branch,
        }

    prompt_file = workspace / RUNTIME_PROMPT_FILENAME
    prompt_file.write_text(combined_prompt, encoding="utf-8")

    cmd = [
        cursor_bin,
        "agent",
        "--print",
        "--output-format",
        "text",
        "--trust",
        "--force",
        "--workspace",
        str(workspace),
    ]
    model = str(req.model or "").strip()
    if model:
        cmd.extend(["--model", model])
    # Prompt text often starts with "---" (markdown). Without "--", many CLIs
    # treat leading dashes as flags/options and fail before the agent runs.
    cmd.append("--")
    cmd.append(combined_prompt)

    print(f"[CursorCLI] Workspace: {workspace}")
    print(f"[CursorCLI] Prompt file: {prompt_file}")
    print(f"[CursorCLI] Running Cursor CLI headlessly...")

    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    collected_output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        collected_output.append(line)

    returncode = proc.wait()
    stdout_text = "".join(collected_output).strip()
    try:
        post_status_lines = _get_git_status(workspace)
        post_paths = _parse_status_paths(post_status_lines)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Cursor CLI finished but failed to read resulting git status: {exc}",
            "output": stdout_text,
        }

    changed_paths = {
        "tracked_paths": _subtract_paths(
            post_paths["tracked_paths"], pre_paths["tracked_paths"]
        ),
        "untracked_paths": _subtract_paths(
            post_paths["untracked_paths"], pre_paths["untracked_paths"]
        ),
    }
    overlapping_tracked_paths = sorted(
        set(post_paths["tracked_paths"]).intersection(set(pre_paths["tracked_paths"]))
    )
    review_safe = not overlapping_tracked_paths

    has_meaningful_changes = bool(
        post_paths["tracked_paths"] or post_paths["untracked_paths"]
    )
    tracked_for_display = sorted(set(changed_paths["tracked_paths"] + overlapping_tracked_paths))
    numstat = _diff_numstat(workspace, tracked_for_display)
    patch_by_file = _diff_patch_by_file(workspace, tracked_for_display)
    code_changes = []
    for rel in tracked_for_display:
        stats = numstat.get(rel, {"insertions": 0, "deletions": 0})
        code_changes.append(
            {
                "path": rel,
                "insertions": stats.get("insertions", 0),
                "deletions": stats.get("deletions", 0),
                "diff": patch_by_file.get(rel, ""),
            }
        )

    review_state = {
        "workspace_path": str(workspace),
        "branch": branch,
        "saved_prompt_path": saved_prompt_path,
        "runtime_prompt_file": str(prompt_file),
        "tracked_paths": changed_paths["tracked_paths"],
        "untracked_paths": changed_paths["untracked_paths"],
        "pending_review": returncode == 0 and has_meaningful_changes,
        "review_safe": review_safe,
        "overlapping_tracked_paths": overlapping_tracked_paths,
        "preexisting_tracked_paths": preexisting_tracked,
    }
    _merge_state({"code_generation_review": review_state})

    return {
        "success": returncode == 0,
        "type": "cursor_cli_generation",
        "workspace_path": str(workspace),
        "branch": branch,
        "saved_prompt_path": saved_prompt_path,
        "prompt_file": str(prompt_file),
        "cursor_command": cmd[:-1] + ["<prompt>"],
        "exit_code": returncode,
        "chat_output": stdout_text,
        "changed_files": changed_paths,
        "review_available": returncode == 0 and has_meaningful_changes,
        "review_safe": review_safe,
        "preexisting_changes": {
            "tracked_paths": preexisting_tracked,
            "untracked_paths": pre_paths["untracked_paths"],
        },
        "overlapping_tracked_paths": overlapping_tracked_paths,
        "code_changes": code_changes,
        "warning": (
            "Code generation ran with preexisting tracked changes. "
            "Accept/Reject is disabled because overlap was detected."
            if overlapping_tracked_paths
            else ""
        ),
    }


def _code_generation_review_core(req: CodeGenerationReviewRequest):
    action = str(req.action or "").strip().lower()
    if action not in {"accept", "reject"}:
        return {"success": False, "error": "action must be 'accept' or 'reject'."}

    orch = shared_state.get_state() or {}
    review = (orch or {}).get("code_generation_review")
    if not isinstance(review, dict) or not review.get("workspace_path"):
        return {"success": False, "error": "No pending code generation review state found."}

    workspace = Path(str(review.get("workspace_path"))).expanduser()
    if not workspace.is_dir():
        return {"success": False, "error": f"Workspace path does not exist: {workspace}"}

    tracked_paths = [str(p) for p in (review.get("tracked_paths") or [])]
    untracked_paths = [str(p) for p in (review.get("untracked_paths") or [])]

    if action == "accept":
        _merge_state({"code_generation_review": None})
        return {
            "success": True,
            "type": "code_generation_review_accept",
            "workspace_path": str(workspace),
            "accepted_changes": {
                "tracked_paths": tracked_paths,
                "untracked_paths": untracked_paths,
            },
        }

    restore_errors: List[str] = []
    if tracked_paths:
        restore_proc = subprocess.run(
            ["git", "restore", "--"] + tracked_paths,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if restore_proc.returncode != 0:
            restore_errors.append((restore_proc.stderr or restore_proc.stdout).strip())

    for rel in untracked_paths:
        target = workspace / rel
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except Exception as exc:
            restore_errors.append(f"{rel}: {exc}")

    _merge_state({"code_generation_review": None})
    return {
        "success": not restore_errors,
        "type": "code_generation_review_reject",
        "workspace_path": str(workspace),
        "reverted_changes": {
            "tracked_paths": tracked_paths,
            "untracked_paths": untracked_paths,
        },
        "errors": restore_errors,
    }


@router.post("/api/code-generation")
def code_generation(req: CodeGenerationRequest):
    with capture_stdio() as cap:
        resp = _code_generation_core(req)
    return merge_server_logs(resp, cap)


@router.post("/api/code-generation/stream")
def code_generation_stream(req: CodeGenerationRequest) -> StreamingResponse:
    return ndjson_stream_from_callable(_code_generation_core, req)


@router.post("/api/code-generation/review")
def code_generation_review(req: CodeGenerationReviewRequest):
    with capture_stdio() as cap:
        resp = _code_generation_review_core(req)
    return merge_server_logs(resp, cap)


@router.post("/api/code-generation/review/stream")
def code_generation_review_stream(req: CodeGenerationReviewRequest) -> StreamingResponse:
    return ndjson_stream_from_callable(_code_generation_review_core, req)
