from pathlib import Path
import hashlib
import json
import queue
import re
import os
import shlex
import shutil
import subprocess
from threading import Lock, Thread
from typing import Any, Callable
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.codegen_diff_utils import parse_git_numstat_output
from services.codegen.pipeline.pipeline import (
    run_end_to_end_from_intent,
    run_resolve_self_learning_session,
)
from services.codegen.store.sqlite_state_store import SqliteStateStore

router = APIRouter()
cursor_cli_router = APIRouter()


def _codegen_root() -> Path:
    # backend/api/codegen_routes.py -> backend/services/codegen
    return Path(__file__).resolve().parent.parent / "services" / "codegen"


def _codegen_outputs_dir() -> Path:
    return _codegen_root() / "outputs"


def _state_store() -> SqliteStateStore:
    return SqliteStateStore(_codegen_outputs_dir() / "session_state.sqlite")


def _manifest_for_session(session_id: str) -> dict[str, Any] | None:
    runs_dir = _codegen_outputs_dir() / "code_gen_runs"
    if not runs_dir.exists():
        return None

    for path in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            import json

            data = json.loads(payload)
        except Exception:
            continue
        if str(data.get("session_id", "")) == session_id:
            data["manifest_path"] = str(path)
            return data
    return None


class GenerateRequest(BaseModel):
    intent: str = Field(..., min_length=1)
    use_llm_review: Any = None


class ResolveAmbiguitiesRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    resolutions: dict[str, str] | None = None
    resolution: str | None = None
    use_llm_review: Any = None


class GenerateCodeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class CommitPushRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    commit_message: str | None = None


class PushRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class UpdatePromptRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)


MILESTONE_TEMPLATE = [
    {"id": "feature_validation", "label": "Feature Validation"},
    {"id": "knowledge_retrieval", "label": "Knowledge Retrieval"},
    {"id": "template_orchestrator", "label": "Template Orchestrator"},
    {"id": "self_learning_agent", "label": "Self Learning Agent"},
    {"id": "prompt_generation", "label": "Prompt Generation"},
    {"id": "code_generation", "label": "Code Generation"},
    {"id": "commit_push", "label": "Commit & Push"},
    {"id": "jenkins_checkout", "label": "Checkout Branch"},
    {"id": "test_script_generation", "label": "Test Script Generation"},
    {"id": "build_compile", "label": "Build / Compile"},
    {"id": "rca_build_fix", "label": "RCA Build Fix Loop"},
    {"id": "runtime_execute", "label": "Run / Execute"},
    {"id": "rca_runtime_fix", "label": "RCA Runtime Fix Loop"},
    {"id": "test_scoring", "label": "Test Execution & Scoring"},
]

STAGE_MESSAGES = {
    "feature_validation_done": "Intent resolution completed.",
    "knowledge_creator_done": "Knowledge base is ready.",
    "retrieval_done": "Knowledge retrieval completed.",
    "template_filled": "Template orchestrator generated final template.",
    "self_learning_done": "Self-learning validation completed.",
    "prompt_generated": "Prompt generation completed.",
}

_runtime_lock = Lock()
_runtime_sessions: dict[str, dict[str, Any]] = {}
_codegen_review_lock = Lock()
_codegen_review_state: dict[str, Any] | None = None
CODEGEN_WORKSPACE = Path(r"C:\Users\ChanduVangala\Desktop\Unified_Platform_UI\OAI-CU")
CODEGEN_PROMPT_FILENAME = ".unified_codegen_prompt.txt"
DEFAULT_CURSOR_CLI_COMMAND = "agent"
FALLBACK_CURSOR_COMMANDS = ("agent",)
SKIP_PROMPT_GENERATION_FOR_TEST = str(
    os.getenv("SKIP_PROMPT_GENERATION_FOR_TEST", "true")
).strip().lower() in {"1", "true", "yes", "on"}
# SKIP_PROMPT_GENERATION_FOR_TEST = False
STRICT_FILE_EDIT_INSTRUCTIONS = (
    "You are operating in a real git workspace. Apply code changes directly to files in this workspace.\n"
    "Do not respond with explanation-only output.\n"
    "Do not ask for confirmation.\n"
    "When finished, provide a short summary of what you changed."
)

class CursorCodeGenerationRequest(BaseModel):
    session_id: str = ""
    prompt: str = ""
    user_message: str = Field(..., min_length=1)
    model: str = ""
    branch: str = "new_feature"


class CursorCodeGenerationReviewRequest(BaseModel):
    action: str = Field(..., min_length=1)


class CodeChangeEntry(BaseModel):
    path: str
    insertions: int
    deletions: int
    diff: str


class ChangedFilesPayload(BaseModel):
    tracked_paths: list[str]
    untracked_paths: list[str]


class CursorCodeGenerationResult(BaseModel):
    success: bool
    type: str = "cursor_cli_generation"
    workspace_path: str = ""
    branch: str = ""
    cursor_command: list[str] = Field(default_factory=list)
    chat_output: str = ""
    changed_files: ChangedFilesPayload = Field(
        default_factory=lambda: ChangedFilesPayload(tracked_paths=[], untracked_paths=[])
    )
    code_changes: list[CodeChangeEntry] = Field(default_factory=list)
    review_available: bool = False
    review_safe: bool = True
    preexisting_changes: dict[str, list[str]] = Field(default_factory=dict)
    overlapping_tracked_paths: list[str] = Field(default_factory=list)
    warning: str = ""
    saved_prompt_path: str = ""
    prompt_file: str = ""
    exit_code: int | None = None
    error: str = ""


def _new_milestones() -> list[dict[str, str]]:
    return [{**item, "status": "not_completed"} for item in MILESTONE_TEMPLATE]


def _append_log(runtime: dict[str, Any], stage: str, message: str, log_type: str = "info") -> None:
    runtime.setdefault("logs", []).append(
        {
            "stage": stage,
            "type": log_type,
            "message": message,
        }
    )


def _workspace_path() -> Path:
    configured = (os.getenv("CODEGEN_WORKSPACE_PATH") or "").strip()
    if configured:
        return Path(configured)
    return CODEGEN_WORKSPACE


def _slugify_feature_name(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "generated-change"
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "generated-change"


def _branch_name_from_session(session: dict[str, Any]) -> str:
    # Uses the resolved intent currently persisted in session state.
    intent = str(session.get("intent") or "").strip()
    slug = _slugify_feature_name(intent)
    parts = [p for p in slug.split("-") if p]
    meaningful = "-".join(parts[:8]).strip("-") or "generated-change"
    short_hash = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    max_base_len = 48
    if len(meaningful) > max_base_len:
        meaningful = meaningful[:max_base_len].rstrip("-")
    return f"feature/{meaningful}-{short_hash}"


def _ensure_feature_branch(workspace: Path, branch_name: str) -> None:
    if not (workspace / ".git").exists():
        raise FileNotFoundError(
            f"Target workspace is not a git repo: {workspace}. "
            "Initialize/clone the repository before code generation."
        )
    try:
        checkout = subprocess.run(
            ["git", "checkout", "-B", branch_name],
            cwd=str(workspace),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Git executable not found in backend process PATH. Install Git and ensure `git` is available."
        ) from exc
    if checkout.returncode != 0:
        raise RuntimeError(
            "Failed to checkout/create feature branch "
            f"{branch_name!r}. git stderr: {(checkout.stderr or '').strip()}"
        )


def _cursor_cli_command() -> str:
    return (os.getenv("CURSOR_CLI_COMMAND") or DEFAULT_CURSOR_CLI_COMMAND).strip()


def _split_command_template(command: str) -> list[str]:
    if not command:
        return []
    # Windows paths contain backslashes; posix=False preserves them while splitting.
    posix_mode = os.name != "nt"
    try:
        return shlex.split(command, posix=posix_mode)
    except ValueError:
        return command.split()


def _is_existing_windows_script(executable: str) -> bool:
    if os.name != "nt":
        return False
    stripped = executable.strip().strip('"')
    if not stripped:
        return False
    suffix = Path(stripped).suffix.lower()
    return suffix in {".cmd", ".bat"} and Path(stripped).exists()


def _default_cursor_agent_prefix_argv() -> list[str]:
    """Resolve how to spawn Cursor Agent without ``CURSOR_CLI_COMMAND``.

    On Windows, ``shutil.which("cursor")`` often points at ``Cursor.exe`` (Electron). Passing
    ``agent`` flags there produces Chromium warnings and usually no edits. Prefer the
    standalone ``agent`` entrypoint (same approach as ``Code_Gen_with_ui`` / local ``agent.cmd``).
    """
    agent_bin = shutil.which("agent")
    if agent_bin:
        return [agent_bin]
    if os.name == "nt":
        local = (os.getenv("LOCALAPPDATA") or "").strip()
        if local:
            for candidate in (
                Path(local) / "Programs" / "agent" / "bin" / "agent.cmd",
                Path(local) / "Programs" / "cursor" / "resources" / "app" / "bin" / "agent.cmd",
            ):
                if candidate.is_file():
                    return [str(candidate)]
    cursor_bin = shutil.which("cursor")
    if cursor_bin:
        return [cursor_bin, "agent"]
    raise FileNotFoundError(
        "Cursor Agent CLI not found. Add `agent` to PATH, install the Cursor CLI, or set "
        "CURSOR_CLI_COMMAND (e.g. full path to `agent.cmd` on Windows)."
    )


def _cursor_agent_prefix_argv() -> list[str]:
    if (os.getenv("CURSOR_CLI_COMMAND") or "").strip():
        return _resolve_cursor_command_tokens()
    return _default_cursor_agent_prefix_argv()


def _prepare_cursor_agent_cmd(cmd: list[str]) -> list[str]:
    """Ensure Windows .cmd/.bat launchers receive argv via cmd.exe."""
    if cmd and _is_existing_windows_script(cmd[0]):
        return ["cmd.exe", "/c", cmd[0], *cmd[1:]]
    return cmd


def _resolve_cursor_command_tokens() -> list[str]:
    cmd_template = _cursor_cli_command()
    template_parts = _split_command_template(cmd_template)
    if template_parts:
        configured_exe = template_parts[0].strip().strip('"')
        if _is_existing_windows_script(configured_exe):
            return [configured_exe] + template_parts[1:]
        if shutil.which(configured_exe) or Path(configured_exe).exists():
            return template_parts
        if os.getenv("CURSOR_CLI_COMMAND"):
            raise FileNotFoundError(
                "Configured CURSOR_CLI_COMMAND executable was not found in PATH: "
                f"{configured_exe!r}. Update CURSOR_CLI_COMMAND or install the CLI."
            )

    for command_candidate in FALLBACK_CURSOR_COMMANDS:
        candidate_parts = shlex.split(command_candidate)
        if candidate_parts and shutil.which(candidate_parts[0]):
            return candidate_parts

    raise FileNotFoundError(
        "Agent CLI executable not found in PATH. Tried commands: "
        f"{', '.join(FALLBACK_CURSOR_COMMANDS)}. "
        "Install Agent CLI or set CURSOR_CLI_COMMAND to the exact command (for example: `agent`)."
    )


def _prompt_from_session(session: dict[str, Any]) -> str:
    prompt = str(session.get("code_generation_prompt") or "").strip()
    if prompt:
        return prompt
    prompt_path = str(session.get("code_generation_prompt_path") or "").strip()
    if not prompt_path:
        return ""
    try:
        return Path(prompt_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _latest_generated_prompt_file() -> Path:
    prompts_dir = _codegen_outputs_dir() / "code_generation_prompts"
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompt outputs directory not found: {prompts_dir}")
    prompt_files = sorted(prompts_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not prompt_files:
        raise FileNotFoundError(f"No generated prompt files found in: {prompts_dir}")
    return prompt_files[0]


def _collect_git_snapshot(workspace: Path) -> dict[str, Any]:
    if not (workspace / ".git").exists():
        return {
            "files_changed": [],
            "diff_preview": "",
            "git_error": "Target path is not a git repository.",
        }
    files_changed: list[str] = []
    diff_preview = ""
    try:
        status_out = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(workspace),
            check=False,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in status_out.stdout.splitlines() if line.strip()]
        files_changed = [line[3:] if len(line) > 3 else line for line in lines]
    except OSError as exc:
        return {"files_changed": [], "diff_preview": "", "git_error": str(exc)}
    try:
        diff_out = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=str(workspace),
            check=False,
            capture_output=True,
            text=True,
        )
        diff_preview = (diff_out.stdout or "").strip()
    except OSError as exc:
        return {"files_changed": files_changed, "diff_preview": "", "git_error": str(exc)}
    return {"files_changed": files_changed, "diff_preview": diff_preview[:12000]}


def _collect_git_numstat(workspace: Path, paths: list[str]) -> dict[str, dict[str, int]]:
    if not paths:
        return {}
    try:
        proc = subprocess.run(
            ["git", "diff", "--numstat", "--", *paths],
            cwd=str(workspace),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    out: dict[str, dict[str, int]] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins = int(parts[0]) if parts[0].isdigit() else 0
        dels = int(parts[1]) if parts[1].isdigit() else 0
        out[parts[2]] = {"insertions": ins, "deletions": dels}
    return out


def _collect_code_changes(workspace: Path, paths: list[str]) -> list[dict[str, Any]]:
    if not paths:
        return []
    numstat = _collect_git_numstat(workspace, paths)
    code_changes: list[dict[str, Any]] = []
    for rel in paths:
        try:
            diff_proc = subprocess.run(
                ["git", "diff", "--", rel],
                cwd=str(workspace),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            diff_text = (diff_proc.stdout or "").strip() if diff_proc.returncode == 0 else ""
        except OSError:
            diff_text = ""
        stats = numstat.get(rel, {"insertions": 0, "deletions": 0})
        code_changes.append(
            {
                "path": rel,
                "insertions": stats.get("insertions", 0),
                "deletions": stats.get("deletions", 0),
                "diff": diff_text,
            }
        )
    return code_changes


def _extract_cli_code_blocks(cli_text: str) -> list[dict[str, str]]:
    text = str(cli_text or "")
    if not text:
        return []
    blocks: list[dict[str, str]] = []
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        language = (match.group(1) or "").strip()
        code = (match.group(2) or "").strip()
        if not code:
            continue
        blocks.append({"language": language, "content": code})
    return blocks


def _resolve_repo_path_from_state() -> str:
    with _runtime_lock:
        snapshots = list(_runtime_sessions.values())
    for runtime in reversed(snapshots):
        for key in ("repo_path", "repository_path", "code_repo_path", "workspace_path"):
            value = str(runtime.get(key) or "").strip()
            if value:
                return value
    return ""


def _resolve_prompt_dir_from_state() -> str:
    with _runtime_lock:
        snapshots = list(_runtime_sessions.values())
    for runtime in reversed(snapshots):
        for key in ("prompt_dir", "prompt_directory", "code_prompt_dir", "code_generation_prompt_dir"):
            value = str(runtime.get(key) or "").strip()
            if value:
                return value
    return ""


def _resolve_target_repo_path() -> Path:
    state_path = _resolve_repo_path_from_state()
    env_path = str(os.getenv("CODE_REPO_PATH") or "").strip()
    fallback = str(CODEGEN_WORKSPACE)
    return Path(state_path or env_path or fallback).expanduser()


def _resolve_prompt_directory() -> Path:
    state_dir = _resolve_prompt_dir_from_state()
    env_dir = str(os.getenv("CODE_PROMPT_DIR") or "").strip()
    fallback = _codegen_outputs_dir() / "code_generation_prompts"
    return Path(state_dir or env_dir or str(fallback)).expanduser()


def _latest_prompt_path_from_dir(prompt_dir: Path) -> Path | None:
    if not prompt_dir.is_dir():
        return None
    files = [p for p in prompt_dir.glob("code_prompt_*.txt") if p.is_file()]
    if not files:
        files = [p for p in prompt_dir.glob("*.txt") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _resolve_generation_prompt(request_prompt: str) -> tuple[str, str]:
    prompt = str(request_prompt or "").strip()
    if prompt:
        return prompt, ""
    prompt_dir = _resolve_prompt_directory()
    latest_prompt = _latest_prompt_path_from_dir(prompt_dir)
    if not latest_prompt:
        return "", ""
    return latest_prompt.read_text(encoding="utf-8").strip(), str(latest_prompt)


def _compose_generation_prompt(base_prompt: str, user_message: str) -> str:
    return (
        f"{STRICT_FILE_EDIT_INSTRUCTIONS}\n\n"
        f"User request:\n{user_message.strip()}\n\n"
        f"Prepared implementation context:\n{base_prompt.strip()}"
    ).strip()


def _run_git_command(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _resolve_push_remote_and_branch(workspace: Path, branch: str) -> tuple[str, str]:
    branch_name = branch.strip()
    if not branch_name:
        raise RuntimeError("Branch name is required for push.")
    upstream_proc = _run_git_command(
        workspace,
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    )
    if upstream_proc.returncode == 0:
        upstream = (upstream_proc.stdout or "").strip()
        if "/" in upstream:
            remote, remote_branch = upstream.split("/", 1)
            if remote and remote_branch:
                return remote, remote_branch

    remotes_proc = _run_git_command(workspace, ["git", "remote"])
    if remotes_proc.returncode != 0:
        raise RuntimeError((remotes_proc.stderr or remotes_proc.stdout or "git remote failed").strip())
    remotes = [line.strip() for line in (remotes_proc.stdout or "").splitlines() if line.strip()]
    if not remotes:
        raise RuntimeError("No git remotes are configured for this repository.")
    if "origin" in remotes:
        return "origin", branch_name
    return remotes[0], branch_name


def _run_commit_and_push(
    *,
    workspace: Path,
    branch: str,
    paths: list[str],
    commit_message: str,
) -> dict[str, Any]:
    if not (workspace / ".git").is_dir():
        return {"success": False, "error": f"Not a git repository: {workspace}"}
    cleaned_paths = sorted({str(p).strip() for p in paths if str(p).strip()})
    if cleaned_paths:
        add_proc = _run_git_command(workspace, ["git", "add", "--", *cleaned_paths])
    else:
        add_proc = _run_git_command(workspace, ["git", "add", "-A"])
    if add_proc.returncode != 0:
        return {"success": False, "error": f"git add failed: {(add_proc.stderr or add_proc.stdout).strip()}"}

    staged_check = _run_git_command(workspace, ["git", "diff", "--cached", "--quiet"])
    if staged_check.returncode == 0:
        return {"success": False, "error": "No staged changes found to commit."}
    if staged_check.returncode not in (0, 1):
        return {
            "success": False,
            "error": f"Failed to validate staged changes: {(staged_check.stderr or staged_check.stdout).strip()}",
        }

    commit_proc = _run_git_command(workspace, ["git", "commit", "-m", commit_message])
    if commit_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git commit failed: {(commit_proc.stderr or commit_proc.stdout).strip()}",
        }

    hash_proc = _run_git_command(workspace, ["git", "rev-parse", "--short", "HEAD"])
    commit_hash = (hash_proc.stdout or "").strip() if hash_proc.returncode == 0 else ""

    try:
        remote, remote_branch = _resolve_push_remote_and_branch(workspace, branch)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc), "commit_hash": commit_hash}

    push_proc = _run_git_command(workspace, ["git", "push", "-u", remote, f"HEAD:{remote_branch}"])
    if push_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git push failed: {(push_proc.stderr or push_proc.stdout).strip()}",
            "commit_hash": commit_hash,
            "remote": remote,
            "remote_branch": remote_branch,
        }

    return {
        "success": True,
        "commit_hash": commit_hash,
        "remote": remote,
        "remote_branch": remote_branch,
        "push_output": (push_proc.stdout or "").strip(),
    }


def _run_commit_only(
    *,
    workspace: Path,
    paths: list[str],
    commit_message: str,
) -> dict[str, Any]:
    if not (workspace / ".git").is_dir():
        return {"success": False, "error": f"Not a git repository: {workspace}"}
    cleaned_paths = sorted({str(p).strip() for p in paths if str(p).strip()})
    if cleaned_paths:
        add_proc = _run_git_command(workspace, ["git", "add", "--", *cleaned_paths])
    else:
        add_proc = _run_git_command(workspace, ["git", "add", "-A"])
    if add_proc.returncode != 0:
        return {"success": False, "error": f"git add failed: {(add_proc.stderr or add_proc.stdout).strip()}"}

    staged_check = _run_git_command(workspace, ["git", "diff", "--cached", "--quiet"])
    if staged_check.returncode == 0:
        return {"success": False, "error": "No staged changes found to commit."}
    if staged_check.returncode not in (0, 1):
        return {
            "success": False,
            "error": f"Failed to validate staged changes: {(staged_check.stderr or staged_check.stdout).strip()}",
        }

    commit_proc = _run_git_command(workspace, ["git", "commit", "-m", commit_message])
    if commit_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git commit failed: {(commit_proc.stderr or commit_proc.stdout).strip()}",
        }

    hash_proc = _run_git_command(workspace, ["git", "rev-parse", "--short", "HEAD"])
    commit_hash = (hash_proc.stdout or "").strip() if hash_proc.returncode == 0 else ""
    return {
        "success": True,
        "commit_hash": commit_hash,
    }


def _run_push_only(
    *,
    workspace: Path,
    branch: str,
) -> dict[str, Any]:
    if not (workspace / ".git").is_dir():
        return {"success": False, "error": f"Not a git repository: {workspace}"}
    try:
        remote, remote_branch = _resolve_push_remote_and_branch(workspace, branch)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}
    push_proc = _run_git_command(workspace, ["git", "push", "-u", remote, f"HEAD:{remote_branch}"])
    if push_proc.returncode != 0:
        return {
            "success": False,
            "error": f"git push failed: {(push_proc.stderr or push_proc.stdout).strip()}",
            "remote": remote,
            "remote_branch": remote_branch,
        }
    return {
        "success": True,
        "remote": remote,
        "remote_branch": remote_branch,
        "push_output": (push_proc.stdout or "").strip(),
    }


def _suggest_commit_message(
    session_id: str,
    tracked_paths: list[str],
    untracked_paths: list[str],
) -> str:
    session = _state_store().get_session(session_id) or {}
    intent = str(session.get("intent") or "").strip()
    clean_intent = re.sub(r"\s+", " ", intent)
    if len(clean_intent) > 72:
        clean_intent = clean_intent[:72].rstrip() + "..."
    file_count = len(set([*tracked_paths, *untracked_paths]))
    if clean_intent:
        return f"feat: {clean_intent}"
    if file_count > 0:
        return f"chore: apply generated code changes ({file_count} files)"
    return f"chore: apply generated code for session {session_id[:8]}"


def _get_git_status_lines(workspace: Path) -> list[str]:
    proc = _run_git_command(workspace, ["git", "status", "--porcelain"])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git status failed").strip())
    return [line.rstrip("\n") for line in (proc.stdout or "").splitlines() if line.strip()]


def _parse_status_paths(lines: list[str]) -> dict[str, list[str]]:
    tracked: list[str] = []
    untracked: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].strip()
        if not path or path == CODEGEN_PROMPT_FILENAME:
            continue
        if status == "??":
            untracked.append(path)
        else:
            tracked.append(path)
    return {"tracked_paths": sorted(set(tracked)), "untracked_paths": sorted(set(untracked))}


def _subtract_paths(current: list[str], baseline: list[str]) -> list[str]:
    baseline_set = set(baseline)
    return sorted([p for p in current if p not in baseline_set])


def _diff_numstat(workspace: Path, paths: list[str]) -> dict[str, dict[str, int]]:
    if not paths:
        return {}
    proc = _run_git_command(workspace, ["git", "diff", "--numstat", "--", *paths])
    if proc.returncode != 0:
        return {}
    return parse_git_numstat_output(proc.stdout or "")


def _diff_patch_by_file(workspace: Path, paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        proc = _run_git_command(workspace, ["git", "diff", "--", p])
        if proc.returncode == 0:
            out[p] = (proc.stdout or "").strip()
    return out


def _run_git_or_raise(workspace: Path, args: list[str], action: str) -> subprocess.CompletedProcess[str]:
    proc = _run_git_command(workspace, args)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or f"{action} failed").strip()
        raise HTTPException(status_code=500, detail=message)
    return proc


def _parse_branch_line(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line:
        return None
    if line.startswith("*"):
        line = line[1:].strip()
    if line.startswith("remotes/"):
        line = line[len("remotes/") :]
    if line == "HEAD":
        return None
    if " -> " in line:
        return None
    return line or None


def _list_local_branches(workspace: Path) -> list[str]:
    local_proc = _run_git_or_raise(
        workspace,
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        "list local branches",
    )
    branches: list[str] = []
    seen: set[str] = set()
    for line in (local_proc.stdout or "").splitlines():
        branch = _parse_branch_line(line)
        if not branch or branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _local_branch_upstream_map(workspace: Path) -> dict[str, str]:
    proc = _run_git_or_raise(
        workspace,
        ["git", "for-each-ref", "--format=%(refname:short)%x1f%(upstream:short)", "refs/heads"],
        "list local branch upstreams",
    )
    upstreams: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\x1f")
        if not parts:
            continue
        branch = (parts[0] or "").strip()
        upstream = (parts[1] or "").strip() if len(parts) > 1 else ""
        if branch:
            upstreams[branch] = upstream
    return upstreams


def _default_git_remote(workspace: Path) -> str:
    proc = _run_git_or_raise(workspace, ["git", "remote"], "list git remotes")
    remotes = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not remotes:
        return ""
    if "origin" in remotes:
        return "origin"
    return remotes[0]


def _current_branch(workspace: Path) -> str:
    proc = _run_git_or_raise(
        workspace,
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "resolve current branch",
    )
    return (proc.stdout or "").strip()


def _list_branch_commits(workspace: Path, branch: str, max_count: int = 150) -> list[dict[str, Any]]:
    # Include parent hashes (%P) so the UI can render merge edges / lane graph.
    format_token = "%H%x1f%h%x1f%P%x1f%an%x1f%ad%x1f%s"
    proc = _run_git_or_raise(
        workspace,
        [
            "git",
            "log",
            branch,
            "--date=short",
            f"--max-count={max(1, min(max_count, 500))}",
            f"--pretty=format:{format_token}",
        ],
        f"list commits for {branch}",
    )
    commits: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\x1f")
        if len(parts) < 6:
            continue
        parents_raw = (parts[2] or "").strip()
        parents = [p for p in parents_raw.split() if p]
        commits.append(
            {
                "hash": parts[0],
                "short_hash": parts[1],
                "parents": parents,
                "author": parts[3],
                "date": parts[4],
                "subject": parts[5],
            }
        )
    return commits


def _upstream_ref_for_branch(workspace: Path, branch: str) -> str:
    proc = _run_git_command(
        workspace,
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{u}}"],
    )
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _rev_parse(workspace: Path, ref: str) -> str:
    if not ref:
        return ""
    proc = _run_git_command(workspace, ["git", "rev-parse", ref])
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _extract_path_from_diff_header(line: str) -> str:
    # diff --git a/path b/path
    match = re.match(r"^diff --git a/(.+?) b/(.+)$", line.strip())
    if not match:
        return ""
    old_path = match.group(1)
    new_path = match.group(2)
    if new_path == "/dev/null":
        return old_path
    return new_path


def _commit_file_diffs(workspace: Path, commit_hash: str) -> list[dict[str, Any]]:
    patch_proc = _run_git_or_raise(
        workspace,
        ["git", "show", "--format=", "--patch", "--no-color", commit_hash],
        "collect commit diff",
    )
    stats_proc = _run_git_or_raise(
        workspace,
        ["git", "show", "--format=", "--numstat", "--no-color", commit_hash],
        "collect commit file stats",
    )
    stats = parse_git_numstat_output(stats_proc.stdout or "")
    lines = (patch_proc.stdout or "").splitlines()
    chunks: list[tuple[str, list[str]]] = []
    current_path = ""
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("diff --git "):
            if current_path and current_lines:
                chunks.append((current_path, current_lines))
            current_path = _extract_path_from_diff_header(line)
            current_lines = [line]
            continue
        if current_path:
            current_lines.append(line)
    if current_path and current_lines:
        chunks.append((current_path, current_lines))

    files: list[dict[str, Any]] = []
    for path, chunk_lines in chunks:
        metric = stats.get(path, {})
        files.append(
            {
                "path": path,
                "insertions": int(metric.get("insertions", 0)),
                "deletions": int(metric.get("deletions", 0)),
                "diff": "\n".join(chunk_lines).strip(),
            }
        )
    return files


def _run_cursor_generation_core(
    req: CursorCodeGenerationRequest,
    emit_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    global _codegen_review_state

    def log(message: str) -> None:
        if emit_log:
            emit_log(message)

    workspace = _resolve_target_repo_path()
    branch = str(req.branch or "new_feature").strip() or "new_feature"
    if not workspace.is_dir():
        return {"success": False, "error": f"Workspace path does not exist: {workspace}"}
    if not (workspace / ".git").is_dir():
        return {"success": False, "error": f"Not a git repository: {workspace}"}
    try:
        agent_prefix = _cursor_agent_prefix_argv()
    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc)}

    saved_prompt_path = ""
    session_id = str(req.session_id or "").strip()
    if session_id:
        session = _state_store().get_session(session_id) or {}
        if not session:
            return {"success": False, "error": f"Unknown session_id: {session_id}"}
        base_prompt = _prompt_from_session(session)
        saved_prompt_path = str(session.get("code_generation_prompt_path") or "").strip()
    else:
        base_prompt, saved_prompt_path = _resolve_generation_prompt(req.prompt)
    if not base_prompt:
        return {"success": False, "error": "No generated prompt available to run."}
    combined_prompt = _compose_generation_prompt(base_prompt, req.user_message)
    if not combined_prompt:
        return {"success": False, "error": "Combined prompt is empty."}

    try:
        pre_status_lines = _get_git_status_lines(workspace)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Failed to read git status before run: {exc}"}
    pre_paths = _parse_status_paths(pre_status_lines)
    preexisting_tracked = pre_paths["tracked_paths"]

    log(f"Checking out branch: {branch}")
    log(f"Target workspace: {workspace}")
    checkout_proc = _run_git_command(workspace, ["git", "checkout", branch])
    if checkout_proc.returncode != 0:
        log(f"Branch '{branch}' was not found. Creating it.")
        create_proc = _run_git_command(workspace, ["git", "checkout", "-b", branch])
        if create_proc.returncode != 0:
            return {
                "success": False,
                "error": (
                    f"Failed to checkout/create branch '{branch}': "
                    f"{(create_proc.stderr or create_proc.stdout or checkout_proc.stderr or checkout_proc.stdout).strip()}"
                ),
            }

    prompt_file = workspace / CODEGEN_PROMPT_FILENAME
    prompt_file.write_text(combined_prompt, encoding="utf-8")
    cmd = [
        *agent_prefix,
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
    # Use @prompt-file indirection to avoid Windows command-length limits.
    prompt_arg = f"@{prompt_file}"
    cmd.extend(["--", prompt_arg])
    cmd = _prepare_cursor_agent_cmd(cmd)
    command_for_response = [*cmd[:-1], "<prompt>"]
    log("Resolved CLI command: " + " ".join(command_for_response))

    log("Running Cursor CLI headlessly.")
    try:
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
    except OSError as exc:
        return {"success": False, "error": f"Failed to start Cursor CLI: {exc}"}
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        clean_line = line.rstrip()
        if clean_line:
            log(clean_line)
    exit_code = proc.wait()
    chat_output = "".join(output_lines).strip()

    try:
        post_status_lines = _get_git_status_lines(workspace)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": f"Cursor CLI finished but failed to read resulting git status: {exc}",
            "chat_output": chat_output,
        }

    post_paths = _parse_status_paths(post_status_lines)
    tracked_new_paths = _subtract_paths(post_paths["tracked_paths"], pre_paths["tracked_paths"])
    untracked_new_paths = _subtract_paths(post_paths["untracked_paths"], pre_paths["untracked_paths"])
    overlapping_tracked_paths = sorted(
        set(post_paths["tracked_paths"]).intersection(set(pre_paths["tracked_paths"]))
    )
    overlap_numstat = _diff_numstat(workspace, overlapping_tracked_paths)
    overlapping_with_diff = sorted(
        [
            rel
            for rel in overlapping_tracked_paths
            if (overlap_numstat.get(rel, {}).get("insertions", 0) + overlap_numstat.get(rel, {}).get("deletions", 0))
            > 0
        ]
    )
    changed_files = {
        "tracked_paths": sorted(set(tracked_new_paths + overlapping_with_diff)),
        "untracked_paths": untracked_new_paths,
    }
    review_safe = not overlapping_tracked_paths
    has_meaningful_changes = bool(changed_files["tracked_paths"] or changed_files["untracked_paths"])
    tracked_paths = changed_files["tracked_paths"]
    numstat = _diff_numstat(workspace, tracked_paths)
    patch_by_file = _diff_patch_by_file(workspace, tracked_paths)
    code_changes: list[CodeChangeEntry] = []
    for rel in tracked_paths:
        code_changes.append(
            CodeChangeEntry(
                path=rel,
                insertions=numstat.get(rel, {}).get("insertions", 0),
                deletions=numstat.get(rel, {}).get("deletions", 0),
                diff=patch_by_file.get(rel, ""),
            )
        )

    with _codegen_review_lock:
        global _codegen_review_state
        _codegen_review_state = {
            "session_id": session_id,
            "workspace_path": str(workspace),
            "branch": branch,
            "tracked_paths": changed_files["tracked_paths"],
            "untracked_paths": changed_files["untracked_paths"],
            "prompt_file": str(prompt_file),
            "pending_review": exit_code == 0 and has_meaningful_changes,
            "review_safe": review_safe,
            "overlapping_tracked_paths": overlapping_tracked_paths,
            "preexisting_changes": {
                "tracked_paths": preexisting_tracked,
                "untracked_paths": pre_paths["untracked_paths"],
            },
        }

    result = CursorCodeGenerationResult(
        success=exit_code == 0 and has_meaningful_changes,
        workspace_path=str(workspace),
        branch=branch,
        saved_prompt_path=saved_prompt_path,
        prompt_file=str(prompt_file),
        cursor_command=command_for_response,
        exit_code=exit_code,
        chat_output=chat_output,
        changed_files=ChangedFilesPayload(**changed_files),
        code_changes=code_changes,
        review_available=exit_code == 0 and has_meaningful_changes,
        review_safe=review_safe,
        preexisting_changes={
            "tracked_paths": preexisting_tracked,
            "untracked_paths": pre_paths["untracked_paths"],
        },
        overlapping_tracked_paths=overlapping_tracked_paths,
        warning=(
            "Code generation touched files that were already modified before the run. "
            "Accept/Reject is disabled because overlap was detected."
            if overlapping_tracked_paths
            else ""
        ),
        error=(
            f"Cursor CLI failed with exit code {exit_code}."
            if exit_code != 0
            else (
                "Cursor CLI completed but no file changes were detected in the target workspace. "
                "This usually means chat-only output or edits attempted outside the workspace."
                if not has_meaningful_changes
                else ""
            )
        ),
    )
    return result.model_dump()


def _run_code_generation_review_core(req: CursorCodeGenerationReviewRequest) -> dict[str, Any]:
    global _codegen_review_state
    action = str(req.action or "").strip().lower()
    if action not in {"accept", "reject"}:
        return {"success": False, "error": "action must be 'accept' or 'reject'."}

    with _codegen_review_lock:
        review = dict(_codegen_review_state or {})
    if not review.get("workspace_path"):
        return {"success": False, "error": "No pending code generation review state found."}
    if not bool(review.get("pending_review")):
        return {"success": False, "error": "No reviewable Cursor generation changes are pending."}

    workspace = Path(str(review.get("workspace_path"))).expanduser()
    tracked_paths = [str(p) for p in (review.get("tracked_paths") or [])]
    untracked_paths = [str(p) for p in (review.get("untracked_paths") or [])]

    if action == "accept":
        session_id = str(review.get("session_id") or "").strip()
        suggested_commit_message = _suggest_commit_message(session_id, tracked_paths, untracked_paths)
        if session_id:
            with _runtime_lock:
                runtime = _runtime_sessions.setdefault(
                    session_id,
                    {
                        "state": "code_ready_for_review",
                        "running": False,
                        "logs": [],
                        "milestones": _new_milestones(),
                        "seen_stage_status": {},
                        "code_changes": [],
                        "cursor_cli_output": "",
                        "cursor_cli_code_blocks": [],
                    },
                )
                runtime["accepted_changes"] = {
                    "tracked_paths": tracked_paths,
                    "untracked_paths": untracked_paths,
                }
                runtime["accepted_branch"] = str(review.get("branch") or "").strip()
                runtime["suggested_commit_message"] = suggested_commit_message
        with _codegen_review_lock:
            _codegen_review_state = None
        return {
            "success": True,
            "type": "code_generation_review_accept",
            "workspace_path": str(workspace),
            "accepted_changes": {"tracked_paths": tracked_paths, "untracked_paths": untracked_paths},
            "suggested_commit_message": suggested_commit_message,
            "error": "",
        }

    restore_errors: list[str] = []
    if tracked_paths:
        restore_proc = _run_git_command(workspace, ["git", "restore", "--", *tracked_paths])
        if restore_proc.returncode != 0:
            restore_errors.append((restore_proc.stderr or restore_proc.stdout).strip())

    for rel in untracked_paths:
        target = workspace / rel
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except Exception as exc:  # noqa: BLE001
            restore_errors.append(f"{rel}: {exc}")

    with _codegen_review_lock:
        _codegen_review_state = None
    return {
        "success": not restore_errors,
        "type": "code_generation_review_reject",
        "workspace_path": str(workspace),
        "reverted_changes": {"tracked_paths": tracked_paths, "untracked_paths": untracked_paths},
        "errors": restore_errors,
        "error": "; ".join(restore_errors) if restore_errors else "",
    }


def _ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _stream_generation(req: CursorCodeGenerationRequest):
    events: "queue.Queue[str | None]" = queue.Queue()

    def _worker() -> None:
        try:
            events.put(_ndjson_line({"type": "log", "text": "Starting Cursor CLI generation run."}))
            result = _run_cursor_generation_core(req, emit_log=lambda text: events.put(_ndjson_line({"type": "log", "text": text})))
            if not result.get("success"):
                events.put(_ndjson_line({"type": "error", "error": str(result.get("error") or "Generation failed.")}))
            events.put(_ndjson_line({"type": "result", "data": result}))
        except Exception as exc:  # noqa: BLE001
            events.put(_ndjson_line({"type": "error", "error": str(exc)}))
        finally:
            events.put(None)

    Thread(target=_worker, daemon=True).start()
    while True:
        item = events.get()
        if item is None:
            break
        yield item


def _stream_review(req: CursorCodeGenerationReviewRequest):
    try:
        yield _ndjson_line({"type": "log", "text": "Starting review action."})
        result = _run_code_generation_review_core(req)
        if not result.get("success"):
            yield _ndjson_line({"type": "error", "error": str(result.get("error") or "Review failed.")})
        yield _ndjson_line({"type": "result", "data": result})
    except Exception as exc:  # noqa: BLE001
        yield _ndjson_line({"type": "error", "error": str(exc)})


@cursor_cli_router.post("/api/code-generation/stream")
def code_generation_stream(req: CursorCodeGenerationRequest) -> StreamingResponse:
    return StreamingResponse(_stream_generation(req), media_type="application/x-ndjson")


@cursor_cli_router.post("/api/code-generation/review/stream")
def code_generation_review_stream(req: CursorCodeGenerationReviewRequest) -> StreamingResponse:
    return StreamingResponse(_stream_review(req), media_type="application/x-ndjson")


def _run_cursor_cli(prompt: str, workspace: Path) -> tuple[int | None, str, str]:
    prompt_path = workspace / CODEGEN_PROMPT_FILENAME
    prompt_path.write_text(prompt, encoding="utf-8")
    cmd_template = _cursor_cli_command()
    base_args = [
        "--print",
        "--output-format",
        "text",
        "--trust",
        "--force",
        "--workspace",
        str(workspace),
        "--",
        f"@{prompt_path}",
    ]

    if (os.getenv("CURSOR_CLI_COMMAND") or "").strip():
        resolved_tokens = _resolve_cursor_command_tokens()
        cmd = [
            part.format(prompt_file=str(prompt_path), prompt=prompt, workspace=str(workspace))
            for part in resolved_tokens
        ]
        has_prompt_placeholder = "{prompt}" in cmd_template or "{prompt_file}" in cmd_template
        if not has_prompt_placeholder:
            cmd.extend(["--", f"@{prompt_path}"])
    else:
        prefix = _default_cursor_agent_prefix_argv()
        cmd = [*prefix, *base_args]
    cmd = _prepare_cursor_agent_cmd(cmd)
    display_cmd = list(cmd)
    if display_cmd:
        for idx, token in enumerate(display_cmd):
            if token == prompt or token == f"@{prompt_path}":
                display_cmd[idx] = "<prompt>"
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _set_milestone(milestones: list[dict[str, str]], milestone_id: str, status: str) -> None:
    for item in milestones:
        if item["id"] == milestone_id:
            item["status"] = _merge_status(item.get("status", "not_completed"), status)
            return


def _mark_failure_on_active_milestone(runtime: dict[str, Any], fallback_id: str) -> str:
    milestones = runtime.get("milestones", [])
    active_ids = [m.get("id") for m in milestones if m.get("status") == "in_progress" and m.get("id")]
    target_id = active_ids[-1] if active_ids else fallback_id
    _set_milestone(milestones, target_id, "failed")
    runtime["failed_milestone_id"] = target_id
    return target_id


def _status_rank(status: str) -> int:
    order = {
        "not_completed": 0,
        "in_progress": 1,
        "completed": 2,
        "failed": 3,
    }
    return order.get(status, 0)


def _merge_status(current: str, incoming: str) -> str:
    return incoming if _status_rank(incoming) >= _status_rank(current) else current


def _update_from_stage_runs(
    milestones: list[dict[str, str]],
    stage_runs: list[dict[str, Any]],
) -> dict[str, str]:
    stage_status: dict[str, str] = {}
    for row in stage_runs:
        stage_name = str(row.get("stage_name", ""))
        stage_state = str(row.get("stage_status", ""))
        mapped = "not_completed"
        if stage_state == "completed":
            mapped = "completed"
        elif stage_state == "started":
            mapped = "in_progress"
        elif stage_state == "failed":
            mapped = "failed"
        stage_status[stage_name] = mapped

    _set_milestone(
        milestones,
        "feature_validation",
        stage_status.get("feature_validation_done", "not_completed"),
    )

    knowledge_status = "not_completed"
    kc = stage_status.get("knowledge_creator_done", "not_completed")
    rt = stage_status.get("retrieval_done", "not_completed")
    if "failed" in (kc, rt):
        knowledge_status = "failed"
    elif rt == "completed":
        knowledge_status = "completed"
    elif "in_progress" in (kc, rt) or kc == "completed":
        knowledge_status = "in_progress"
    _set_milestone(milestones, "knowledge_retrieval", knowledge_status)

    _set_milestone(
        milestones,
        "template_orchestrator",
        stage_status.get("template_filled", "not_completed"),
    )
    _set_milestone(
        milestones,
        "self_learning_agent",
        stage_status.get("self_learning_done", "not_completed"),
    )
    _set_milestone(
        milestones,
        "prompt_generation",
        stage_status.get("prompt_generated", "not_completed"),
    )
    return stage_status


def _extract_ambiguity_items(session: dict[str, Any] | None) -> list[Any]:
    if not session:
        return []
    items = session.get("ambiguity_questions")
    if isinstance(items, list):
        return items
    return []


def _state_from_session(runtime: dict[str, Any], session: dict[str, Any] | None) -> str:
    runtime_state = str(runtime.get("state") or "").strip()
    if runtime_state == "failed":
        return "failed"
    # Runtime state must take precedence for post-prompt stages, otherwise snapshots regress to
    # prompt_ready whenever the persisted session remains final_prompt_ready.
    if runtime_state in {
        "prompt_generation_in_progress",
        "prompt_ready",
        "code_generation_in_progress",
        "code_ready_for_review",
        "commit_push_in_progress",
        "commit_ready_for_push",
        "pipeline_done",
    }:
        return runtime_state
    if runtime.get("running"):
        return runtime_state or "prompt_generation_in_progress"
    if not session:
        return runtime_state or "prompt_generation_in_progress"
    session_status = str(session.get("status", ""))
    if session_status == "awaiting_user":
        return "ambiguity_required"
    if session_status == "final_prompt_ready":
        return "prompt_ready"
    return runtime_state or "prompt_generation_in_progress"


def _build_ui_snapshot(session_id: str) -> dict[str, Any]:
    store = _state_store()
    session = store.get_session(session_id)
    stage_runs = store.get_stage_runs(session_id) if session else []
    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_generation_in_progress",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
            },
        )
        milestones = _new_milestones()
        for m in runtime.get("milestones", []):
            _set_milestone(milestones, m["id"], m.get("status", "not_completed"))
        stage_status = _update_from_stage_runs(milestones, stage_runs)
        if session and str(session.get("status", "")) == "final_prompt_ready":
            _set_milestone(milestones, "prompt_generation", "completed")
        runtime["milestones"] = milestones

        seen = runtime.setdefault("seen_stage_status", {})
        for stage_name, status in stage_status.items():
            if seen.get(stage_name) != status and stage_name in STAGE_MESSAGES:
                if status == "in_progress":
                    _append_log(runtime, stage_name, f"{STAGE_MESSAGES[stage_name]} (in progress)")
                elif status == "completed":
                    _append_log(runtime, stage_name, STAGE_MESSAGES[stage_name])
                elif status == "failed":
                    _append_log(runtime, stage_name, f"{STAGE_MESSAGES[stage_name]} (failed)", "error")
                seen[stage_name] = status

        state = _state_from_session(runtime, session)
        prompt = (session or {}).get("code_generation_prompt") or ""
        ambiguity_items = _extract_ambiguity_items(session)

        return {
            "session_id": session_id,
            "state": state,
            "milestones": milestones,
            "logs": runtime.get("logs", []),
            "prompt": prompt,
            "codePreview": runtime.get("code_preview", ""),
            "generatedFiles": runtime.get("generated_files", []),
            "codeChanges": runtime.get("code_changes", []),
            "cursorCliOutput": runtime.get("cursor_cli_output", ""),
            "cursorCliCodeBlocks": runtime.get("cursor_cli_code_blocks", []),
            "ambiguity_items": ambiguity_items,
            "ambiguity_questions": ambiguity_items,
        }


def _build_resolutions_from_text(
    session: dict[str, Any] | None,
    resolution_text: str,
) -> dict[str, str]:
    text = resolution_text.strip()
    if not text:
        return {}
    ambiguities = _extract_ambiguity_items(session)
    if not ambiguities:
        return {"general_resolution": text}
    out: dict[str, str] = {}
    for idx, item in enumerate(ambiguities, 1):
        if isinstance(item, dict) and item.get("id"):
            key = str(item.get("id"))
        else:
            key = f"ambiguity_{idx}"
        out[key] = text
    return out


def _start_generate_task(session_id: str, request: GenerateRequest) -> None:
    def _runner() -> None:
        try:
            run_end_to_end_from_intent(
                user_intent=request.intent,
                use_llm_review=request.use_llm_review,
                session_id=session_id,
            )
            session = _state_store().get_session(session_id) or {}
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                if str(session.get("status", "")) == "awaiting_user":
                    runtime["state"] = "ambiguity_required"
                    _append_log(
                        runtime,
                        "self_learning_agent",
                        "Ambiguities found. Provide resolution and submit.",
                        "warning",
                    )
                else:
                    runtime["state"] = "prompt_ready"
                    _append_log(runtime, "prompt_generation", "Prompt is ready.")
        except Exception as exc:  # noqa: BLE001
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                runtime["state"] = "failed"
                failed_stage = _mark_failure_on_active_milestone(runtime, "prompt_generation")
                _append_log(runtime, failed_stage, str(exc), "error")
        finally:
            with _runtime_lock:
                _runtime_sessions[session_id]["running"] = False

    Thread(target=_runner, daemon=True).start()


def _start_resolve_task(
    session_id: str,
    resolutions: dict[str, str],
    use_llm_review: Any,
) -> None:
    def _runner() -> None:
        try:
            run_resolve_self_learning_session(
                session_id=session_id,
                user_resolutions=resolutions,
                use_llm_review=use_llm_review,
            )
            session = _state_store().get_session(session_id) or {}
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                if str(session.get("status", "")) == "awaiting_user":
                    runtime["state"] = "ambiguity_required"
                    _append_log(
                        runtime,
                        "self_learning_agent",
                        "More ambiguities remain. Submit updated resolution.",
                        "warning",
                    )
                else:
                    runtime["state"] = "prompt_ready"
                    _append_log(runtime, "prompt_generation", "Prompt regenerated after resolution.")
        except Exception as exc:  # noqa: BLE001
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                runtime["state"] = "failed"
                failed_stage = _mark_failure_on_active_milestone(runtime, "self_learning_agent")
                _append_log(runtime, failed_stage, str(exc), "error")
        finally:
            with _runtime_lock:
                _runtime_sessions[session_id]["running"] = False

    Thread(target=_runner, daemon=True).start()


def _start_generate_code_task(session_id: str) -> None:
    def _runner() -> None:
        store = _state_store()
        session = store.get_session(session_id) or {}
        prompt = _prompt_from_session(session)
        workspace = _workspace_path()
        branch_name = _branch_name_from_session(session)
        try:
            if not prompt:
                raise ValueError("No prompt available for this session. Regenerate or update prompt first.")
            if not workspace.exists() or not workspace.is_dir():
                raise FileNotFoundError(f"Codegen workspace not found: {workspace}")
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                resolved_cmd = " ".join(_resolve_cursor_command_tokens())
                _append_log(runtime, "code_generation", f"Resolved CLI command: {resolved_cmd}")
            _ensure_feature_branch(workspace, branch_name)
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                _append_log(runtime, "code_generation", f"Checked out branch: {branch_name}")

            return_code, stdout, stderr = _run_cursor_cli(prompt, workspace)
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                runtime["cursor_cli_output"] = stdout or ""
                runtime["cursor_cli_code_blocks"] = _extract_cli_code_blocks(stdout or "")
                if stdout.strip():
                    _append_log(
                        runtime,
                        "code_generation",
                        "Cursor CLI response captured. Use `cursorCliOutput` for full text output.",
                    )
                if stderr.strip():
                    _append_log(runtime, "code_generation", stderr.strip(), "warning")
                if return_code not in (0, None):
                    runtime["state"] = "failed"
                    _set_milestone(runtime["milestones"], "code_generation", "failed")
                    _append_log(runtime, "code_generation", f"Cursor CLI failed with exit code {return_code}.", "error")
                    return
                git_snapshot = _collect_git_snapshot(workspace)
                runtime["generated_files"] = git_snapshot.get("files_changed", [])
                runtime["code_preview"] = git_snapshot.get("diff_preview", "")
                runtime["code_changes"] = _collect_code_changes(
                    workspace,
                    list(runtime["generated_files"]),
                )
                _set_milestone(runtime["milestones"], "code_generation", "completed")
                runtime["state"] = "code_ready_for_review"
                _append_log(runtime, "code_generation", "Code generation completed and changes collected.")
                if runtime["code_changes"]:
                    _append_log(
                        runtime,
                        "code_generation",
                        f"Detected {len(runtime['code_changes'])} changed file(s) with exact additions/deletions.",
                    )
                if git_snapshot.get("git_error"):
                    _append_log(runtime, "code_generation", str(git_snapshot["git_error"]), "warning")
        except Exception as exc:  # noqa: BLE001
            with _runtime_lock:
                runtime = _runtime_sessions[session_id]
                runtime["state"] = "failed"
                _set_milestone(runtime["milestones"], "code_generation", "failed")
                _append_log(runtime, "code_generation", str(exc), "error")
        finally:
            with _runtime_lock:
                _runtime_sessions[session_id]["running"] = False

    Thread(target=_runner, daemon=True).start()


@router.post("/generate")
def generate_code(request: GenerateRequest) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    store = _state_store()
    if SKIP_PROMPT_GENERATION_FOR_TEST:
        latest_prompt_path = _latest_generated_prompt_file()
        latest_prompt = latest_prompt_path.read_text(encoding="utf-8").strip()
        if not latest_prompt:
            raise HTTPException(status_code=500, detail=f"Latest prompt file is empty: {latest_prompt_path}")
        store.ensure_session(
            session_id=session_id,
            intent=request.intent.strip(),
            status="final_prompt_ready",
            template_path="",
            code_generation_prompt=latest_prompt,
            code_generation_prompt_path=str(latest_prompt_path),
        )
        with _runtime_lock:
            _runtime_sessions[session_id] = {
                "state": "prompt_ready",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            }
            _set_milestone(_runtime_sessions[session_id]["milestones"], "prompt_generation", "completed")
            _append_log(
                _runtime_sessions[session_id],
                "prompt_generation",
                f"Skipped prompt pipeline for testing. Loaded latest prompt: {latest_prompt_path.name}",
                "warning",
            )
        return _build_ui_snapshot(session_id)

    with _runtime_lock:
        _runtime_sessions[session_id] = {
            "state": "prompt_generation_in_progress",
            "running": True,
            "logs": [],
            "milestones": _new_milestones(),
            "seen_stage_status": {},
            "code_changes": [],
            "cursor_cli_output": "",
            "cursor_cli_code_blocks": [],
        }
        _set_milestone(_runtime_sessions[session_id]["milestones"], "feature_validation", "in_progress")
        _append_log(_runtime_sessions[session_id], "feature_validation", "Starting intent resolution pipeline.")
    _start_generate_task(session_id, request)
    return _build_ui_snapshot(session_id)


@router.post("/resolve-ambiguities")
def resolve_ambiguities(request: ResolveAmbiguitiesRequest) -> dict[str, Any]:
    session_id = request.session_id.strip()
    session = _state_store().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    resolutions = request.resolutions or _build_resolutions_from_text(session, request.resolution or "")
    if not resolutions:
        raise HTTPException(status_code=400, detail="Provide either non-empty resolutions or resolution.")

    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_generation_in_progress",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            },
        )
        runtime["running"] = True
        runtime["state"] = "prompt_generation_in_progress"
        _set_milestone(runtime["milestones"], "self_learning_agent", "in_progress")
        _append_log(runtime, "self_learning_agent", "Applying ambiguity resolution.")
    _start_resolve_task(session_id, resolutions, request.use_llm_review)
    return _build_ui_snapshot(session_id)


@router.post("/update-prompt")
def update_prompt(request: UpdatePromptRequest) -> dict[str, Any]:
    session_id = request.session_id.strip()
    new_prompt = request.prompt.strip()
    store = _state_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    store.ensure_session(
        session_id=session_id,
        status=str(session.get("status") or "final_prompt_ready"),
        code_generation_prompt=new_prompt,
    )
    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_ready",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            },
        )
        _append_log(runtime, "prompt_generation", "Prompt updated from UI.")
    return _build_ui_snapshot(session_id)


@router.post("/generate-code")
def generate_code_from_prompt(request: GenerateCodeRequest) -> dict[str, Any]:
    session_id = request.session_id.strip()
    store = _state_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_ready",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            },
        )
        if runtime.get("running"):
            raise HTTPException(status_code=409, detail="A run is already in progress for this session.")
        runtime["running"] = True
        runtime["state"] = "code_generation_in_progress"
        runtime["code_preview"] = ""
        runtime["generated_files"] = []
        runtime["code_changes"] = []
        runtime["cursor_cli_output"] = ""
        runtime["cursor_cli_code_blocks"] = []
        _set_milestone(runtime["milestones"], "code_generation", "in_progress")
        _append_log(runtime, "code_generation", "Running Cursor CLI using session prompt.")
    _start_generate_code_task(session_id)
    return _build_ui_snapshot(session_id)


@router.post("/commit-push")
def commit_push(request: CommitPushRequest) -> dict[str, Any]:
    session_id = request.session_id.strip()
    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_ready",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            },
        )
        if runtime.get("running"):
            raise HTTPException(status_code=409, detail="A run is already in progress for this session.")
        runtime["running"] = True
        runtime["state"] = "commit_push_in_progress"
        _set_milestone(runtime["milestones"], "commit_push", "in_progress")
        _append_log(runtime, "commit_push", "Starting commit and push for accepted changes.")

    try:
        with _runtime_lock:
            current = _runtime_sessions.get(session_id, {})
            accepted = dict(current.get("accepted_changes") or {})
            branch = str(current.get("accepted_branch") or "new_feature").strip() or "new_feature"
            suggested_message = str(current.get("suggested_commit_message") or "").strip()
        tracked_paths = [str(p) for p in (accepted.get("tracked_paths") or [])]
        untracked_paths = [str(p) for p in (accepted.get("untracked_paths") or [])]
        commit_paths = sorted(set(tracked_paths + untracked_paths))
        workspace = _resolve_target_repo_path()
        commit_message = (
            str(request.commit_message or "").strip()
            or suggested_message
            or _suggest_commit_message(session_id, tracked_paths, untracked_paths)
        )
        commit_result = _run_commit_only(
            workspace=workspace,
            paths=commit_paths,
            commit_message=commit_message,
        )
        with _runtime_lock:
            runtime = _runtime_sessions[session_id]
            if commit_result.get("success"):
                runtime["state"] = "commit_ready_for_push"
                runtime["committed_branch"] = branch
                runtime["committed_hash"] = str(commit_result.get("commit_hash") or "")
                _append_log(runtime, "commit_push", "Commit completed. Review commit details before pushing.")
                if commit_result.get("commit_hash"):
                    _append_log(runtime, "commit_push", f"Commit hash: {commit_result['commit_hash']}")
                _append_log(runtime, "commit_push", f"Commit message: {commit_message}")
            else:
                runtime["state"] = "failed"
                _set_milestone(runtime["milestones"], "commit_push", "failed")
                _append_log(
                    runtime,
                    "commit_push",
                    str(commit_result.get("error") or "Commit failed."),
                    "error",
                )
    except Exception as exc:  # noqa: BLE001
        with _runtime_lock:
            runtime = _runtime_sessions[session_id]
            runtime["state"] = "failed"
            _set_milestone(runtime["milestones"], "commit_push", "failed")
            _append_log(runtime, "commit_push", str(exc), "error")
    finally:
        with _runtime_lock:
            _runtime_sessions[session_id]["running"] = False
    return _build_ui_snapshot(session_id)


@router.post("/push")
def push_after_commit(request: PushRequest) -> dict[str, Any]:
    session_id = request.session_id.strip()
    with _runtime_lock:
        runtime = _runtime_sessions.setdefault(
            session_id,
            {
                "state": "prompt_ready",
                "running": False,
                "logs": [],
                "milestones": _new_milestones(),
                "seen_stage_status": {},
                "code_changes": [],
                "cursor_cli_output": "",
                "cursor_cli_code_blocks": [],
            },
        )
        if runtime.get("running"):
            raise HTTPException(status_code=409, detail="A run is already in progress for this session.")
        if not str(runtime.get("committed_hash") or "").strip():
            raise HTTPException(status_code=400, detail="Commit step is required before push.")
        runtime["running"] = True
        runtime["state"] = "commit_push_in_progress"
        _set_milestone(runtime["milestones"], "commit_push", "in_progress")
        _append_log(runtime, "commit_push", "Pushing committed changes to remote.")
    try:
        with _runtime_lock:
            current = _runtime_sessions.get(session_id, {})
            branch = str(current.get("committed_branch") or current.get("accepted_branch") or "new_feature").strip() or "new_feature"
        workspace = _resolve_target_repo_path()
        push_result = _run_push_only(workspace=workspace, branch=branch)
        with _runtime_lock:
            runtime = _runtime_sessions[session_id]
            if push_result.get("success"):
                _set_milestone(runtime["milestones"], "commit_push", "completed")
                runtime["state"] = "pipeline_done"
                _append_log(
                    runtime,
                    "commit_push",
                    "Push completed "
                    f"({push_result.get('remote', 'origin')}/{push_result.get('remote_branch', branch)}).",
                )
            else:
                runtime["state"] = "failed"
                _set_milestone(runtime["milestones"], "commit_push", "failed")
                _append_log(runtime, "commit_push", str(push_result.get("error") or "Push failed."), "error")
    except Exception as exc:  # noqa: BLE001
        with _runtime_lock:
            runtime = _runtime_sessions[session_id]
            runtime["state"] = "failed"
            _set_milestone(runtime["milestones"], "commit_push", "failed")
            _append_log(runtime, "commit_push", str(exc), "error")
    finally:
        with _runtime_lock:
            _runtime_sessions[session_id]["running"] = False
    return _build_ui_snapshot(session_id)


@router.get("/git/branches")
def git_history_branches() -> dict[str, Any]:
    workspace = _resolve_target_repo_path()
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace path does not exist: {workspace}")
    if not (workspace / ".git").is_dir():
        raise HTTPException(status_code=400, detail=f"Not a git repository: {workspace}")
    current = _current_branch(workspace)
    branches = _list_local_branches(workspace)
    upstreams = _local_branch_upstream_map(workspace)
    default_remote = _default_git_remote(workspace)
    if current and current not in branches:
        branches.insert(0, current)
    return {
        "current_branch": current,
        "branches": branches,
        "upstreams": upstreams,
        "default_remote": default_remote,
    }


@router.get("/git/commits")
def git_history_commits(branch: str | None = None, limit: int = 150) -> dict[str, Any]:
    workspace = _resolve_target_repo_path()
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace path does not exist: {workspace}")
    if not (workspace / ".git").is_dir():
        raise HTTPException(status_code=400, detail=f"Not a git repository: {workspace}")
    selected_branch = (branch or "").strip() or _current_branch(workspace)
    commits = _list_branch_commits(workspace, selected_branch, max_count=limit)
    upstream_ref = _upstream_ref_for_branch(workspace, selected_branch)
    upstream_head = _rev_parse(workspace, upstream_ref) if upstream_ref else ""
    return {
        "branch": selected_branch,
        "commits": commits,
        "upstream_ref": upstream_ref,
        "upstream_head": upstream_head,
    }


@router.get("/git/commit/{commit_hash}")
def git_history_commit_details(commit_hash: str) -> dict[str, Any]:
    workspace = _resolve_target_repo_path()
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace path does not exist: {workspace}")
    if not (workspace / ".git").is_dir():
        raise HTTPException(status_code=400, detail=f"Not a git repository: {workspace}")

    meta_proc = _run_git_or_raise(
        workspace,
        [
            "git",
            "show",
            "-s",
            "--date=iso",
            "--no-color",
            "--pretty=format:%H%x1f%h%x1f%an%x1f%ae%x1f%ad%x1f%s",
            commit_hash,
        ],
        "resolve commit metadata",
    )
    parts = (meta_proc.stdout or "").strip().split("\x1f")
    if len(parts) < 6:
        raise HTTPException(status_code=500, detail="Failed to parse commit metadata.")
    files = _commit_file_diffs(workspace, commit_hash)
    return {
        "commit": {
            "hash": parts[0],
            "short_hash": parts[1],
            "author": parts[2],
            "author_email": parts[3],
            "date": parts[4],
            "subject": parts[5],
        },
        "files": files,
    }


@router.get("/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _state_store().get_session(session_id)
    with _runtime_lock:
        has_runtime = session_id in _runtime_sessions
    if not session and not has_runtime:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return _build_ui_snapshot(session_id)


@router.get("/artifacts/{session_id}")
def get_artifacts(session_id: str) -> dict[str, Any]:
    session = _state_store().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    manifest = _manifest_for_session(session_id)
    template_outputs = (manifest or {}).get("template_outputs", {}) or {}
    self_learning = (manifest or {}).get("self_learning", {}) or {}

    return {
        "session_id": session_id,
        "status": session.get("status"),
        "template_path": session.get("template_path"),
        "prompt_path": session.get("code_generation_prompt_path"),
        "run_manifest_path": (manifest or {}).get("manifest_path", ""),
        "final_filled_template_path": template_outputs.get("final_filled_template_path", ""),
        "ambiguities": session.get("ambiguity_questions", []) or [],
        "ambiguity_count": len(session.get("ambiguity_questions", []) or []),
        "self_learning": self_learning,
    }
