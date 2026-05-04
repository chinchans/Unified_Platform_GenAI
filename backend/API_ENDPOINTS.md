# Unified Platform API Endpoints

## Framework Coverage

- **FastAPI**: All currently implemented backend endpoints are defined using FastAPI `APIRouter` and mounted in `backend/main.py`.
- **Flask / Django / Express (or other frameworks)**: No endpoint definitions found in this backend codebase.

## Base Routing

All routes are mounted under module prefixes in `backend/main.py`:

- `/api/common`
- `/api/codegen`
- `/api/test-automation`
- `/api/rca`
- `/api/code-assistant`
- `/api/error-fixing`
- `/api/code-review`

---

## Common (`/api/common`)

| Method | Endpoint | Functionality |
|---|---|---|
| GET | `/api/common/health` | Health-check endpoint that returns service status (`ok`). |
| GET | `/api/common/version` | Returns service identity and current API version metadata. |

## Code Generation (`/api/codegen`)

| Method | Endpoint | Functionality |
|---|---|---|
| POST | `/api/codegen/generate` | Starts end-to-end code generation from a user intent and returns either draft prompt output or ambiguity review state. |
| POST | `/api/codegen/resolve-ambiguities` | Resolves pending ambiguity questions for a codegen session and continues generation flow. |
| GET | `/api/codegen/session/{session_id}` | Fetches persisted codegen session state by session ID. |
| GET | `/api/codegen/artifacts/{session_id}` | Returns generated artifact metadata and paths (prompt/template/session outputs) for a session. |

## Test Automation (`/api/test-automation`)

| Method | Endpoint | Functionality |
|---|---|---|
| POST | `/api/test-automation/generate_5g_tests_from_json` | Generates 5G test cases from chunks/CodeGen JSON payload and saves sanitized test cases. |
| POST | `/api/test-automation/upload_5g_chunks_json` | Uploads 5G chunks JSON, extracts text, creates DOCX content output, and stores normalized JSON. |
| POST | `/api/test-automation/analyze_5g_test_coverage` | Computes coverage of generated tests against required 5G call-flow steps. |
| POST | `/api/test-automation/analyze_5g_test_gaps` | Computes requirement/context gaps by comparing uploaded chunk context and generated tests. |
| POST | `/api/test-automation/generate_5g_test_script_from_test_cases` | Generates a Python test script from available test cases via the legacy LLM script-generation pipeline. |

## RCA (`/api/rca`)

| Method | Endpoint | Functionality |
|---|---|---|
| POST | `/api/rca/upload-logs` | Uploads RCA log files and stores file metadata for later analysis. |
| POST | `/api/rca/analyze` | Runs RCA/error-fixing analysis from log text, log file path/name, or crash analysis inputs. |
| POST | `/api/rca/save-analysis` | Persists RCA analysis records to storage/history. |

## Code Assistant (`/api/code-assistant`)

| Method | Endpoint | Functionality |
|---|---|---|
| GET | `/api/code-assistant/bug-history` | Lists saved bug-analysis history with summary metadata and patch counts. |
| GET | `/api/code-assistant/load-analysis/{filename}` | Loads a specific saved bug-analysis file and returns formatted patch/command details. |
| POST | `/api/code-assistant/apply-patches` | Applies selected code/config patches from a saved analysis using the unified patch applicator. |
| POST | `/api/code-assistant/run-investigation` | Executes investigation terminal commands associated with a saved analysis and returns execution results. |
| POST | `/api/code-assistant/git-history/search` | Searches indexed Git commit history for similar fixes and returns ranked matches/selection metadata. |
| POST | `/api/code-assistant/git-history/select` | Selects a specific commit from history and prepares fix-suggestion payloads from commit metadata/diff. |
| POST | `/api/code-assistant/git-commit-push` | Stages changes, commits them, and optionally pushes to remote in the resolved source repository. |

## Error Fixing (`/api/error-fixing`)

| Method | Endpoint | Functionality |
|---|---|---|
| POST | `/api/error-fixing/analyze` | Legacy-compatible direct entrypoint to run the error-fixing analysis pipeline. |

## Code Review (`/api/code-review`)

| Method | Endpoint | Functionality |
|---|---|---|
| POST | `/api/code-review/run` | Runs multi-layer code review orchestration and returns review output. |

---

## Notes

- Endpoint list is derived from FastAPI route decorators in `backend/api/*.py` and router mounting in `backend/main.py`.
- This inventory reflects currently declared routes in code, not external gateways/proxies.
