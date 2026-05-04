import json
import os
import sys
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.test_automation.fiveg_test_case_generator import FiveGTestCaseGenerator

router = APIRouter()

RESOURCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
os.makedirs(RESOURCES_DIR, exist_ok=True)

EXTRACTED_CONTENT_PATH = os.path.join(RESOURCES_DIR, "extracted_content.docx")
TEST_CASES_PATH = os.path.join(RESOURCES_DIR, "test_cases.json")
_LEGACY_TSG = None
_NON_5G_KEYWORDS = (
    "wifi",
    "wi-fi",
    "802.11",
    "dot11",
    "dot11be",
    "wlc",
    "access point",
    "ap ",
    "ssid",
    "beacon",
    "probe",
    "association frame",
    "mlo",
)


class Generate5GTestsRequest(BaseModel):
    raw_json: Any
    system_type: Optional[str] = None


class CoverageGapRequest(BaseModel):
    raw_json: Any


class GenerateScriptFromTestCasesRequest(BaseModel):
    test_cases: Optional[list[dict[str, Any]]] = None
    output_filename: Optional[str] = None


def _normalize_chunks_payload(payload: Any) -> dict[str, Any]:
    """
    Normalize different input payloads into the chunks structure expected by generator.

    Supported shapes:
    1) {"chunks": {"expanded"/"reranked"/"semantic_search": [...]}}
    2) CodeGen retrieval output with {"final_context": [{"content": "..."}]}
    """
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    chunks = payload.get("chunks")
    if isinstance(chunks, dict):
        return payload

    final_context = payload.get("final_context")
    if isinstance(final_context, list):
        converted_chunks: list[dict[str, str]] = []
        for item in final_context:
            if not isinstance(item, dict):
                continue
            text = item.get("content") or item.get("text") or ""
            if isinstance(text, str) and text.strip():
                converted_chunks.append({"chunk_text": text.strip()})

        if not converted_chunks:
            raise ValueError("final_context found but no usable content/text values.")

        return {
            "chunks": {"expanded": converted_chunks},
            "intent": payload.get("intent"),
            "timestamp": payload.get("timestamp"),
        }

    raise ValueError(
        "Unsupported JSON structure. Expected either 'chunks' object or CodeGen 'final_context' list."
    )


def _extract_chunk_texts(payload: dict[str, Any]) -> list[str]:
    chunks = payload.get("chunks", {})
    if not isinstance(chunks, dict):
        return []

    texts: list[str] = []

    def collect(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("chunk_text") or item.get("text") or ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    for phase_key in ["expanded", "reranked", "semantic_search"]:
        phase_chunks = chunks.get(phase_key)
        if isinstance(phase_chunks, list):
            collect(phase_chunks)
        elif isinstance(phase_chunks, dict):
            for key in sorted(phase_chunks.keys()):
                collect(phase_chunks.get(key))

    return texts


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _load_test_cases() -> list[dict[str, Any]]:
    if not os.path.exists(TEST_CASES_PATH):
        return []
    try:
        with open(TEST_CASES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except Exception:
        return []


def _save_test_cases(test_cases: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(TEST_CASES_PATH), exist_ok=True)
    with open(TEST_CASES_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_cases, fh, indent=2, ensure_ascii=False)


def _test_case_text(tc: dict[str, Any]) -> str:
    values = [
        tc.get("title", ""),
        tc.get("objective", ""),
        " ".join(tc.get("steps", []) if isinstance(tc.get("steps"), list) else []),
        " ".join(
            tc.get("expected_results", []) if isinstance(tc.get("expected_results"), list) else []
        ),
        " ".join(tc.get("preconditions", []) if isinstance(tc.get("preconditions"), list) else []),
    ]
    return _normalize_text(" ".join(str(v) for v in values))


def _looks_non_5g(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(keyword in normalized for keyword in _NON_5G_KEYWORDS)


def _sanitize_5g_text(value: str, fallback: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if _looks_non_5g(text):
        return fallback
    return text


def _sanitize_5g_steps(values: list[str]) -> list[str]:
    sanitized: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        if _looks_non_5g(text):
            continue
        sanitized.append(text)
    return sanitized


def _sanitize_5g_test_cases(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for idx, tc in enumerate(test_cases, start=1):
        title = _sanitize_5g_text(tc.get("title", ""), fallback=f"5G Test Case {idx}")
        objective = _sanitize_5g_text(tc.get("objective", ""), fallback="Validate 5G procedure behavior")
        preconditions = _sanitize_5g_steps(
            [str(x) for x in tc.get("preconditions", [])] if isinstance(tc.get("preconditions"), list) else []
        )
        steps = _sanitize_5g_steps(
            [str(x) for x in tc.get("steps", [])] if isinstance(tc.get("steps"), list) else []
        )
        expected_results = _sanitize_5g_steps(
            [str(x) for x in tc.get("expected_results", [])]
            if isinstance(tc.get("expected_results"), list)
            else []
        )

        # Keep only meaningful 5G cases.
        if not steps and not expected_results:
            continue

        cloned = dict(tc)
        cloned["title"] = title
        cloned["objective"] = objective
        cloned["preconditions"] = preconditions
        cloned["steps"] = steps
        cloned["expected_results"] = expected_results
        sanitized.append(cloned)
    return sanitized


def _compute_gap_and_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    chunk_texts = _extract_chunk_texts(payload)
    normalized_chunks = [_normalize_text(t) for t in chunk_texts]
    combined_chunk_text = _normalize_text(" ".join(chunk_texts))

    test_cases = _load_test_cases()
    test_case_texts = [_test_case_text(tc) for tc in test_cases]
    combined_test_text = _normalize_text(" ".join(test_case_texts))

    call_flow_steps = FiveGTestCaseGenerator(test_cases_path=TEST_CASES_PATH).CALL_FLOW_STEPS if hasattr(FiveGTestCaseGenerator, "CALL_FLOW_STEPS") else None
    # Fallback to module-level constant if class attribute is absent.
    if call_flow_steps is None:
        from services.test_automation.fiveg_test_case_generator import CALL_FLOW_STEPS as _CALL_FLOW_STEPS

        call_flow_steps = _CALL_FLOW_STEPS

    required_steps: list[dict[str, Any]] = []
    for idx, step in enumerate(call_flow_steps or [], start=1):
        msg = str(step.get("message") or step.get("Message") or "").strip()
        if not msg:
            continue
        required_steps.append({"id": idx, "message": msg})

    covered_in_context: list[dict[str, Any]] = []
    missing_in_context: list[dict[str, Any]] = []
    covered_by_tests: list[dict[str, Any]] = []
    uncovered_by_tests: list[dict[str, Any]] = []

    for step in required_steps:
        normalized_msg = _normalize_text(step["message"])
        in_context = normalized_msg in combined_chunk_text
        in_tests = normalized_msg in combined_test_text

        if in_context:
            covered_in_context.append(step)
        else:
            missing_in_context.append(step)

        if in_tests:
            covered_by_tests.append(step)
        else:
            uncovered_by_tests.append(step)

    total_steps = len(required_steps)
    context_coverage_percent = round((len(covered_in_context) / total_steps) * 100, 2) if total_steps else 0.0
    test_coverage_percent = round((len(covered_by_tests) / total_steps) * 100, 2) if total_steps else 0.0

    return {
        "summary": {
            "required_call_flow_steps": total_steps,
            "uploaded_chunks_count": len(chunk_texts),
            "generated_test_cases_count": len(test_cases),
            "context_coverage_percent": context_coverage_percent,
            "test_coverage_percent": test_coverage_percent,
        },
        "context_analysis": {
            "covered_steps": covered_in_context,
            "missing_steps": missing_in_context,
        },
        "test_coverage_analysis": {
            "covered_steps": covered_by_tests,
            "uncovered_steps": uncovered_by_tests,
        },
    }


def _load_or_validate_test_cases(
    explicit_cases: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if explicit_cases is not None:
        return [x for x in explicit_cases if isinstance(x, dict)]
    return _load_test_cases()


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_test_case_shape(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize multiple testcase schemas into a single internal format.
    Supported input fields include:
    - title/testCaseTitle
    - objective/testObjective
    - preconditions/preConditions
    - steps/testSteps
    - expected_results/expectedResult
    """
    normalized: list[dict[str, Any]] = []
    for idx, tc in enumerate(test_cases, start=1):
        title = (
            tc.get("title")
            or tc.get("testCaseTitle")
            or tc.get("test_case_title")
            or f"Test Case {idx}"
        )
        objective = tc.get("objective") or tc.get("testObjective") or tc.get("description") or ""
        preconditions = _ensure_list(tc.get("preconditions"))
        if not preconditions:
            preconditions = _ensure_list(tc.get("preConditions"))
        steps = _ensure_list(tc.get("steps"))
        if not steps:
            steps = _ensure_list(tc.get("testSteps"))
        expected_results = _ensure_list(tc.get("expected_results"))
        if not expected_results:
            expected_results = _ensure_list(tc.get("expectedResult"))

        title = _sanitize_5g_text(str(title), fallback=f"5G Test Case {idx}")
        objective = _sanitize_5g_text(str(objective), fallback="Validate 5G procedure behavior")
        preconditions = _sanitize_5g_steps(preconditions)
        steps = _sanitize_5g_steps(steps)
        expected_results = _sanitize_5g_steps(expected_results)
        if not steps and not expected_results:
            continue

        normalized.append(
            {
                "id": tc.get("id") or tc.get("testCaseId") or f"TC-{idx:03d}",
                "title": title,
                "objective": objective,
                "preconditions": preconditions,
                "steps": steps,
                "expected_results": expected_results,
                "category": tc.get("category") or tc.get("testType") or "",
                "priority": tc.get("priority") or "",
                "raw": tc,
            }
        )
    return normalized


def _safe_python_identifier(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"tc_{cleaned}"
    return cleaned


def _build_python_script_from_test_cases(test_cases: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        '"""Auto-generated 5G test script from test cases."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Dict, Any",
        "",
        "",
        "def _run_step(step: str) -> None:",
        '    print(f"[STEP] {step}")',
        "",
        "",
        "def _assert_expected(expected: str) -> None:",
        '    print(f"[EXPECT] {expected}")',
        "",
        "",
    ]

    runner_calls: list[str] = []
    for idx, tc in enumerate(test_cases, start=1):
        title = str(tc.get("title") or f"Test Case {idx}").strip()
        objective = str(tc.get("objective") or "").strip()
        func_name = _safe_python_identifier(title, f"test_case_{idx}")
        steps = tc.get("steps", [])
        expected_results = tc.get("expected_results", [])
        preconditions = tc.get("preconditions", [])

        lines.extend(
            [
                f"def {func_name}() -> Dict[str, Any]:",
                f'    """{title}"""',
                f'    print("\\n=== {title} ===")',
            ]
        )
        if objective:
            lines.append(f'    print("Objective: {objective}")')
        if isinstance(preconditions, list) and preconditions:
            lines.append("    print('Preconditions:')")
            for pre in preconditions:
                lines.append(f"    _run_step({pre!r})")
        if isinstance(steps, list) and steps:
            lines.append("    print('Execution Steps:')")
            for step in steps:
                lines.append(f"    _run_step({step!r})")
        if isinstance(expected_results, list) and expected_results:
            lines.append("    print('Expected Results:')")
            for exp in expected_results:
                lines.append(f"    _assert_expected({exp!r})")
        lines.extend(
            [
                "    return {",
                f"        'title': {title!r},",
                "        'status': 'PASS',",
                "    }",
                "",
                "",
            ]
        )
        runner_calls.append(f"    results.append({func_name}())")

    lines.extend(
        [
            "def run_all_tests() -> list[Dict[str, Any]]:",
            "    results: list[Dict[str, Any]] = []",
            *runner_calls,
            "    return results",
            "",
            "",
            "if __name__ == '__main__':",
            "    summary = run_all_tests()",
            '    print("\\n=== TEST SUMMARY ===")',
            "    for item in summary:",
            "        print(f\"- {item['title']}: {item['status']}\")",
            "",
        ]
    )

    return "\n".join(lines)


def _get_legacy_test_script_generator():
    global _LEGACY_TSG
    if _LEGACY_TSG is not None:
        return _LEGACY_TSG

    # Reuse original implementation from legacy backend "as-is".
    workspace_dir = os.path.dirname(os.path.dirname(RESOURCES_DIR))
    legacy_backend = os.path.join(workspace_dir, "temp_intake", "5G_RCA_Electron-main", "Backend")
    if not os.path.exists(legacy_backend):
        raise RuntimeError(f"Legacy backend path not found: {legacy_backend}")

    if legacy_backend not in sys.path:
        sys.path.insert(0, legacy_backend)

    from app.services.test_script_generator import TestScriptGenerator  # type: ignore

    _LEGACY_TSG = TestScriptGenerator()
    return _LEGACY_TSG


@router.post("/generate_5g_tests_from_json")
def generate_5g_tests_from_json(req: dict[str, Any]) -> dict[str, Any]:
    """
    Generate 5G test cases from:
    1) wrapped body: {"raw_json": <string|object>, "system_type": "..."}
    2) direct CodeGen/chunks JSON pasted as request body
    """
    try:
        raw_json_value = req.get("raw_json")
        if raw_json_value is None:
            payload_raw = req
        elif isinstance(raw_json_value, str):
            payload_raw = json.loads(raw_json_value)
        elif isinstance(raw_json_value, dict):
            payload_raw = raw_json_value
        else:
            raise HTTPException(
                status_code=400,
                detail="raw_json must be either a JSON string or a JSON object.",
            )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in raw_json string: {exc}") from exc

    try:
        payload = _normalize_chunks_payload(payload_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    system_type = str(req.get("system_type") or "5G F1AP UE CONTEXT SETUP")
    try:
        generator = FiveGTestCaseGenerator(test_cases_path=TEST_CASES_PATH)
        result = generator.generate_from_chunks(payload, system_type=system_type)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"5G test generation failed: {exc}") from exc

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    generated_cases = result.get("test_cases", [])
    if isinstance(generated_cases, list):
        sanitized_cases = _sanitize_5g_test_cases([x for x in generated_cases if isinstance(x, dict)])
        _save_test_cases(sanitized_cases)
        result["test_cases"] = sanitized_cases

    return {
        "status": result.get("status", "success"),
        "test_case_count": len(result.get("test_cases", [])),
        "message": result.get("message", "5G Test Cases generated"),
    }


@router.post("/upload_5g_chunks_json")
async def upload_5g_chunks_json(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Upload 5G chunks JSON, create extracted DOCX, and persist uploaded JSON.
    """
    global EXTRACTED_CONTENT_PATH
    try:
        raw_bytes = await file.read()
        payload_raw = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

    try:
        payload = _normalize_chunks_payload(payload_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = payload.get("chunks", {})
    if not isinstance(chunks, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON structure: 'chunks' must be an object.")

    texts: list[str] = []

    def collect_from_list(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("chunk_text") or item.get("text") or ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    for phase_key in ["expanded", "reranked", "semantic_search"]:
        phase_chunks = chunks.get(phase_key)
        if isinstance(phase_chunks, list):
            collect_from_list(phase_chunks)
        elif isinstance(phase_chunks, dict):
            for key in sorted(phase_chunks.keys()):
                collect_from_list(phase_chunks.get(key))

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="No chunk_text found under chunks.expanded/reranked/semantic_search.",
        )

    try:
        from docx import Document

        doc = Document()
        for chunk_text in texts:
            for para in chunk_text.split("\n\n"):
                para = para.strip()
                if para:
                    doc.add_paragraph(para)

        timestamp = payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_timestamp = str(timestamp).replace(":", "").replace(" ", "_")
        output_filename = f"extracted_5g_{safe_timestamp}.docx"
        output_path = os.path.join(RESOURCES_DIR, output_filename)
        doc.save(output_path)

        uploaded_5g_json_path = os.path.join(RESOURCES_DIR, "uploaded_5g_chunks.json")
        with open(uploaded_5g_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        EXTRACTED_CONTENT_PATH = output_path
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"5G chunks JSON upload failed: {exc}") from exc

    return {
        "status": "success",
        "message": f"5G chunks JSON uploaded and converted to DOCX at {output_path}",
        "content_path": output_path,
        "chunks_json_path": uploaded_5g_json_path,
    }


@router.post("/analyze_5g_test_coverage")
def analyze_5g_test_coverage(req: CoverageGapRequest) -> dict[str, Any]:
    """
    Analyze end-to-end coverage of generated test cases against the 5G call-flow steps.
    """
    try:
        payload = _normalize_chunks_payload(req.raw_json)
        analysis = _compute_gap_and_coverage(payload)
        return {
            "status": "success",
            "message": "5G test coverage analysis completed",
            "coverage": analysis["summary"],
            "test_coverage_analysis": analysis["test_coverage_analysis"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Coverage analysis failed: {exc}") from exc


@router.post("/analyze_5g_test_gaps")
def analyze_5g_test_gaps(req: CoverageGapRequest) -> dict[str, Any]:
    """
    Analyze requirement gaps by comparing uploaded chunk context with 5G call-flow steps.
    """
    try:
        payload = _normalize_chunks_payload(req.raw_json)
        analysis = _compute_gap_and_coverage(payload)
        return {
            "status": "success",
            "message": "5G test gap analysis completed",
            "coverage": analysis["summary"],
            "context_analysis": analysis["context_analysis"],
            "test_coverage_analysis": analysis["test_coverage_analysis"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {exc}") from exc


@router.post("/generate_5g_test_script_from_test_cases")
def generate_5g_test_script_from_test_cases(
    req: GenerateScriptFromTestCasesRequest,
) -> dict[str, Any]:
    """
    Generate test script from test cases using the original legacy LLM pipeline.
    Uses request test_cases if provided; otherwise reads backend/resources/test_cases.json.
    """
    try:
        test_cases = _load_or_validate_test_cases(req.test_cases)
        if not test_cases:
            raise HTTPException(
                status_code=400,
                detail="No test cases available. Generate test cases first or provide test_cases in request body.",
            )

        normalized_cases = _normalize_test_case_shape(test_cases)
        generator = _get_legacy_test_script_generator()

        # Feed normalized testcases as dataset text to legacy generator.
        text_content = json.dumps(normalized_cases, indent=2, ensure_ascii=False)
        prompts = generator.get_prompts()
        selected_prompt = prompts.get("Test Script")
        if isinstance(selected_prompt, dict):
            # Defensive handling if storage format changed unexpectedly.
            selected_prompt = json.dumps(selected_prompt, ensure_ascii=False)
        if not isinstance(selected_prompt, str) or not selected_prompt.strip():
            raise RuntimeError("Legacy 'Test Script' prompt template is not available.")

        generator.current_prompt_key = "Test Script"
        generator.testcases_name = req.output_filename or "provided_test_cases"
        generator.set_variables(
            domain="Network Infrastructure",
            system_type="5G",
            primary_feature="Attach",
            connection_method="SSH",
            login_credentials="admin@<IP>",
            access_mode="CLI mode",
            language="Python",
        )
        script_content = generator.generate_response_from_text(text_content, selected_prompt)
        if not script_content or str(script_content).startswith("Error:"):
            raise RuntimeError(f"LLM generation failed: {script_content}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = req.output_filename or f"generated_5g_test_script_{ts}.py"
        if not output_filename.endswith(".py"):
            output_filename += ".py"
        output_path = os.path.join(RESOURCES_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(script_content)

        return {
            "status": "success",
            "message": "5G test script generated successfully from test cases (LLM pipeline)",
            "test_case_count": len(normalized_cases),
            "script_path": output_path,
            "script_preview": script_content[:1500],
            "generation_mode": "legacy_llm_pipeline",
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Test script generation failed: {exc}") from exc
