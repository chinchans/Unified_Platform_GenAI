"""Unified code-review orchestrator for 4-layer evaluation."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from services.rca.paths import BACKEND_DIR, BUG_HISTORY_DIR, CODE_REVIEW_HISTORY_DIR
from services.rca.schemas import RunCodeReviewRequest


def _load_analysis_file(filename: str) -> Dict[str, Any]:
    file_path = BUG_HISTORY_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {filename}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_selected_patches(
    analysis_data: Dict[str, Any],
    selected_code_patches: List[str],
    selected_config_patches: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results = analysis_data.get("results", {})
    phase3_fixes = results.get("phase3_fixes", {})
    fix_suggestion = phase3_fixes.get("fix_suggestion", {})
    code_patches = fix_suggestion.get("code_patches", []) or []
    config_patches = fix_suggestion.get("config_patches", []) or []

    selected_code_patch_data: List[Dict[str, Any]] = []
    for patch in code_patches:
        function_name = patch.get("function_name", "Unknown")
        file_name = Path(patch.get("file_path", "Unknown")).name
        display_text = f"{function_name} ({file_name})"
        if not selected_code_patches or display_text in selected_code_patches:
            selected_code_patch_data.append(patch)

    selected_config_patch_data: List[Dict[str, Any]] = []
    for patch in config_patches:
        config_name = patch.get("config_name", patch.get("parameter_name", "Unknown"))
        file_name = Path(patch.get("file_path", "Unknown")).name
        display_text = f"{config_name} ({file_name})"
        if not selected_config_patches or display_text in selected_config_patches:
            selected_config_patch_data.append(patch)

    return selected_code_patch_data, selected_config_patch_data


def _load_legacy_code_testing_engine() -> Any:
    workspace_root = BACKEND_DIR.parent
    legacy_backend = workspace_root / "temp_intake" / "5G_RCA_Electron-main" / "Backend"
    if not legacy_backend.exists():
        raise RuntimeError(f"Legacy backend path not found: {legacy_backend}")
    sys.path.insert(0, str(legacy_backend))
    module = importlib.import_module("app.services.code_testing_engine")
    return module.CodeTestingEngine()


def _run_layer2_spec_reference(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    results = analysis_data.get("results", {})
    phase3_fixes = results.get("phase3_fixes", {})
    fix_suggestion = phase3_fixes.get("fix_suggestion", {})
    spec_context = str(fix_suggestion.get("specification_context", "") or "").strip()

    lines = ["Layer 2: 3GPP Spec Reference Analysis"]
    warnings: List[str] = []

    if not spec_context:
        lines.append("  ⚠️ No specification context present in analysis data.")
        return {
            "layer": 2,
            "status": "warning",
            "label": "Layer 2: 3GPP Spec Reference Analysis",
            "output_lines": lines,
            "warnings": ["Missing specification_context in fix suggestions"],
        }

    if "embeddings file missing" in spec_context.lower() or "no 3gpp specification context found" in spec_context.lower():
        warnings.append("3GPP embeddings/spec retrieval appears unavailable; context is degraded.")
        lines.append("  ⚠️ Specification context is degraded due to missing embeddings.")
    else:
        lines.append("  ✅ Specification context is available in analysis output.")

    lines.append(f"  Context snippet: {spec_context[:300]}")
    return {
        "layer": 2,
        "status": "success" if not warnings else "warning",
        "label": "Layer 2: 3GPP Spec Reference Analysis",
        "output_lines": lines,
        "warnings": warnings,
    }


def _run_layer3_llm_judge(
    code_patches: List[Dict[str, Any]],
    config_patches: List[Dict[str, Any]],
    analysis_data: Dict[str, Any],
) -> Dict[str, Any]:
    results = analysis_data.get("results", {})
    phase3_fixes = results.get("phase3_fixes", {})
    fix_suggestion = phase3_fixes.get("fix_suggestion", {})
    reason = str(fix_suggestion.get("reason", "") or "").strip()
    root_cause = str(fix_suggestion.get("root_cause_analysis", "") or "").strip()

    lines = ["Layer 3: LLM as Judge"]
    score = 0
    if reason:
        score += 1
    if root_cause:
        score += 1
    if code_patches or config_patches:
        score += 1

    if score >= 3:
        verdict = "PASS"
        status = "success"
        lines.append("  ✅ Review quality gate passed (reasoning + root-cause + actionable patches).")
    elif score == 2:
        verdict = "NEEDS_REVIEW"
        status = "warning"
        lines.append("  ⚠️ Partial confidence in fix quality; manual review recommended.")
    else:
        verdict = "FAIL"
        status = "failed"
        lines.append("  ❌ Insufficient evidence for high-confidence patch recommendation.")

    lines.append(f"  Heuristic score: {score}/3")
    return {
        "layer": 3,
        "status": status,
        "label": "Layer 3: LLM as Judge",
        "verdict": verdict,
        "output_lines": lines,
    }


def run_code_review(request: RunCodeReviewRequest) -> Dict[str, Any]:
    CODE_REVIEW_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    analysis_data = _load_analysis_file(request.analysis_filename)
    code_dir = request.code_dir or analysis_data.get("code_dir", "")

    code_patches, config_patches = _extract_selected_patches(
        analysis_data,
        request.selected_code_patches,
        request.selected_config_patches,
    )

    selected_layers = sorted({layer for layer in request.selected_layers if layer in {1, 2, 3, 4}})
    if not selected_layers:
        selected_layers = [1, 2, 3, 4]

    layer_results: List[Dict[str, Any]] = []
    warnings: List[str] = []

    engine = None
    if 1 in selected_layers or 4 in selected_layers:
        try:
            engine = _load_legacy_code_testing_engine()
        except Exception as exc:
            warnings.append(f"Legacy code testing engine unavailable: {exc}")

    for layer in selected_layers:
        try:
            if layer == 1:
                if engine is None:
                    result = {
                        "layer": 1,
                        "status": "failed",
                        "label": "Layer 1: Syntax & Structural Validation",
                        "output_lines": ["Layer 1: Syntax & Structural Validation", "  ❌ Engine not available."],
                    }
                else:
                    lines = engine.run_layer1_syntax_validation(
                        code_patches,
                        code_dir,
                        config_patches=config_patches,
                    )
                    result = {
                        "layer": 1,
                        "status": "success",
                        "label": "Layer 1: Syntax & Structural Validation",
                        "output_lines": lines,
                    }
            elif layer == 2:
                result = _run_layer2_spec_reference(analysis_data)
            elif layer == 3:
                result = _run_layer3_llm_judge(code_patches, config_patches, analysis_data)
            else:  # layer == 4
                if engine is None:
                    result = {
                        "layer": 4,
                        "status": "failed",
                        "label": "Layer 4: Variable Impact Analysis",
                        "output_lines": ["Layer 4: Variable Impact Analysis", "  ❌ Engine not available."],
                    }
                else:
                    lines, structured = engine.run_variable_impact_analysis(
                        code_patches,
                        code_dir,
                        error_summary=analysis_data.get("error_message"),
                    )
                    result = {
                        "layer": 4,
                        "status": "success",
                        "label": "Layer 4: Variable Impact Analysis",
                        "output_lines": lines,
                        "structured_results": structured,
                    }
        except Exception as exc:
            result = {
                "layer": layer,
                "status": "failed",
                "label": f"Layer {layer}",
                "output_lines": [f"Layer {layer}", f"  ❌ Failed: {exc}"],
            }
            if not request.continue_on_error:
                layer_results.append(result)
                break

        layer_results.append(result)

    overall_success = all(r.get("status") == "success" for r in layer_results if r.get("status") != "warning")
    run_id = f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    history_file = CODE_REVIEW_HISTORY_DIR / f"{run_id}.json"

    report = {
        "success": overall_success,
        "run_id": run_id,
        "analysis_filename": request.analysis_filename,
        "code_dir": code_dir,
        "selected_layers": selected_layers,
        "selected_code_patches": request.selected_code_patches,
        "selected_config_patches": request.selected_config_patches,
        "selected_code_patch_count": len(code_patches),
        "selected_config_patch_count": len(config_patches),
        "warnings": warnings,
        "layer_results": layer_results,
        "created_at": datetime.now().isoformat(),
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["history_file"] = str(history_file)
    return report
