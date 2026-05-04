"""Orchestrate CompleteErrorFixingPipeline (ported from legacy Electron backend)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from services.rca.paths import PIPELINE_DIR
from services.rca.schemas import ErrorAnalysisRequest


def is_pipeline_available() -> bool:
    try:
        from services.rca.Error_fixing_pipelin import CompleteErrorFixingPipeline  # noqa: F401

        return True
    except Exception:
        return False


def run_error_fixing_analyze(request: ErrorAnalysisRequest) -> Dict[str, Any]:
    """
    Run the full error-fixing / RCA pipeline.
    Raises RuntimeError if the pipeline package cannot be loaded or inputs are invalid.
    """
    if not is_pipeline_available():
        raise RuntimeError("Error fixing pipeline not available (import failed)")

    from services.rca.Error_fixing_pipelin import CompleteErrorFixingPipeline
    from services.rca.Error_fixing_pipelin.complete_error_fixing_pipeline import (
        generate_terminal_commands,
    )
    from services.rca.Error_fixing_pipelin.fix_suggestion_pipeline import FixSuggestionPipeline
    from services.rca.Error_fixing_pipelin.parse_log_context import LogContextParser

    pipeline_dir = str(PIPELINE_DIR.resolve())
    if not os.path.isdir(pipeline_dir):
        raise RuntimeError(f"Error fixing pipeline directory not found: {pipeline_dir}")

    if request.crash_analysis and not request.log_file_path:
        raise RuntimeError("crash_analysis requires log_file_path")

    original_cwd = os.getcwd()
    os.chdir(pipeline_dir)
    result: Dict[str, Any] = {}
    try:
        pipeline = CompleteErrorFixingPipeline(
            openair_codebase_file_name=request.openair_codebase_name
        )

        error_message = request.error_message
        log_file_path = request.log_file_path

        if not error_message and log_file_path:
            try:
                log_parser = LogContextParser(
                    openair_codebase_file_name=request.openair_codebase_name
                )
                error_message = log_parser.extract_error_message(log_file_path)
            except Exception:
                pass

            # Integration fallback: if parser returns no error (or import/runtime failure),
            # still extract a best-effort line from the log before invoking pipeline.
            if not error_message:
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    error_keywords = [
                        "error",
                        "ERROR",
                        "fail",
                        "FAIL",
                        "exception",
                        "EXCEPTION",
                        "fatal",
                        "FATAL",
                    ]
                    for line in content.split("\n"):
                        if any(k in line for k in error_keywords):
                            error_message = line.strip()
                            if error_message:
                                break
                except Exception:
                    pass

        relative_log_path = log_file_path
        if log_file_path and os.path.isabs(log_file_path):
            relative_log_path = os.path.relpath(log_file_path, pipeline_dir)

        if request.crash_analysis:
            print("Running complete crash analysis pipeline...")
            result = pipeline.process_crash_analysis(relative_log_path, phase="full")
            error_message = result.get("error_message", "Segmentation Fault Detected")

            crash_output_file = os.path.join(pipeline_dir, "output/crash_phase3_fixes.json")
            if os.path.exists(crash_output_file):
                try:
                    with open(crash_output_file, "r", encoding="utf-8") as f:
                        crash_detailed = json.load(f)
                        result["crash_detailed_fixes"] = crash_detailed
                except Exception:
                    pass

            if "crash_detailed_fixes" in result:
                crash_fixes = result["crash_detailed_fixes"]
                result["phase3_fixes"] = {"fix_suggestion": crash_fixes.get("fix_suggestion", {})}
                result["fix_suggestions"] = {
                    "fix_suggestion": crash_fixes.get("fix_suggestion", {}),
                    "error_text": error_message,
                }

            if "crash_info" not in result:
                result["crash_info"] = result.get("extraction_summary", {})
            if "backtrace" not in result and "crash_info" in result:
                result["backtrace"] = result["crash_info"].get("backtrace", [])
            if "scenario_flow" not in result and "crash_info" in result:
                result["scenario_flow"] = result["crash_info"].get("scenario_flow", [])
        else:
            custom_context = (
                request.custom_deployment_context
                if request.custom_deployment_context
                else None
            )
            result = pipeline.process_error_with_context(
                error_message,
                relative_log_path,
                custom_deployment_context=custom_context,
            )

        phase3_fixes = result.get("phase3_fixes", {})
        fix_suggestion = phase3_fixes.get("fix_suggestion", {})

        try:
            phase3_fixes = result.get("phase3_fixes", {})
            fix_suggestion = phase3_fixes.get("fix_suggestion", {})
            investigation_steps = fix_suggestion.get("investigation_steps", [])
            deployment_context = result.get("deployment_context")

            troubleshooting_hints: list = []
            try:
                patterns_file = os.path.join(pipeline_dir, "database", "error_patterns_structured.json")
                with open(patterns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    patterns = data.get("patterns", {})
                    err_l = (error_message or "").lower()
                    pattern_found = False
                    for _name, pattern_data in patterns.items():
                        keywords = pattern_data.get("keywords", [])
                        if any(k in err_l for k in keywords):
                            troubleshooting_hints.extend(pattern_data.get("suggested_fixes", []))
                            pattern_found = True
                            break
                    if not pattern_found and error_message:
                        fix_pipeline = FixSuggestionPipeline(
                            openair_codebase_file_name=request.openair_codebase_name
                        )
                        dynamic_pattern = fix_pipeline._generate_dynamic_error_pattern(error_message)
                        fix_pipeline._add_pattern_to_json(error_message, dynamic_pattern)
                        troubleshooting_hints.extend(dynamic_pattern.get("suggested_fixes", []))
            except Exception:
                troubleshooting_hints = [
                    "Validate network configuration and parameters in config files",
                    "Check network reachability between endpoints",
                    "Verify protocol-specific configuration settings",
                    "Review error logs for additional context",
                ]

            terminal_commands = generate_terminal_commands(
                error_message=error_message or "",
                investigation_steps=investigation_steps,
                deployment_context=deployment_context,
                troubleshooting_hints=troubleshooting_hints,
                openair_codebase_file_name=request.openair_codebase_name,
            )

            result["phase4_commands"] = {
                "terminal_commands": terminal_commands,
                "command_count": len(terminal_commands),
            }

            fix_suggestions_file = os.path.join(pipeline_dir, "output", "fix_suggestions.json")
            fix_suggestions_data = result.get("phase3_fixes", {}).copy()
            fix_suggestions_data["terminal_commands"] = result.get("phase4_commands", {})
            os.makedirs(os.path.dirname(fix_suggestions_file), exist_ok=True)
            with open(fix_suggestions_file, "w", encoding="utf-8") as f:
                json.dump(fix_suggestions_data, f, indent=2, ensure_ascii=False)

            output_file = os.path.join(pipeline_dir, "output", "complete_error_analysis.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            if result.get("deployment_context"):
                context_file = os.path.join(pipeline_dir, "output", "deployment_context.json")
                os.makedirs(os.path.dirname(context_file), exist_ok=True)
                with open(context_file, "w", encoding="utf-8") as f:
                    json.dump(result["deployment_context"], f, indent=2, ensure_ascii=False)

            summary_file = os.path.join(pipeline_dir, "output", "error_fix_summary.txt")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("Error Fix Summary Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Error: {error_message}\n")
                f.write(f"Log File: {log_file_path or 'None'}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if result.get("deployment_context"):
                    ctx = result["deployment_context"]
                    f.write("Deployment Context:\n")
                    f.write(f"- Role: {ctx.get('role', 'Unknown')}\n")
                    f.write(f"- Active Configs: {len(ctx.get('active_configs', []))}\n")
                    network_params = ctx.get("network_params", {})
                    f.write(
                        f"- Network: gNB={network_params.get('gnb_ipv4', 'Unknown')}, "
                        f"AMF={network_params.get('amf_ipv4', 'Unknown')}\n\n"
                    )
                phase2 = result.get("phase2_analysis", {})
                f.write("Phase 2 Results:\n")
                f.write(f"- Retrieval Method: {phase2.get('retrieval_method', 'standard')}\n")
                f.write(f"- Functions: {len(phase2.get('suspected_functions', []))}\n")
                f.write(f"- Configs: {len(phase2.get('suspected_configs', []))}\n\n")
                phase3 = result.get("phase3_fixes", {})
                fs = phase3.get("fix_suggestion", {})
                reason = fs.get("reason", "Not provided") or "Not provided"
                if isinstance(reason, str):
                    reason_snip = reason[:200]
                else:
                    reason_snip = str(reason)[:200]
                f.write("Phase 3 Results:\n")
                f.write(f"- Root Cause: {reason_snip}...\n")
                f.write(
                    "- Fix Available: "
                    f"{'Yes' if fs.get('config_fix') or fs.get('code_patch') else 'No'}\n"
                )

            phase3_fixes = result.get("phase3_fixes", {})
            if "fix_suggestion" not in phase3_fixes:
                phase3_fixes["fix_suggestion"] = {}
            phase3_fixes["fix_suggestion"]["investigation_commands"] = [
                {"command": cmd.get("command", ""), "hint": cmd.get("explanation", "")}
                for cmd in terminal_commands
            ]

            result["summary"] = {
                "total_functions_analyzed": len(
                    phase3_fixes.get("fix_suggestion", {}).get("suspected_functions", [])
                ),
                "total_configs_analyzed": len(
                    phase3_fixes.get("fix_suggestion", {}).get("suspected_configs", [])
                ),
                "call_graph_entries": len(result.get("call_graph_context", [])),
                "pattern_matched": True,
                "analysis_completed": True,
            }
            result["timestamp"] = datetime.now().isoformat()
            result["log_file"] = request.log_file_path
        except Exception as e:
            print(f"Could not generate investigation commands: {e}")
            import traceback

            traceback.print_exc()

        formatted_output = ""
        try:
            output_dir = os.path.join(pipeline_dir, "output")
            llm_files = [
                f
                for f in os.listdir(output_dir)
                if f.startswith("llm_prompts_") and f.endswith(".txt")
            ]
            if llm_files:
                latest_file = max(llm_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
                with open(os.path.join(output_dir, latest_file), "r", encoding="utf-8") as f:
                    content = f.read()
                if "RCA ANALYSIS RESULTS" in content:
                    formatted_output = content[content.find("RCA ANALYSIS RESULTS") :]
                elif "SUSPECTED FUNCTIONS" in content:
                    formatted_output = content[content.find("SUSPECTED FUNCTIONS") :]
        except Exception:
            formatted_output = ""

    finally:
        os.chdir(original_cwd)

    fix_suggestions_data = None
    try:
        fix_suggestions_file = os.path.join(pipeline_dir, "output", "fix_suggestions.json")
        if os.path.exists(fix_suggestions_file):
            with open(fix_suggestions_file, "r", encoding="utf-8") as f:
                fix_suggestions_data = json.load(f)
    except Exception:
        pass

    deployment_context_extended: Dict[str, Any] = {}
    try:
        patterns_file = os.path.join(pipeline_dir, "database", "error_patterns_structured.json")
        if os.path.exists(patterns_file):
            with open(patterns_file, "r", encoding="utf-8") as f:
                patterns_data = json.load(f)
                deployment_context_extended = patterns_data.get("deployment_context", {})
    except Exception:
        pass

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "formatted_output": formatted_output,
        "fix_suggestions": fix_suggestions_data,
        "deployment_context_extended": deployment_context_extended,
    }
