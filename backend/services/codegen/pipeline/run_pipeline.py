from __future__ import annotations

import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Allow running as: python backend/services/codegen/pipeline/run_pipeline.py
CODEGEN_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

def main() -> None:
    output_dir = CODEGEN_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_log = output_dir / f"run_pipeline_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    user_intent = "gNB-CU has to prepare and send F1AP 'UE CONTEXT SETUP REQUEST' message to the candidate gNB-DU and candidate gNB-DU has to respond with F1AP 'UE CONTEXT SETUP RESPONSE' message and this message has to be handled on gNB-CU for Inter-gNB-DU LTM handover."
    

    def _log(msg: str) -> None:
        print(msg, flush=True)
        with open(debug_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log(f"[Code_Gen] Starting end-to-end pipeline from intent.")
    _log(f"[Code_Gen] Debug log: {debug_log}")

    try:
        from backend.services.codegen.pipeline.pipeline import (
            run_end_to_end_from_intent,
            run_resolve_self_learning_session,
        )
    except BaseException:
        _log("[Code_Gen] Failed while importing codegen pipeline entrypoint")
        _log(traceback.format_exc())
        raise

    def _collect_plain_text_resolutions(ambiguities: list[Dict[str, Any]]) -> Dict[str, str]:
        resolutions: Dict[str, str] = {}
        print("\n[Code_Gen] Enter plain-text resolutions for ambiguities.", flush=True)
        print("[Code_Gen] Leave blank to skip an item.\n", flush=True)
        for idx, item in enumerate(ambiguities, start=1):
            amb_id = str(item.get("id", "")).strip()
            question = str(item.get("question", "")).strip()
            if not amb_id:
                continue
            print(f"[{idx}] id={amb_id}", flush=True)
            if question:
                print(f"    question: {question}", flush=True)
            value = input("    resolution> ").strip()
            if value:
                resolutions[amb_id] = value
        return resolutions

    result = run_end_to_end_from_intent(user_intent)
    _log("[Code_Gen] Pipeline completed.")
    _log(f"[Code_Gen] Run manifest: {result.get('run_manifest_path', '')}")
    _log(f"[Code_Gen] Final template: {result.get('final_filled_template_path', '')}")
    _log(f"[Code_Gen] Prompt file: {result.get('code_generation_prompt_path', '')}")

    # Surface ambiguity status clearly for terminal users.
    session_id = str(result.get("session_id", "") or "")
    self_learning = result.get("self_learning", {}) or {}
    has_ambiguities = bool(
        result.get("self_learning_has_ambiguities")
        or self_learning.get("has_ambiguities")
    )
    ambiguities = (
        result.get("self_learning_ambiguities")
        or self_learning.get("ambiguities")
        or []
    )

    if has_ambiguities:
        _log("[Code_Gen] Ambiguity review required before prompt generation.")
        _log(f"[Code_Gen] Session ID: {session_id}")
        _log(f"[Code_Gen] Ambiguities count: {len(ambiguities)}")
        for idx, item in enumerate(ambiguities, start=1):
            _log(f"[Code_Gen][Ambiguity {idx}] {item}")

        apply_now = input(
            "\n[Code_Gen] Provide resolutions now in terminal? (y/n): "
        ).strip().lower()
        while apply_now == "y" and has_ambiguities:
            resolutions = _collect_plain_text_resolutions(ambiguities)
            if not resolutions:
                _log("[Code_Gen] No resolutions entered. Stopping interactive resolution.")
                break

            result = run_resolve_self_learning_session(
                session_id=session_id,
                user_resolutions=resolutions,
            )
            has_ambiguities = bool(result.get("self_learning_has_ambiguities"))
            ambiguities = result.get("self_learning_ambiguities") or []
            _log(f"[Code_Gen] Resolution applied. Remaining ambiguities: {len(ambiguities)}")

            if has_ambiguities:
                for idx, item in enumerate(ambiguities, start=1):
                    _log(f"[Code_Gen][Remaining {idx}] {item}")
                apply_now = input(
                    "\n[Code_Gen] More ambiguities remain. Provide more resolutions? (y/n): "
                ).strip().lower()

        if not has_ambiguities:
            _log("[Code_Gen] All ambiguities resolved.")
            _log(f"[Code_Gen] Prompt file: {result.get('code_generation_prompt_path', '')}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[Code_Gen] Pipeline failed with exception:", flush=True)
        traceback.print_exc()
        raise
