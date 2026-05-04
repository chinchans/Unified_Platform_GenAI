"""
Minimal 5G test case generator for chunk JSON payloads.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AzureOpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None


logger = logging.getLogger(__name__)


def _load_call_flow_steps() -> List[Dict[str, Any]]:
    """
    Preferred path: backend/resources/5g_call_flow_sequence.json
    Fallback path: temp_intake/test_automation/5g_call_flow_sequence.json
    """
    module_file = os.path.abspath(__file__)
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(module_file)))
    workspace_dir = os.path.dirname(backend_dir)

    preferred = os.path.join(backend_dir, "resources", "5g_call_flow_sequence.json")
    fallback = os.path.join(
        workspace_dir, "temp_intake", "test_automation", "5g_call_flow_sequence.json"
    )

    for path in [preferred, fallback]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                steps = [x for x in data if isinstance(x, dict)]
                logger.info("Loaded %d call-flow steps from %s", len(steps), path)
                return steps
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load call-flow file %s: %s", path, exc)
    return []


CALL_FLOW_STEPS: List[Dict[str, Any]] = _load_call_flow_steps()


class FiveGTestCaseGenerator:
    _MAX_SECTION_CHARS = 15000

    def __init__(self, test_cases_path: str) -> None:
        self.test_cases_path = test_cases_path
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_model = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.azure_client: Optional[AzureOpenAI] = None

        if self.azure_endpoint and self.azure_api_key and AzureOpenAI is not None:
            self.azure_client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version="2024-02-15-preview",
            )

    def _select_candidate_chunks(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks_root = payload.get("chunks", {}) or {}
        expanded = chunks_root.get("expanded") or []
        reranked = chunks_root.get("reranked") or []
        semantic = chunks_root.get("semantic_search") or []

        if isinstance(expanded, list) and expanded:
            return expanded
        if isinstance(reranked, list) and reranked:
            return reranked
        if isinstance(semantic, list) and semantic:
            return semantic

        raise ValueError(
            "No valid chunks found. Expected non-empty chunks.expanded/reranked/semantic_search."
        )

    def _split_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self._MAX_SECTION_CHARS:
            return [text]

        pieces: List[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start : start + self._MAX_SECTION_CHARS].strip())
            start += self._MAX_SECTION_CHARS
        return [p for p in pieces if p]

    def _extract_signal_features(self, text: str) -> Dict[str, Any]:
        low = text.lower()
        protocols = []
        for p in ["f1ap", "ngap", "e1ap", "rrc", "nas"]:
            if p in low:
                protocols.append(p.upper())

        phases = []
        for ph in ["setup", "registration", "session", "release", "handover"]:
            if ph in low:
                phases.append(ph)

        nodes = []
        for node in ["gnb-cu", "gnb-du", "amf", "ue", "cu-up", "cu-cp"]:
            if node in low:
                nodes.append(node)

        return {"protocols": protocols, "phases": phases, "nodes": nodes}

    def _call_flow_hint(self, text: str) -> str:
        if not CALL_FLOW_STEPS:
            return ""
        low = text.lower()
        selected: List[str] = []
        for step in CALL_FLOW_STEPS:
            msg = str(step.get("message") or step.get("Message") or "").strip()
            if not msg:
                continue
            msg_low = msg.lower()
            if msg_low and (msg_low in low or any(tok in low for tok in msg_low.split()[:2])):
                selected.append(msg)
            if len(selected) >= 8:
                break
        return "\n".join(f"- {x}" for x in selected)

    def _build_prompts(self, system_type: str, text: str) -> Tuple[str, str]:
        features = self._extract_signal_features(text)
        flow_hint = self._call_flow_hint(text)

        system_prompt = (
            "You are an expert 5G test engineer. Generate high-quality, implementation-ready test cases. "
            "Return strict JSON only."
        )

        user_prompt = (
            f"System type: {system_type}\n"
            f"Detected protocols: {features['protocols']}\n"
            f"Detected phases: {features['phases']}\n"
            f"Detected nodes: {features['nodes']}\n"
            f"Relevant call-flow steps:\n{flow_hint}\n\n"
            "From the chunk below, generate test cases in JSON format:\n"
            "{\n"
            '  "test_cases": [\n'
            "    {\n"
            '      "title": "...",\n'
            '      "objective": "...",\n'
            '      "preconditions": ["..."],\n'
            '      "steps": ["..."],\n'
            '      "expected_results": ["..."],\n'
            '      "category": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Chunk text:\n{text}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def _extract_json_blob(raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {"test_cases": []}
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return {"test_cases": []}

    def _generate_cases_for_text(self, text: str, system_type: str) -> List[Dict[str, Any]]:
        if not self.azure_client:
            raise RuntimeError(
                "Azure OpenAI client is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )

        system_prompt, user_prompt = self._build_prompts(system_type, text)
        response = self.azure_client.chat.completions.create(
            model=self.azure_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""
        parsed = self._extract_json_blob(content or "")
        cases = parsed.get("test_cases", [])
        return [c for c in cases if isinstance(c, dict)]

    @staticmethod
    def _deduplicate(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for tc in test_cases:
            title = str(tc.get("title", "")).strip().lower()
            objective = str(tc.get("objective", "")).strip().lower()
            key = (title, objective)
            if key in seen:
                continue
            seen.add(key)
            out.append(tc)
        return out

    def _save(self, test_cases: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.test_cases_path), exist_ok=True)
        with open(self.test_cases_path, "w", encoding="utf-8") as fh:
            json.dump(test_cases, fh, indent=2, ensure_ascii=False)

    def generate_from_chunks(self, payload: Dict[str, Any], system_type: str) -> Dict[str, Any]:
        try:
            chunks = self._select_candidate_chunks(payload)

            generated: List[Dict[str, Any]] = []
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("chunk_text") or chunk.get("text") or "").strip()
                if not text:
                    continue

                for sub_text in self._split_text(text):
                    generated.extend(self._generate_cases_for_text(sub_text, system_type))

            generated = self._deduplicate(generated)
            for idx, tc in enumerate(generated, start=1):
                tc["id"] = f"TC-5G-{idx:03d}"

            self._save(generated)
            return {
                "status": "success",
                "message": "5G test cases generated successfully",
                "test_cases": generated,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("5G test generation failed: %s", exc)
            return {"error": str(exc)}
