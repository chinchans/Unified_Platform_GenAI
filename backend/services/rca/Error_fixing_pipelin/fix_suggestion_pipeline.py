#!/usr/bin/env python3
"""
Phase 3 – Fix Suggestion Pipeline

This script implements the final phase of the error-fixing pipeline, taking the outputs
from Phase 2 (error analysis, candidate functions/configs, call graph context) and using
Azure OpenAI GPT-4o-mini to generate specific fix suggestions.

Author: AI Assistant
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase 3 completion limits (output tokens). Build logs need larger budgets to avoid truncated JSON.
_DEFAULT_MAX_PATCH_TOKENS = 8000
_BUILD_ERROR_MAX_PATCH_TOKENS = 16384
_RUNTIME_CONFIG_FILES = ("cu_gnb.conf", "du_gnb.conf", "ue.conf", "5g_sa_ue.conf")


def _parse_fix_suggestion_json(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM output into the fix-suggestion dict. Returns None if no valid JSON object.
    """
    if not response_content or not response_content.strip():
        return None
    text = response_content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            try:
                cleaned = json_content.replace(",\n}", "\n}").replace(",\n]", "\n]")
                cleaned = re.sub(r"//.*?(?=\n|$)", "", cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        json_text = json_match.group(0)
        try:
            json_text_cleaned = re.sub(r"//.*?(?=\n|$)", "", json_text)
            return json.loads(json_text_cleaned)
        except json.JSONDecodeError:
            pass
    return None


def _parse_cmake_diagnosis_json(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse Stage-1 CMake diagnosis JSON (same tolerant parser, stricter structure checked later).
    """
    parsed = _parse_fix_suggestion_json(response_content)
    if not isinstance(parsed, dict):
        return None
    return parsed


@dataclass
class FixContext:
    """Data structure for fix suggestion context"""
    error_text: str
    candidate_functions: List[Dict[str, Any]]
    candidate_configs: List[Dict[str, Any]]
    call_graph_context: List[Dict[str, Any]]
    matched_pattern: Optional[Dict[str, Any]]
    
@dataclass
class CodePatch:
    """Data structure for individual code patches"""
    function_name: str
    file_path: str
    patch_type: str  # "modification", "addition", "replacement"
    original_code: str
    patched_code: str
    line_numbers: str
    description: str

@dataclass
class ConfigPatch:
    """Configuration patch details"""
    config_name: str
    file_path: str
    patch_type: str  # "set_value", "add_line", "modify_value"
    current_value: str
    new_value: str
    line_number: str
    relevance_score: float
    description: str

@dataclass
class FixSuggestion:
    """Data structure for fix suggestions"""
    suspected_functions: List[str]
    suspected_configs: List[str]
    reason: str
    config_fix: str
    code_patches: List[CodePatch]
    config_patches: List[ConfigPatch]
    root_cause_analysis: str
    investigation_steps: List[str]
    specification_context: str

class FixSuggestionPipeline:
    """Main pipeline for generating fix suggestions"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the fix suggestion pipeline.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🔧 Initializing Fix Suggestion Pipeline...")
        
        # Store the codebase folder name for path construction
        self.openair_codebase_file_name = openair_codebase_file_name
        
        # Setup Azure OpenAI client
        self._setup_azure_client()
        
        # Load functions database for function definition lookup
        self._load_functions_database()
        
        logger.info("✅ Fix Suggestion Pipeline initialized successfully")

    @staticmethod
    def _is_build_error_mode(deployment_context: Optional[Dict[str, Any]]) -> bool:
        """True for elevated patch budget / build-time handling (compile or CMake log kind)."""
        if not isinstance(deployment_context, dict):
            return False
        if deployment_context.get("dependency_advice_mode"):
            return False
        return bool(
            deployment_context.get("build_error_mode")
            or deployment_context.get("cmake_build_system_mode")
        )

    def _is_compile_diagnostic_log(self, deployment_context: Optional[Dict[str, Any]]) -> bool:
        """True for compile-fix prompts: build kind and/or parsed compiler lines, never CMake mode."""
        return self._compile_mode_relax_patch_grounding(deployment_context)

    @staticmethod
    def _is_supported_runtime_config_file(file_path: str) -> bool:
        """Allow-list for runtime deployment config files."""
        p = (file_path or "").replace("\\", "/").lower()
        return any(name in p for name in _RUNTIME_CONFIG_FILES)

    @staticmethod
    def _is_linker_undefined_reference_error(error_text: str) -> bool:
        """True when error text is primarily a linker undefined-reference failure."""
        t = (error_text or "").lower()
        if not t:
            return False
        return ("undefined reference to" in t) or ("/usr/bin/ld:" in t)

    def _is_supported_config_patch_file(
        self, file_path: str, deployment_context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Allow config patch files:
        - Always: known runtime deployment configs (.conf)
        - Only when log kind is CMake/build-system: CMakeLists.txt / *.cmake
        (Compile-only build logs do not accept CMake file patches here — keeps paths separate.)
        """
        p = (file_path or "").replace("\\", "/").lower()
        if self._is_supported_runtime_config_file(p):
            return True
        dc = deployment_context if isinstance(deployment_context, dict) else {}
        if dc.get("cmake_build_system_mode") and (
            p.endswith(".cmake") or p.endswith("/cmakelists.txt") or p == "cmakelists.txt"
        ):
            return True
        return False

    @staticmethod
    def _normalize_code_for_compare(code: str) -> str:
        """
        Normalize code snippets for semantic-ish equality checks.
        We intentionally keep this conservative (whitespace/comments only),
        because we only use it to detect obvious no-op / duplicate patches.
        """
        if not code:
            return ""
        # Normalize newlines and strip surrounding whitespace.
        s = str(code).replace("\r\n", "\n").replace("\r", "\n").strip()
        # Drop empty lines and collapse runs of whitespace.
        s = "\n".join([ln.rstrip() for ln in s.split("\n") if ln.strip() != ""])
        s = re.sub(r"[ \t]+", " ", s)
        return s

    def _filter_code_patches(self, code_patches: List["CodePatch"]) -> List["CodePatch"]:
        """
        Filter out clearly bad patches:
        - no-op patches where original_code == patched_code (after normalization)
        - empty original/patched snippets
        - explicit 'no change needed' descriptions
        - duplicates (same normalized original+patched for same file/function)
        """
        if not code_patches:
            return []

        kept: List[CodePatch] = []
        seen: set = set()
        filtered_noop = 0
        filtered_empty = 0
        filtered_desc = 0
        filtered_dupe = 0

        for p in code_patches:
            if not p:
                continue
            orig = (p.original_code or "").strip()
            pat = (p.patched_code or "").strip()
            if not orig or not pat:
                filtered_empty += 1
                continue

            desc = (p.description or "").strip().lower()
            if "no change needed" in desc or "no changes needed" in desc:
                filtered_desc += 1
                continue

            n_orig = self._normalize_code_for_compare(orig)
            n_pat = self._normalize_code_for_compare(pat)
            if n_orig == n_pat:
                filtered_noop += 1
                continue

            dupe_key = (
                (p.file_path or "").strip().lower(),
                (p.function_name or "").strip().lower(),
                n_orig,
                n_pat,
            )
            if dupe_key in seen:
                filtered_dupe += 1
                continue
            seen.add(dupe_key)
            kept.append(p)

        if filtered_empty or filtered_desc or filtered_noop or filtered_dupe:
            logger.warning(
                "🧹 Filtered code patches: empty=%s, desc_no_change=%s, noop=%s, dupes=%s (kept=%s/%s)",
                filtered_empty,
                filtered_desc,
                filtered_noop,
                filtered_dupe,
                len(kept),
                len(code_patches),
            )

        return kept

    def _pipeline_package_dir(self) -> str:
        """Directory containing this module (Error_fixing_pipelin package root)."""
        return os.path.dirname(os.path.abspath(__file__))

    def _codebase_root_dir(self) -> str:
        """
        Absolute path to the checked-out OAI tree used for grounding patches.
        Must match complete_error_fixing_pipeline path joins (dirname(__file__) + codebase folder).
        """
        name = (self.openair_codebase_file_name or "").strip()
        if not name:
            return self._pipeline_package_dir()
        return os.path.join(self._pipeline_package_dir(), name)

    def _compile_mode_relax_patch_grounding(
        self, deployment_context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        When local paths/logs do not match the workspace (or the model paraphrased snippets),
        still surface compile-fix suggestions instead of dropping everything after validation.
        Never relax in CMake build-system mode (stricter file anchors there).
        """
        if not isinstance(deployment_context, dict):
            return False
        if deployment_context.get("cmake_build_system_mode"):
            return False
        k = str(deployment_context.get("log_error_kind") or "").strip().lower()
        if k == "build":
            return True
        gt = deployment_context.get("compiler_ground_truth_errors")
        return bool(gt and isinstance(gt, list) and len(gt) > 0)

    def _resolve_patch_file_abs_path(self, patch_file_path: str) -> str:
        """Resolve patch file path to absolute path under the OAI tree next to this package."""
        p = (patch_file_path or "").strip().replace("\\", "/")
        if not p:
            return ""
        if os.path.isabs(p):
            return p
        if p.startswith("./"):
            p = p[2:]
        root = self._codebase_root_dir()
        codebase_prefix = (self.openair_codebase_file_name or "").strip().replace("\\", "/")
        if codebase_prefix and p.startswith(codebase_prefix + "/"):
            rel_within = p[len(codebase_prefix) + 1 :]
        else:
            rel_within = p
        return os.path.normpath(os.path.join(root, rel_within))

    @staticmethod
    def _snippet_in_file(snippet: str, file_content: str) -> bool:
        """Check whether snippet exists exactly or with whitespace normalization."""
        s = (snippet or "").strip()
        if not s:
            return False
        if s in file_content:
            return True
        s_norm = re.sub(r"\s+", " ", s)
        c_norm = re.sub(r"\s+", " ", file_content)
        return s_norm in c_norm

    @staticmethod
    def _snippet_exact_count(snippet: str, file_content: str) -> int:
        """Return exact count of a snippet in file content (newline-normalized)."""
        s = (snippet or "").strip().replace("\r\n", "\n")
        if not s:
            return 0
        c = (file_content or "").replace("\r\n", "\n")
        return c.count(s)

    def _validate_patch_grounding(
        self,
        code_patches: List["CodePatch"],
        config_patches: List["ConfigPatch"],
        deployment_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[List["CodePatch"], List["ConfigPatch"]]:
        """
        Remove hallucinated patches by checking they are grounded in target file content.
        """
        dc = deployment_context if isinstance(deployment_context, dict) else {}
        cmake_mode = bool(dc.get("cmake_build_system_mode"))
        relax_compile = self._compile_mode_relax_patch_grounding(dc)

        grounded_code: List[CodePatch] = []
        for patch in code_patches or []:
            abs_path = self._resolve_patch_file_abs_path(patch.file_path)
            if not abs_path or not os.path.exists(abs_path):
                if relax_compile:
                    logger.warning(
                        "⚠️ Keeping code patch without local file (compile mode; path may be "
                        "from another machine or tree): %s",
                        patch.file_path,
                    )
                    grounded_code.append(patch)
                else:
                    logger.warning("🚫 Rejected code patch: file not found (%s)", patch.file_path)
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                if relax_compile:
                    logger.warning(
                        "⚠️ Keeping code patch; could not read file for grounding (%s)",
                        patch.file_path,
                    )
                    grounded_code.append(patch)
                else:
                    logger.warning("🚫 Rejected code patch: cannot read file (%s)", patch.file_path)
                continue
            if not self._snippet_in_file(patch.original_code, content):
                if relax_compile:
                    logger.warning(
                        "⚠️ Keeping code patch though original_code not found in file "
                        "(compile mode — verify manually): %s",
                        patch.file_path,
                    )
                    grounded_code.append(patch)
                else:
                    logger.warning(
                        "🚫 Rejected code patch: original_code not found in target file (%s)",
                        patch.file_path,
                    )
                continue
            grounded_code.append(patch)

        grounded_config: List[ConfigPatch] = []
        for patch in config_patches or []:
            abs_path = self._resolve_patch_file_abs_path(patch.file_path)
            if not abs_path or not os.path.exists(abs_path):
                logger.warning("🚫 Rejected config patch: file not found (%s)", patch.file_path)
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                logger.warning("🚫 Rejected config patch: cannot read file (%s)", patch.file_path)
                continue

            current_value = (patch.current_value or "").strip()
            new_value = (patch.new_value or "").strip()
            if cmake_mode:
                if not current_value:
                    logger.warning(
                        "🚫 Rejected config patch in cmake mode: empty current_value (%s)",
                        patch.file_path,
                    )
                    continue
                if not self._snippet_in_file(current_value, content):
                    logger.warning(
                        "🚫 Rejected config patch in cmake mode: current_value not found in file (%s)",
                        patch.file_path,
                    )
                    continue
                exact_count = self._snippet_exact_count(current_value, content)
                if exact_count != 1:
                    logger.warning(
                        "🚫 Rejected config patch in cmake mode: current_value exact match count=%s (need exactly 1) (%s)",
                        exact_count,
                        patch.file_path,
                    )
                    continue
                # Reject huge synthetic blobs even when partially matching.
                if (
                    len(current_value) > 2000
                    or len(new_value) > 2000
                    or current_value.count("\n") > 60
                    or new_value.count("\n") > 60
                ):
                    logger.warning(
                        "🚫 Rejected config patch in cmake mode: patch snippet too large (%s)",
                        patch.file_path,
                    )
                    continue

            if not cmake_mode and current_value:
                too_large = (len(current_value) > 4000) or (current_value.count("\n") > 120)
                if too_large and not self._snippet_in_file(current_value, content):
                    logger.warning(
                        "🚫 Rejected config patch: oversized ungrounded current_value (%s)",
                        patch.file_path,
                    )
                    continue

            grounded_config.append(patch)

        return grounded_code, grounded_config

    @staticmethod
    def _extract_missing_asn1_symbols(error_text: str) -> List[Tuple[str, str]]:
        """
        From linker lines like `undefined reference to asn_DEF_F1AP_LTMConfiguration`,
        return list of (bundle_prefix, type_suffix) e.g. ("F1AP", "LTMConfiguration")
        so generated filename is `{bundle_prefix}_{type_suffix}.c`.
        """
        if not error_text:
            return []
        out: List[Tuple[str, str]] = []
        seen: set = set()
        for m in re.finditer(
            r"asn_DEF_([A-Za-z0-9]+)_([A-Za-z0-9_-]+)", str(error_text)
        ):
            prefix = (m.group(1) or "").strip()
            suffix = (m.group(2) or "").strip()
            if not prefix or not suffix:
                continue
            key = (prefix, suffix)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _cmake_var_for_bundle(self, bundle_prefix: str) -> str:
        """OAI ASN.1 bundle variable names inside *-bundle.cmake files."""
        p = (bundle_prefix or "").upper()
        if p == "NGAP":
            return "ngap_source"
        if p == "F1AP":
            return "f1ap_source"
        if p == "X2AP":
            return "x2ap_source"
        if p == "S1AP":
            return "s1ap_source"
        return f"{p.lower()}_source"

    @staticmethod
    def _asn1_bundle_version_tuple(rel_path: str) -> Tuple[int, ...]:
        """
        Parse `.../f1ap-18.6.0.cmake` -> (18, 6, 0) for ordering. Unknown / non-standard names -> (0,).
        Prefer **highest** tuple when multiple protocol bundles exist (18.6.0 over 16.3.1).
        """
        name = os.path.basename((rel_path or "").replace("\\", "/"))
        # Typical: f1ap-18.6.0.cmake, ngap-16.3.0.cmake
        m = re.search(r"-(\d+)\.(\d+)\.(\d+)\.cmake$", name, flags=re.I)
        if m:
            return tuple(int(x) for x in m.groups())
        m2 = re.search(r"-(\d+)\.(\d+)\.cmake$", name, flags=re.I)
        if m2:
            return tuple(int(x) for x in m2.groups()) + (0,)
        return (0,)

    def _sort_bundle_cmake_paths_newest_first(self, paths: List[str]) -> List[str]:
        """Deduplicate while preserving highest-version preference for same basename family."""
        if not paths:
            return []
        # Sort by version tuple descending so e.g. f1ap-18.6.0.cmake is tried before f1ap-16.3.1.cmake.
        return sorted(
            paths,
            key=lambda p: self._asn1_bundle_version_tuple(p),
            reverse=True,
        )

    def _discover_asn1_bundle_cmake_paths(
        self,
        candidate_configs: List[Dict[str, Any]],
        bundle_prefix: str,
    ) -> List[str]:
        """
        Resolve ASN1 `*-*.cmake` bundle paths: merge retrieval candidates + filesystem glob
        when embeddings / FAISS retrieval returned nothing.
        """
        paths: List[str] = []
        seen: set = set()

        def add(p: str) -> None:
            p = (p or "").replace("\\", "/").strip()
            if p and p not in seen:
                seen.add(p)
                paths.append(p)

        for c in candidate_configs or []:
            if not isinstance(c, dict):
                continue
            fp = (c.get("file_path") or "").replace("\\", "/")
            low = fp.lower()
            if low.endswith(".cmake") and "/asn1/" in low:
                add(fp)

        bp = (bundle_prefix or "").upper()
        subdir_map = {
            "F1AP": ("openair2", "F1AP", "MESSAGES", "ASN1"),
            "NGAP": ("openair2", "NGAP", "MESSAGES", "ASN1"),
            "X2AP": ("openair2", "X2AP", "MESSAGES", "ASN1"),
            "S1AP": ("openair2", "S1AP", "MESSAGES", "ASN1"),
        }
        parts = subdir_map.get(bp)
        if parts:
            asn1_dir = os.path.join(self._codebase_root_dir(), *parts)
            if os.path.isdir(asn1_dir):
                prefix = bp.lower()
                try:
                    for name in sorted(os.listdir(asn1_dir)):
                        if not name.endswith(".cmake"):
                            continue
                        if name.startswith(prefix + "-"):
                            add("/".join(parts + (name,)))
                except OSError:
                    pass

        dc = getattr(self, "_current_deployment_context", None) or {}
        hints = dc.get("linker_derived_hints") if isinstance(dc, dict) else None
        if isinstance(hints, dict):
            for q in hints.get("extra_search_queries") or []:
                if not isinstance(q, str):
                    continue
                for m in re.finditer(
                    r"(openair2/[A-Za-z0-9]+/MESSAGES/ASN1/[A-Za-z0-9.-]+\.cmake)",
                    q.replace("\\", "/"),
                ):
                    add(m.group(1))

        return self._sort_bundle_cmake_paths_newest_first(paths)

    @staticmethod
    def _find_cmake_set_block_span(
        lines: List[str], var_name: str
    ) -> Optional[Tuple[int, int]]:
        """Return [start_line_idx, end_line_idx] inclusive for `set(var` ... `)` block."""
        start_re = re.compile(rf"^\s*set\s*\(\s*{re.escape(var_name)}\s*$", re.I)
        for i, line in enumerate(lines):
            if start_re.match(line):
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == ")":
                        return (i, j)
                return None
        return None

    def _missing_asn1_tuples_for_linker_error(
        self, error_text: str
    ) -> List[Tuple[str, str]]:
        """
        Collect (bundle_prefix, type_suffix) tuples for asn_DEF_* linker errors, merging
        error_text with linker_derived_hints from deployment context.
        """
        missing = self._extract_missing_asn1_symbols(error_text)
        dc = getattr(self, "_current_deployment_context", None) or {}
        hints = dc.get("linker_derived_hints") if isinstance(dc, dict) else None
        if isinstance(hints, dict):
            for sym in hints.get("asn_def_types") or []:
                s = str(sym or "").strip()
                if not s:
                    continue
                if s.startswith("asn_DEF_"):
                    s = s[len("asn_DEF_") :]
                if "_" not in s:
                    continue
                prefix, suffix = s.split("_", 1)
                if prefix and suffix:
                    missing.append((prefix, suffix))

        seen_missing: set = set()
        dedup_missing: List[Tuple[str, str]] = []
        for item in missing:
            if item in seen_missing:
                continue
            seen_missing.add(item)
            dedup_missing.append(item)
        return dedup_missing

    def _cmake_bundle_source_list_complete_guidance(
        self,
        error_text: str,
        candidate_configs: List[Dict[str, Any]],
    ) -> Optional[FixSuggestion]:
        """
        When the newest ASN bundle(s) on disk already list the generated `.c` file(s),
        no CMake list edit is appropriate — return grounded guidance without calling the LLM
        (avoids invalid JSON + empty patches for stale-link / wrong-archive cases).
        """
        missing = self._missing_asn1_tuples_for_linker_error(error_text)
        if not missing:
            return None
        steps: List[str] = []
        suspected_paths: List[str] = []
        for bundle_prefix, type_suffix in missing:
            missing_file = f"{bundle_prefix}_{type_suffix}.c"
            cmake_paths = self._discover_asn1_bundle_cmake_paths(
                candidate_configs, bundle_prefix
            )
            if not cmake_paths:
                return None
            newest = cmake_paths[0]
            abs_path = self._resolve_patch_file_abs_path(newest)
            if not abs_path or not os.path.exists(abs_path):
                return None
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                return None
            if missing_file not in content:
                return None
            if newest not in suspected_paths:
                suspected_paths.append(newest)
            steps.append(
                f"Checked `{newest}`: it already lists `{missing_file}`. "
                "If linking still fails, run a clean rebuild of the ASN.1 / `libasn1_*` "
                "targets, re-run codegen if you changed `.asn`, and ensure one bundle version "
                "is used consistently."
            )
        return FixSuggestion(
            suspected_functions=[],
            suspected_configs=suspected_paths,
            reason=(
                "The ASN.1 bundle CMake list on disk already includes the generated `.c` "
                "for the missing `asn_DEF_*` symbol; a CMake source-list patch is not indicated."
            ),
            config_fix=(
                "Clean and rebuild `libasn1_*` and the final link target; verify codegen "
                "and that the same bundle version is used end-to-end."
            ),
            code_patches=[],
            config_patches=[],
            root_cause_analysis=(
                "Linker errors can persist when static archives or object files are stale, "
                "when a different protocol bundle version is picked at link time than the one "
                "in your tree, or when generated sources were not recompiled after ASN.1 changes."
            ),
            investigation_steps=steps,
            specification_context="",
        )

    def _derive_cmake_asn1_fallback_patches(
        self,
        error_text: str,
        candidate_configs: List[Dict[str, Any]],
    ) -> List["ConfigPatch"]:
        """
        Deterministic fallback for CMake linker ASN.1 missing symbol errors.
        Inserts missing `PROTO_Type.c` into the correct `set(<proto>_source ...)` list
        using line-accurate anchors (works when retrieval returns no candidate_configs).
        """
        missing = self._missing_asn1_tuples_for_linker_error(error_text)
        if not missing:
            return []


        out: List[ConfigPatch] = []
        grouped: Dict[
            Tuple[str, str, str, str, int, int],
            Dict[str, Any],
        ] = {}
        for bundle_prefix, type_suffix in missing:
            missing_file = f"{bundle_prefix}_{type_suffix}.c"
            cmake_var = self._cmake_var_for_bundle(bundle_prefix)
            cmake_paths = self._discover_asn1_bundle_cmake_paths(
                candidate_configs, bundle_prefix
            )
            if not cmake_paths:
                continue

            # Newest bundle first (e.g. f1ap-18.6.0.cmake before f1ap-16.3.1.cmake).
            for cmake_fp in cmake_paths:
                abs_path = self._resolve_patch_file_abs_path(cmake_fp)
                if not abs_path or not os.path.exists(abs_path):
                    continue
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue

                if missing_file in content:
                    # Newest matching bundle already lists this TU; do not patch older `*-*.cmake` files.
                    break

                lines = content.splitlines()
                span = self._find_cmake_set_block_span(lines, cmake_var)
                if not span:
                    continue
                start_idx, end_idx = span
                inner = lines[start_idx + 1 : end_idx]
                if not inner:
                    continue

                insert_at = end_idx
                for k, ln in enumerate(inner):
                    st = ln.strip()
                    if st.endswith(".c") and st > missing_file:
                        insert_at = start_idx + 1 + k
                        break

                if insert_at <= start_idx or insert_at > end_idx:
                    continue

                line_after = lines[insert_at]
                if insert_at == end_idx:
                    # Insert immediately before closing `)` of the set(...).
                    line_before = lines[end_idx - 1]
                elif insert_at == start_idx + 1:
                    line_before = lines[start_idx]
                else:
                    line_before = lines[insert_at - 1]
                indent_m = re.match(r"^(\s*)", line_after)
                indent = indent_m.group(1) if indent_m else "  "
                new_line = f"{indent}{missing_file}"

                current_value = f"{line_before}\n{line_after}"
                block_line = start_idx + 1
                group_key = (
                    cmake_fp,
                    cmake_var,
                    current_value,
                    indent,
                    insert_at,
                    block_line,
                )
                group = grouped.get(group_key)
                if group is None:
                    group = {
                        "line_before": line_before,
                        "line_after": line_after,
                        "missing_files": [],
                        "symbols": [],
                    }
                    grouped[group_key] = group
                if missing_file not in group["missing_files"]:
                    group["missing_files"].append(missing_file)
                sym = f"asn_DEF_{bundle_prefix}_{type_suffix}"
                if sym not in group["symbols"]:
                    group["symbols"].append(sym)
                # Only patch the newest matching bundle for this missing file.
                break
        for (cmake_fp, cmake_var, current_value, indent, insert_at, block_line), group in grouped.items():
            line_before = group["line_before"]
            line_after = group["line_after"]
            missing_files = group["missing_files"]
            symbols = group["symbols"]
            insert_lines = [f"{indent}{mf}" for mf in missing_files]
            new_value = f"{line_before}\n" + "\n".join(insert_lines) + f"\n{line_after}"
            files_txt = ", ".join(f"`{mf}`" for mf in missing_files)
            syms_txt = ", ".join(f"`{s}`" for s in symbols)
            out.append(
                ConfigPatch(
                    config_name=cmake_var,
                    file_path=cmake_fp,
                    patch_type="targeted_insertion",
                    current_value=current_value,
                    new_value=new_value,
                    line_number=(
                        f"insert {files_txt} before line {insert_at + 1} "
                        f"in `{cmake_var}` list (bundle ~line {block_line})"
                    ),
                    relevance_score=0.99,
                    description=(
                        f"Add generated ASN.1 sources {files_txt} so linker symbols "
                        f"{syms_txt} are included in libasn1."
                    ),
                )
            )
        return out

    @staticmethod
    def _extract_compiler_suggestions_from_text(text: str) -> List[Dict[str, str]]:
        """
        Extract compiler-provided suggestions like:
        - "has no member named ‘X’; did you mean ‘Y’?"
        - "unknown type name ‘X’; did you mean ‘Y’?"
        Returns list of {kind, wrong, suggested}.
        """
        if not text:
            return []

        t = str(text)
        out: List[Dict[str, str]] = []

        # Match both unicode quotes ‘ ’ and ascii quotes ' '.
        patterns = [
            # member suggestion
            (r"has no member named [‘']([^’']+)[’'].*?did you mean [‘']([^’']+)[’']\?", "member"),
            # unknown type suggestion
            (r"unknown type name [‘']([^’']+)[’'].*?did you mean [‘']([^’']+)[’']\?", "type"),
            # undeclared identifier suggestion (gcc/clang sometimes say: did you mean 'X'?)
            (r"undeclared.*?[‘']([^’']+)[’'].*?did you mean [‘']([^’']+)[’']\?", "identifier"),
        ]

        for pat, kind in patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE | re.DOTALL):
                wrong = (m.group(1) or "").strip()
                suggested = (m.group(2) or "").strip()
                if wrong and suggested and wrong != suggested:
                    out.append({"kind": kind, "wrong": wrong, "suggested": suggested})

        # De-duplicate while preserving order.
        seen = set()
        deduped: List[Dict[str, str]] = []
        for item in out:
            key = (item.get("kind"), item.get("wrong"), item.get("suggested"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped

    def _extract_compiler_suggestions(self, error_text: str, deployment_context: Optional[Dict]) -> List[Dict[str, str]]:
        """
        Pull compiler suggestions from multiple sources (authoritative):
        - error_text passed into Phase 3
        - deployment_context.log_anchors
        - deployment_context.detailed_log_context.error_sequences[].context
        """
        blobs: List[str] = []
        if error_text:
            blobs.append(str(error_text))
        if deployment_context and isinstance(deployment_context, dict):
            anchors = deployment_context.get("log_anchors") or []
            if isinstance(anchors, list):
                blobs.extend([str(a) for a in anchors if a])
            detailed = deployment_context.get("detailed_log_context") or {}
            if isinstance(detailed, dict):
                seqs = detailed.get("error_sequences") or []
                if isinstance(seqs, list):
                    for s in seqs:
                        if not isinstance(s, dict):
                            continue
                        ctx = s.get("context") or []
                        if isinstance(ctx, list):
                            blobs.extend([str(x) for x in ctx if x])
                        err_line = s.get("error_line")
                        if err_line:
                            blobs.append(str(err_line))

        merged = "\n".join(blobs)
        return self._extract_compiler_suggestions_from_text(merged)
    
    def _load_functions_database(self):
        """Load functions database for function definition lookup"""
        try:
            # Try database/functions.json first
            functions_db_path = "database/functions.json"
            if os.path.exists(functions_db_path):
                with open(functions_db_path, 'r', encoding='utf-8') as f:
                    self.functions_db = json.load(f)
                logger.info(f"✅ Functions database loaded: {len(self.functions_db)} entries")
            else:
                # Fallback: try faiss_indices/functions_mapping.json
                fallback_path = "faiss_indices/functions_mapping.json"
                if os.path.exists(fallback_path):
                    with open(fallback_path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                        # Convert mapping dict to list format
                        if isinstance(mapping, dict):
                            self.functions_db = list(mapping.values())
                        else:
                            self.functions_db = mapping
                    logger.info(f"✅ Functions mapping loaded: {len(self.functions_db)} entries")
                else:
                    logger.warning("⚠️  Functions database not found (database/functions.json or faiss_indices/functions_mapping.json)")
                    self.functions_db = []
        except Exception as e:
            logger.error(f"Failed to load functions database: {e}")
            self.functions_db = []
    
    def _lookup_function_definition(self, function_name: str) -> Optional[Dict]:
        """Look up function definition/signature from functions database"""
        if not self.functions_db:
            return None
        
        # Search for function by name
        for func_data in self.functions_db:
            if isinstance(func_data, dict) and func_data.get('function_name') == function_name:
                return func_data
        
        return None
    
    def _setup_azure_client(self):
        """Setup Azure OpenAI client"""
        logger.info("🔧 Setting up Azure OpenAI client...")
        
        # Prefer AZURE_OPENAI_API_KEY, but also accept AZURE_OPENAI_KEY for backward compatibility
        api_key = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        missing_vars = []
        if not api_key:
            missing_vars.append('AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY for backward compatibility)')
        if not endpoint:
            missing_vars.append('AZURE_OPENAI_ENDPOINT')
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Initialize Azure OpenAI client
        self.azure_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint
        )
        
        # Use hardcoded deployment name
        self.model = "gpt-4o-mini"
        
        logger.info("✅ Azure OpenAI client initialized successfully")
        logger.info(f"📦 Using deployment: {self.model}")
    
    def _smart_truncate_code(self, code_body: str) -> str:
        """
        Smart truncation that preserves function signature and core logic
        
        Args:
            code_body: Complete function code
            
        Returns:
            Truncated code that preserves important parts
        """
        if not code_body:
            return ""
        
        # If function is reasonable size, return complete code
        if len(code_body) <= 5000:
            return code_body
        
        # For huge functions, preserve signature and first part
        lines = code_body.split('\n')
        
        # Find function signature (first line with parentheses)
        signature_end = 0
        for i, line in enumerate(lines):
            if '(' in line and ')' in line and ('static' in line or 'void' in line or 'int' in line or 'char' in line or 'uint' in line):
                signature_end = i
                break
        
        # Take signature + first 60 lines + middle 40 lines + last 30 lines for better coverage
        preserved_lines = lines[:signature_end + 1]  # Include signature
        preserved_lines.extend(lines[signature_end + 1:signature_end + 61])  # First 60 lines after signature
        
        if len(lines) > 120:  # Only add ending if function is long enough
            # Add some middle lines to capture if-else structures
            middle_start = signature_end + 61
            middle_end = min(middle_start + 40, len(lines) - 30)
            if middle_start < middle_end:
                preserved_lines.append("... [early part truncated] ...")
                preserved_lines.extend(lines[middle_start:middle_end])
            
            preserved_lines.append("... [middle of function truncated] ...")
            preserved_lines.extend(lines[-30:])  # Last 30 lines (increased from 20)
        
        result = '\n'.join(preserved_lines)
        
        # Still truncate if result is too long
        if len(result) > 8000:
            result = result[:8000] + "\n... [further truncated]"
        
        return result
    
    def assemble_context(self, 
                        error: str, 
                        candidate_functions: List[Dict], 
                        candidate_configs: List[Dict], 
                        call_graph_context: List[Dict], 
                        matched_pattern: Optional[Dict] = None,
                        deployment_context: Optional[Dict] = None) -> str:
        """
        Step 3.1 - Context Assembly
        
        Assemble all context information into a structured prompt for GPT-4o-mini
        
        Args:
            error: Original error text
            candidate_functions: List of suspected functions
            candidate_configs: List of suspected config parameters
            call_graph_context: Call graph relationships
            matched_pattern: Matched error pattern (if any)
            
        Returns:
            Assembled context string for LLM
        """
        logger.info("🏗️ Assembling context for fix suggestion...")
        
        context_parts = []
        
        # Header
        context_parts.append("# Error Fix Analysis Context")
        context_parts.append("=" * 50)
        
        # 1. Error Information
        context_parts.append("\n## 🔥 ERROR DETAILS")
        context_parts.append(f"**Original Error:** {error}")

        cmake_mode = bool(
            deployment_context
            and isinstance(deployment_context, dict)
            and deployment_context.get("cmake_build_system_mode")
        )
        if cmake_mode:
            excerpt = (deployment_context or {}).get("full_build_log_excerpt") or ""
            if len(excerpt) > 20000:
                excerpt = excerpt[:20000] + "\n... [truncated for LLM context]"
            context_parts.append("\n## 📜 FULL BUILD LOG (EXCERPT)")
            context_parts.append(
                "The following text is the build log (or tail). Use it as primary evidence."
            )
            context_parts.append(f"\n```text\n{excerpt}\n```")
            context_parts.append(
                "\n**🔧 CMake / build-system mode:** Prefer concrete `config_patches` targeting "
                "**CMakeLists.txt** and ***.cmake** from the suspected configurations below. "
                "Do **not** prioritize unrelated runtime protocol functions (e.g. RRC reject helpers) "
                "unless the log explicitly implicates them."
            )
            ld_hints = (deployment_context or {}).get("linker_derived_hints") or {}
            pb = ld_hints.get("prompt_bullets") or []
            if pb:
                context_parts.append(
                    "\n## 🧭 LINKER-INFERRED TARGETS (log often omits the exact `.cmake` path)\n"
                    "Use these together with **SUSPECTED CONFIGURATIONS** rows whose `file_path` "
                    "ends in **`.cmake`** under **`.../MESSAGES/ASN1/`**:\n"
                )
                for i, line in enumerate(pb, 1):
                    context_parts.append(f"   {i}. {line}")
                context_parts.append(
                    "\n**Patch rule:** If a row shows **`f1ap-*.cmake`** (or similar), your "
                    "`config_patches[].file_path` should reference **that same path**, editing the "
                    "generated-source list (e.g. add missing **`F1AP_<Type>.c`** next to sibling `.c` "
                    "lines). Do **not** invent only `CMakeLists.txt` when the evidence points to an "
                    "**ASN1 bundle `.cmake`** file."
                )
        
        if matched_pattern:
            context_parts.append(f"**Pattern Matched:** {matched_pattern.get('name', 'Unknown')}")
            context_parts.append(f"**Pattern Category:** {matched_pattern.get('category', 'General')}")
        
        # 2. Call Graph Context (Enhanced - shows execution flow)
        context_parts.append("\n## 🔗 CALL GRAPH CONTEXT")
        call_chain_summary = self._build_call_chain_summary(call_graph_context, candidate_functions)
        context_parts.append(call_chain_summary)
        
        # 3. Suspected Functions
        context_parts.append("\n## 🔧 SUSPECTED FUNCTIONS")
        if cmake_mode and not candidate_functions:
            context_parts.append(
                "*(None — CMake/build-system mode uses the log + CMake index rows below; "
                "avoid guessing unrelated C functions.)*"
            )
        elif candidate_functions:
            # BOOST SCORE FOR rrc_gNB_generate_RRCReject - CRITICAL FOR RRC REJECTION SCENARIOS
            for func in candidate_functions:
                # Type safety: skip non-dict items
                if not isinstance(func, dict):
                    logger.warning(f"⚠️ Skipping non-dict function item: {type(func)}")
                    continue

                if (
                    not cmake_mode
                    and func.get("function_name") == "rrc_gNB_generate_RRCReject"
                ):
                    func['relevance_score'] = 0.7  # Boost to highest priority
                    func['reason'] = "CRITICAL RRC REJECTION FUNCTION: This function is essential for handling RRC setup failures. When an RRC setup request fails or encounters errors, this function generates and sends an RRCReject message to the UE, properly handling the rejection scenario. For segmentation faults in rrc_handle_RRCSetupRequest, this function should be called in error handling paths to gracefully reject the connection instead of crashing."
                    break
            
            for i, func in enumerate(candidate_functions, 1):
                # Type safety: skip non-dict items
                if not isinstance(func, dict):
                    continue
                    
                context_parts.append(f"\n### Function {i}: {func.get('function_name', 'Unknown')}")
                context_parts.append(f"**File:** {func.get('file_path', 'Unknown')}")
                context_parts.append(f"**Relevance Score:** {func.get('relevance_score', 0):.2f}")
                context_parts.append(f"**Reason:** {func.get('reason', 'No reason provided')}")
                
                # Include code snippet
                code_snippet = func.get('code_snippet', func.get('code_body', ''))
                if code_snippet:
                    # Use smart truncation for very long code
                    if len(code_snippet) > 8000:
                        code_snippet = self._smart_truncate_code(code_snippet)
                    context_parts.append(f"**Code:**\n```c\n{code_snippet}\n```")
        else:
            context_parts.append("No suspected functions identified.")
        
        # 4. Suspected Configurations
        context_parts.append("\n## ⚙️ SUSPECTED CONFIGURATIONS")
        cmake_ctx = bool(isinstance(deployment_context, dict) and deployment_context.get("cmake_build_system_mode"))
        if candidate_configs:
            # Filter configs to supported files based on mode
            filtered_configs = []
            for config in candidate_configs:
                # Type safety: skip non-dict items
                if not isinstance(config, dict):
                    logger.warning(f"⚠️ Skipping non-dict config item: {type(config)}")
                    continue
                    
                file_path = config.get('file_path', '').lower()
                if self._is_supported_config_patch_file(file_path, deployment_context):
                    filtered_configs.append(config)
            
            if filtered_configs:
                context_parts.append(f"\n**🎯 FILTERED CONFIGS: {len(filtered_configs)} out of {len(candidate_configs)} total configs**")
                if cmake_ctx:
                    context_parts.append("**📁 Showing configs from: cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf, CMakeLists.txt, *.cmake**")
                else:
                    context_parts.append("**📁 Showing configs from: cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf**")
                
                # Add consolidated guidance for ALL config patches (ONCE, not repeated)
                context_parts.append("\n**🚨 CRITICAL GUIDANCE FOR CONFIG PATCHES:**")
                context_parts.append("**📌 MANDATORY RULE: ONLY suggest config patches when ALL three conditions are met:**")
                context_parts.append("1. ✅ The parameter EXISTS in deployment context (shown at the top)")
                context_parts.append("2. ✅ The current value is DIFFERENT from deployment context value")
                context_parts.append("3. ✅ The parameter is directly related to the error")
                context_parts.append("\n**📋 EXAMPLES OF WHEN TO SUGGEST PATCHES:**")
                context_parts.append("- ✅ Deployment context has `DNN: oai`, current config has `dnn = \"internet\"` → SUGGEST: change to `\"oai\"`")
                context_parts.append("- ✅ Deployment context has `NSSAI SST: 1`, current config has `nssai_sst = 2` → SUGGEST: change to `1`")
                context_parts.append("- ✅ Deployment context has `NSSAI SD: 0xc`, current config has `nssai_sd = 0x1` → SUGGEST: change to `0xc`")
                context_parts.append("- ✅ Deployment context has `AMF IP: 192.168.70.132`, current config has `amf_ip_address = \"88.168.89.198\"` → SUGGEST: change to `\"192.168.70.132\"`")
                context_parts.append("\n**❌ EXAMPLES OF WHEN NOT TO SUGGEST PATCHES:**")
                context_parts.append("- ❌ Parameter `gNB_name` is NOT in deployment context → DO NOT suggest changes")
                context_parts.append("- ❌ Parameter `gNB_ID` is NOT in deployment context → DO NOT suggest changes")
                context_parts.append("- ❌ Parameter `Active_gNBs` is NOT in deployment context → DO NOT suggest changes")
                context_parts.append("- ❌ Current value matches deployment context → DO NOT suggest (no-op)")
                context_parts.append("\n**📝 For each config_patch you DO suggest:**")
                context_parts.append("- Use the exact file path, current value, and line number shown below")
                context_parts.append("- Set `new_value` to the EXACT value from deployment context (not placeholders, not \"improvements\")")
                context_parts.append("- Description must say: \"Must match deployment context value\"")
                context_parts.append("- If parameter is NOT in deployment context, skip it entirely")
                
                for i, config in enumerate(filtered_configs, 1):
                    context_parts.append(f"\n### Config {i}: {config.get('param_name', 'Unknown')}")
                    context_parts.append(f"**File:** {config.get('file_path', 'Unknown')}")
                    context_parts.append(f"**Current Value:** {config.get('param_value', 'Unknown')}")
                    context_parts.append(f"**Relevance Score:** {config.get('relevance_score', 0):.2f}")
                    context_parts.append(f"**Reason:** {config.get('reason', 'No reason provided')}")
                    
                    # Add line number information if available
                    line_number = config.get('line_number', 'Unknown')
                    context_parts.append(f"**Line Number:** {line_number}")
                    
                    # Add config context if available
                    config_context = config.get('config_context', '')
                    if config_context:
                        context_parts.append(f"**Config Context:**\n```\n{config_context}\n```")
            else:
                context_parts.append("**⚠️ NO CONFIGS FOUND: All configs filtered out (not in supported config files)**")
                context_parts.append("**Available configs were in other files and have been excluded from analysis.**")
                if cmake_ctx:
                    context_parts.append("**📁 Supported files: cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf, CMakeLists.txt, *.cmake**")
                else:
                    context_parts.append("**📁 Supported files: cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf**")
        else:
            context_parts.append("No suspected configurations identified.")
        
        # 5. Enhanced Runtime Log Context (NEW)
        if deployment_context and deployment_context.get('log_anchors'):
            context_parts.append("\n## 📋 RUNTIME LOG CONTEXT")
            context_parts.append("**Key log messages from the actual runtime:**")
            log_anchors = deployment_context.get('log_anchors', [])
            for i, anchor in enumerate(log_anchors[:10], 1):  # Limit to top 10 most relevant
                context_parts.append(f"   {i}. {anchor}")
        
        # 6. Detailed Log Analysis (NEW)
        if deployment_context and deployment_context.get('detailed_log_context'):
            detailed_context = deployment_context.get('detailed_log_context', {})
            
            # Debug values (enum constants, etc.)
            if detailed_context.get('debug_values'):
                context_parts.append("\n## 🔍 DEBUG VALUES FROM LOG")
                context_parts.append("**Critical runtime values extracted from logs:**")
                for key, values in detailed_context['debug_values'].items():
                    if values:  # Only show if we have values
                        context_parts.append(f"   • **{key}**: {values[0]['value']} (from: {values[0]['line'][:100]}...)")
            
            # Error sequences with context
            if detailed_context.get('error_sequences'):
                context_parts.append("\n## ⚠️ ERROR SEQUENCES WITH CONTEXT")
                context_parts.append("**Error occurrences with surrounding log lines:**")
                for i, seq in enumerate(detailed_context['error_sequences'][:3], 1):  # Limit to top 3 sequences
                    context_parts.append(f"   **Error {i} (Line {seq['line_number']})**: {seq['error_line']}")
                    context_parts.append("   Context:")
                    for ctx_line in seq['context']:
                        context_parts.append(f"   {ctx_line}")
                    context_parts.append("")
            
            # Add network context if available
            network_params = deployment_context.get('network_params', {})
            if network_params:
                context_parts.append(f"\n**Network Status:**")
                for param, value in network_params.items():
                    if value is not None:
                        context_parts.append(f"   - {param}: {value}")
            
            context_parts.append("\n**💡 Use these log messages to understand the exact runtime state and sequence of events that led to the error.**")

        # 6.5 Compiler "did you mean" suggestions (AUTHORITATIVE) — not for CMake/linker mode
        compiler_suggestions = (
            []
            if cmake_mode
            else self._extract_compiler_suggestions(error, deployment_context)
        )
        if compiler_suggestions:
            context_parts.append("\n## ✅ COMPILER-SUGGESTED FIXES (AUTHORITATIVE)")
            context_parts.append(
                "The compiler log contains explicit replacement hints (e.g., \"did you mean ...?\"). "
                "Treat these as **authoritative**. When generating code patches, apply these exact "
                "identifier/member/type replacements first wherever they match the broken code."
            )
            # Group by kind for readability.
            by_kind: Dict[str, List[Dict[str, str]]] = {}
            for s in compiler_suggestions:
                by_kind.setdefault(s.get("kind", "other"), []).append(s)
            for kind, items in by_kind.items():
                context_parts.append(f"\n**{kind.upper()} replacements:**")
                for s in items[:25]:  # keep bounded
                    context_parts.append(f"- Replace `{s['wrong']}` → `{s['suggested']}`")
        
        # 7. Analysis Instructions
        context_parts.append("\n## 📋 ANALYSIS INSTRUCTIONS")
        context_parts.append("""
Please analyze this error context and provide:

1. **Root Cause Analysis:** What is the most likely cause of this error?
2. **Suspected Functions:** Which functions are most likely responsible?
3. **Suspected Configs:** Which configuration parameters need to be checked/fixed?
4. **Fix Suggestions:**
   - Specific code patches for identified functions (using exact file paths provided)
   - Detailed config_patches with exact file paths, current values, and relevance scores from above
   - Step-by-step investigation procedure
5. **Call Chain Analysis:** How does the error propagation occur?

For config_patches, you MUST include:
- config_name: exact parameter name
- file_path: exact path from the suspected configurations above  
- current_value: exact current value from above
- new_value: your recommended fix value
- line_number: approximate line or section
- relevance_score: use the score from suspected configurations above
- description: detailed explanation of why this fix resolves the error

Focus on providing actionable, specific fixes that address the root cause.
""")
        
        assembled_context = "\n".join(context_parts)
        logger.info(f"✅ Context assembled: {len(assembled_context)} characters")
        
        return assembled_context
    
    def _build_call_chain_summary(self, call_graph_context: List[Dict], candidate_functions: List[Dict]) -> str:
        """
        Build enhanced call chain summary showing execution flow with function definitions
        
        Args:
            call_graph_context: Call graph data
            candidate_functions: Suspected functions
            
        Returns:
            Human-readable call chain summary with function definitions
        """
        if not call_graph_context:
            return "No call graph context available."
        
        summary_parts = []
        
        # Extract function names from candidates for priority (with type safety)
        candidate_names = {func.get('function_name', '') for func in candidate_functions if isinstance(func, dict)}
        
        # Collect downstream function names for definition lookup
        downstream_func_names = set()
        
        # Build call chains
        call_chains = []
        
        for entry in call_graph_context:
            # Type safety: skip non-dict entries
            if not isinstance(entry, dict):
                logger.warning(f"⚠️ Skipping non-dict call graph entry: {type(entry)}")
                continue
                
            func_name = entry.get('function', '')
            calls = entry.get('calls', [])
            called_by = entry.get('called_by', [])
            
            # Build upstream chain (who calls this function)
            if called_by:
                upstream_chain = f"{' → '.join(called_by)} → **{func_name}**"
                call_chains.append(f"📥 Upstream: {upstream_chain}")
            
            # Build downstream chain (what this function calls)
            if calls:
                downstream_chain = f"**{func_name}** → {' → '.join(calls[:10])}"  # Limit to first 10
                if len(calls) > 10:
                    downstream_chain += f" ... (+{len(calls)-10} more)"
                call_chains.append(f"📤 Downstream: {downstream_chain}")
                
                # Collect downstream function names for definition lookup
                for called_func in calls[:10]:  # Limit to first 10
                    if called_func not in candidate_names:  # Only add if not already a candidate
                        downstream_func_names.add(called_func)
        
        if call_chains:
            summary_parts.append("**Call Flow Analysis:**")
            # Prioritize chains involving candidate functions
            priority_chains = [chain for chain in call_chains if any(name in chain for name in candidate_names)]
            other_chains = [chain for chain in call_chains if chain not in priority_chains]
            
            summary_parts.extend(priority_chains[:10])  # Top 10 priority chains
            if other_chains:
                summary_parts.append("**Additional Chains:**")
                summary_parts.extend(other_chains[:10])  # Top 10 additional chains
            
            # Add function definitions for downstream functions
            if downstream_func_names:
                summary_parts.append("\n**📚 Function Definitions (Downstream Functions):**")
                for func_name in sorted(list(downstream_func_names)[:15]):  # Limit to 15 most relevant
                    func_def = self._lookup_function_definition(func_name)
                    if func_def:
                        code_snippet = func_def.get('code_snippet', func_def.get('code_body', ''))
                        file_path = func_def.get('file_path', 'Unknown')
                        
                        # Extract function signature (first few lines)
                        if code_snippet:
                            lines = code_snippet.split('\n')
                            signature_lines = []
                            brace_count = 0
                            for line in lines[:20]:  # First 20 lines should contain signature
                                signature_lines.append(line)
                                brace_count += line.count('{') - line.count('}')
                                if brace_count > 0 and '{' in line:
                                    break
                            
                            signature = '\n'.join(signature_lines[:5])  # First 5 lines for signature
                            summary_parts.append(f"\n**{func_name}** (from {file_path}):")
                            summary_parts.append(f"```c\n{signature}\n```")
        else:
            summary_parts.append("No clear call chains identified.")
        
        return "\n".join(summary_parts)
    
    def _get_dynamic_troubleshooting_hints(self, error_text: str) -> str:
        """Get dynamic troubleshooting hints based on error type from JSON patterns."""
        try:
            # Load error patterns
            with open("database/error_patterns_structured.json", 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            patterns = patterns_data.get('patterns', {})
            error_lower = error_text.lower()
            
            # Find matching pattern
            for pattern_name, pattern_data in patterns.items():
                keywords = pattern_data.get('keywords', [])
                
                # Check if any keyword matches
                if any(keyword in error_lower for keyword in keywords):
                    suggested_fixes = pattern_data.get('suggested_fixes', [])
                    
                    # Format troubleshooting hints
                    hints = ["### NETWORK TROUBLESHOOTING HINTS:"]
                    for fix in suggested_fixes:
                        hints.append(f"- {fix}")
                    
                    return "\n".join(hints)
            
            # No pattern matches - generate dynamic pattern
            logger.info(f"🔄 No pattern found for '{error_text}', generating dynamic pattern...")
            dynamic_pattern = self._generate_dynamic_error_pattern(error_text)
            
            # Add pattern to JSON for future use
            self._add_pattern_to_json(error_text, dynamic_pattern)
            
            # Use the generated pattern
            suggested_fixes = dynamic_pattern.get('suggested_fixes', [])
            hints = ["### NETWORK TROUBLESHOOTING HINTS (Dynamically Generated):"]
            for fix in suggested_fixes:
                hints.append(f"- {fix}")
            
            return "\n".join(hints)
            
        except Exception as e:
            logger.warning(f"Could not load dynamic troubleshooting hints: {e}")
            # Fallback to default hints
            return """### NETWORK TROUBLESHOOTING HINTS:
- Validate network configuration and parameters in config files
- Check network reachability between endpoints
- Verify protocol-specific configuration settings"""
    
    def _generate_dynamic_error_pattern(self, error_text: str) -> dict:
        """Generate a new error pattern using LLM when no match is found."""
        try:
            logger.info(f"🤖 Generating dynamic error pattern for: {error_text}")
            
            prompt = f"""You are an expert in 5G/LTE telecommunications error analysis. Generate a structured error pattern for the following error that is not currently in our database.

Error: "{error_text}"

Generate a JSON pattern with the following structure:
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "root_cause_template": "Brief description of what this error indicates and common causes",
  "suggested_fixes": [
    "1. First suggested fix",
    "2. Second suggested fix", 
    "3. Third suggested fix",
    "4. Fourth suggested fix"
  ],
  "specifications": [
    "Relevant 3GPP specification 1",
    "Relevant 3GPP specification 2"
  ],
  "function_analysis": [
    "Function analysis point 1",
    "Function analysis point 2",
    "Function analysis point 3"
  ]
}}

Guidelines:
- Keywords should be 2-4 relevant terms that would match this error
- Root cause should explain what the error means and common causes
- Suggested fixes should be specific, actionable steps
- Specifications should reference relevant 3GPP standards
- Function analysis should focus on code-level debugging steps

Return ONLY the JSON object, no other text."""

            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            pattern_text = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown formatting if present)
            if pattern_text.startswith('```json'):
                pattern_text = pattern_text[7:]
            if pattern_text.endswith('```'):
                pattern_text = pattern_text[:-3]
            
            # Parse the JSON
            import json
            pattern = json.loads(pattern_text)
            
            logger.info(f"✅ Generated dynamic pattern with {len(pattern.get('keywords', []))} keywords")
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic error pattern: {e}")
            # Return a generic pattern as fallback
            return {
                "keywords": [error_text.lower().split()[0] if error_text.split() else "unknown"],
                "root_cause_template": f"Error analysis for: {error_text}. This requires investigation of network connectivity, configuration parameters, and protocol-specific issues.",
                "suggested_fixes": [
                    "1. Check network connectivity and configuration",
                    "2. Verify protocol-specific parameters",
                    "3. Review error logs for additional context",
                    "4. Validate system configuration files"
                ],
                "specifications": [
                    "3GPP TS 38.300 - NR and NG-RAN Overall Description",
                    "3GPP TS 38.331 - NR RRC Protocol Specification"
                ],
                "function_analysis": [
                    "Review error handling in relevant functions",
                    "Check parameter validation and initialization",
                    "Verify protocol state machine transitions"
                ]
            }

    def _add_pattern_to_json(self, error_text: str, pattern: dict) -> None:
        """Add the generated pattern to error_patterns_structured.json"""
        try:
            # Load existing patterns
            with open('database/error_patterns_structured.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Generate a unique pattern name
            pattern_name = error_text.lower().replace(' ', '_').replace(':', '').replace(',', '').replace('.', '')[:30]
            pattern_name = f"dynamic_{pattern_name}"
            
            # Add the new pattern
            if 'patterns' not in data:
                data['patterns'] = {}
            
            data['patterns'][pattern_name] = pattern
            
            # Save back to file
            with open('database/error_patterns_structured.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Added dynamic pattern '{pattern_name}' to error_patterns_structured.json")
            
        except Exception as e:
            logger.error(f"Failed to add pattern to JSON: {e}")

    def _get_deployment_context(self) -> str:
        """Load deployment context from error_patterns_structured.json or use custom context"""
        try:
            # Check if custom deployment context was provided via process_fix_request
            has_current = hasattr(self, '_current_deployment_context')
            current_value = self._current_deployment_context if has_current else None
            
            logger.info(f"🔍 _get_deployment_context called:")
            logger.info(f"   - Has _current_deployment_context attribute: {has_current}")
            logger.info(f"   - _current_deployment_context value: {current_value}")
            logger.info(f"   - _current_deployment_context is truthy: {bool(current_value)}")
            
            if has_current and current_value:
                deployment_context = self._current_deployment_context
                logger.info(f"✅ Using deployment context passed from pipeline ({len(deployment_context)} values)")
            else:
                # Load default from JSON file
                logger.info("📄 Loading deployment context from JSON file (no context passed or empty)")
                with open('database/error_patterns_structured.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    deployment_context = data.get('deployment_context', {})
                    logger.info(f"✅ Loaded {len(deployment_context)} values from JSON")
            
            # Build context lines (works for both custom and JSON deployment context)
            context_lines = []
            context_lines.append(f"- CU IP Address: {deployment_context.get('cu_ip_address', 'Unknown')}")
            context_lines.append(f"- DU IP Address: {deployment_context.get('du_ip_address', 'Unknown')}")
            context_lines.append(f"- gNB IP Address: {deployment_context.get('gnb_ip_address', 'Unknown')}")
            context_lines.append(f"- GNB IP Address for NG AMF: {deployment_context.get('gnb_ip_address', 'Unknown')}")
            context_lines.append(f"- AMF IP Address: {deployment_context.get('amf_ip_address', 'Unknown')}")
            context_lines.append(f"- Core Network Machine IP: {deployment_context.get('core_network_machine_ip', 'Unknown')}")
            context_lines.append(f"- local_s_portc= {deployment_context.get('local_s_portc', 'Unknown')}")
            context_lines.append(f"- local_s_portd = {deployment_context.get('local_s_portd', 'Unknown')}")
            context_lines.append(f"- remote_s_portc = {deployment_context.get('remote_s_portc', 'Unknown')}")
            context_lines.append(f"- remote_s_portd= {deployment_context.get('remote_s_portd', 'Unknown')}")
            context_lines.append(f"- NSSAI SST: {deployment_context.get('nssai_sst', 'Unknown')}")
            context_lines.append(f"- NSSAI SD: {deployment_context.get('nssai_sd', 'Unknown')}")
            context_lines.append(f"- NMC Size: {deployment_context.get('nmc_size', 'Unknown')}")
            context_lines.append(f"- DNN: {deployment_context.get('dnn', 'Unknown')}")
            
            # Add deployment commands to help LLM understand which config files to use
            cu_command = deployment_context.get('Deploy_command_cu_gnb_conf', '')
            du_command = deployment_context.get('Deploy_command_du_gnb_conf', '')
            
            if cu_command:
                context_lines.append(f"\n**🚀 CU Deployment Command:** {cu_command}")
                context_lines.append("**📁 This indicates CU configs should be in: cu_gnb.conf**")
            
            if du_command:
                context_lines.append(f"\n**🚀 DU Deployment Command:** {du_command}")
                context_lines.append("**📁 This indicates DU configs should be in: du_gnb.conf**")
            
            context_lines.append("\n**🎯 CONFIG FILE GUIDANCE:**")
            context_lines.append("- For CU/gNB-related errors: Suggest config patches for cu_gnb.conf or du_gnb.conf")
            context_lines.append("- For UE-related errors: Suggest config patches for ue.conf or 5g_sa_ue.conf")
            context_lines.append("- For UE security/authentication errors: Prefer 5g_sa_ue.conf for IMSI, key, OPC, DNN, NSSAI, etc.")
            context_lines.append("- The deployment commands above show which config files are used for each component")
            
            return "\n".join(context_lines)
        except Exception as e:
            logger.warning(f"Could not load deployment context: {e}")
            return "- Deployment context not available"
    
    def _get_3gpp_specification_context(self, error_text: str) -> str:
        """Get relevant 3GPP specification context based on error type using TF-IDF approach"""    
        try:
            import numpy as np
            import json
            import pickle
            from sklearn.metrics.pairwise import cosine_similarity
            import os

            # --- Load TF-IDF embeddings and metadata ---
            embeddings_path = "faiss_indices/embeddings.npy"
            chunks_path = "faiss_indices/chunks_metadata.json"
            vectorizer_path = "faiss_indices/tfidf_vectorizer.pkl"
            
            # Check if files exist
            if not os.path.exists(embeddings_path):
                logger.error(f"Embeddings file not found: {embeddings_path}")
                return "- No 3GPP specification context found (embeddings file missing)"
            
            if not os.path.exists(chunks_path):
                logger.error(f"Chunks metadata file not found: {chunks_path}")
                return "- No 3GPP specification context found (metadata file missing)"
                
            if not os.path.exists(vectorizer_path):
                logger.error(f"TF-IDF vectorizer file not found: {vectorizer_path}")
                return "- No 3GPP specification context found (vectorizer file missing)"

            logger.info(f"Loading 3GPP embeddings from: {embeddings_path}")
            embeddings = np.load(embeddings_path)   # (num_chunks, embedding_dim)
            
            logger.info(f"Loading chunks metadata from: {chunks_path}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            logger.info(f"Loading TF-IDF vectorizer from: {vectorizer_path}")
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
                
            logger.info(f"Loaded {len(chunks)} chunks with embeddings shape: {embeddings.shape}")

            # --- TF-IDF search function ---
            def get_context(query: str, top_k=10):
                logger.info(f"Searching for query: '{query}'")
                query_vector = vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, embeddings)[0]
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                logger.info(f"Top similarities: {similarities[top_indices[:5]]}")

                results = []
                for idx in top_indices:
                    results.append({
                        "similarity": float(similarities[idx]),
                        "chunk_id": chunks[idx]["chunk_id"],
                        "page_number": chunks[idx]["page_number"],
                        "section_number": chunks[idx]["section_number"],
                        "section_title": chunks[idx]["section_title"],
                        "text": chunks[idx]["text"]
                    })
                logger.info(f"Found {len(results)} results with similarities: {[r['similarity'] for r in results[:5]]}")
                return results

            # --- Extract key terms from error text for better search ---
            import re
            
            # Extract key terms from error text
            key_terms = []
            
            # Look for function names (e.g., rrc_handle_RRCSetupRequest)
            function_matches = re.findall(r'(\w+_handle_\w+)', error_text)
            key_terms.extend(function_matches)
            
            # Look for RRC message types (e.g., RRCSetupRequest)
            rrc_matches = re.findall(r'(RRC\w+)', error_text)
            key_terms.extend(rrc_matches)
            
            # Look for procedure names
            procedure_matches = re.findall(r'(\w+_\w+)', error_text)
            key_terms.extend(procedure_matches)
            
            # Add generic terms based on error type
            if 'segmentation' in error_text.lower() or 'fault' in error_text.lower():
                key_terms.extend(['RRC connection establishment', 'RRCSetup', 'RRCSetupRequest'])
            
            if 'AMF' in error_text:
                key_terms.extend(['AMF', 'gNB-AMF', 'connection establishment'])
            
            if 'RRC' in error_text:
                key_terms.extend(['RRC connection', 'RRC establishment', 'RRC procedures'])
            
            # Create a more targeted search query
            if key_terms:
                search_query = ' '.join(key_terms[:5])  # Use top 5 key terms
                logger.info(f"Extracted key terms: {key_terms[:5]}")
            else:
                search_query = error_text
            
            logger.info(f"Searching 3GPP context for error: {error_text}")
            logger.info(f"Using search query: {search_query}")
            results = get_context(search_query, top_k=10)
            logger.info(f"Found {len(results)} 3GPP specification results")

            specification_context = ""

            if not results:
                logger.warning("No results returned from 3GPP search")
                return "- No relevant 3GPP specification context found for this error"
            
            # Check if we have meaningful results (similarity > 0.01)
            meaningful_results = [r for r in results if r['similarity'] > 0.01]
            if not meaningful_results:
                logger.warning(f"All results have very low similarity scores. Top scores: {[r['similarity'] for r in results[:5]]}")
                # Try with a more general query
                logger.info("Trying with more general query terms...")
                general_terms = ["AMF", "gNB", "connection", "establishment", "error"]
                for term in general_terms:
                    if term.lower() in error_text.lower():
                        logger.info(f"Trying search with term: {term}")
                        term_results = get_context(term, top_k=10)
                        if term_results and any(r['similarity'] > 0.01 for r in term_results):
                            results = term_results
                            logger.info(f"Found better results with term '{term}': {len(results)} results")
                            break
                else:
                    logger.warning("No meaningful results found even with general terms")
                    return "- No relevant 3GPP specification context found for this error"

            for i, res in enumerate(results, start=1):
                specification_context += f"\n=== Extraction Section No. : {i} ===\n"
                specification_context += f"Page: {res['page_number']}\n"
                specification_context += f"Section Title: {res['section_title']}\n"
                specification_context += f"Section Number: {res['section_number']}\n"
                
                # Handle truncated text from the embeddings data
                text_content = res['text']
                if '[... omitted end of long line]' in text_content:
                    text_content = text_content.replace('[... omitted end of long line]', '\n[Note: Content truncated in source data]')
                
                specification_context += f"Content:\n{text_content}\n"

            # Filter the context using LLM to keep only relevant parts
            logger.info(f"Raw 3GPP context length: {len(specification_context)} characters")
            
            # Save debug file with raw context
            debug_data = {
                "error_text": error_text,
                "raw_context": specification_context,
                "raw_context_length": len(specification_context),
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            
            try:
                filtered_context = self._filter_3gpp_context_with_llm(error_text, specification_context)
                logger.info(f"Filtered 3GPP context length: {len(filtered_context)} characters")
                
                # Add filtered context to debug data
                debug_data["filtered_context"] = filtered_context
                debug_data["filtered_context_length"] = len(filtered_context)
                debug_data["filtering_successful"] = True
                
                # Check if filtering actually worked (not just error message)
                if filtered_context and not filtered_context.startswith("- No relevant") and len(filtered_context) > 100:
                    # Additional check: if filtered content is still too long, truncate it
                    if len(filtered_context) > 2000:
                        logger.warning(f"Filtered content still too long ({len(filtered_context)} chars), truncating to 2000 chars")
                        filtered_context = filtered_context[:2000] + "\n\n[Content truncated for display]"
                    
                    # Post-process to clean up the format
                    final_context = self._clean_3gpp_context_format(filtered_context)
                    
                    debug_data["final_context"] = final_context
                    debug_data["using_filtered"] = True
                else:
                    logger.warning("LLM filtering returned minimal content, using raw context")
                    # Add a note about truncation if present
                    if "[Note: Content truncated in source data]" in specification_context:
                        final_context = specification_context + "\n\n[Note: Some content was truncated in the original 3GPP specification data. This may limit the completeness of the analysis.]"
                    else:
                        final_context = specification_context
                    debug_data["final_context"] = final_context
                    debug_data["using_filtered"] = False
                    debug_data["reason"] = "LLM filtering returned minimal content"
                    
            except Exception as e:
                logger.warning(f"LLM filtering failed: {e}, using raw context")
                debug_data["filtered_context"] = None
                debug_data["filtered_context_length"] = 0
                debug_data["filtering_successful"] = False
                debug_data["error"] = str(e)
                debug_data["final_context"] = specification_context
                debug_data["using_filtered"] = False
                debug_data["reason"] = f"LLM filtering failed: {e}"
                final_context = specification_context
            
            # Save debug data to file
            try:
                import json
                debug_filename = f"output/3gpp_context_debug_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(debug_filename, 'w', encoding='utf-8') as f:
                    json.dump(debug_data, f, indent=2, ensure_ascii=False)
                logger.info(f"3GPP context debug data saved to: {debug_filename}")
            except Exception as debug_error:
                logger.warning(f"Could not save debug file: {debug_error}")
            
            return final_context
                
        except Exception as e:
            logger.warning(f"Could not load 3GPP specification context: {e}")
            return "- 3GPP specification context not available"
    
    def _clean_3gpp_context_format(self, context: str) -> str:
        """Clean and format the 3GPP context for better display"""
        if not context or context.strip() == "":
            return context
        
        # Remove excessive dashes and clean up formatting
        lines = context.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive dashes at the beginning
                if line.startswith('- '):
                    line = line[2:]  # Remove '- ' prefix
                cleaned_lines.append(line)
        
        # Join lines with proper spacing
        cleaned_context = '\n'.join(cleaned_lines)
        
        # Add proper section headers if missing
        if not cleaned_context.startswith('3GPP SPECIFICATION REFERENCE'):
            cleaned_context = f"3GPP SPECIFICATION REFERENCE\n{'='*80}\n\n{cleaned_context}"
        
        return cleaned_context
    
    def _filter_3gpp_context_with_llm(self, error_text: str, raw_context: str) -> str:
        """Filter 3GPP specification context to keep only relevant parts for error fixing."""
        try:
            if not raw_context or raw_context.strip() == "":
                return "- No 3GPP specification context available"
            
            # Truncate context if too long to avoid token limits
            max_context_length = 8000  # Leave room for prompt and response
            if len(raw_context) > max_context_length:
                raw_context = raw_context[:max_context_length] + "\n[Context truncated for processing]"
            
            # Prepare a more intelligent prompt for filtering
            filter_prompt = f"""You are a 5G/4G network expert. Analyze this 3GPP specification context and extract the MOST RELEVANT parts for fixing this specific error.

ERROR TO FIX: {error_text}

3GPP SPECIFICATION CONTEXT:
{raw_context}

ANALYSIS TASK:
Extract the most relevant sections that directly help understand and fix this error. Prioritize content that matches the error context.

PRIORITY ORDER (highest to lowest):
1. **Exact procedure/function mentioned in error** (e.g., if error mentions "RRCSetupRequest", prioritize RRC connection establishment procedures)
2. **Core procedures related to the error type** (RRC, AMF, connection establishment, etc.)
3. **Error handling and failure recovery procedures**
4. **Message definitions and signaling for the specific error**
5. **Configuration parameters and troubleshooting steps**

KEEP:
- Procedures that directly match the error context
- Core functionality related to the error
- Error handling and failure scenarios
- Message definitions and signaling
- Configuration parameters

REMOVE:
- Completely unrelated procedures
- Excessive background information
- Redundant content
- Low-priority procedures that don't match the error context

OUTPUT FORMAT:
- Maximum 3-4 most relevant sections
- Each section should be concise (max 200 words)
- Focus on actionable information for fixing the error
- Use clear, structured format with bullet points
- Include specific page numbers and section references

If no relevant content is found, return: "- No relevant 3GPP specification context found for this error"

Return only the filtered content, no explanations."""

            # Call LLM to filter the context
            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a 5G/4G network expert specializing in 3GPP specifications and error analysis."},
                    {"role": "user", "content": filter_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            filtered_context = response.choices[0].message.content.strip()
            logger.info(f"Filtered 3GPP context: {len(filtered_context)} characters")
            
            return filtered_context
            
        except Exception as e:
            logger.error(f"LLM filtering failed with error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Raw context length: {len(raw_context)}")
            
            # Return a clean truncated version if LLM filtering fails
            if len(raw_context) > 1500:
                return raw_context[:1500] + f"\n\n[Note: LLM filtering failed ({type(e).__name__}: {str(e)[:100]}). Showing raw context.]"
            return raw_context
    
    def _get_dynamic_code_examples(self, error_text: str) -> str:
        """Get dynamic code fix examples based on error type."""
        try:
            # Load error patterns
            with open("database/error_patterns_structured.json", 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            patterns = patterns_data.get('patterns', {})
            error_lower = error_text.lower()
            
            # Find matching pattern and get function analysis
            for pattern_name, pattern_data in patterns.items():
                keywords = pattern_data.get('keywords', [])
                
                if any(keyword in error_lower for keyword in keywords):
                    function_analysis = pattern_data.get('function_analysis', [])
                    
                    if function_analysis:
                        examples = ["### EXAMPLES OF GOOD CODE FIXES:"]
                        for analysis in function_analysis[:3]:  # Limit to 3 examples
                            if "function" in analysis.lower():
                                # Convert function analysis to actionable code fix examples
                                if "ngap" in analysis.lower():
                                    examples.append("- Add NGAP message validation and error handling")
                                elif "amf" in analysis.lower():
                                    examples.append("- Add AMF connectivity checks and selection logic")
                                elif "security" in analysis.lower():
                                    examples.append("- Add security context validation and key handling")
                                elif "handover" in analysis.lower():
                                    examples.append("- Add handover preparation and execution validation")
                                else:
                                    examples.append(f"- {analysis}")
                        return "\n".join(examples)
            
            # No pattern matches - check if we already generated a dynamic pattern
            # (This will be called after _get_dynamic_troubleshooting_hints, so pattern should exist)
            for pattern_name, pattern_data in patterns.items():
                if pattern_name.startswith('dynamic_'):
                    function_analysis = pattern_data.get('function_analysis', [])
                    if function_analysis:
                        examples = ["### EXAMPLES OF GOOD CODE FIXES (Dynamically Generated):"]
                        for analysis in function_analysis[:3]:
                            examples.append(f"- {analysis}")
                        return "\n".join(examples)
            
            # Default examples if no pattern matches
            return """### EXAMPLES OF GOOD CODE FIXES:
- Add network reachability validation before AMF registration
- Insert connectivity checks in AMF selection logic
- Add proper error handling for network failures
- Correct IP address validation in configuration parsing"""
            
        except Exception as e:
            logger.warning(f"Could not load dynamic code examples: {e}")
            return """### EXAMPLES OF GOOD CODE FIXES:
- Add network reachability validation before AMF registration
- Insert connectivity checks in AMF selection logic
- Add proper error handling for network failures
- Correct IP address validation in configuration parsing"""

    def _patch_completion_max_tokens(self, build_error_mode: bool) -> int:
        raw = os.getenv("MAX_PATCH_COMPLETION_TOKENS", "").strip()
        if raw.isdigit():
            return max(1024, min(int(raw), 32000))
        return _BUILD_ERROR_MAX_PATCH_TOKENS if build_error_mode else _DEFAULT_MAX_PATCH_TOKENS

    def _invoke_patch_completion(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        use_json_object: bool,
    ) -> Tuple[str, Optional[str]]:
        """Call Azure chat completions; returns (message_content, finish_reason)."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "seed": 99,
        }
        if use_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.azure_client.chat.completions.create(**kwargs)
        except Exception as e:
            if use_json_object:
                logger.warning(
                    "⚠️  Chat completion with JSON mode failed (%s); retrying without response_format",
                    e,
                )
                kwargs.pop("response_format", None)
                response = self.azure_client.chat.completions.create(**kwargs)
            else:
                raise
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        return content, finish

    @staticmethod
    def _normalize_candidate_file_path(file_path: str) -> str:
        """Normalize candidate file paths for matching and existence checks."""
        return (file_path or "").replace("\\", "/").strip()

    def _resolve_candidate_file_to_disk(self, file_path: str) -> Optional[str]:
        """
        Resolve a candidate config file path to an on-disk file path.
        Tries path relative to the OAI tree root next to this package (same as patch grounding).
        """
        rel = self._normalize_candidate_file_path(file_path)
        if not rel:
            return None
        root = self._codebase_root_dir()
        prefix = (self.openair_codebase_file_name or "").strip().replace("\\", "/")
        if prefix and rel.startswith(prefix + "/"):
            rel_within = rel[len(prefix) + 1 :]
        else:
            rel_within = rel
        p_norm = os.path.normpath(os.path.join(root, rel_within))
        if os.path.isfile(p_norm):
            return p_norm
        return None

    @staticmethod
    def _count_non_overlapping(haystack: str, needle: str) -> int:
        """Count non-overlapping occurrences of needle in haystack."""
        if not haystack or not needle:
            return 0
        return haystack.count(needle)

    def _run_cmake_diagnosis_pass(
        self,
        context: str,
        error_text: str,
        candidate_configs: List[Dict[str, Any]],
        deployment_context_info: str,
    ) -> Dict[str, Any]:
        """
        Stage 1: diagnosis-only pass for CMake/build-system errors.
        Returns validated evidence rows with file hashes and anchor checks.
        """
        diagnosis_prompt = f"""
You are running STAGE-1 DIAGNOSIS ONLY for CMake/build linker failures.

Return ONLY JSON in this exact schema:
{{
  "suspected_issues": [
    {{
      "error_symbol": "asn_DEF_F1AP_LTMConfiguration",
      "file_path": "openair2/F1AP/MESSAGES/ASN1/f1ap-18.6.0.cmake",
      "anchor_snippet": "exact text copied from Config Context",
      "confidence": 0.0,
      "why": "short reason"
    }}
  ],
  "diagnosis_summary": "short diagnosis"
}}

Rules:
- DIAGNOSIS ONLY. Do not output fixes, patches, new_value, or edit instructions.
- Use file paths present in SUSPECTED CONFIGURATIONS / Config Context, or standard OAI paths like
  `openair2/<PROTO>/MESSAGES/ASN1/<bundle>-<rel>.cmake` when the error names `asn_DEF_*` / `libasn1_*`.
- `anchor_snippet` must be copied exactly from provided context (3-15 lines).
- Prefer a single highest-confidence issue when evidence is narrow.

Deployment Context:
{deployment_context_info}

Error:
{error_text}
"""
        response_content, finish = self._invoke_patch_completion(
            diagnosis_prompt,
            context,
            max_tokens=3000,
            use_json_object=True,
        )
        diagnosis_data = _parse_cmake_diagnosis_json(response_content) or {}
        rows = diagnosis_data.get("suspected_issues", [])
        if not isinstance(rows, list):
            rows = []

        # Trust FAISS candidate paths, plus on-disk ASN.1 bundle `.cmake` files that match linker
        # symbols (retrieval often returns zero rows for pure linker logs).
        allowed_paths = {
            self._normalize_candidate_file_path((c or {}).get("file_path", ""))
            for c in (candidate_configs or [])
            if isinstance(c, dict)
        }
        allowed_paths = {p for p in allowed_paths if p}
        et = error_text or ""
        for prefix in ("F1AP", "NGAP", "X2AP", "S1AP"):
            if re.search(
                rf"(asn_DEF_{prefix}_|libasn1_{prefix.lower()}\.a)",
                et,
                re.I,
            ):
                for p in self._discover_asn1_bundle_cmake_paths(
                    candidate_configs, prefix
                ):
                    allowed_paths.add(self._normalize_candidate_file_path(p))

        validated_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rel_path = self._normalize_candidate_file_path(row.get("file_path", ""))
            anchor = str(row.get("anchor_snippet", "") or "")
            if not rel_path or not anchor or rel_path not in allowed_paths:
                continue

            disk_path = self._resolve_candidate_file_to_disk(rel_path)
            if not disk_path:
                continue
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            occ = self._count_non_overlapping(content, anchor)
            if occ <= 0:
                continue

            validated_rows.append(
                {
                    "error_symbol": str(row.get("error_symbol", "") or ""),
                    "file_path": rel_path,
                    "anchor_snippet": anchor,
                    "anchor_match_count": occ,
                    "file_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "why": str(row.get("why", "") or ""),
                }
            )

        return {
            "raw_finish_reason": finish,
            "diagnosis_summary": str(diagnosis_data.get("diagnosis_summary", "") or ""),
            "validated_issues": validated_rows,
        }

    def generate_patch(
        self,
        context: str,
        error_text: str = "",
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> FixSuggestion:
        """
        Step 3.2 - Patch Generation
        
        Use GPT-4o-mini to generate fix suggestions based on assembled context
        
        Args:
            context: Assembled context string
            
        Returns:
            FixSuggestion object with all fix details
        """
        logger.info("🧠 Generating fix suggestions with GPT-4o-mini...")

        dc = getattr(self, "_current_deployment_context", None) or {}
        build_error_mode = self._is_build_error_mode(dc)
        compile_diagnostic_mode = self._is_compile_diagnostic_log(dc)
        cmake_sys_mode = bool(isinstance(dc, dict) and dc.get("cmake_build_system_mode"))
        candidate_configs = candidate_configs or []

        deployment_context_info = self._get_deployment_context()
        # CMake/link mode: skip 3GPP TF-IDF spec retrieval (often missing embeddings; adds noise and tokens).
        if cmake_sys_mode:
            specification_context = ""
        else:
            specification_context = self._get_3gpp_specification_context(error_text)
        self._current_specification_context = specification_context

        # Deterministic ASN.1 bundle patch first — no LLM (avoids truncated JSON + empty retrieval).
        if cmake_sys_mode:
            det_patches = self._derive_cmake_asn1_fallback_patches(
                error_text, candidate_configs
            )
            if det_patches:
                _, grounded_cfg = self._validate_patch_grounding(
                    [], det_patches, dc
                )
                if grounded_cfg:
                    self._phase3_cmake_diagnosis = {
                        "validated_issues": [],
                        "diagnosis_summary": "deterministic_asn1_bundle_patch",
                        "source": "filesystem",
                    }
                    self._phase3_llm_meta = {
                        "skipped_llm": True,
                        "deterministic_cmake": True,
                        "parse_ok": True,
                        "max_output_tokens": 0,
                        "json_mode": False,
                        "build_error_mode": build_error_mode,
                        "finish_reason": None,
                        "attempts": [],
                    }
                    return FixSuggestion(
                        suspected_functions=[],
                        suspected_configs=[p.file_path for p in grounded_cfg],
                        reason=(
                            "Linker undefined reference to asn_DEF_* — added missing generated "
                            "`.c` to the ASN.1 bundle source list in the matching `*.cmake` file."
                        ),
                        config_fix=(
                            "Ensure the generated ASN.1 source file for the missing type is listed "
                            "under `f1ap_source` / `ngap_source` in the protocol bundle `.cmake`."
                        ),
                        code_patches=[],
                        config_patches=grounded_cfg,
                        root_cause_analysis=(
                            "The static archive was linked without the object file that defines "
                            "the ASN.1 descriptor for the missing type — usually because the "
                            "generated `.c` was not listed in the bundle CMake source list."
                        ),
                        investigation_steps=[
                            "Save the `.cmake` change, reconfigure if needed, and rebuild the target "
                            "that links `libasn1_*`.",
                        ],
                        specification_context=specification_context,
                    )

            # If bundle(s) already list the generated `.c`, enrich context but still run Phase 3 so
            # the model can propose link-order / target / rebuild fixes (skipping LLM left users
            # with suspected_configs but zero patches).
            bundle_hint = self._cmake_bundle_source_list_complete_guidance(
                error_text, candidate_configs
            )
            if bundle_hint is not None:
                context = (
                    context
                    + "\n\n## 📎 ASN.1 BUNDLE PRE-CHECK (workspace files)\n"
                    + (bundle_hint.reason or "")
                    + "\n\n**Pre-check steps:**\n"
                    + "\n".join(
                        f"- {s}" for s in (bundle_hint.investigation_steps or [])
                    )
                    + "\n\n**Response rules:** The missing `asn_DEF_*` may already be covered by a "
                    "listed `.c` in `f1ap-*.cmake`. Still emit **`config_patches`** when "
                    "*SUSPECTED CONFIGURATIONS* include grounded anchors for `CMakeLists.txt` or "
                    "`*.cmake` (e.g. `target_link_libraries`, `add_dependencies`, link order, or "
                    "wrong bundle version). If no safe anchored edit exists, use **`investigation_steps`** "
                    "with concrete rebuild/codegen commands; avoid empty output."
                )

        troubleshooting_hints = (
            "" if cmake_sys_mode else self._get_dynamic_troubleshooting_hints(error_text)
        )
        code_examples = "" if cmake_sys_mode else self._get_dynamic_code_examples(error_text)

        cmake_diagnosis = {}
        if cmake_sys_mode:
            # Two-stage flow:
            # Stage 1 -> diagnosis-only with grounded file+anchor evidence
            # Stage 2 -> patch generation constrained to validated evidence
            cmake_diagnosis = self._run_cmake_diagnosis_pass(
                context=context,
                error_text=error_text,
                candidate_configs=candidate_configs,
                deployment_context_info=deployment_context_info,
            )
            validated = cmake_diagnosis.get("validated_issues", [])
            if validated:
                context = (
                    context
                    + "\n\n## 🧪 STAGE-1 VALIDATED CMAKE DIAGNOSIS (AUTHORITATIVE)\n"
                    + "Use ONLY the validated issues below to generate config patches. "
                    + "If an anchor is ambiguous (`anchor_match_count` > 1), generate a conservative patch "
                    + "or return no patch for that row.\n\n```json\n"
                    + json.dumps(cmake_diagnosis, indent=2)
                    + "\n```"
                )
            self._phase3_cmake_diagnosis = cmake_diagnosis
        log_error_kind = ""
        if isinstance(dc, dict):
            log_error_kind = str(dc.get("log_error_kind") or "").strip().lower()
        rrc_rejection_priority_block = (
            ""
            if cmake_sys_mode
            else r"""
🚨 **CRITICAL RRC REJECTION FUNCTION**: If `rrc_gNB_generate_RRCReject` appears in the suspected functions list, it has HIGHEST PRIORITY (score 0.9). This function is ESSENTIAL for handling RRC setup failures gracefully. For segmentation faults in `rrc_handle_RRCSetupRequest`, you MUST include code that calls `rrc_gNB_generate_RRCReject` in error handling paths to properly reject the connection instead of crashing. This function generates and sends RRCReject messages to UEs when setup requests fail.
"""
        )
        log_type_instructions = ""
        if log_error_kind == "runtime":
            log_type_instructions = """
### LOG TYPE: RUNTIME (MANDATORY)
- Prioritize state-machine, protocol-flow, and data-lifetime fixes in C/C++ logic.
- Use `code_patches` as primary output; propose `config_patches` only when runtime configs clearly mismatch deployment context.
- Ground each patch in observed runtime evidence (anchors, sequences, fault path), not speculative refactors.
"""
        elif log_error_kind == "build":
            log_type_instructions = """
### LOG TYPE: BUILD (MANDATORY)
- Prioritize compiler diagnostics and include/type/signature mismatches from build output.
- Focus `code_patches` on minimal compile-fix edits (headers, symbols, types, API usage), preserving behavior.
- Add `config_patches` only when build wiring/config content directly causes the compile failure.
"""
        elif log_error_kind == "other":
            log_type_instructions = """
### LOG TYPE: OTHER / UNKNOWN (MANDATORY)
- Start with conservative root-cause analysis and avoid overfitting to a single subsystem.
- Prefer minimal, high-confidence edits and include investigation steps when certainty is medium/low.
- If evidence points to build-system or runtime only after analysis, bias fixes toward that discovered path.
"""
        cmake_mode_instructions = (
            """

### CMAKE / BUILD-SYSTEM MODE (MANDATORY — OVERRIDES RUNTIME/RRC PRIORITIES)
- Set `suspected_functions` to **[]** unless the build log explicitly points to a specific C/C++ edit.
- **Primary fixes:** `config_patches` on the **exact build files** shown in *SUSPECTED CONFIGURATIONS* (paths + line context). Prefer **`*.cmake` under `openair2/.../MESSAGES/ASN1/`** (e.g. `f1ap-18.6.0.cmake`) when the linker mentions **`libasn1_f1ap.a`** or **`asn_DEF_F1AP_*`** — those bundles list generated ASN.1 `.c` files; **`undefined reference to asn_DEF_<Type>`** usually means **`<Type>.c` is missing from that list** (or codegen was not re-run).
- **Do not** default to only top-level **`CMakeLists.txt`** if retrieved rows include a matching **`f1ap-*.cmake` / `ngap-*.cmake`** with relevant `param_value` lines — patch **that** file using the provided **Config Context** snippets.
- Linker lines often **omit** the `.cmake` filename; use **LINKER-INFERRED TARGETS** + **suspected `file_path`** to choose the patch target.
- **Do not** apply RRC rejection / UE call-flow rules here; they are irrelevant to CMake/link failures.
"""
            if cmake_sys_mode
            else ""
        )
        patch_max_tokens = self._patch_completion_max_tokens(build_error_mode)
        patch_use_json_object = os.getenv("PATCH_JSON_MODE", "1").strip() != "0"
        if cmake_sys_mode:
            config_file_restriction_guidance = """**📁 CONFIG FILE RESTRICTION (CMAKE / LINKER MODE ONLY):**
- **🚨 CRITICAL: You MAY suggest config patches for:**
  - **CMakeLists.txt**
  - **.cmake files** (especially **`.../MESSAGES/ASN1/<bundle>-<rel>.cmake`** for OAI ASN.1 codegen source lists)
  - **cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf** (only when directly relevant to the failure)
- For **`undefined reference to asn_DEF_*`** / **`libasn1_*.a`**: prefer patches on the **ASN1 `*.cmake`** file that lists generated `.c` sources for that library, when such a path appears in suspected configurations."""
        elif compile_diagnostic_mode:
            config_file_restriction_guidance = """**📁 CONFIG FILE RESTRICTION (COMPILE / DIAGNOSTIC LOG — NOT CMAKE MODE):**
- **Primary output:** `code_patches` on the **exact C/C++ source or headers** implicated by compiler diagnostics.
- **🚨 CRITICAL: ONLY suggest config patches for these runtime deployment files when the compile error is clearly caused by a wrong macro/value from them:**
  - **cu_gnb.conf, du_gnb.conf, ue.conf, 5g_sa_ue.conf**
- **DO NOT** suggest **CMakeLists.txt** or **\*.cmake** changes in this mode; linker/ASN bundle wiring is handled by the separate CMake log path.
- **DO NOT suggest config patches for any other files.**"""
        else:
            config_file_restriction_guidance = """**📁 CONFIG FILE RESTRICTION:**
- **🚨 CRITICAL: ONLY suggest config patches for these files:**
  - **cu_gnb.conf** (for CU/gNB network configs)
  - **du_gnb.conf** (for DU configs)
  - **ue.conf** (for UE configs)
  - **5g_sa_ue.conf** (for UE security/authentication configs)
- **🚨 CRITICAL: DO NOT suggest config patches for any other config files**
- **Role-based guidance:**
  - CU/gNB errors → cu_gnb.conf or du_gnb.conf
  - UE errors → ue.conf or 5g_sa_ue.conf
  - Security/authentication errors → 5g_sa_ue.conf (IMSI, key, OPC, DNN, NSSAI, etc.)"""
        build_compiler_section = ""
        if compile_diagnostic_mode:
            build_compiler_section = """

### COMPILER / BUILD-ERROR MODE (MANDATORY)
- **You MUST populate `code_patches` with at least one concrete C/C++ edit** when the log shows compiler errors, using the **Code:** snippets and file paths from *SUSPECTED FUNCTIONS* above. Copy `original_code` **verbatim** from those snippets when possible so the patch can be applied.
- In `reason`, state the **systemic root cause first** when diagnostics repeat (e.g. ASN.1 codegen vs hand-edited types, mismatched `ProtocolExtensionContainer_*` names). Then summarize impact by file.
- **Each `original_code` and `patched_code` MUST stay at or under ~35 lines.** Include only the edited span plus 2–4 lines of surrounding context. Do **not** paste entire functions.
- Prefer **compiler-suggested fixes** (`did you mean ...`) and typedef/header corrections.
- Combine identical mechanical renames (same old→new type) into **one** patch when possible.
- The JSON object must be **complete and parseable** (every string and bracket closed) within your reply.

"""

#         system_prompt = f"""You are an expert telecommunications software debugger and fix engineer specializing in 5G/LTE systems. You have deep knowledge of:

# - NGAP (Next Generation Application Protocol)
# - RRC (Radio Resource Control) 
# - NAS (Non-Access Stratum)
# - AMF (Access and Mobility Management Function)
# - gNB (Next Generation Node B) architecture
# - SCTP, network configuration, and protocol stacks
# - Network connectivity, subnet analysis, and routing

# Your task is to analyze error contexts and propose SURGICAL but MEANINGFUL fixes that address the ACTUAL root cause.  
# Fixes should not be trivial placeholders — they must directly resolve the issue while preserving existing functionality.

# ### ROOT CAUSE ANALYSIS REQUIREMENTS:
# - **ALWAYS analyze network connectivity first** - most 5G errors are network/infrastructure issues
# - **Check subnet compatibility** between gNB and AMF configurations
# - **Verify IP address reachability** and routing between endpoints
# - **CRITICAL: Consider Docker/Container networking** - AMF may be running in Docker container on different machine
# - **Check for container isolation issues** - gNB cannot directly reach container IPs without proper routing
# - **Distinguish between network issues** and code logic issues
# - **Trace the exact error flow** from error message to root cause
# - **Prioritize network fixes** over code logic fixes when connectivity is the issue
# - **USE RUNTIME LOG CONTEXT** - analyze the actual log messages to understand the sequence of events and runtime state
# - **CORRELATE LOG EVENTS** - match log messages with code execution flow to identify where the failure occurs

# ### CRITICAL REQUIREMENTS:
# - **Preserve all existing function logic completely** (do not rewrite entire functions).
# - **Add minimal but effective corrections**: validation, missing assignments, proper error handling, correct parameter usage, or protocol checks.
# - **Do not limit to null checks only** — include logic adjustments or missing branches if they clearly fix the error.
# - **Never call non-existent functions** — use only available APIs and structures.
# - **Focus on the specific error scenario**, not generic safety checks.
# - **CRITICAL: Provide EXACT line numbers or specific context** for where to insert/replace code.
# - **CRITICAL: Check variable scope** - only reference variables that are declared and in scope at the insertion point.
# - **CRITICAL: Use specific context** like "after line containing 'amf_desc_p = ngap_gNB_get_AMF'" instead of vague descriptions.
# - **CRITICAL: NEVER reference function names in context** - use specific code lines or variable assignments instead.
# - **CRITICAL: Specify insertion points INSIDE function bodies**, not at function signatures.

# ### DEPLOYMENT CONTEXT (ALWAYS USE THIS DATA):
# {deployment_context_info}

# {specification_context}

# {troubleshooting_hints}

# ### NETWORK CONFIGURATION VALIDATION:
# - **Ensure IP addresses are valid** and on compatible subnets
# - **Verify network reachability** between configured endpoints
# - **Check for subnet conflicts** or routing issues
# - **Do not suggest invalid network configurations**
# - **Preserve subnet masks** unless there's a specific routing issue
# - **Ensure gNB and AMF can communicate** on the configured network
# - **Use the actual deployment IP addresses** from the context above
# - **Consider the specific network topology** with CU/DU on 10.138.77.131 and AMF on 192.168.70.132

# ### CONFIGURATION PARAMETER SEMANTICS (CRITICAL):
# **Apply these semantic rules to avoid configuration mistakes:**

# **LOCAL ENDPOINT PARAMETERS** (gNB's own identity):
# - Parameters containing "GNB_IPV4_ADDRESS", "FOR_NG_AMF", "LOCAL", "GNB_IP": Use gNB's IP from deployment context
# - These specify the gNB's own IP address when communicating with other nodes

# **REMOTE ENDPOINT PARAMETERS** (target service identity):  
# - Parameters containing "amf_ip_address", "AMF_IP", "REMOTE", "TARGET": Use AMF's IP from deployment context  
# - These specify the IP address of the service the gNB connects to

# **PORT PARAMETERS**:
# - Parameters ending with "PORT", "PORTC", "PORTD": Use corresponding ports from deployment context
# - Local ports: use local_s_portc/local_s_portd, Remote ports: use remote_s_portc/remote_s_portd

# **TIMER PARAMETERS**:
# - Parameters ending with "Timer", "TIMER", containing "timeout": Use reasonable values (8-64, never 0)
# - Zero values typically disable timers and cause failures

# **CRITICAL RULE**: Never set a gNB's own IP parameter to the AMF's IP address or vice versa!

# ### CONFIGURATION PATCH REQUIREMENTS:
# - **CRITICAL**: ONLY suggest config patches when the parameter EXISTS in deployment context AND values differ
# - Use ONLY actual config names, file paths, and current values from candidate configs.
# - Always include `relevance_score` and a clear explanation of why this parameter must change.
# - Set `new_value` to the EXACT value from deployment context (never use placeholders or "improvements")
# - Do NOT output patches for parameters not in deployment context (like gNB_name, gNB_ID, Active_gNBs, etc.)
# - Do NOT output no-op patches (new value == current value).
# - **Validate that suggested IP addresses are reachable** from the gNB's network

# ### RESPONSE FORMAT:
# Respond with ONLY valid JSON (no markdown, no explanation outside JSON):

# {{
#     "suspected_functions": ["function1", "function2"],
#     "suspected_configs": ["config1", "config2"], 
#     "reason": "Detailed root cause explanation",
#     "config_fix": "Specific configuration corrections with exact values",
#     "code_patches": [
#         {{
#             "function_name": "function_name",
#             "file_path": "path/to/file.c",
#             "patch_type": "targeted_insertion_or_adjustment",
#             "original_code": "// Line(s) around the issue",
#             "patched_code": "// The exact corrected code snippet",
#             "line_numbers": "EXACT line numbers (e.g., '120-125') or specific context (e.g., 'after line containing \\"amf_desc_p = ngap_gNB_get_AMF\\"')",
#             "description": "Why this correction resolves the error"
#         }}
#     ],
#     "config_patches": [
#         {{
#             "config_name": "parameter_name",
#             "file_path": "path/to/config.conf",
#             "patch_type": "set_value",
#             "current_value": "current_value",
#             "new_value": "corrected_value",
#             "line_number": "approximate_line_or_section",
#             "relevance_score": "confidence_score_from_analysis",
#             "description": "Why this config change resolves the error"
#         }}
#     ],
#     "root_cause_analysis": "Deep technical analysis of why this error occurs",
#     "investigation_steps": ["step1", "step2", "step3"]
# }}

# ### INVESTIGATION STEPS REQUIREMENTS:
# - **ALWAYS include Docker/container networking steps** when AMF is on different subnet
# - **MUST include the exact static route command** when container networking is suspected
# - **Include specific IP addresses** from deployment context in all steps
# - **Provide actionable commands** that can be executed directly

# ### EXAMPLES OF GOOD NETWORK FIXES:
# - **Docker/Container Networking**: "AMF (192.168.70.132) is running in Docker container on Core Network Machine (10.138.77.217). Add static route: `ip route add 192.168.70.132 via 10.138.77.217 dev <interface>` to route gNB traffic through the host machine"
# - **Container Isolation Issue**: "gNB (10.138.77.131) cannot reach AMF container (192.168.70.132) directly due to Docker networking isolation. Configure routing through Core Network Machine (10.138.77.217)"
# - **Network connectivity**: "AMF IP 192.168.70.132 is on different subnet than gNB 10.138.77.131 - add static route or configure gNB to use 192.168.70.x subnet"
# - **Subnet configuration**: "gNB (10.138.77.131) and AMF (192.168.70.132) are on different subnets - ensure routing via Core Network Machine (10.138.77.217)"
# - **Routing issues**: "Add network reachability check before AMF registration - ping 192.168.70.132 from gNB machine"
# - **IP validation**: "Validate AMF IP address 192.168.70.132 is reachable before attempting connection"
# - **Port configuration**: "Ensure SCTP ports match: local_s_portc=501, local_s_portd=2152, remote_s_portc=500, remote_s_portd=2152"

# {code_examples}

# ### EXAMPLES OF GOOD CONTEXT SPECIFICATIONS:
# - `"after line containing 'amf_desc_p = ngap_gNB_get_AMF'"` ✅
# - `"after line containing 'if (msg->ue_identity.presenceMask & NGAP_UE_IDENTITIES_guami)'"` ✅
# - `"after line containing 'memcpy(&sctp_new_association_req_p->remote_address'"` ✅
# - `"after line containing 'ngap_amf_data_p->assoc_id = -1'"` ✅

# ### EXAMPLES OF BAD CONTEXT SPECIFICATIONS:
# - `"after line containing 'select_amf'"` ❌ (function name)
# - `"after line containing 'ngap_gNB_register_amf'"` ❌ (function name)
# - `"at function start"` ❌ (too vague)
# - `"after variable declarations"` ❌ (too vague)

# ### EXAMPLES OF BAD FIXES:
# - Adding only generic null checks without addressing the root cause
# - Rewriting the full function
# - Removing existing working logic
# - Using placeholder or invented parameters/functions
# - No-op config changes (same value as before)
# - **Removing subnet masks** without understanding routing requirements
# - **Making gNB and AMF share the same IP address**
# - **Ignoring network connectivity issues** and focusing only on code logic"""        
        system_prompt = f"""
You are an expert telecommunications software debugger and fix engineer specializing in 5G/LTE systems.  
You have deep knowledge of:

- NGAP (Next Generation Application Protocol)
- RRC (Radio Resource Control)
- NAS (Non-Access Stratum)
- AMF (Access and Mobility Management Function)
- gNB (Next Generation Node B) architecture
- SCTP, network configuration, and protocol stacks
- Network connectivity, subnet analysis, and routing

Your task is to analyze error contexts and propose **SURGICAL but MEANINGFUL** fixes that address the **ACTUAL** root cause.  
Fixes must intelligently **reconstruct missing or incorrect control-flow** when evidence shows the function's logic is incomplete.  
Do **NOT** rely solely on defensive null checks unless they are truly the cause.

🚨 **COMPILER-HINTS-FIRST RULE (HIGHEST PRIORITY)**:
If the provided log/error context contains compiler suggestions such as:
- "has no member named 'X'; did you mean 'Y'?"
- "unknown type name 'X'; did you mean 'Y'?"
Then you MUST first propose fixes that apply these exact replacements (`X` → `Y`) in the relevant code snippets.
Only if no compiler hint applies should you invent or infer alternative fixes.

🚨 **MANDATORY REQUIREMENT**: If you find any incomplete if-else if chains (missing final else clause), you MUST add the missing else branch to handle unrecognized cases. This is CRITICAL for preventing segmentation faults.
{rrc_rejection_priority_block}
{log_type_instructions}
{cmake_mode_instructions}
🚨 **CRITICAL RULE FOR NULL POINTER FIXES IN ELSE BLOCKS**: When adding an `else` block that calls a function requiring a non-NULL pointer parameter, you MUST:
1. **Identify the function call** that requires a non-NULL pointer (e.g., `rrc_gNB_generate_RRCReject(rrc, ue_context_p)` where `ue_context_p` must not be NULL)
2. **Check the call graph context** and code snippet to find functions that initialize/create this pointer (e.g., `rrc_gNB_create_ue_context` creates `ue_context_p`)
3. **Before calling the function**, you MUST first call the initialization function to create the pointer
4. **Follow the pattern** shown in other branches of the code (if `if` or `else if` branches create the pointer, the `else` block must also create it)
5. **NEVER** call a function with a NULL or uninitialized pointer - this will cause a segmentation fault

🚨 **CRITICAL PLACEMENT RULE**: When adding missing else clauses, the `line_numbers` must reference the line AFTER the closing brace `}}` of the last else if block, NOT the opening else if line!

🔍 **STEP-BY-STEP SEARCH PROCESS**:
1. **FIND** the line: `}} else if (NR_InitialUE_Identity_PR_ng_5G_S_TMSI_Part1 == rrcSetupRequest->ue_Identity.present) {{`
2. **SCAN DOWN** to find the ACTUAL LAST line in this else if block (look for assignments like `UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1;`)
3. **FIND** the closing brace `}}` right after that last assignment
4. **USE** that closing section as your original_code

⚠️ **EXAMPLE OF MISSING ELSE**: If you see code like:
```
if (condition1) {{
    // some code
}} else if (condition2) {{
    // some code
    UE->some_field = value;  // <- FIND THE ACTUAL LAST ASSIGNMENT
}}
// <-- MISSING ELSE CLAUSE HERE!
NR_CellGroupConfig_t *cellGroupConfig = NULL;  // Code continues
```
You MUST find the LAST assignment (like `UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1;`) + closing brace and add the else clause there.

### DEPLOYMENT CONTEXT (ALWAYS USE THIS DATA - DO NOT CHANGE THESE VALUES):
{deployment_context_info}

**🚨 CRITICAL: THESE ARE THE ONLY PARAMETERS YOU CAN SUGGEST CONFIG PATCHES FOR:**
The deployment context above contains these specific parameters. ONLY suggest patches for parameters that:
1. Are listed here with their correct values
2. Have different current values in the candidate configs
3. Are related to the error

**✅ PARAMETERS IN DEPLOYMENT CONTEXT (check these for mismatches):**
- CU IP Address, DU IP Address, gNB IP Address, AMF IP Address, Core Network Machine IP
- local_s_portc, local_s_portd, remote_s_portc, remote_s_portd
- NSSAI SST, NSSAI SD, NMC Size, DNN
- (Any other parameters explicitly shown in deployment context above)

**❌ DO NOT suggest patches for parameters NOT in the list above** (like gNB_name, gNB_ID, Active_gNBs, GNB_PORT_FOR_S1U, cell_type, tracking_area_code, mobile_country_code, etc.)

**🚨 MATCHING RULES:**
- For each candidate config parameter, check if it exists in deployment context
- If YES and values differ → suggest changing to deployment context value
- If NO (not in deployment context) → skip it, DO NOT suggest any patch
- If YES and values match → skip it (no-op)

{config_file_restriction_guidance}

{troubleshooting_hints}

### ROOT CAUSE ANALYSIS REQUIREMENTS
- Always check **network connectivity first** for 5G issues.  
- Distinguish between **network vs. code logic** failures.  
- Trace error flow from **log messages** to code path.  
- **CRITICAL**: Look for **incomplete conditional logic patterns**:
  * Hardcoded constant comparisons (e.g., `== 3`) instead of runtime field checks
  * Missing `else` branches in multi-case scenarios (enum handling, protocol states)
  * Unvalidated assumptions about input data structure/presence
- If branches or identity handling are clearly missing or malformed, **reconstruct the expected flow** using:
  * 3GPP specifications
  * Variable names and surrounding code context
  * Typical OpenAirInterface gNB patterns  
- Do **NOT** require the original code to be supplied—derive expected behavior from specs and context.

### CODE PATCH RULES
- **CRITICAL**: Preserve existing correct logic; **insert or adjust only the minimal lines** necessary to restore intended behavior.  
- Use **exact surrounding code lines or variable names** for placement (`after line containing "..."`).  
- **CRITICAL**: Look for **hardcoded constant comparisons** (like `== 3`, `== 1`) and replace with **proper dynamic checks** using available variables/fields.
- **CRITICAL**: If the function has incomplete conditional logic (missing `else` branches, unhandled enum cases), **reconstruct the missing branches** instead of adding trivial guards.  
- **CRITICAL**: For segmentation faults, trace the **actual cause** (uninitialized variables, missing validation, incomplete state handling) rather than adding generic null checks.
- Never invent functions or config parameters that don't exist.  
- Validate variable scope before referencing them.  
- **AVOID GENERIC NULL CHECKS**: If a segmentation fault is due to skipped handling of input cases, **add the missing handling**, not just a pointer check.

### COMMON SEGMENTATION FAULT PATTERNS TO LOOK FOR
- **Hardcoded enum comparisons**: `if (ENUM_CONSTANT == hardcoded_value)` → should be `if (data->field == ENUM_CONSTANT)`
- **CRITICAL: Missing else/default cases**: Incomplete switch/if-else chains for enum/state handling
- **MANDATORY: Always add else branch to any if-else if chain that lacks final else clause**
- **Uninitialized pointers**: Variables that remain NULL when certain conditions aren't met
- **Protocol state violations**: Functions proceeding without validating required protocol fields
- **Array/structure access**: Accessing fields without checking structure validity first

### 🚨 CRITICAL REQUIREMENT: MISSING ELSE BRANCHES 🚨
**🚨 READ THIS ENTIRE SECTION CAREFULLY - IT IS CRITICAL FOR PREVENTING SEGMENTATION FAULTS 🚨**

**EXAMINE THE PROVIDED CODE CAREFULLY**: Look for incomplete if-else if chains and add the missing else clause:

**⚠️ MOST COMMON ERROR: Calling a function with a NULL pointer in the else block**
**⚠️ IF YOU ADD AN ELSE BLOCK AND CALL A FUNCTION THAT USES A POINTER VARIABLE, YOU MUST INITIALIZE THAT VARIABLE FIRST!**
**⚠️ DO NOT CALL `error_function(context, variable)` IF `variable` IS NULL - INITIALIZE IT FIRST!**

**PATTERN TO LOOK FOR:**
1. Code that starts with: `SomeType *variable = NULL;` (a pointer variable initialized to NULL)
2. Followed by: `if (condition1) {{ ... variable = initialize_function(...); ... }} else if (condition2) {{ ... variable = initialize_function(...); ... }}`
3. The if-else if chain ends with just a closing brace `}}` (NO final `else` block)
4. After the closing brace, the code continues and uses `variable` (which could still be NULL if neither condition matched)

**EXAMPLE OF THE PROBLEM PATTERN:**
```c
SomeType *variable = NULL;
if (condition1) {{
  variable = initialize_function(...);
}} else if (condition2) {{
  variable = initialize_function(...);
  // some other assignments
}}
// ⚠️ PROBLEM: If neither condition matches, variable is still NULL!
// Code continues here and might use variable, causing NULL pointer dereference
if (some_other_condition) {{
  function_that_uses_variable(context, variable); // ⚠️ CRASH if variable is NULL!
}}
```

**REQUIRED FIX:**
- For missing else clauses, you MUST create a SEPARATE patch that adds an `else` block:
  * `"original_code": "FIND THE ACTUAL LAST STATEMENT in the else if block (the last assignment or statement before the closing brace) + closing brace }}"` 
  * `"patched_code": "SAME ENDING LINES + }} else {{\\n    // STEP 1: Initialize the variable that was NULL (MANDATORY)\\n    // STEP 2: Then call the error handling function\\n}}"`
  * `"line_numbers": "after the closing brace of the else if block"`
  * **🚨 MANDATORY CHECKLIST FOR ELSE BLOCK (VERIFY EACH STEP):**
    
    **BEFORE writing the else block, answer these questions:**
    1. What variable is initialized in the `if` block? (e.g., `ue_context_p`)
    2. What function is called to initialize it in the `if` block? (e.g., `rrc_gNB_create_ue_context`)
    3. What parameters are passed to that function in the `if` block? (e.g., `assoc_id, msg->crnti, rrc, random_value, msg->gNB_DU_ue_id`)
    4. What function is called to initialize it in the `else if` block? (should be the SAME function)
    5. What parameters are passed to that function in the `else if` block?
    6. What function is called later that uses this variable? (e.g., `rrc_gNB_generate_RRCReject`)
    
    **NOW write the else block following this EXACT structure:**
    ```c
    }} else {{
      // STEP 1: Initialize the variable using the SAME function from if/else if blocks
      // Extract or generate any required parameters
      // Example: uint64_t random_value = 0;
      //          memcpy(...); // if needed to extract from input
      variable = initialization_function(param1, param2, param3, param4, param5);
      
      // STEP 2: Log the error (optional but recommended)
      LOG_E(..., "Error message...");
      
      // STEP 3: Call the error handling function that uses the variable
      error_handling_function(context, variable);
      return;
    }}
    ```
    
    **VERIFICATION CHECKLIST (your else block MUST have):**
    - [ ] The initialization function call (same as in if/else if blocks)
    - [ ] The variable assignment (e.g., `variable = initialization_function(...)`)
    - [ ] The error handling function call AFTER initialization
    - [ ] NO direct calls to functions that use the variable BEFORE initialization
    
    **COMMON MISTAKES TO AVOID:**
    - ❌ Calling `error_handling_function(context, variable)` when `variable` is still NULL
    - ❌ Forgetting to call the initialization function
    - ❌ Calling the initialization function AFTER the error handling function
    - ✅ CORRECT: Initialize first, THEN call error handling function

- **🚨 CRITICAL FOR NULL POINTER FIXES**: If you need to call a function that requires a non-NULL pointer, you MUST first initialize that pointer. 
  - **MANDATORY RULE**: Before calling any function that dereferences a pointer parameter, check if that pointer could be NULL. If it could be NULL (e.g., because it was only initialized in certain branches of an if-else if chain), you MUST initialize it first.
  - **REQUIRED PATTERN**: 
    1. Identify what function calls appear later in the code that use the variable that might be NULL
    2. Look at the call graph context and code snippet to find the appropriate initialization function (check what functions are called in the `if`/`else if` blocks to initialize the variable)
    3. In the `else` block, call the initialization function first, then call the function that uses the variable
  - **NEVER** call a function with a NULL or uninitialized pointer parameter - this will cause a segmentation fault.
- **IMPORTANT**: If the code snippet already contains function calls for initialization, you MUST use them in your patch when they are needed to prevent NULL pointer dereferences. Check the provided code snippet and call graph context to see what functions are already being used in similar contexts.

### CONFIGURATION VALIDATION
- **CRITICAL**: ONLY suggest patches when parameter exists in deployment context AND values differ
- Use actual config names and values from candidate configs.  
- Set new_value to EXACT deployment context value (no placeholders, no "improvements")
- DO NOT suggest patches for parameters not in deployment context
- DO NOT suggest no-op patches (same value).

{build_compiler_section}
### RESPONSE FORMAT
Respond with **ONLY valid JSON** in this EXACT structure:
**🚨 CRITICAL: Do NOT use JavaScript comments (//) in JSON - they are not valid JSON syntax!**

{{
    "suspected_functions": ["function1", "function2"],
    "suspected_configs": ["config1", "config2"], 
    "reason": "Detailed root cause explanation",
    "config_fix": "Specific configuration corrections with exact values",
    "code_patches": [
#         {{
#             "function_name": "function_name",
#             "file_path": "path/to/file.c",
#             "patch_type": "targeted_insertion_or_adjustment",
#             "original_code": "// Line(s) around the issue",
#             "patched_code": "// The exact corrected code snippet",
#             "line_numbers": "EXACT line numbers (e.g., '120-125') or specific context (e.g., 'after line containing \\"amf_desc_p = ngap_gNB_get_AMF\\"')",
#             "description": "Why this correction resolves the error"
#         }}
#     ],
#     "config_patches": [
#         {{
#             "config_name": "parameter_name",
#             "file_path": "path/to/config.conf",
#             "patch_type": "set_value",
#             "current_value": "current_value",
#             "new_value": "corrected_value",
#             "line_number": "approximate_line_or_section",
#             "relevance_score": "confidence_score_from_analysis",
#             "description": "Why this config change resolves the error"
#         }}
#     ],
#     "root_cause_analysis": "Deep technical analysis of why this error occurs",
#     "investigation_steps": ["step1", "step2", "step3"]
# }}

"Specifically use this context from the 3GPP specifications to understand the expected behavior and mandatory validations, to give the best fix suggestions."
{specification_context}

### ADDITIONAL HINTS
- **Protocol Compliance**: Use 3GPP specifications and RFCs to understand expected behavior and mandatory validations.
- **Network Connectivity**: For 5G/LTE issues, verify IP reachability, routing, and port accessibility between components.
- **State Validation**: Ensure all possible input states/cases are handled with appropriate error responses.
- **Container Networking**: If using Docker/containers, add static routes or network configuration as needed.

### GOOD PATCH EXAMPLES
- **CRITICAL**: For missing else clauses - EXAMPLE PATCH (shows ACTUAL CLOSING LINES):
  ```
  {{
    "original_code": "    UE->Initialue_identity_5g_s_TMSI.presence = true;\\n    UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1;\\n  }}",
    "patched_code": "    UE->Initialue_identity_5g_s_TMSI.presence = true;\\n    UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1;\\n  }} else {{\\n    LOG_E(NR_RRC, \\"Unhandled ue_Identity.present value: %d\\", rrcSetupRequest->ue_Identity.present);\\n    return;\\n  }}",
    "line_numbers": "replace the section ending with UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1; }}"
  }}
  ```
  **KEY**: Find the ACTUAL LAST assignment in the else if block (like UE->ng_5G_S_TMSI_Part1 = s_tmsi_part1;) + closing brace

- **🚨 CRITICAL EXAMPLE FOR NULL POINTER FIXES**: When adding an `else` block that calls a function requiring a non-NULL pointer, follow this pattern:
  * **Step 1**: Identify the function call that needs a non-NULL pointer (e.g., `rrc_gNB_generate_RRCReject(rrc, ue_context_p)`)
  * **Step 2**: Check the call graph context and code snippet to find the function that creates this pointer (e.g., `rrc_gNB_create_ue_context` appears in call graph and is used in other branches)
  * **Step 3**: Extract any required parameters from input data (e.g., extract `random_value` from `rrcSetupRequest->ue_Identity.choice.randomValue` if needed)
  * **Step 4**: Call the initialization function first (e.g., `ue_context_p = rrc_gNB_create_ue_context(...)`)
  * **Step 5**: Then call the function that requires the pointer (e.g., `rrc_gNB_generate_RRCReject(rrc, ue_context_p)`)
  * **MANDATORY**: You MUST initialize pointers before using them in function calls. Check the call graph and code snippet to identify the correct initialization function.
- Replace hardcoded comparisons: `if (enum_constant == 3)` → `if (variable->field == enum_constant)`
- Adding **routing checks** or static route commands for Dockerized AMF  

### BAD PATCH EXAMPLES
- Only adding generic NULL checks like `if (ptr == NULL) {{ return; }}` without addressing the root cause.
- Rewriting entire functions when only specific lines need adjustment.
- Using invented APIs, functions, or configuration parameters.
- Keeping hardcoded constants instead of using proper runtime checks.
- Adding defensive code without understanding why the error occurs.  

"""
        if cmake_sys_mode:
            # Dedicated CMake-only prompt to avoid runtime-style overgeneration/hallucinated code blocks.
            system_prompt = f"""
You are a strict build-system fix generator for OpenAirInterface CMake/ASN.1 link errors.

You are STAGE-2 (fix generation). Stage-1 diagnosis has already run.
You must output ONLY valid JSON and ONLY CMake/config edits.

MANDATORY RULES:
- This is CMAKE MODE. Set `code_patches` to [] always.
- Do not output C/C++ function edits.
- Only suggest `config_patches` for exact retrieved files (`CMakeLists.txt` or `*.cmake`).
- If STAGE-1 validated diagnosis is present in the user context, use ONLY those `file_path` + `anchor_snippet` rows.
- Never emit a patch when `current_value` does not exactly match a validated anchor snippet.
- Build `current_value` from the given `Config Context` block in the prompt (copy exact nearby lines; do not invent long lists).
- Keep `current_value`/`new_value` concise (<= 20 lines each), single-edit focused.
- Prefer one high-confidence patch over many speculative patches.
- If multiple missing ASN.1 symbols map to the same `file_path` + `config_name` + `current_value` anchor, emit ONE merged `config_patch` with all new `.c` entries added in that single `new_value` block.
- Do not emit multiple patches that share identical `current_value` for targeted insertion (they will conflict at apply time).
- Use the real set/list identifier from the file context for `config_name` (example: `f1ap_source`), never invent names like `F1AP_ASN1_SRCS`.
- If the user context includes **📎 ASN.1 BUNDLE PRE-CHECK**, the needed `.c` may already appear in `f1ap-*.cmake` on disk but the link still fails (stale objects, wrong bundle picked, codegen). Then either output **grounded** `config_patches` from *SUSPECTED CONFIGURATIONS* (exact `current_value` from **Config Context**), **or** set `config_patches` to [] and fill **`investigation_steps`** with at least **3** concrete commands (e.g. `ninja clean`, rebuild `libasn1_f1ap`, full rebuild). **Never** leave both `config_patches` and `investigation_steps` empty for a linker error.

Expected issue type:
- Linker undefined reference like `asn_DEF_*` usually means missing ASN.1 generated `.c` in `.../ASN1/<bundle>.cmake`.

DEPLOYMENT CONTEXT:
{deployment_context_info}

Return JSON in this shape:
{{
  "suspected_functions": [],
  "suspected_configs": ["path/to/file.cmake"],
  "reason": "short root cause",
  "config_fix": "short fix summary",
  "code_patches": [],
  "config_patches": [
    {{
      "config_name": "f1ap_source",
      "file_path": "openair2/F1AP/MESSAGES/ASN1/f1ap-18.6.0.cmake",
      "patch_type": "targeted_insertion",
      "current_value": "existing nearby lines from file",
      "new_value": "patched nearby lines",
      "line_number": "anchor context",
      "relevance_score": 0.95,
      "description": "why this resolves undefined reference"
    }}
  ],
  "root_cause_analysis": "short technical explanation",
  "investigation_steps": ["step1", "step2"]
}}
The above shown json is just a simple example to show the json structure not the source truth. So you should not use this json as the source truth.
"""
        # Save system and user prompts to a readable file for manual verification
        try:
            timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
            prompts_filename = f"output/llm_prompts_{timestamp}.txt"
            
            with open(prompts_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LLM PROMPTS FOR MANUAL VERIFICATION\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Error Text: {error_text}\n")
                f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("SYSTEM PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(system_prompt)
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("USER PROMPT (CONTEXT)\n")
                f.write("=" * 80 + "\n")
                f.write(context)
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("PROMPT STATISTICS\n")
                f.write("=" * 80 + "\n")
                f.write(f"System prompt length: {len(system_prompt):,} characters\n")
                f.write(f"User prompt length: {len(context):,} characters\n")
                f.write(f"Total prompt length: {len(system_prompt) + len(context):,} characters\n")
                f.write(f"System prompt lines: {len(system_prompt.splitlines()):,} lines\n")
                f.write(f"User prompt lines: {len(context.splitlines()):,} lines\n")
            
            logger.info(f"✅ LLM prompts saved to: {prompts_filename}")
            print(f"📝 LLM prompts saved to: {prompts_filename}")
            
        except Exception as e:
            logger.warning(f"Could not save prompts to file: {e}")
        
        # print("*"*100,"the system prompt is :", system_prompt)

        if cmake_sys_mode:
            _COMPACT_JSON_RETRY = (
                "\n\n---\nCRITICAL FOLLOW-UP: Invalid or incomplete JSON. Reply with ONE json_object only. "
                "Set code_patches to [] and suspected_functions to []. "
                "At most 3 config_patches; keep current_value and new_value under 22 lines each. "
                "No markdown fences. Close all strings and brackets."
            )
        else:
            _COMPACT_JSON_RETRY = (
                "\n\n---\nCRITICAL FOLLOW-UP: Your previous reply was missing or invalid JSON. "
                "Respond with ONE complete JSON object only. "
                "Keep each code_patches[].original_code and code_patches[].patched_code at most 25 lines; "
                "use // ... truncated ... inside a line comment in the C snippet if needed. "
                "Close every string and bracket. Include all top-level keys even if arrays are empty."
            )

        try:
            fix_data = None
            response_content = ""
            last_finish: Optional[str] = None
            attempts_meta: List[Dict[str, Any]] = []
            raw_by_attempt: List[str] = []

            for attempt in range(2):
                user_msg = context if attempt == 0 else context + _COMPACT_JSON_RETRY
                response_content, last_finish = self._invoke_patch_completion(
                    system_prompt,
                    user_msg,
                    patch_max_tokens,
                    patch_use_json_object,
                )
                response_content = response_content or ""
                raw_by_attempt.append(response_content)
                attempts_meta.append(
                    {
                        "attempt": attempt + 1,
                        "finish_reason": last_finish,
                        "chars": len(response_content),
                    }
                )
                if last_finish == "length":
                    logger.warning(
                        "⚠️  Phase 3 LLM finish_reason=length (output may be truncated); max_tokens=%s",
                        patch_max_tokens,
                    )

                fix_data = _parse_fix_suggestion_json(response_content)
                if fix_data:
                    if attempt > 0:
                        logger.info("✅ Phase 3 JSON parsed after compact retry")
                    break
                if attempt == 0:
                    logger.warning(
                        "⚠️  Phase 3 JSON parse failed; retrying with compact-output instruction"
                    )

            self._phase3_llm_meta = {
                "max_output_tokens": patch_max_tokens,
                "json_mode": patch_use_json_object,
                "build_error_mode": build_error_mode,
                "finish_reason": last_finish,
                "parse_ok": bool(fix_data),
                "attempts": attempts_meta,
            }

            # Debug: Save raw response to a file for inspection (append to the prompts file)
            try:
                with open(prompts_filename, "a", encoding="utf-8") as f:
                    for i, raw in enumerate(raw_by_attempt):
                        f.write("=" * 80 + "\n")
                        f.write(f"LLM RESPONSE (attempt {i + 1})\n")
                        f.write("=" * 80 + "\n")
                        f.write(raw)
                        f.write("\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("RESPONSE STATISTICS\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Response length (final): {len(response_content):,} characters\n")
                    f.write(f"Response lines (final): {len(response_content.splitlines()):,} lines\n")
                    f.write("Temperature: 0.1\n")
                    f.write(f"Max tokens: {patch_max_tokens}\n")
                    f.write(f"Finish reason (final): {last_finish}\n")
                    f.write("Seed: 99\n")
                    f.write(f"JSON mode: {patch_use_json_object}\n")
                    f.write(f"Parse OK: {bool(fix_data)}\n")
                    f.write(f"Attempts: {json.dumps(attempts_meta)}\n")

                logger.info(f"✅ LLM response appended to: {prompts_filename}")

            except Exception as e:
                logger.warning(f"Could not append response to prompts file: {e}")
                try:
                    with open("debug_llm_response.txt", "w", encoding="utf-8") as f:
                        f.write("=== LLM RESPONSE (last attempt) ===\n")
                        f.write(response_content)
                except OSError:
                    pass

            logger.info("✅ Fix suggestions generated successfully")

            if not fix_data:
                logger.warning(
                    "⚠️  No valid JSON in Phase 3 response after retry"
                )

            if fix_data:
                # Parse code patches - handle both flat and nested structures
                code_patches = []
                # Check for nested structure first (with type safety)
                fix_suggestions = fix_data.get('fix_suggestions', {})
                
                # Handle case where fix_suggestions is an array of patches (not a dict)
                if isinstance(fix_suggestions, list):
                    logger.info(f"✅ fix_suggestions is an array with {len(fix_suggestions)} patches")
                    patches_data = fix_suggestions
                elif isinstance(fix_suggestions, dict):
                    # fix_suggestions is a dict, check for code_patches inside it
                    patches_data = fix_suggestions.get('code_patches', [])
                else:
                    logger.warning(f"⚠️ fix_suggestions is {type(fix_suggestions)}, treating as empty")
                    patches_data = []
                
                # Fallback to top-level code_patches if patches_data is empty
                if not patches_data:
                    patches_data = fix_data.get('code_patches', [])
                if cmake_sys_mode:
                    # Never accept code patches in CMake mode.
                    patches_data = []
                if patches_data:
                    for patch_data in patches_data:
                        code_patch = CodePatch(
                            function_name=patch_data.get('function_name', ''),
                            file_path=patch_data.get('file_path', ''),
                            patch_type=patch_data.get('patch_type', 'modification'),
                            original_code=patch_data.get('original_code', ''),
                            patched_code=patch_data.get('patched_code', ''),
                            line_numbers=patch_data.get('line_numbers', ''),
                            description=patch_data.get('description', '')
                        )
                        code_patches.append(code_patch)

                # Deterministically remove no-op/empty/duplicate patches.
                code_patches = self._filter_code_patches(code_patches)
                
                # Parse config patches - handle both flat and nested structures with mode-aware filtering
                config_patches = []
                _dc = getattr(self, "_current_deployment_context", None)
                # Handle fix_suggestions being either dict or list
                if isinstance(fix_suggestions, dict):
                    config_patches_data = fix_suggestions.get('config_patches', [])
                else:
                    config_patches_data = []
                # Fallback to top-level config_patches
                if not config_patches_data:
                    config_patches_data = fix_data.get('config_patches', [])
                for config_patch_data in config_patches_data:
                    # Mode-aware filtering: runtime configs always; CMake files only in CMake log kind.
                    file_path = config_patch_data.get('file_path', '').lower()
                    if self._is_supported_config_patch_file(file_path, _dc):
                        config_patch = ConfigPatch(
                            config_name=config_patch_data.get('config_name', ''),
                            file_path=config_patch_data.get('file_path', ''),
                            patch_type=config_patch_data.get('patch_type', 'set_value'),
                            current_value=config_patch_data.get('current_value', ''),
                            new_value=config_patch_data.get('new_value', ''),
                            line_number=config_patch_data.get('line_number', ''),
                            relevance_score=float(config_patch_data.get('relevance_score', 0.0)),
                            description=config_patch_data.get('description', '')
                        )
                        config_patches.append(config_patch)
                        logger.info(f"✅ Included config patch for supported file: {file_path}")
                    else:
                        logger.warning(f"🚫 FILTERED OUT config patch for unsupported file: {file_path}")
                        logger.warning(
                            "🚫 Allowed: runtime .conf files; CMakeLists.txt/*.cmake only when "
                            "log_error_kind is cmake (CMake/build-system mode)."
                        )

                # No hard rejection filter here; keep model-proposed patches and rely on prompt discipline.
                # Hard grounding gate: remove hallucinated or unanchored patches before building FixSuggestion.
                code_patches, config_patches = self._validate_patch_grounding(
                    code_patches=code_patches,
                    config_patches=config_patches,
                    deployment_context=getattr(self, "_current_deployment_context", None),
                )

                # Create FixSuggestion object from parsed JSON - handle both flat and nested structures
                # fix_suggestions already validated as dict above
                fix_suggestion = FixSuggestion(
                    suspected_functions=fix_data.get('suspected_functions', []),
                    suspected_configs=fix_data.get('suspected_configs', []),
                    reason=fix_data.get('reason', ''),
                    config_fix=str(fix_data.get('config_fix', '')),  # Convert to string if dict
                    code_patches=code_patches,
                    config_patches=config_patches,
                    root_cause_analysis=fix_data.get('root_cause_analysis', ''),
                    investigation_steps=fix_data.get('investigation_steps', []) or fix_suggestions.get('investigation_steps', []),
                    specification_context=getattr(self, '_current_specification_context', '')
                )
                
                return fix_suggestion
            else:
                # Safe fallback: never surface raw LLM blob (prevents hallucinated JSON/text from appearing as analysis).
                logger.warning("⚠️  Using safe empty fallback")
                safe_reason = (
                    "Phase 3 returned invalid JSON for CMake/build generation. "
                    "No patch suggestions returned to avoid hallucinated edits."
                    if cmake_sys_mode
                    else "Phase 3 returned invalid JSON. No patch suggestions generated."
                )
                return FixSuggestion(
                    suspected_functions=[],
                    suspected_configs=[],
                    reason=safe_reason,
                    config_fix="",
                    code_patches=[],
                    config_patches=[],
                    root_cause_analysis=safe_reason,
                    investigation_steps=[],
                    specification_context=getattr(self, '_current_specification_context', '')
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to generate fix suggestions: {e}")
            raise

    def generate_dependency_advice_only(
        self,
        error: str,
        deployment_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Environment / toolchain / missing package / version mismatch: full-log LLM analysis only.
        No code or CMake patches (returns empty patch lists).
        """
        dc = deployment_context if isinstance(deployment_context, dict) else {}
        full_log = (dc.get("full_dependency_log_text") or "").strip()
        if not full_log:
            full_log = (error or "").strip()
        user = (
            "You help resolve C/C++ development environment and dependency problems.\n\n"
            "The log may show: missing executables (cmake, ninja, gcc/clang), missing system "
            "packages or dev libraries, shared library load failures, tool version mismatches, or "
            "vcpkg/Conan resolution failures.\n\n"
            "Do **not** propose source-code patches or CMakeLists/source edits. The fix should be "
            "installation, PATH, toolchain selection, or package manager actions.\n\n"
            "**Operating system (mandatory):**\n"
            "- Infer the OS from the log when possible (examples: `/usr/`, `/home/`, `apt`, `dpkg`, "
            "`dnf`, `yum`, `pacman` → Linux; `C:\\\\`, `Program Files`, `cmd.exe`, `PowerShell`, "
            "`choco`, `winget`, `Visual Studio` paths → Windows; `/Users/`, `Darwin`, `brew`, "
            "`xcodebuild`, `.app` bundles → macOS).\n"
            "- Set `os_detected` to one of: `linux`, `windows`, `macos`, or `unknown`.\n"
            "- Set `os_detection_confidence` to one of: `high`, `medium`, `low`.\n"
            "- Set `os_detection_rationale` to one short sentence explaining what in the log supports "
            "the choice, or empty if `unknown`.\n"
            "- **If** `os_detected` is `linux`, `windows`, or `macos` **and** confidence is `high` "
            "or `medium`: give install/PATH/verification steps **only for that OS**. In "
            "`commands_by_os`, **only** populate the matching key (`linux`, `windows`, or `macos`); "
            "use empty arrays `[]` for the other two keys.\n"
            "- **If** `os_detected` is `unknown` **or** confidence is `low`: you **must** provide "
            "parallel guidance for **all three**: Linux, Windows, and macOS. Fill `commands_by_os` "
            "with non-empty `linux`, `windows`, and `macos` arrays (each a list of command strings) "
            "with equivalent fixes where possible.\n\n"
            "Return **JSON only** with exactly these keys:\n"
            '{"root_cause":"","recommended_actions":"","os_detected":"unknown",'
            '"os_detection_confidence":"low","os_detection_rationale":"",'
            '"commands_by_os":{"linux":[],"windows":[],"macos":[]},'
            '"commands_to_run":[],"verification_steps":[],"additional_notes":""}\n'
            "- `commands_to_run`: optional legacy flat list; prefer filling `commands_by_os`.\n"
            "- `verification_steps`: how to confirm the fix; scope them to the same OS rule as above.\n"
            "- `additional_notes`: distro nuances (e.g. Ubuntu vs Fedora) only when relevant.\n\n"
            f"## Extracted error line / summary\n{error or ''}\n\n"
            f"## Complete build/tool log\n```text\n{full_log}\n```\n"
        )
        system = (
            "You output valid JSON only. Be concise and actionable. No markdown outside the JSON. "
            "Support Linux (apt/dnf/pacman etc.), Windows (winget/choco/Visual Studio installers), "
            "and macOS (brew/Xcode CLT) as appropriate."
        )
        self._phase3_llm_meta = {
            "mode": "dependency_advice",
            "parse_ok": False,
            "model": self.model,
        }
        os_det = "unknown"
        linux_c: List[str] = []
        win_c: List[str] = []
        mac_c: List[str] = []
        try:
            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:120000]},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            raw = (response.choices[0].message.content or "").strip()
            data = _parse_fix_suggestion_json(raw)
            if not isinstance(data, dict):
                data = {}

            def _norm_cmd_list(val: Any) -> List[str]:
                if val is None:
                    return []
                if isinstance(val, list):
                    return [str(x).strip() for x in val if str(x).strip()]
                s = str(val).strip()
                return [s] if s else []

            root = (data.get("root_cause") or "").strip()
            actions = (data.get("recommended_actions") or "").strip()
            notes = (data.get("additional_notes") or "").strip()
            os_det = (data.get("os_detected") or "unknown").strip().lower()
            conf = (data.get("os_detection_confidence") or "low").strip().lower()
            rationale = (data.get("os_detection_rationale") or "").strip()
            if os_det not in ("linux", "windows", "macos", "unknown"):
                os_det = "unknown"
            if conf not in ("high", "medium", "low"):
                conf = "low"

            cbo = data.get("commands_by_os")
            if isinstance(cbo, dict):
                linux_c = _norm_cmd_list(cbo.get("linux"))
                win_c = _norm_cmd_list(cbo.get("windows"))
                mac_c = _norm_cmd_list(cbo.get("macos"))

            legacy_cmds = _norm_cmd_list(data.get("commands_to_run"))
            if not (linux_c or win_c or mac_c) and legacy_cmds:
                single_os = os_det in ("linux", "windows", "macos") and conf in ("high", "medium")
                if single_os:
                    if os_det == "linux":
                        linux_c = legacy_cmds
                    elif os_det == "windows":
                        win_c = legacy_cmds
                    else:
                        mac_c = legacy_cmds
                else:
                    linux_c = legacy_cmds

            verify = _norm_cmd_list(data.get("verification_steps"))

            reason_parts = [p for p in (root, actions) if p]
            reason = "\n\n".join(reason_parts) if reason_parts else (
                "See dependency / toolchain analysis (JSON parse produced limited fields)."
            )

            single_target_os = (
                os_det in ("linux", "windows", "macos")
                and conf in ("high", "medium")
            )
            show_all_os = (
                os_det == "unknown"
                or conf == "low"
                or not single_target_os
            )

            investigation_steps = []
            if os_det != "unknown" or rationale:
                hdr = f"OS inference: {os_det} (confidence: {conf})"
                if rationale:
                    hdr += f" — {rationale}"
                investigation_steps.append(hdr)

            def _append_cmds_block(title: str, lines: List[str], use_dollar: bool) -> None:
                if not lines:
                    return
                investigation_steps.append(title)
                prefix = "  $ " if use_dollar else "  > "
                for c in lines:
                    investigation_steps.append(prefix + c)

            if single_target_os:
                if os_det == "linux" and linux_c:
                    _append_cmds_block("Suggested commands (Linux):", linux_c, True)
                elif os_det == "windows" and win_c:
                    _append_cmds_block("Suggested commands (Windows):", win_c, False)
                elif os_det == "macos" and mac_c:
                    _append_cmds_block("Suggested commands (macOS):", mac_c, True)
                elif legacy_cmds and not (linux_c or win_c or mac_c):
                    _append_cmds_block("Suggested commands:", legacy_cmds, os_det != "windows")
            else:
                if linux_c or win_c or mac_c:
                    investigation_steps.append(
                        "Install / PATH fixes (all platforms — log did not pin OS reliably):"
                        if show_all_os
                        else "Install / PATH fixes:"
                    )
                    _append_cmds_block("Linux:", linux_c, True)
                    _append_cmds_block("Windows (cmd/PowerShell):", win_c, False)
                    _append_cmds_block("macOS:", mac_c, True)
                elif legacy_cmds:
                    _append_cmds_block("Suggested commands:", legacy_cmds, True)

            if verify:
                if investigation_steps:
                    investigation_steps.append("")
                investigation_steps.append("Verification:")
                investigation_steps.extend([f"  - {v}" for v in verify])
            if notes:
                investigation_steps.append("")
                investigation_steps.append(f"Notes: {notes}")

            self._phase3_llm_meta["parse_ok"] = True
            self._phase3_llm_meta["os_detected"] = os_det
            self._phase3_llm_meta["os_detection_confidence"] = conf
        except Exception as e:
            logger.error("❌ dependency advice LLM failed: %s", e)
            reason = (
                f"Dependency / toolchain advice could not be generated automatically ({e}). "
                "Review the full log and install missing tools or libraries indicated there."
            )
            root = reason
            actions = ""
            investigation_steps = [
                "OS inference: unknown (LLM unavailable) — apply the fix on the OS you use:",
                "Linux: install missing packages with apt/dnf/pacman (e.g. cmake, ninja, build-essential, "
                "or distro-specific *-dev packages).",
                "Windows: install CMake/ninja/VS Build Tools (Visual Studio Installer), or winget/choco; "
                "ensure tools are on PATH.",
                "macOS: xcode-select --install for CLI tools; brew install cmake ninja for Homebrew.",
            ]
            self._phase3_llm_meta["parse_ok"] = False
            self._phase3_llm_meta["error"] = str(e)

        fix_suggestion = FixSuggestion(
            suspected_functions=[],
            suspected_configs=[],
            reason=reason,
            config_fix=actions or "No CMake/config file changes — environment or package install only.",
            code_patches=[],
            config_patches=[],
            root_cause_analysis=root,
            investigation_steps=investigation_steps,
            specification_context="",
        )
        results = {
            "error_text": error,
            "fix_suggestion": {
                "suspected_functions": fix_suggestion.suspected_functions,
                "suspected_configs": fix_suggestion.suspected_configs,
                "reason": fix_suggestion.reason,
                "config_fix": fix_suggestion.config_fix,
                "code_patches": [],
                "config_patches": [],
                "root_cause_analysis": fix_suggestion.root_cause_analysis,
                "investigation_steps": fix_suggestion.investigation_steps,
                "specification_context": fix_suggestion.specification_context,
                "dependency_advice_only": True,
                "dependency_os_detected": os_det,
                "dependency_commands_by_os": {
                    "linux": linux_c,
                    "windows": win_c,
                    "macos": mac_c,
                },
            },
            "context_summary": {
                "candidate_functions_count": 0,
                "candidate_configs_count": 0,
                "call_graph_entries": 0,
                "pattern_matched": False,
            },
            "phase3_llm": getattr(self, "_phase3_llm_meta", None) or {},
            "dependency_advice_mode": True,
        }
        logger.info("✅ Dependency advice-only Phase 3 completed (no patches)")
        return results
    
    def process_fix_request(self, 
                           error: str,
                           candidate_functions: List[Dict],
                           candidate_configs: List[Dict],
                           call_graph_context: List[Dict] = None,
                           matched_pattern: Dict = None,
                           deployment_context: Dict = None) -> Dict[str, Any]:
        """
        Complete fix suggestion pipeline
        
        Args:
            error: Original error text
            candidate_functions: Suspected functions from Phase 2
            candidate_configs: Suspected configs from Phase 2
            call_graph_context: Call graph relationships
            matched_pattern: Matched error pattern
            deployment_context: Deployment context with log anchors and network params
            
        Returns:
            Complete fix suggestion results
        """
        logger.info("🚀 Starting Fix Suggestion Pipeline")
        logger.info("=" * 60)
        
        try:
            # Store deployment_context for use in _get_deployment_context
            self._current_deployment_context = deployment_context
            
            # Debug logging
            logger.info(f"📦 Deployment context received in process_fix_request:")
            logger.info(f"   Type: {type(deployment_context)}")
            logger.info(f"   Is None: {deployment_context is None}")
            logger.info(f"   Is truthy: {bool(deployment_context)}")
            if deployment_context and isinstance(deployment_context, dict):
                logger.info(f"   Keys: {list(deployment_context.keys())}")
                logger.info(f"   Length: {len(deployment_context)}")
            
            # Step 3.1: Assemble Context
            logger.info("🏗️ STEP 3.1 - CONTEXT ASSEMBLY")
            
            # Type validation before processing
            logger.info(f"📊 Input validation:")
            logger.info(f"   candidate_functions type: {type(candidate_functions)}, count: {len(candidate_functions) if isinstance(candidate_functions, list) else 'N/A'}")
            logger.info(f"   candidate_configs type: {type(candidate_configs)}, count: {len(candidate_configs) if isinstance(candidate_configs, list) else 'N/A'}")
            logger.info(f"   call_graph_context type: {type(call_graph_context)}, count: {len(call_graph_context) if call_graph_context and isinstance(call_graph_context, list) else 'N/A'}")
            
            # Ensure inputs are lists
            if not isinstance(candidate_functions, list):
                logger.error(f"❌ candidate_functions is not a list: {type(candidate_functions)}")
                candidate_functions = []
            if not isinstance(candidate_configs, list):
                logger.error(f"❌ candidate_configs is not a list: {type(candidate_configs)}")
                candidate_configs = []
            if call_graph_context is not None and not isinstance(call_graph_context, list):
                logger.error(f"❌ call_graph_context is not a list: {type(call_graph_context)}")
                call_graph_context = []

            _dc_dep = deployment_context if isinstance(deployment_context, dict) else {}
            if _dc_dep.get("dependency_advice_mode"):
                logger.info("📦 DEPENDENCY ADVICE MODE — skipping patch context assembly")
                return self.generate_dependency_advice_only(
                    error=error,
                    deployment_context=deployment_context,
                )
            
            context = self.assemble_context(
                error=error,
                candidate_functions=candidate_functions,
                candidate_configs=candidate_configs,
                call_graph_context=call_graph_context or [],
                matched_pattern=matched_pattern,
                deployment_context=deployment_context
            )
            
            # Step 3.2: Generate Patch
            logger.info("🧠 STEP 3.2 - PATCH GENERATION")
            fix_suggestion = self.generate_patch(
                context,
                error,
                candidate_configs=candidate_configs,
            )
            dc = getattr(self, "_current_deployment_context", None)
            cmake_mode = bool(isinstance(dc, dict) and dc.get("cmake_build_system_mode"))
            if cmake_mode:
                deterministic_cmake_patches = self._derive_cmake_asn1_fallback_patches(
                    error_text=error,
                    candidate_configs=candidate_configs,
                )
                if deterministic_cmake_patches:
                    # Strict mode for CMake: prefer deterministic, file-grounded patches over free-form LLM patch text.
                    fix_suggestion.config_patches = deterministic_cmake_patches
                    fix_suggestion.code_patches = []
                    if not fix_suggestion.config_fix:
                        fix_suggestion.config_fix = (
                            "Generated deterministic CMake patch from linker symbol + actual ASN1 cmake source list."
                        )
                    logger.info(
                        "✅ Using deterministic CMake patch set (%s patch(es)); overriding LLM config patches",
                        len(deterministic_cmake_patches),
                    )
                elif not fix_suggestion.config_patches:
                    logger.info(
                        "ℹ️ No deterministic CMake patch could be derived; keeping empty config patches"
                    )
            
            # Format results
            # Convert code patches to dict format for JSON serialization
            code_patches_dict = []
            for patch in fix_suggestion.code_patches:
                code_patches_dict.append({
                    "function_name": patch.function_name,
                    "file_path": self.openair_codebase_file_name + "/" + patch.file_path,
                    "patch_type": patch.patch_type,
                    "original_code": patch.original_code,
                    "patched_code": patch.patched_code,
                    "line_numbers": patch.line_numbers,
                    "description": patch.description
                })
            
            # Convert config patches to dict format for JSON serialization with FINAL VALIDATION
            config_patches_dict = []
            _dc_final = getattr(self, "_current_deployment_context", None)
            for patch in fix_suggestion.config_patches:
                # FINAL VALIDATION: Double-check that only supported files are included
                file_path = patch.file_path.lower()
                if self._is_supported_config_patch_file(file_path, _dc_final):
                    config_patches_dict.append({
                        "config_name": patch.config_name,
                        "file_path": self.openair_codebase_file_name + "/" + patch.file_path,
                        "patch_type": patch.patch_type,
                        "current_value": patch.current_value,
                        "new_value": patch.new_value,
                        "line_number": patch.line_number,
                        "relevance_score": patch.relevance_score,
                        "description": patch.description
                    })
                    logger.info(f"✅ FINAL VALIDATION: Included config patch for supported file: {file_path}")
                else:
                    logger.error(f"🚫 FINAL VALIDATION FAILED: Rejected config patch for unsupported file: {file_path}")
                    logger.error("🚫 This should not happen - config patch was not properly filtered earlier")

            # Last-resort deterministic fallback for CMake ASN.1 linker errors.
            # This runs after all filtering to guarantee at least one grounded patch when evidence is strong.
            dc = getattr(self, "_current_deployment_context", None)
            cmake_mode = bool(isinstance(dc, dict) and dc.get("cmake_build_system_mode"))
            if cmake_mode and not config_patches_dict:
                fallback_patches = self._derive_cmake_asn1_fallback_patches(
                    error_text=error,
                    candidate_configs=candidate_configs,
                )
                if fallback_patches:
                    logger.info(
                        "✅ Applying final deterministic CMake fallback patch(es): %s",
                        len(fallback_patches),
                    )
                    for patch in fallback_patches:
                        config_patches_dict.append({
                            "config_name": patch.config_name,
                            "file_path": self.openair_codebase_file_name + "/" + patch.file_path,
                            "patch_type": patch.patch_type,
                            "current_value": patch.current_value,
                            "new_value": patch.new_value,
                            "line_number": patch.line_number,
                            "relevance_score": patch.relevance_score,
                            "description": patch.description
                        })
                    # Keep top-level narrative aligned with emitted patches.
                    if not fix_suggestion.config_fix:
                        fix_suggestion.config_fix = (
                            "Deterministic fallback patch generated from linker symbol evidence."
                        )
            # If strict grounding produced no file patch in CMake linker mode, emit one
            # explicit advisory "config patch" row so the UI does not appear empty.
            if (
                cmake_mode
                and not config_patches_dict
                and self._is_linker_undefined_reference_error(error)
            ):
                advisory_file = "CMakeLists.txt"
                for c in candidate_configs or []:
                    if not isinstance(c, dict):
                        continue
                    fp = (c.get("file_path") or "").replace("\\", "/").strip()
                    low = fp.lower()
                    if low.endswith(".cmake") or low.endswith("/cmakelists.txt") or low == "cmakelists.txt":
                        advisory_file = fp
                        break
                config_patches_dict.append(
                    {
                        "config_name": "Build relink cleanup (advisory)",
                        "file_path": self.openair_codebase_file_name + "/" + advisory_file,
                        "patch_type": "manual_build_relink",
                        "current_value": "No deterministic in-file edit was safely grounded.",
                        "new_value": (
                            "Clean ASN.1 objects/static library and rebuild linker targets "
                            "(see investigation_steps)."
                        ),
                        "line_number": "N/A",
                        "relevance_score": 0.7,
                        "description": (
                            "Advisory build-system action: linker unresolved ASN.1 symbols detected. "
                            "No safe auto-edit found in local workspace, so apply clean/rebuild steps."
                        ),
                    }
                )
                if not fix_suggestion.config_fix:
                    fix_suggestion.config_fix = (
                        "No safe in-file CMake edit was grounded; use the build relink cleanup steps."
                    )
            
            phase3_llm = getattr(self, "_phase3_llm_meta", None)
            if not isinstance(phase3_llm, dict):
                phase3_llm = {}

            results = {
                "error_text": error,
                "fix_suggestion": {
                    "suspected_functions": fix_suggestion.suspected_functions,
                    "suspected_configs": fix_suggestion.suspected_configs,
                    "reason": fix_suggestion.reason,
                    "config_fix": fix_suggestion.config_fix,
                    "code_patches": code_patches_dict,
                    "config_patches": config_patches_dict,
                    "root_cause_analysis": fix_suggestion.root_cause_analysis,
                    "investigation_steps": fix_suggestion.investigation_steps,
                    "specification_context": fix_suggestion.specification_context
                },
                "context_summary": {
                    "candidate_functions_count": len(candidate_functions),
                    "candidate_configs_count": len(candidate_configs),
                    "call_graph_entries": len(call_graph_context or []),
                    "pattern_matched": matched_pattern is not None
                },
                "phase3_llm": phase3_llm,
            }
            phase3_cmake_diagnosis = getattr(self, "_phase3_cmake_diagnosis", None)
            if isinstance(phase3_cmake_diagnosis, dict) and phase3_cmake_diagnosis:
                results["phase3_cmake_diagnosis"] = phase3_cmake_diagnosis
            if phase3_llm and not phase3_llm.get("parse_ok", True):
                results["phase3_parse_failed"] = True
            
            logger.info("✅ Fix Suggestion Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fix Suggestion Pipeline failed: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "error_text": error,
                "pipeline_error": str(e),
                "fix_suggestion": {
                    "suspected_functions": [],
                    "suspected_configs": [],
                    "reason": f"Pipeline processing failed: {e}",
                    "config_fix": "",
                    "code_patches": [],
                    "config_patches": [],
                    "root_cause_analysis": "",
                    "investigation_steps": []
                }
            }

def load_call_graph_context(candidate_functions: List[Dict], function_calls_file: str = "database/function_calls.json") -> List[Dict]:
    """
    Load relevant call graph context for candidate functions, including downstream functions
    
    Args:
        candidate_functions: List of suspected functions
        function_calls_file: Path to function_calls.json
        
    Returns:
        Relevant call graph entries (including entries for candidate functions and their downstream calls)
    """
    try:
        with open(function_calls_file, 'r', encoding='utf-8') as f:
            all_function_calls = json.load(f)
        
        # Extract candidate function names (with type safety)
        candidate_names = {func.get('function_name', '') for func in candidate_functions if isinstance(func, dict)}
        
        # Collect downstream function names (functions called by candidates)
        downstream_names = set()
        
        # Find relevant call graph entries
        relevant_entries = []
        for entry in all_function_calls:
            # Type safety: skip non-dict entries
            if not isinstance(entry, dict):
                continue
                
            func_name = entry.get('function', '')
            # Include entries for candidate functions
            if func_name in candidate_names:
                relevant_entries.append(entry)
                # Collect downstream functions (functions called by this candidate)
                calls = entry.get('calls', [])
                if isinstance(calls, list):
                    downstream_names.update(calls)
        
        # Also include entries for downstream functions (functions called by candidates)
        for entry in all_function_calls:
            if not isinstance(entry, dict):
                continue
            func_name = entry.get('function', '')
            if func_name in downstream_names and entry not in relevant_entries:
                relevant_entries.append(entry)
        
        logger.info(f"📊 Loaded {len(relevant_entries)} relevant call graph entries (including {len(downstream_names)} downstream functions)")
        return relevant_entries
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to load call graph context: {e}")
        return []

def main():
    """Demo the fix suggestion pipeline"""
    print("🔧 Fix Suggestion Pipeline Demo")
    print("=" * 50)
    
    # Example usage
    try:
        pipeline = FixSuggestionPipeline()
        
        # Example error and candidates (from Phase 2 output)
        example_error = "No AMF associated to gNB"
        
        example_functions = [
            {
                "function_name": "select_amf",
                "file_path": "openair3/NGAP/ngap_gNB_nas_procedures.c",
                "relevance_score": 0.90,
                "reason": "This function is directly related to selecting an AMF for the gNB",
                "code_snippet": "static ngap_gNB_amf_data_t *select_amf(ngap_gNB_instance_t *instance_p, const ngap_nas_first_req_t *msg)\n{\n  ngap_gNB_amf_data_t *amf = NULL;\n  // Select AMF logic here\n  return amf;\n}"
            }
        ]
        
        example_configs = [
            {
                "param_name": "GNB_IPV4_ADDRESS_FOR_NG_AMF",
                "param_value": "192.168.71.150/24",
                "file_path": "ci-scripts/conf_files/gnb-cucp.sa.f1.conf",
                "relevance_score": 0.70,
                "reason": "This configuration parameter specifies the IPv4 address for the NG AMF"
            }
        ]
        
        # Load call graph context
        call_graph = load_call_graph_context(example_functions)
        
        # Process fix request
        results = pipeline.process_fix_request(
            error=example_error,
            candidate_functions=example_functions,
            candidate_configs=example_configs,
            call_graph_context=call_graph
        )
        
        # Display results
        print("\n📊 FIX SUGGESTION RESULTS:")
        print("=" * 40)
        
        fix = results.get('fix_suggestion', {})
        print(f"🔧 Suspected Functions: {fix.get('suspected_functions', [])}")
        print(f"⚙️  Suspected Configs: {fix.get('suspected_configs', [])}")
        print(f"💡 Reason: {fix.get('reason', 'No reason provided')[:200]}...")
        
        if fix.get('config_fix'):
            print(f"\n🔧 Config Fix:")
            print(fix['config_fix'][:300] + "..." if len(fix['config_fix']) > 300 else fix['config_fix'])
        
        # Display code patches in new format
        # Handle both flat and nested structures
        fix_suggestions = fix.get('fix_suggestions', {})
        code_patches = fix.get('code_patches', []) or fix_suggestions.get('code_patches', [])
        if code_patches:
            print(f"\n💻 Code Patches ({len(code_patches)}):")
            for i, patch in enumerate(code_patches, 1):
                print(f"   {i}. {patch.get('function_name', 'Unknown')} ({patch.get('patch_type', 'modification')})")
                print(f"      File: {patch.get('file_path', 'Unknown')}")
                print(f"      Lines: {patch.get('line_numbers', 'Unknown')}")
                print(f"      Description: {patch.get('description', 'No description')}")
                if patch.get('patched_code'):
                    code_preview = patch['patched_code'][:200] + "..." if len(patch['patched_code']) > 200 else patch['patched_code']
                    print(f"      Code: {code_preview}")
        elif fix.get('code_patch'):  # Fallback for old format
            print(f"\n💻 Code Patch (Legacy):")
            print(fix['code_patch'][:300] + "..." if len(fix['code_patch']) > 300 else fix['code_patch'])
        
        # Display config patches
        config_patches = fix.get('config_patches', []) or fix_suggestions.get('config_patches', [])
        if config_patches:
            print(f"\n⚙️ Config Patches ({len(config_patches)}):")
            for i, patch in enumerate(config_patches, 1):
                print(f"   {i}. {patch.get('config_name', 'Unknown')} ({patch.get('patch_type', 'set_value')})")
                print(f"      File: {patch.get('file_path', 'Unknown')}")
                print(f"      Current: {patch.get('current_value', 'Unknown')}")
                print(f"      New: {patch.get('new_value', 'Unknown')}")
                print(f"      Description: {patch.get('description', 'No description')}")
        
        # Display investigation steps
        investigation_steps = fix.get('investigation_steps', []) or fix_suggestions.get('investigation_steps', [])
        if investigation_steps:
            print(f"\n📋 Investigation Steps ({len(investigation_steps)}):")
            for i, step in enumerate(investigation_steps, 1):
                print(f"   {i}. {step}")
        
        # Save results
        with open('output/fix_suggestions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to output/fix_suggestions.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
