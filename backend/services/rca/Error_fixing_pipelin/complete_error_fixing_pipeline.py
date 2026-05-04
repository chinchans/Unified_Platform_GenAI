#!/usr/bin/env python3
"""
Complete Error Fixing Pipeline with Context-Aware Features

This script integrates:
- Log parsing and deployment context extraction (Step 2.1)
- Missing config resolution (Step 2.2) 
- Context-aware candidate retrieval (Step 2.4)
- Error analysis (Phase 2)
- Fix suggestions (Phase 3)

Usage:
    python complete_error_fixing_pipeline.py
    
    Update the error_message and log_file_path variables in main() as needed.

Author: AI Assistant
"""

import json
import logging
import os
import pickle
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import faiss
from .error_handling_pipeline import ErrorHandlingPipeline
from .fix_suggestion_pipeline import FixSuggestionPipeline, load_call_graph_context
from .parse_log_context import LogContextParser
from .segmentation_fault_analyzer import SegmentationFaultAnalyzer
from .scripts.extract_functions_improved import ImprovedFunctionExtractor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase 3 chunk size when compiler errors were parsed and narrow Phase 2 is active.
# Smaller batches → shorter JSON per LLM call, fewer truncated responses.
_COMPILER_GROUND_TRUTH_PHASE3_BATCH_SIZE = 6


class CompleteErrorFixingPipeline:
    """ complete error fixing pipeline with context-aware features"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the pipeline.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🚀 Initializing Complete Error Fixing Pipeline...")
        
        load_dotenv()
        
        # Store the codebase folder name
        self.openair_codebase_file_name = openair_codebase_file_name
        
        # Initialize all components with the dynamic folder name
        self.phase2_pipeline = ErrorHandlingPipeline(openair_codebase_file_name=openair_codebase_file_name)
        self.phase3_pipeline = FixSuggestionPipeline(openair_codebase_file_name=openair_codebase_file_name)
        self.log_parser = LogContextParser(openair_codebase_file_name=openair_codebase_file_name)
        self.crash_analyzer = SegmentationFaultAnalyzer(openair_codebase_file_name=openair_codebase_file_name)

        # Optional Knowledge Graph (KG) enhancement for Phase 2 function expansion.
        self._kg_graph = None
        self._kg_func_nodes_by_name = None
        self._kg_functions_db_cache = None
        self._kg_functions_db_index = None

        # One-time baseline metadata bootstrap for future incremental indexing.
        # This only sets last_indexed_commit if metadata is missing/incomplete.
        self._bootstrap_index_state_commit()
        
        logger.info("✅ Complete Error Fixing Pipeline initialized")

    def _get_index_state_path(self) -> str:
        """Return absolute path to incremental index state metadata file."""
        return os.path.join(os.path.dirname(__file__), "faiss_indices", "index_state.json")

    def _resolve_code_repo_path(self) -> str:
        """
        Resolve repository path for the codebase we index.
        Prefers Error_fixing_pipelin/<openair_codebase_file_name>.
        """
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), self.openair_codebase_file_name)
        )

    def _get_head_commit(self, repo_path: str) -> str:
        """Get current HEAD commit hash for the given repository path."""
        if not repo_path or not os.path.isdir(repo_path):
            return ""
        git_dir = os.path.join(repo_path, ".git")
        if not os.path.exists(git_dir):
            return ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=8,
                check=False
            )
            if result.returncode != 0:
                return ""
            return (result.stdout or "").strip()
        except Exception:
            return ""

    def _bootstrap_index_state_commit(self) -> None:
        """
        One-time bootstrap:
        If index_state.json has no last_indexed_commit, store current HEAD commit.
        """
        state_path = self._get_index_state_path()
        state_dir = os.path.dirname(state_path)
        os.makedirs(state_dir, exist_ok=True)

        state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded
            except Exception as e:
                logger.warning(f"⚠️ Failed to read index state metadata; recreating: {e}")
                state = {}

        if state.get("last_indexed_commit"):
            logger.info("ℹ️ Index state already has last_indexed_commit; bootstrap skipped")
            return

        repo_path = self._resolve_code_repo_path()
        head_commit = self._get_head_commit(repo_path)
        if not head_commit:
            logger.warning(
                "⚠️ Could not resolve HEAD commit for one-time index metadata bootstrap"
            )
            return

        state["last_indexed_commit"] = head_commit
        state["metadata_version"] = state.get("metadata_version", 1)
        state["baseline_initialized_at"] = datetime.now().isoformat()
        state["openair_codebase_file_name"] = self.openair_codebase_file_name
        state["repo_path"] = repo_path

        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info(
                "✅ One-time index baseline commit stored in metadata: %s",
                state.get("last_indexed_commit", "")[:12]
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to write index state metadata: {e}")

    @staticmethod
    def _normalize_rel_path(path_str: str) -> str:
        """Normalize path into a repo-relative canonical form."""
        if not path_str:
            return ""
        p = str(path_str).replace("\\", "/").strip().lower()
        return p.lstrip("./")

    def _normalize_function_db_path_to_repo_rel(self, db_path: str) -> str:
        """
        Convert DB path like openairinterface5g-develop\\a\\b.c to repo-relative a/b.c.
        """
        p = self._normalize_rel_path(db_path)
        prefix = self._normalize_rel_path(self.openair_codebase_file_name) + "/"
        if p.startswith(prefix):
            return p[len(prefix):]
        return p

    def _repo_rel_to_db_path(self, repo_rel: str) -> str:
        """Convert repo-relative path to functions.json style path."""
        repo_rel = str(repo_rel).replace("/", "\\").replace("\\\\", "\\").strip("\\")
        return f"{self.openair_codebase_file_name}\\{repo_rel}"

    def _git_name_only(self, repo_path: str, args: list) -> list:
        """Run git name-only command and return normalized file list."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=12,
                check=False
            )
            if result.returncode != 0:
                return []
            files = []
            for line in (result.stdout or "").splitlines():
                rel = self._normalize_rel_path(line)
                if rel:
                    files.append(rel)
            return files
        except Exception:
            return []

    def _collect_changed_source_files(self, repo_path: str, last_commit: str, head_commit: str) -> list:
        """Collect changed C/C++ files from commit diff + staged + unstaged changes."""
        changed = set()
        if last_commit and head_commit and last_commit != head_commit:
            for f in self._git_name_only(repo_path, ["diff", "--name-only", f"{last_commit}..{head_commit}"]):
                changed.add(f)
        for f in self._git_name_only(repo_path, ["diff", "--name-only"]):
            changed.add(f)
        for f in self._git_name_only(repo_path, ["diff", "--name-only", "--cached"]):
            changed.add(f)

        exts = (".c", ".h", ".cpp", ".cc", ".cxx")
        return sorted([f for f in changed if f.endswith(exts)])

    @staticmethod
    def _function_text_for_embedding(func_obj: dict) -> str:
        """Embedding text format aligned with index build script."""
        function_name = func_obj.get("function_name", "")
        code_body = (func_obj.get("code_body", "") or "").strip()
        file_path = func_obj.get("file_path", "")
        if len(code_body) > 2000:
            code_body = code_body[:2000] + "..."
        return f"{function_name}\n{code_body}\n{file_path}"

    def _load_functions_mapping_list(self, mapping_path: str) -> list:
        """Load mapping.json and return ordered list by vector id."""
        if not os.path.exists(mapping_path):
            return []
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, list):
                return mapping
            if isinstance(mapping, dict):
                out = []
                for k in sorted(mapping.keys(), key=lambda x: int(x)):
                    out.append(mapping[k])
                return out
        except Exception as e:
            logger.warning(f"⚠️ Failed to load functions mapping list: {e}")
        return []

    @staticmethod
    def _save_mapping_from_list(data_list: list, mapping_path: str) -> None:
        mapping = {str(i): obj for i, obj in enumerate(data_list)}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    def _incremental_sync_functions_and_embeddings(self) -> dict:
        """
        Incrementally sync function DB + function embeddings/index based on git diff.
        Uses last_indexed_commit from index_state.json as baseline.
        """
        state_path = self._get_index_state_path()
        state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}

        repo_path = self._resolve_code_repo_path()
        head_commit = self._get_head_commit(repo_path)
        last_commit = state.get("last_indexed_commit", "")
        if not head_commit:
            logger.warning("⚠️ Skipping incremental sync: unable to resolve repo HEAD")
            return {"updated": False, "reason": "no_head"}

        changed_files = self._collect_changed_source_files(repo_path, last_commit, head_commit)
        if not changed_files:
            state["last_indexed_commit"] = head_commit
            state["last_sync_at"] = datetime.now().isoformat()
            state["last_sync_changed_files_count"] = 0
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info("ℹ️ Incremental sync skipped: no changed source files detected")
            return {"updated": False, "reason": "no_changes"}

        functions_db_path = os.path.join(os.path.dirname(__file__), "database", "functions.json")
        mapping_path = os.path.join(os.path.dirname(__file__), "faiss_indices", "functions_mapping.json")
        index_path = os.path.join(os.path.dirname(__file__), "faiss_indices", "functions_index.faiss")
        embeddings_path = os.path.join(os.path.dirname(__file__), "faiss_indices", "functions_embeddings.npy")

        if not os.path.exists(functions_db_path):
            logger.warning("⚠️ Skipping incremental sync: functions.json not found")
            return {"updated": False, "reason": "missing_functions_db"}

        with open(functions_db_path, "r", encoding="utf-8") as f:
            functions_db = json.load(f)
        if not isinstance(functions_db, list):
            logger.warning("⚠️ Skipping incremental sync: invalid functions.json format")
            return {"updated": False, "reason": "invalid_functions_db"}

        changed_set = {self._normalize_rel_path(p) for p in changed_files}

        # Remove stale DB entries for changed files.
        kept_db = []
        for entry in functions_db:
            if not isinstance(entry, dict):
                continue
            rel = self._normalize_function_db_path_to_repo_rel(entry.get("file_path", ""))
            if rel in changed_set:
                continue
            kept_db.append(entry)

        # Re-extract only changed files.
        extractor = ImprovedFunctionExtractor(repo_path)
        new_entries = []
        for rel_path in sorted(changed_set):
            abs_path = os.path.join(repo_path, rel_path)
            if not os.path.exists(abs_path):
                continue  # deleted file
            parsed = extractor._parse_file(Path(rel_path))
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                item["file_path"] = self._repo_rel_to_db_path(rel_path)
                new_entries.append(item)

        updated_db = kept_db + new_entries
        with open(functions_db_path, "w", encoding="utf-8") as f:
            json.dump(updated_db, f, indent=2, ensure_ascii=False)

        # Update functions mapping + index incrementally when possible.
        old_mapping_list = self._load_functions_mapping_list(mapping_path)
        if not old_mapping_list:
            old_mapping_list = list(functions_db)

        # Keep old mapping entries from unchanged files.
        keep_indices = []
        new_mapping_list = []
        for i, entry in enumerate(old_mapping_list):
            if not isinstance(entry, dict):
                continue
            rel = self._normalize_function_db_path_to_repo_rel(entry.get("file_path", ""))
            if rel in changed_set:
                continue
            keep_indices.append(i)
            new_mapping_list.append(entry)
        new_mapping_list.extend(new_entries)

        model = getattr(self.phase2_pipeline, "embedding_model", None)
        if model is None:
            logger.warning("⚠️ Embedding model unavailable, skipping function index refresh")
            self._save_mapping_from_list(new_mapping_list, mapping_path)
        else:
            base_embeddings = None

            if os.path.exists(embeddings_path):
                try:
                    base_embeddings = np.load(embeddings_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cached function embeddings: {e}")

            if base_embeddings is None and os.path.exists(index_path):
                try:
                    old_index = faiss.read_index(index_path)
                    if old_index.ntotal == len(old_mapping_list):
                        dim = old_index.d
                        base_embeddings = np.zeros((old_index.ntotal, dim), dtype=np.float32)
                        old_index.reconstruct_n(0, old_index.ntotal, base_embeddings)
                except Exception as e:
                    logger.warning(f"⚠️ Could not reconstruct embeddings from FAISS index: {e}")

            if base_embeddings is not None and len(base_embeddings) == len(old_mapping_list):
                kept_embeddings = base_embeddings[keep_indices] if keep_indices else np.zeros((0, base_embeddings.shape[1]), dtype=np.float32)
                if new_entries:
                    texts = [self._function_text_for_embedding(x) for x in new_entries]
                    new_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    new_emb = np.asarray(new_emb, dtype=np.float32)
                    final_embeddings = np.vstack([kept_embeddings, new_emb]) if len(kept_embeddings) else new_emb
                else:
                    final_embeddings = kept_embeddings
            else:
                # One-time fallback: re-embed all mapping entries if baseline embeddings are unavailable.
                texts = [self._function_text_for_embedding(x) for x in new_mapping_list]
                final_embeddings = np.asarray(
                    model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32
                ) if texts else np.zeros((0, 768), dtype=np.float32)

            self._save_mapping_from_list(new_mapping_list, mapping_path)
            np.save(embeddings_path, final_embeddings)

            if len(final_embeddings) > 0:
                idx = faiss.IndexFlatIP(final_embeddings.shape[1])
                idx.add(final_embeddings.astype(np.float32))
                faiss.write_index(idx, index_path)
                logger.info(
                    "✅ Incremental function index refreshed: total=%d changed_files=%d new_entries=%d",
                    idx.ntotal,
                    len(changed_set),
                    len(new_entries),
                )
            else:
                logger.warning("⚠️ No function embeddings available after incremental sync")

        state["last_indexed_commit"] = head_commit
        state["last_sync_at"] = datetime.now().isoformat()
        state["last_sync_changed_files_count"] = len(changed_set)
        state["last_sync_new_functions"] = len(new_entries)
        state["last_sync_changed_files"] = sorted(changed_set)[:200]
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        return {"updated": True, "changed_files": len(changed_set), "new_entries": len(new_entries)}

    def _sync_git_commit_embeddings_at_rca_start(self) -> None:
        """Refresh git-commit embedding store once per analysis if repo HEAD moved."""
        try:
            from .ui_integration import sync_git_commit_embeddings_at_rca_start

            repo_path = self._resolve_code_repo_path()
            if not repo_path or not os.path.isdir(repo_path):
                return
            ok, msg = sync_git_commit_embeddings_at_rca_start(
                self.openair_codebase_file_name,
                code_dir=repo_path,
                progress_callback=None,
            )
            logger.info("Git commit embedding sync at RCA start: %s — %s", ok, msg)
        except Exception as e:
            logger.warning("Git commit embedding sync at RCA start failed: %s", e)
    
    def _load_functions_db_for_expansion(self) -> list:
        """
        Load functions database for retrieving code_snippet/function definitions.
        Cached to avoid repeated JSON reads during a single analysis.
        """
        if self._kg_functions_db_cache is not None:
            return self._kg_functions_db_cache

        functions_db = []
        try:
            if os.path.exists("database/functions.json"):
                with open("database/functions.json", 'r', encoding='utf-8') as f:
                    functions_db = json.load(f)
            elif os.path.exists("faiss_indices/functions_mapping.json"):
                with open("faiss_indices/functions_mapping.json", 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    if isinstance(mapping, dict):
                        functions_db = list(mapping.values())
                    else:
                        functions_db = mapping
        except Exception as e:
            logger.warning(f"⚠️ Failed to load functions database for KG expansion: {e}")
            functions_db = []

        # Build index by function_name -> list of entries (for quick lookup)
        index = {}
        for func_data in functions_db:
            if not isinstance(func_data, dict):
                continue
            name = func_data.get("function_name")
            if not name:
                continue
            index.setdefault(name, []).append(func_data)

        self._kg_functions_db_cache = functions_db
        self._kg_functions_db_index = index
        return functions_db

    def _load_knowledge_graph(self):
        """Load the KG from disk and build a name->nodes index."""
        if self._kg_graph is not None:
            return

        kg_path = os.path.join(
            os.path.dirname(__file__),
            "resources",
            "embeddings",
            "knowledge_graph.pkl",
        )

        if not os.path.exists(kg_path):
            logger.warning(f"⚠️ KG file not found, skipping KG expansion: {kg_path}")
            self._kg_graph = None
            self._kg_func_nodes_by_name = {}
            return

        try:
            with open(kg_path, "rb") as f:
                self._kg_graph = pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load KG pickle, skipping KG expansion: {e}")
            self._kg_graph = None
            self._kg_func_nodes_by_name = {}
            return

        # Build index for mapping candidate function_name -> KG function nodes.
        func_nodes_by_name = {}
        try:
            for node, data in self._kg_graph.nodes(data=True):
                if not isinstance(data, dict):
                    continue
                if data.get("entity_type") != "function":
                    continue
                fn = data.get("name")
                if not fn:
                    continue
                func_nodes_by_name.setdefault(fn, set()).add(node)
        except Exception as e:
            logger.warning(f"⚠️ Failed to index KG functions by name: {e}")
            func_nodes_by_name = {}

        self._kg_func_nodes_by_name = func_nodes_by_name
        logger.info(
            f"✅ KG loaded for Phase-2 expansion: nodes={self._kg_graph.number_of_nodes()}, "
            f"edges={self._kg_graph.number_of_edges()}"
        )

    @staticmethod
    def _normalize_path_for_match(path_str: str) -> str:
        if not path_str:
            return ""
        return str(path_str).replace("/", "\\").strip().lower()

    def _kg_expand_functions(self, candidate_functions: list, max_iters: int = 2, max_new: int = 25) -> list:
        """
        Enhance Phase-2 by adding more suspected functions using the KG.

        Expansion logic (non-replacing):
        - function_calls: add called functions
        - function_uses_struct/function_uses_variable: add other functions that use the same structs/variables

        This runs AFTER your existing call-graph downstream expansion, and only appends new functions.
        """
        self._load_knowledge_graph()
        if not self._kg_graph:
            return []
        if not isinstance(candidate_functions, list) or not candidate_functions:
            return []

        functions_db = self._load_functions_db_for_expansion()
        functions_index = self._kg_functions_db_index or {}

        # Seed KG function nodes from current candidate_functions
        seed_func_nodes = set()
        for func in candidate_functions:
            if not isinstance(func, dict):
                continue
            fn_name = func.get("function_name", "")
            if not fn_name:
                continue

            kg_nodes = self._kg_func_nodes_by_name.get(fn_name, set())
            if not kg_nodes:
                continue

            # If file_path is known, filter KG nodes to match file.
            cfile = func.get("file_path", "")
            if cfile:
                cfile_norm = self._normalize_path_for_match(cfile)
                filtered = set()
                for node in kg_nodes:
                    ndata = self._kg_graph.nodes[node] if hasattr(self._kg_graph, "nodes") else {}
                    kfile = self._normalize_path_for_match(ndata.get("file_path", ""))
                    if kfile and (kfile == cfile_norm or kfile.endswith(cfile_norm) or cfile_norm.endswith(kfile)):
                        filtered.add(node)
                if filtered:
                    kg_nodes = filtered

            seed_func_nodes |= set(kg_nodes)

        if not seed_func_nodes:
            return []

        discovered_functions = set(seed_func_nodes)
        current_functions = set(seed_func_nodes)
        struct_var_nodes = set()

        # Track best evidence for each discovered function node.
        # func_node_id -> dict(function_name, file_path, relevance_score, source, reason)
        evidence = {}

        def _node_to_candidate(node_id: str, relevance_score: float, source: str, reason: str):
            ndata = self._kg_graph.nodes[node_id] if hasattr(self._kg_graph, "nodes") else {}
            fn_name = ndata.get("name") or node_id
            file_path = ndata.get("file_path", "Unknown")

            # Try to attach a code_snippet from the functions db.
            code_snippet = ""
            if fn_name and fn_name in functions_index:
                # Prefer file_path match if possible.
                want_path_norm = self._normalize_path_for_match(file_path)
                best = None
                for fd in functions_index.get(fn_name, []):
                    fd_path = fd.get("file_path", "")
                    if not fd_path:
                        continue
                    fd_path_norm = self._normalize_path_for_match(fd_path)
                    if want_path_norm and (fd_path_norm == want_path_norm or fd_path_norm.endswith(want_path_norm) or want_path_norm.endswith(fd_path_norm)):
                        best = fd
                        break
                    if best is None:
                        best = fd
                if best:
                    code_snippet = best.get("code_snippet", best.get("code_body", "")) or ""

            return {
                "function_name": fn_name if isinstance(fn_name, str) else str(fn_name),
                "file_path": file_path,
                "relevance_score": float(relevance_score),
                "source": source,
                "code_snippet": code_snippet,
                "reason": reason,
            }

        for _depth in range(max_iters):
            next_functions = set()
            next_struct_vars = set()

            # From functions, collect:
            # - called functions
            # - structs/variables used
            for func_node in current_functions:
                for _u, v, edata in self._kg_graph.out_edges(func_node, data=True):
                    if not isinstance(edata, dict):
                        continue
                    rel = edata.get("relationship_type")
                    vtype = self._kg_graph.nodes[v].get("entity_type") if v in self._kg_graph.nodes else None

                    if rel == "function_calls" and vtype == "function":
                        if v not in discovered_functions:
                            next_functions.add(v)
                            discovered_functions.add(v)
                            if v not in evidence:
                                parent_name = self._kg_graph.nodes[func_node].get("name", "")
                                evidence[v] = _node_to_candidate(
                                    v,
                                    relevance_score=0.55,
                                    source="kg_expansion_function_calls",
                                    reason=f"KG evidence: function calls '{parent_name}' -> called function expands execution flow."
                                )

                    elif rel in ("function_uses_struct", "function_uses_variable") and vtype in ("struct", "variable"):
                        next_struct_vars.add(v)

            struct_var_nodes |= next_struct_vars

            # From structs/variables, add other functions that use them (incoming edges into struct/variable)
            for sv_node in struct_var_nodes:
                for u, _v, edata in self._kg_graph.in_edges(sv_node, data=True):
                    if not isinstance(edata, dict):
                        continue
                    rel = edata.get("relationship_type")
                    utype = self._kg_graph.nodes[u].get("entity_type") if u in self._kg_graph.nodes else None
                    if utype != "function":
                        continue
                    if rel not in ("function_uses_struct", "function_uses_variable"):
                        continue
                    if u in discovered_functions:
                        continue

                    next_functions.add(u)
                    discovered_functions.add(u)
                    if u not in evidence:
                        source_name = self._kg_graph.nodes[u].get("name", "")
                        evidence[u] = _node_to_candidate(
                            u,
                            relevance_score=0.45,
                            source="kg_expansion_shared_struct_variable",
                            reason=f"KG evidence: shares struct/variable usage with candidate-related code (struct/variable -> other using functions)."
                        )

            if len(evidence) >= max_new:
                break
            current_functions = next_functions

        # Convert evidence nodes into candidate dicts, avoiding duplicates by function_name.
        existing_names = {f.get("function_name", "") for f in candidate_functions if isinstance(f, dict)}

        new_candidates = []
        for node_id, cand in evidence.items():
            if not isinstance(cand, dict):
                continue
            fn = cand.get("function_name", "")
            if not fn or fn in existing_names:
                continue
            new_candidates.append(cand)
            if len(new_candidates) >= max_new:
                break

        # Sort by relevance_score descending so more relevant functions appear earlier to Phase 3.
        new_candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return new_candidates

    def _extract_function_name(self, func):
        """Extract function name from nested structure."""
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            return inner_candidate.get("function_name", inner_candidate.get("name", "Unknown"))
        return "Unknown"
    
    def _extract_file_path(self, func):
        """Extract file path from nested structure."""
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            return inner_candidate.get("file_path", "Unknown")
        return "Unknown"
    
    def _extract_code_snippet(self, func):
        """Extract code snippet from nested structure."""
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            return inner_candidate.get("code_snippet", inner_candidate.get("code_body", ""))
        return ""
    
    def _extract_relevance_score(self, func):
        """Extract relevance score from nested structure."""
        # First check top level
        score = func.get("relevance_score", func.get("score", 0.0))
        if score != 0.0:
            return score
        
        # Then check candidate level
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            score = candidate.get("relevance_score", candidate.get("score", 0.0))
            if score != 0.0:
                return score
            
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            score = inner_candidate.get("relevance_score", inner_candidate.get("score", 0.0))
            if score != 0.0:
                return score
        
        return 0.0
    
    def _extract_grade_reason(self, func):
        """Extract grade reason from nested structure."""
        # First check top level
        reason = func.get("grade_reason", "No reason provided")
        if reason != "No reason provided":
            return reason
        
        # Then check candidate level
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            reason = candidate.get("grade_reason", "No reason provided")
            if reason != "No reason provided":
                return reason
            
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            reason = inner_candidate.get("grade_reason", "No reason provided")
            if reason != "No reason provided":
                return reason
        
        return "No reason provided"
    
    def _extract_source(self, func):
        """Extract source from nested structure."""
        # First check top level - ALWAYS return if exists
        if "source" in func and func["source"]:
            return func["source"]
        
        # Then check candidate level
        candidate = func.get("candidate", {})
        if isinstance(candidate, dict):
            if "source" in candidate and candidate["source"]:
                return candidate["source"]
            
            # Handle double-nested structure
            inner_candidate = candidate.get("candidate", candidate)
            if isinstance(inner_candidate, dict) and "source" in inner_candidate and inner_candidate["source"]:
                return inner_candidate["source"]
        
        return "context_aware_retrieval"

    def _normalize_path(self, path: str) -> str:
        if not path:
            return ""
        return str(path).replace("\\", "/").lower()

    def _parse_function_from_compiler_context(self, line: str) -> str:
        if not line:
            return None
        match = re.search(r"In function [‘']([^’']+)[’']:", line)
        return match.group(1) if match else None

    def _collect_ground_truth_clusters(self, ground_truth_errors: list) -> list:
        """
        Group compiler errors by (file, function) to provide stable, high-signal
        context to Phase 2/3.
        """
        clusters = {}
        for err in ground_truth_errors or []:
            if not isinstance(err, dict):
                continue
            key = (
                self._normalize_path(err.get("file_path", "")),
                err.get("function_name") or "Unknown"
            )
            clusters.setdefault(key, {
                "file_path": err.get("file_path", ""),
                "function_name": err.get("function_name") or "Unknown",
                "errors": [],
                "priority_reason": "direct_compiler_error"
            })
            clusters[key]["errors"].append({
                "line_number": err.get("line_number"),
                "error_code_category": err.get("error_code_category"),
                "error_text": err.get("error_text")
            })

        # Keep only compact top clusters.
        output = []
        for _, c in clusters.items():
            output.append({
                "file_path": c["file_path"],
                "function_name": c["function_name"],
                "error_count": len(c["errors"]),
                "priority_reason": c["priority_reason"],
                "top_errors": c["errors"][:4]
            })
        output.sort(key=lambda x: x.get("error_count", 0), reverse=True)
        return output[:10]

    def _inject_ground_truth_priority(self, candidate_functions: list, ground_truth_errors: list) -> list:
        """
        Boost and inject candidates using compiler ground-truth errors.
        """
        if not isinstance(candidate_functions, list):
            candidate_functions = []
        if not isinstance(ground_truth_errors, list) or not ground_truth_errors:
            return candidate_functions

        # Build function/file evidence maps.
        gt_by_function = {}
        gt_by_file = {}
        for err in ground_truth_errors:
            if not isinstance(err, dict):
                continue
            fn = err.get("function_name")
            fp = self._normalize_path(err.get("file_path", ""))
            if fn:
                gt_by_function[fn] = gt_by_function.get(fn, 0) + 1
            if fp:
                gt_by_file[fp] = gt_by_file.get(fp, 0) + 1

        existing_by_name = {}
        for func in candidate_functions:
            if isinstance(func, dict) and func.get("function_name"):
                existing_by_name[func["function_name"]] = func

        # Inject missing function candidates from ground truth.
        for err in ground_truth_errors:
            if not isinstance(err, dict):
                continue
            fn = err.get("function_name")
            if not fn or fn in existing_by_name:
                continue
            candidate = {
                "function_name": fn,
                "file_path": err.get("file_path", "Unknown"),
                "relevance_score": 1.0,
                "source": "compiler_ground_truth",
                "code_snippet": "",
                "reason": f"Direct compiler error at line {err.get('line_number')}: {err.get('error_text', '')}"
            }
            candidate_functions.append(candidate)
            existing_by_name[fn] = candidate

        # Boost candidates that match ground truth function/file evidence.
        for func in candidate_functions:
            if not isinstance(func, dict):
                continue
            fn = func.get("function_name")
            fp = self._normalize_path(func.get("file_path", ""))

            matched = False
            if fn and fn in gt_by_function:
                matched = True
            elif fp:
                for gt_fp in gt_by_file:
                    if fp.endswith(gt_fp) or gt_fp.endswith(fp):
                        matched = True
                        break

            if matched:
                old_score = float(func.get("relevance_score", 0.0) or 0.0)
                func["relevance_score"] = max(old_score, 0.99)
                func["source"] = "compiler_ground_truth"
                reason = func.get("reason", "")
                if "direct compiler error" not in reason.lower():
                    func["reason"] = (reason + " | " if reason else "") + "Backed by direct compiler error evidence."

        candidate_functions.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return candidate_functions

    def _log_path_to_relative_source(self, log_file_path: str) -> str:
        """Map an absolute log path to a path under the OAI tree (e.g. openair2/...)."""
        if not log_file_path:
            return ""
        p = log_file_path.replace("\\", "/")
        lower = p.lower()
        for marker in (
            "openair2/",
            "openair3/",
            "openair1/",
            "/common/",
            "nfapi/",
            "executables/",
        ):
            i = lower.find(marker)
            if i != -1:
                return p[i:]
        return ""

    def _read_snippet_around_line(
        self, rel_path: str, line_no: int, window: int = 55
    ) -> str:
        if not rel_path or not line_no:
            return ""
        base = os.path.join(
            os.path.dirname(__file__), self.openair_codebase_file_name
        )
        full = os.path.join(base, rel_path.replace("\\", "/"))
        if not os.path.isfile(full):
            return ""
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            lo = max(0, line_no - 1 - window)
            hi = min(len(lines), line_no - 1 + window)
            return "".join(lines[lo:hi])
        except OSError:
            return ""

    def _snippet_from_functions_db_for_compiler(self, fn: str, log_fp: str) -> str:
        self._load_functions_db_for_expansion()
        idx = self._kg_functions_db_index or {}
        entries = idx.get(fn) or []
        nfp = self._normalize_path(log_fp or "")
        for e in entries:
            if not isinstance(e, dict):
                continue
            ep = self._normalize_path(e.get("file_path", ""))
            if nfp and ep and (nfp.endswith(ep) or ep.endswith(nfp) or ep in nfp):
                return (e.get("code_snippet") or e.get("source_code") or "") or ""
        if entries and isinstance(entries[0], dict):
            return (
                entries[0].get("code_snippet")
                or entries[0].get("source_code")
                or ""
            ) or ""
        return ""

    def _build_candidates_from_primary_compiler_clusters(
        self, compiler_cluster_analysis: dict
    ) -> list:
        """
        Build Phase 2 suspected_functions from primary compiler clusters only
        (no CRAG / downstream / KG expansion).
        """
        if not isinstance(compiler_cluster_analysis, dict):
            return []
        pids = set(compiler_cluster_analysis.get("primary_cluster_ids") or [])
        if not pids:
            return []
        seen = set()
        candidates = []
        for c in compiler_cluster_analysis.get("clusters") or []:
            if not isinstance(c, dict):
                continue
            if c.get("cluster_id") not in pids:
                continue
            for err in c.get("errors") or []:
                if not isinstance(err, dict):
                    continue
                fn = err.get("function_name")
                if not fn:
                    continue
                fp = err.get("file_path") or ""
                key = (fn, self._normalize_path(fp))
                if key in seen:
                    continue
                seen.add(key)
                rel = self._log_path_to_relative_source(fp)
                ln = err.get("line_number")
                snippet = ""
                if rel and ln:
                    snippet = self._read_snippet_around_line(rel, int(ln))
                if not snippet:
                    snippet = self._snippet_from_functions_db_for_compiler(fn, fp)
                candidates.append(
                    {
                        "function_name": fn,
                        "file_path": rel or fp or "Unknown",
                        "relevance_score": 1.0,
                        "source": "compiler_ground_truth_primary",
                        "code_snippet": snippet or "",
                        "reason": (
                            f"Primary compiler cluster [{c.get('category')}]: "
                            f"{(err.get('error_text') or '')[:220]}"
                        ),
                    }
                )
        candidates.sort(
            key=lambda x: float(x.get("relevance_score", 0.0) or 0.0), reverse=True
        )
        return candidates

    def _call_graph_subset_for_functions(
        self, call_graph_context: list, function_names: set
    ) -> list:
        if not call_graph_context or not function_names:
            return []
        out = []
        for entry in call_graph_context:
            if isinstance(entry, dict) and entry.get("function") in function_names:
                out.append(entry)
        return out

    def _merge_phase3_compiler_batch_results(
        self,
        batch_results: list,
        base_error_text: str,
        total_candidate_functions: int,
    ) -> dict:
        """
        Merge multiple process_fix_request outputs from compiler-ground-truth batches.
        """
        empty_fix = {
            "suspected_functions": [],
            "suspected_configs": [],
            "reason": "",
            "config_fix": "",
            "code_patches": [],
            "config_patches": [],
            "root_cause_analysis": "",
            "investigation_steps": [],
            "specification_context": "",
        }
        if not batch_results:
            return {
                "error_text": base_error_text,
                "fix_suggestion": empty_fix,
                "context_summary": {
                    "candidate_functions_count": 0,
                    "candidate_configs_count": 0,
                    "call_graph_entries": 0,
                    "pattern_matched": False,
                },
                "phase3_batching": {
                    "enabled": False,
                    "batch_count": 0,
                    "batch_size": _COMPILER_GROUND_TRUTH_PHASE3_BATCH_SIZE,
                },
            }

        if len(batch_results) == 1:
            one = dict(batch_results[0])
            one["phase3_batching"] = {
                "enabled": False,
                "batch_count": 1,
                "batch_size": _COMPILER_GROUND_TRUTH_PHASE3_BATCH_SIZE,
            }
            return one

        merged_patches = []
        patch_keys_seen = set()
        merged_config_patches = []
        suspected_fn_order = []
        suspected_fn_seen = set()
        reasons = []
        root_causes = []
        inv_steps = []
        inv_seen = set()
        spec_chunks = []
        config_fix_bits = []
        max_cg = 0

        for i, br in enumerate(batch_results):
            if not isinstance(br, dict):
                continue
            if br.get("pipeline_error"):
                logger.warning(
                    "Phase 3 compiler batch %s failed: %s",
                    i + 1,
                    br.get("pipeline_error"),
                )
            fs = br.get("fix_suggestion") if isinstance(br.get("fix_suggestion"), dict) else {}
            cs = br.get("context_summary") if isinstance(br.get("context_summary"), dict) else {}
            max_cg = max(max_cg, int(cs.get("call_graph_entries") or 0))

            for name in fs.get("suspected_functions") or []:
                if name and name not in suspected_fn_seen:
                    suspected_fn_seen.add(name)
                    suspected_fn_order.append(name)

            for p in fs.get("code_patches") or []:
                if not isinstance(p, dict):
                    continue
                key = (
                    p.get("function_name"),
                    p.get("file_path"),
                    str(p.get("line_numbers", "")),
                    (p.get("original_code") or "")[:400],
                )
                if key in patch_keys_seen:
                    continue
                patch_keys_seen.add(key)
                merged_patches.append(p)

            for cp in fs.get("config_patches") or []:
                if isinstance(cp, dict):
                    merged_config_patches.append(cp)

            r = (fs.get("reason") or "").strip()
            if r:
                reasons.append(r)
            rc = (fs.get("root_cause_analysis") or "").strip()
            if rc:
                root_causes.append(rc)
            for step in fs.get("investigation_steps") or []:
                if isinstance(step, str) and step.strip() and step not in inv_seen:
                    inv_seen.add(step)
                    inv_steps.append(step)
            sc = (fs.get("specification_context") or "").strip()
            if sc:
                spec_chunks.append(sc)
            cf = (fs.get("config_fix") or "").strip()
            if cf:
                config_fix_bits.append(cf)

        merged_spec = ""
        if spec_chunks:
            merged_spec = max(spec_chunks, key=len)

        phase3_llm_batches: list = []
        any_parse_fail = False
        for br in batch_results:
            if not isinstance(br, dict):
                continue
            meta = br.get("phase3_llm")
            if isinstance(meta, dict):
                phase3_llm_batches.append(meta)
                if not meta.get("parse_ok", True):
                    any_parse_fail = True

        fix_suggestion = {
            "suspected_functions": suspected_fn_order,
            "suspected_configs": [],
            "reason": "\n\n---\n\n".join(reasons),
            "config_fix": "\n\n".join(config_fix_bits),
            "code_patches": merged_patches,
            "config_patches": merged_config_patches,
            "root_cause_analysis": "\n\n---\n\n".join(root_causes),
            "investigation_steps": inv_steps,
            "specification_context": merged_spec,
        }

        out = {
            "error_text": base_error_text,
            "fix_suggestion": fix_suggestion,
            "context_summary": {
                "candidate_functions_count": total_candidate_functions,
                "candidate_configs_count": 0,
                "call_graph_entries": max_cg,
                "pattern_matched": False,
            },
            "phase3_batching": {
                "enabled": True,
                "batch_count": len(batch_results),
                "batch_size": _COMPILER_GROUND_TRUTH_PHASE3_BATCH_SIZE,
            },
            "phase3_llm": {"batches": phase3_llm_batches},
        }
        if any_parse_fail:
            out["phase3_parse_failed"] = True
        return out

    def _phase3_compiler_ground_truth_block(
        self,
        compiler_ground_truth_errors: list,
        compiler_cluster_analysis: dict,
        primary_only: bool,
    ) -> str:
        """Append compiler ground truth to Phase 3 error text (primary-only when narrow Phase 2)."""
        if not compiler_ground_truth_errors:
            return ""
        lines = []
        if primary_only and compiler_cluster_analysis.get("primary_cluster_ids"):
            pid = set(compiler_cluster_analysis["primary_cluster_ids"])
            for c in compiler_cluster_analysis.get("clusters") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("cluster_id") not in pid:
                    continue
                for gt in (c.get("errors") or [])[:4]:
                    if not isinstance(gt, dict):
                        continue
                    lines.append(
                        f"- {gt.get('file_path')}:{gt.get('line_number')} "
                        f"[{gt.get('function_name') or 'Unknown'}] "
                        f"({c.get('category')}) {gt.get('error_text')}"
                    )
            sec = compiler_cluster_analysis.get("secondary_cluster_ids") or []
            if sec:
                lines.append(
                    f"(Omitted {len(sec)} secondary/cascade compiler cluster(s); "
                    "fix primary issues first.)"
                )
        else:
            for gt in compiler_ground_truth_errors[:8]:
                if not isinstance(gt, dict):
                    continue
                lines.append(
                    f"- {gt.get('file_path')}:{gt.get('line_number')} "
                    f"[{gt.get('function_name') or 'Unknown'}] {gt.get('error_text')}"
                )
        if not lines:
            return ""
        return "\n\nHighest Priority Compiler Ground Truth:\n" + "\n".join(lines)

    def _extract_downstream_functions_from_call_graph(self, candidate_functions: list, call_graph_context: list) -> list:
        """
        Extract downstream functions from call graph and add them as candidates.
        This ensures functions like rrc_gNB_create_ue_context are available for fix generation.
        
        Args:
            candidate_functions: List of candidate functions from Phase 2
            call_graph_context: Call graph context loaded from function_calls.json
            
        Returns:
            List of downstream function candidates with their definitions
        """
        downstream_candidates = []
        candidate_names = {f.get('function_name', '') for f in candidate_functions if isinstance(f, dict)}
        
        # Collect downstream function names from call graph
        downstream_names = set()
        for entry in call_graph_context:
            if not isinstance(entry, dict):
                continue
            func_name = entry.get('function', '')
            if func_name in candidate_names:
                # This is a candidate function, get its downstream calls
                calls = entry.get('calls', [])
                if isinstance(calls, list):
                    for called_func in calls:
                        if called_func not in candidate_names:
                            downstream_names.add(called_func)
        
        # Load functions database to get definitions
        functions_db = []
        try:
            if os.path.exists("database/functions.json"):
                with open("database/functions.json", 'r', encoding='utf-8') as f:
                    functions_db = json.load(f)
            elif os.path.exists("faiss_indices/functions_mapping.json"):
                with open("faiss_indices/functions_mapping.json", 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    if isinstance(mapping, dict):
                        functions_db = list(mapping.values())
                    else:
                        functions_db = mapping
        except Exception as e:
            logger.warning(f"⚠️  Failed to load functions database for downstream extraction: {e}")
        
        # Create candidate entries for downstream functions
        for func_name in downstream_names:
            # Find function definition in database
            func_def = None
            for func_data in functions_db:
                if isinstance(func_data, dict) and func_data.get('function_name') == func_name:
                    func_def = func_data
                    break
            
            # Also check call graph context for file path if not found in database
            file_path = 'Unknown'
            code_snippet = ''
            if func_def:
                file_path = func_def.get('file_path', 'Unknown')
                code_snippet = func_def.get('code_snippet', func_def.get('code_body', ''))
            else:
                # Try to get file path from call graph entry
                for entry in call_graph_context:
                    if isinstance(entry, dict) and entry.get('function') == func_name:
                        file_path = entry.get('file', 'Unknown')
                        # Normalize path separators
                        if file_path and '\\' in file_path:
                            file_path = file_path.replace('\\', '/')
                        break
            
            # Create candidate entry even if we don't have full definition
            # This ensures downstream functions are included in Phase 2
            downstream_candidates.append({
                "function_name": func_name,
                "file_path": file_path,
                "relevance_score": 0.7,  # Medium-high relevance for downstream functions
                "source": "call_graph_downstream",
                "code_snippet": code_snippet if code_snippet else f"Function {func_name} called by candidate functions",
                "reason": f"Downstream function called by candidate functions. Important for understanding context initialization and error handling."
            })
            logger.info(f"📥 Extracted downstream function: {func_name} from {file_path}")
        
        logger.info(f"✅ Extracted {len(downstream_candidates)} downstream functions from call graph")
        return downstream_candidates
    
    def process_crash_analysis(self, log_file_path: str, phase: str = "extraction") -> dict:
        """
        Phase 1: Extract error and traceback from segmentation fault log.
        
        Args:
            log_file_path: Path to the segmentation fault log file
            phase: Analysis phase - "extraction" for Phase 1, "full" for complete analysis
            
        Returns:
            Crash extraction results
        """
        logger.info("🔬 CRASH ANALYSIS PIPELINE - PHASE 1: EXTRACTION")
        logger.info("=" * 70)
        try:
            sync_result = self._incremental_sync_functions_and_embeddings()
            if sync_result.get("updated"):
                # Reinitialize pipelines so they reload refreshed indices/databases.
                self.phase2_pipeline = ErrorHandlingPipeline(openair_codebase_file_name=self.openair_codebase_file_name)
                self.phase3_pipeline = FixSuggestionPipeline(openair_codebase_file_name=self.openair_codebase_file_name)
        except Exception as e:
            logger.warning(f"⚠️ Incremental sync before crash analysis failed: {e}")

        self._sync_git_commit_embeddings_at_rca_start()
        
        if not log_file_path or not os.path.exists(log_file_path):
            logger.error(f"❌ Log file not found: {log_file_path}")
            return {
                "error": "Log file not found",
                "log_file": log_file_path,
                "pipeline_error": "Segmentation fault log file must be provided"
            }
        
        try:
            # Extract deployment context from log file (same as normal flow)
            logger.info(f"📁 Parsing log file for deployment context: {log_file_path}")
            deployment_context = self.log_parser.parse_log_file(log_file_path)
            
            # Phase 1: Extract error and traceback only (no LLM analysis yet)
            logger.info(f"📄 Extracting error and traceback from: {log_file_path}")
            extract_only = (phase == "extraction")
            crash_results = self.crash_analyzer.process_segmentation_fault(log_file_path, extract_only=extract_only)
            
            if "error" in crash_results:
                logger.error(f"❌ Crash extraction failed: {crash_results['error']}")
                return crash_results
            
            # Format results for UI compatibility
            crash_info = crash_results.get("crash_info", {})
            
            # Create error message from extracted crash info
            error_message = f"Segmentation Fault in {crash_info.get('faulting_function', 'Unknown')} at {crash_info.get('fault_location', 'Unknown')}"
            
            # Convert to similar format as regular pipeline for UI display
            formatted_results = {
                "error_message": error_message,
                "crash_analysis": True,
                "phase": "extraction",
                "timestamp": datetime.now().isoformat(),
                "log_file": log_file_path,
                "deployment_context": deployment_context,  # Add deployment context for artifacts view
                "crash_info": crash_info,
                "extraction_summary": {
                    "crash_detected": crash_info.get("crash_detected", False),
                    "signal": crash_info.get("signal"),
                    "crash_type": crash_info.get("crash_type"),
                    "fault_location": crash_info.get("fault_location"),
                    "faulting_function": crash_info.get("faulting_function"),
                    "faulting_file": crash_info.get("faulting_file"),
                    "faulting_line": crash_info.get("faulting_line"),
                    "crash_thread": crash_info.get("crash_thread"),
                    "backtrace_frames": len(crash_info.get("backtrace", [])),
                    "scenario_steps_before_crash": len(crash_info.get("scenario_flow", []))
                },
                "backtrace": crash_info.get("backtrace", []),
                "scenario_flow": crash_info.get("scenario_flow", []),
                "pre_crash_logs": crash_info.get("pre_crash_logs", []),
                "source_code_at_fault": crash_info.get("source_code_at_fault"),
                "summary": {
                    "phase": "Phase 1: Error & Traceback Extraction",
                    "crash_detected": crash_info.get("crash_detected", False),
                    "fault_location": crash_info.get("fault_location"),
                    "faulting_function": crash_info.get("faulting_function"),
                    "backtrace_frames": len(crash_info.get("backtrace", [])),
                    "scenario_steps_before_crash": len(crash_info.get("scenario_flow", [])),
                    "next_phase": "Phase 2: Root Cause Analysis (not implemented yet)"
                }
            }
            
            logger.info("\n✅ PHASE 1: ERROR & TRACEBACK EXTRACTION COMPLETED")
            logger.info("=" * 70)
            logger.info(f"📊 Extracted:")
            logger.info(f"   - Signal: {crash_info.get('signal')}")
            logger.info(f"   - Fault Location: {crash_info.get('fault_location')}")
            logger.info(f"   - Faulting Function: {crash_info.get('faulting_function')}")
            logger.info(f"   - Backtrace Frames: {len(crash_info.get('backtrace', []))}")
            logger.info(f"   - Scenario Steps Before Crash: {len(crash_info.get('scenario_flow', []))}")
            
            # Run Phase 2: Targeted Candidate Retrieval
            logger.info("\n🔗 Running Phase 2: Targeted Candidate Retrieval...")
            phase2_results = self.crash_analyzer.run_phase2_retrieval("output/segmentation_fault_extraction.json")
            
            if phase2_results:
                formatted_results["phase2_crash_retrieval"] = phase2_results
                total_funcs = len(phase2_results.get('prioritized_functions', [])) + len(phase2_results.get('call_chain_expansion', []))
                logger.info("✅ Phase 2 crash retrieval completed")
                logger.info(f"   - Total Functions Collected: {total_funcs}")
                logger.info(f"   - Prioritized (Backtrace): {len(phase2_results.get('prioritized_functions', []))}")
                logger.info(f"   - Call Chain Expansion: {len(phase2_results.get('call_chain_expansion', []))}")
                
                # Run Phase 2.5: Intelligent Grading
                logger.info("\n🧠 Running Phase 2.5: Intelligent Function Grading...")
                phase25_results = self.crash_analyzer.run_phase2_grading(
                    "output/crash_phase2_retrieval.json",
                    "output/segmentation_fault_extraction.json"
                )
                
                if phase25_results:
                    # Replace phase2 data with graded results (Phase 3 compatible format)
                    formatted_results["phase2_analysis"] = {
                        "suspected_functions": phase25_results.get("suspected_functions", []),
                        "suspected_configs": phase25_results.get("suspected_configs", []),
                        "crash_analysis": True,
                        "retrieval_method": "crash_intelligent_grading"
                    }
                    logger.info("✅ Phase 2.5 intelligent grading completed")
                    logger.info(f"   - Selected TOP {len(phase25_results.get('suspected_functions', []))} functions for Phase 3")
                    
                    # Run Phase 3: Generate crash fixes
                    logger.info("\n🔧 Running Phase 3: Crash Fix Generation...")
                    phase3_results = self.crash_analyzer.run_phase3_fix_generation(
                        "output/crash_phase2_graded.json",
                        "output/segmentation_fault_extraction.json"
                    )
                    
                    if phase3_results and "error" not in phase3_results:
                        # Add Phase 3 results in format compatible with display_bug_analysis_results
                        formatted_results["phase3_fixes"] = {
                            "fix_suggestion": phase3_results.get("fix_suggestion", {}),
                            "context_summary": phase3_results.get("context_summary", {})
                        }
                        
                        fix_suggestion = phase3_results.get("fix_suggestion", {})
                        logger.info("✅ Phase 3 crash fix generation completed")
                        logger.info(f"   - Code Patches Generated: {len(fix_suggestion.get('code_patches', []))}")
                        logger.info(f"   - Suspected Functions: {len(fix_suggestion.get('suspected_functions', []))}")
                    else:
                        logger.warning("⚠️  Phase 3 fix generation not available or failed")
                else:
                    logger.warning("⚠️  Phase 2.5 grading not available or failed")
            else:
                logger.warning("⚠️  Phase 2 retrieval not available or failed")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Crash extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "log_file": log_file_path,
                "pipeline_error": "Crash extraction failed",
                "crash_analysis": True
            }

    @staticmethod
    def _is_cmake_linker_structural_log(error_message: str, log_content: str) -> bool:
        """
        Linker / ASN.1 / CMake structural failures that must stay on the existing CMake patch path,
        not the dependency-advice-only path.
        """
        text = f"{error_message or ''}\n{log_content or ''}"
        if not text.strip():
            return False
        low = text.lower()
        if any(
            x in low
            for x in (
                "cmake error",
                "errors occurred in cmake",
                "cmake configuration failed",
                "could not find cmake",
                "cmakelists.txt",
                ".cmake:",
            )
        ):
            return True
        if "/usr/bin/ld:" in text or "undefined reference" in low:
            return True
        if "collect2:" in low and "ld returned" in low:
            return True
        if "ninja:" in low and "error" in low and "undefined" in low:
            return True
        if "libasn1_" in low and ".a" in low:
            return True
        if "asn_def_" in low and ("ld" in low or "reference" in low or "undefined" in low):
            return True
        return False

    @staticmethod
    def _should_force_dependency_log_kind(error_message: str, log_content: str) -> bool:
        """
        C/C++ toolchain / missing tool / missing package / version mismatch — advice-only path
        (no code or CMake patching). Excludes linker/structural CMake failures that use the
        existing CMake flow.
        """
        text = f"{error_message or ''}\n{log_content or ''}"
        if not text.strip():
            return False
        low = text.lower()
        if CompleteErrorFixingPipeline._is_cmake_linker_structural_log(error_message, log_content):
            return False
        # Mixed logs often contain benign "No package ... found" lines during configure, then fail
        # later with concrete compiler diagnostics. Preserve original build flow for those logs.
        if CompleteErrorFixingPipeline._has_compile_diagnostics_in_text(error_message, log_content):
            return False
        # Runtime crashes stay on runtime path
        if any(
            x in low
            for x in (
                "segmentation fault",
                "segfault",
                "sigsegv",
                "sigabrt",
            )
        ):
            return False
        if any(
            x in low
            for x in (
                "command not found",
                "is not recognized as an internal or external command",
            )
        ):
            return True
        if "unable to locate package" in low or "e: unable to locate package" in low:
            return True
        if "no package '" in low and "found" in low:
            return True
        if "package '" in low and "not found" in low and "pkg-config" in low:
            return True
        if "error while loading shared libraries" in low and "cannot open shared object" in low:
            return True
        if "could not find " in low and ("compiler" in low or "cxx compiler" in low):
            return True
        if "no cmake_cxx_compiler could be found" in low or "no cmake_c_compiler could be found" in low:
            return True
        if "the cxx compiler" in low and "not able to compile" in low:
            return True
        if "unsupported compiler" in low or ("compiler" in low and "too old" in low):
            return True
        if "version conflict" in low or "unsatisfiable" in low:
            return True
        if "vcpkg" in low and any(
            x in low for x in ("not installed", "unknown package", "was not found", "cannot find")
        ):
            return True
        if "conan" in low and "not found" in low:
            return True
        if "failed downloading" in low and ("http" in low or "https" in low):
            return True
        return False

    @staticmethod
    def _has_compile_diagnostics_in_text(error_message: str, log_content: str) -> bool:
        """Detect C/C++ compiler diagnostics directly from raw text."""
        text = f"{error_message or ''}\n{log_content or ''}"
        if not text.strip():
            return False
        low = text.lower()
        if any(
            x in low
            for x in (
                "error: unknown type name",
                "error: request for member",
                "error: implicit declaration of function",
                "compilation terminated",
            )
        ):
            return True
        if re.search(r":\d+:\d+:\s*error:\s", text):
            return True
        if re.search(r":\d+:\s*error:\s", text):
            return True
        return False

    @staticmethod
    def _heuristic_log_error_kind(error_message: str, log_content: str) -> Optional[str]:
        """
        Fast path: return one of:
        - dependency
        - runtime
        - build
        - cmake
        - other
        - None (use LLM fallback)
        """
        text = f"{error_message or ''}\n{log_content or ''}"
        if not text.strip():
            return None
        low = text.lower()
        # Toolchain / missing package / version mismatch (before runtime/cmake/build)
        if CompleteErrorFixingPipeline._should_force_dependency_log_kind(error_message, log_content):
            return "dependency"
        # Runtime / crash / protocol
        if any(
            x in low
            for x in (
                "segmentation fault",
                "segfault",
                "sigsegv",
                "sigabrt",
                "assertion",
                "assert failed",
            )
        ):
            return "runtime"
        # Build system: CMake configure / generator / linker wiring
        if any(
            x in low
            for x in (
                "cmake error",
                "errors occurred in cmake",
                "cmake configuration failed",
                "could not find cmake",
                "cmakelists.txt",
                ".cmake:",
            )
        ):
            return "cmake"
        if "/usr/bin/ld:" in text or "undefined reference" in low:
            return "cmake"
        if "collect2:" in low and "ld returned" in low:
            return "cmake"
        if "ninja:" in low and "error" in low and "undefined" in low:
            return "cmake"
        # Build diagnostics
        if any(
            x in low
            for x in (
                "compilation terminated",
                "fatal error:",
                "error: unknown type name",
                "error: implicit declaration of function",
                "did you mean",
            )
        ):
            return "build"
        # Classic compiler diagnostic (source file:line:col:error:)
        if re.search(r":\d+:\d+:\s*error:\s", text):
            return "build"
        # GCC/clang without column: file:line: error:
        if re.search(r":\d+:\s*error:\s", text):
            return "build"
        return None

    def _classify_log_error_kind_llm(self, error_message: str, log_content: str) -> str:
        """LLM fallback when heuristics are ambiguous."""
        snippet = (log_content or "")[-6500:]
        user = (
            f"Error message:\n{error_message or ''}\n\nLog excerpt (tail):\n{snippet}\n\n"
            'Return JSON only: {"kind":"runtime"} or {"kind":"build"} or {"kind":"cmake"} or '
            '{"kind":"dependency"} or {"kind":"other"}.\n'
            "- cmake: fix mainly involves CMakeLists.txt / *.cmake / build (find_package, "
            "target_link_libraries, add sources to targets, linker undefined reference, ASN.1 codegen not linked).\n"
            "- build: fix mainly involves compiler diagnostics in C/C++ source or headers during build.\n"
            "- runtime: fix mainly involves runtime behavior, protocol state, crashes, assertions, or invalid state.\n"
            "- dependency: missing toolchain (cmake/gcc not in PATH), missing OS/dev package or library, "
            "version mismatch, or package manager (vcpkg/conan) cannot resolve/install — fix is usually "
            "install/configure environment, not source patches.\n"
            "- other: logs do not cleanly fit build/runtime/cmake/dependency categories.\n"
        )
        response = self.phase2_pipeline.azure_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": "You classify engineering logs. Reply with JSON only."},
                {"role": "user", "content": user},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            # tolerate markdown fences
            if "```" in raw:
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:].lstrip()
            data = json.loads(raw)
            k = data.get("kind")
            if k in ("runtime", "build", "cmake", "dependency", "other"):
                return k
        except Exception:
            pass
        low = raw.lower()
        if '"kind":"cmake"' in low or "cmake" in low:
            return "cmake"
        if '"kind":"dependency"' in low or "dependency" in low:
            return "dependency"
        if '"kind":"build"' in low or "build" in low or "compiler" in low:
            return "build"
        if '"kind":"runtime"' in low or "runtime" in low:
            return "runtime"
        return "other"

    def _classify_log_error_kind(self, error_message: str, log_content: str) -> str:
        h = self._heuristic_log_error_kind(error_message, log_content)
        if h:
            logger.info(f"📌 Log error kind (heuristic): {h}")
            return h
        try:
            k = self._classify_log_error_kind_llm(error_message, log_content)
            logger.info(f"📌 Log error kind (LLM): {k}")
            return k
        except Exception as e:
            logger.warning(f"⚠️ Log kind LLM failed ({e}); defaulting to other")
            return "other"

    @staticmethod
    def _should_force_cmake_log_kind(error_message: str, log_content: str) -> bool:
        """
        Linker / CMake / ASN.1 build-system signals that must stay on the CMake path even when
        an LLM classifies the log as generic 'build' or compiler diagnostics appear elsewhere.
        Does not match pure runtime crashes without these signals.
        """
        text = f"{error_message or ''}\n{log_content or ''}"
        if not text.strip():
            return False
        low = text.lower()
        if any(
            x in low
            for x in (
                "cmake error",
                "errors occurred in cmake",
                "cmake configuration failed",
                "could not find cmake",
                "cmakelists.txt",
                ".cmake:",
            )
        ):
            return True
        if "/usr/bin/ld:" in text or "undefined reference" in low:
            return True
        if "collect2:" in low and "ld returned" in low:
            return True
        if "ninja:" in low and "error" in low and "undefined" in low:
            return True
        if "libasn1_" in low and ".a" in low:
            return True
        if "asn_def_" in low and ("ld" in low or "reference" in low or "undefined" in low):
            return True
        return False
    
    def process_error_with_context(self, error_message: str = None, log_file_path: str = None, custom_deployment_context: dict = None) -> dict:
        """
        Process error through the complete pipeline with context awareness
        
        Args:
            error_message: The error to analyze and fix (optional - will be extracted from log if not provided)
            log_file_path: Optional path to log file for context extraction
            custom_deployment_context: Optional custom deployment context to use instead of JSON defaults
            
        Returns:
            Complete results including analysis and fix suggestions
        """
        logger.info("COMPLETE ERROR FIXING PIPELINE")
        logger.info("=" * 70)
        try:
            sync_result = self._incremental_sync_functions_and_embeddings()
            if sync_result.get("updated"):
                # Reinitialize pipelines so they reload refreshed indices/databases.
                self.phase2_pipeline = ErrorHandlingPipeline(openair_codebase_file_name=self.openair_codebase_file_name)
                self.phase3_pipeline = FixSuggestionPipeline(openair_codebase_file_name=self.openair_codebase_file_name)
        except Exception as e:
            logger.warning(f"⚠️ Incremental sync before error analysis failed: {e}")

        self._sync_git_commit_embeddings_at_rca_start()
        
        # Step 0: Extract error message from log if not provided
        extracted_error = None
        if not error_message and log_file_path and os.path.exists(log_file_path):
            logger.info(f"🔍 STEP 0 - ERROR MESSAGE EXTRACTION")
            logger.info("-" * 50)
            logger.info(f"📄 No error message provided, extracting from log: {log_file_path}")
            
            extracted_error = self.log_parser.extract_error_message(log_file_path)
            if extracted_error:
                error_message = extracted_error
                logger.info(f"✅ Extracted error message: {error_message}")
            else:
                logger.warning("⚠️  Could not extract error message from log file")
                return {
                    "error": "Could not extract error message from log file",
                    "log_file": log_file_path,
                    "extracted_error": None,
                    "pipeline_error": "No error message provided and extraction failed"
                }
        elif not error_message:
            logger.error("❌ No error message provided and no log file specified")
            return {
                "error": "No error message provided and no log file specified",
                "log_file": log_file_path,
                "pipeline_error": "Either error_message or log_file_path must be provided"
            }
        
        logger.info(f"📥 Processing error: {error_message}")
        if extracted_error:
            logger.info(f"📄 Error source: Extracted from log file")
        
        try:
            deployment_context = None
            compiler_ground_truth_errors = []
            log_content = ""
            log_was_parsed = False
            
            # Step 2.1: Parse log file for deployment context (if provided)
            if log_file_path and os.path.exists(log_file_path):
                logger.info(f"\n📄 STEP 2.1 - LOG PARSING & DEPLOYMENT CONTEXT")
                logger.info("-" * 50)
                logger.info(f"📁 Parsing log file: {log_file_path}")
                
                deployment_context = self.log_parser.parse_log_file(log_file_path)
                
                # Extract detailed log context for better analysis
                logger.info("🔍 Extracting detailed log context...")
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                detailed_log_context = self.log_parser.extract_detailed_log_context(log_content, error_message)
                deployment_context['detailed_log_context'] = detailed_log_context
                compiler_ground_truth_errors = self.log_parser.extract_compiler_ground_truth_errors(log_content)
                deployment_context['compiler_ground_truth_errors'] = compiler_ground_truth_errors
                
                log_was_parsed = True
                self.log_parser.save_context(deployment_context, "deployment_context.json")
                
                logger.info(f"✅ Deployment context extracted:")
                logger.info(f"   Role: {deployment_context.get('role', 'Unknown')}")
                logger.info(f"   Active configs: {len(deployment_context.get('active_configs', []))}")
                logger.info(f"   Log anchors: {len(deployment_context.get('log_anchors', []))}")
                logger.info(f"   Debug values: {len(detailed_log_context.get('debug_values', {}))}")
                logger.info(f"   Error sequences: {len(detailed_log_context.get('error_sequences', []))}")
                logger.info(f"   Compiler ground-truth errors: {len(compiler_ground_truth_errors)}")
                logger.info(f"   Network params: {deployment_context.get('network_params', {})}")
            else:
                logger.info(f"\n⚠️  No log file provided or file not found: {log_file_path}")
                logger.info("   Proceeding with standard error analysis...")

            # Optional artifact: save ALL primary compiler ground-truth error lines.
            # This does not change any retrieval / fix logic; it's only for inspection.
            try:
                if compiler_ground_truth_errors:
                    primary_compiler_errors = self.log_parser.extract_primary_compiler_errors(
                        compiler_ground_truth_errors
                    )
                    if primary_compiler_errors:
                        out_json = "output/primary_compiler_errors.json"
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(primary_compiler_errors, f, indent=2, ensure_ascii=False)

                        out_txt = "output/primary_compiler_errors.txt"
                        # Sort to make the file stable/readable.
                        primary_sorted = sorted(
                            primary_compiler_errors,
                            key=lambda e: (
                                (e.get("file_path") or ""),
                                int(e.get("line_number") or 0),
                                int(e.get("column_number") or 0),
                                e.get("function_name") or "",
                            ),
                        )
                        with open(out_txt, "w", encoding="utf-8") as f:
                            for e in primary_sorted:
                                fp = e.get("file_path") or ""
                                ln = e.get("line_number") or ""
                                cn = e.get("column_number") or ""
                                fn = e.get("function_name") or ""
                                cat = e.get("error_code_category") or ""
                                et = e.get("error_text") or ""
                                f.write(f"{fp}:{ln}:{cn}: [{fn}] ({cat}) {et}\n")

                        logger.info(
                            f"💾 Saved primary compiler errors: "
                            f"{len(primary_compiler_errors)} lines to {out_json}"
                        )
            except Exception as e:
                logger.warning(f"⚠️ Failed to save primary compiler errors artifact: {e}")
            
            # Override with custom deployment context if provided
            if custom_deployment_context:
                logger.info(f"\n⚙️  CUSTOM DEPLOYMENT CONTEXT OVERRIDE")
                logger.info("-" * 50)
                logger.info("📝 Using custom deployment context provided by user")
                if deployment_context is None:
                    deployment_context = {}
                # Merge custom deployment context with log-parsed context (custom takes priority)
                deployment_context.update(custom_deployment_context)
                logger.info(f"✅ Custom deployment context applied: {len(custom_deployment_context)} values")
            
            # Check if deployment context has the required network parameters
            # If not, load from JSON as fallback
            has_network_params = False
            if deployment_context and isinstance(deployment_context, dict):
                # Check if it has any of the key network parameters
                network_keys = ['cu_ip_address', 'du_ip_address', 'gnb_ip_address', 'amf_ip_address']
                has_network_params = any(key in deployment_context for key in network_keys)
            
            if not has_network_params:
                logger.info(f"\n📄 LOADING DEFAULT DEPLOYMENT CONTEXT FROM JSON")
                logger.info("-" * 50)
                logger.info(f"   Current deployment_context: {deployment_context}")
                logger.info(f"   Reason: Missing network parameters (IP addresses)")
                try:
                    json_path = 'database/error_patterns_structured.json'
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            json_deployment = data.get('deployment_context', {})
                            
                            # Merge JSON defaults with existing deployment_context
                            # Existing values (from log parsing) take priority
                            if deployment_context is None:
                                deployment_context = {}
                            
                            # Add JSON values for missing keys
                            for key, value in json_deployment.items():
                                if key not in deployment_context:
                                    deployment_context[key] = value
                            
                            logger.info(f"✅ Merged {len(json_deployment)} default deployment values from JSON")
                            logger.info(f"   Final deployment_context has {len(deployment_context)} total values")
                    else:
                        logger.warning(f"⚠️  JSON file not found at {json_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to load deployment context from JSON: {e}")
            
            # True when we had parsed log and/or merged deployment context before classification keys
            pre_classify_had_context = (
                deployment_context is not None and len(deployment_context) > 0
            )
            
            # Classify log: CMake/build-system vs compile/runtime (controls Phase 2 + Phase 3)
            if deployment_context is None:
                deployment_context = {}
            classified_log_error_kind = self._classify_log_error_kind(error_message, log_content)
            has_compiler_diagnostics = bool(compiler_ground_truth_errors) or self._has_compile_diagnostics_in_text(
                error_message, log_content
            )
            # Dependency / toolchain / missing package (C/C++): advice-only path — evaluated before
            # CMake structural path so e.g. "cmake: command not found" is not treated like a CMakeLists fix.
            dep_candidate = (
                classified_log_error_kind == "dependency"
                or self._should_force_dependency_log_kind(error_message, log_content)
            )
            if dep_candidate and not has_compiler_diagnostics and not self._is_cmake_linker_structural_log(
                error_message, log_content
            ):
                log_error_kind = "dependency"
                logger.info(
                    "📌 Log error kind set to dependency (toolchain / missing package / version mismatch; "
                    "advice-only path)."
                )
            # CMake / linker / ASN.1 wiring: keep a dedicated path so Phase 2/3 do not follow
            # compile-only retrieval when the log is actually a link or CMake failure.
            elif self._should_force_cmake_log_kind(error_message, log_content) and classified_log_error_kind != "runtime":
                log_error_kind = "cmake"
                logger.info(
                    "📌 Log error kind set to cmake (CMake / linker / ASN.1 build-system signals; "
                    "does not override pure runtime crash classification)."
                )
            elif classified_log_error_kind in ("runtime", "other", "dependency") and has_compiler_diagnostics:
                log_error_kind = "build"
                logger.info(
                    "📌 Log error kind adjusted to build (compiler diagnostics parsed from log)"
                )
            else:
                log_error_kind = classified_log_error_kind

            deployment_context["log_error_kind_raw"] = classified_log_error_kind
            deployment_context["log_error_kind"] = log_error_kind
            deployment_context["build_log_mode"] = (log_error_kind == "build")
            deployment_context["runtime_log_mode"] = (log_error_kind == "runtime")
            deployment_context["other_log_mode"] = (log_error_kind == "other")
            # Never leak CMake mode from merged JSON/custom context into compile/runtime logs.
            deployment_context["cmake_build_system_mode"] = log_error_kind == "cmake"
            deployment_context["dependency_advice_mode"] = log_error_kind == "dependency"
            if log_error_kind == "dependency":
                max_dep = 48000
                deployment_context["full_dependency_log_text"] = (
                    log_content[-max_dep:] if len(log_content) > max_dep else log_content
                )

            if log_error_kind == "cmake":
                deployment_context["full_build_log_excerpt"] = (
                    log_content[-22000:] if len(log_content) > 22000 else log_content
                )
                try:
                    from .error_handling_pipeline import linker_derived_cmake_hints

                    deployment_context["linker_derived_hints"] = linker_derived_cmake_hints(
                        error_message, log_content
                    )
                    nq = len(
                        deployment_context["linker_derived_hints"].get(
                            "extra_search_queries", []
                        )
                    )
                    logger.info(
                        "🏗️ Linker-derived CMake hints: %s ASN type(s), %s extra FAISS queries",
                        len(
                            deployment_context["linker_derived_hints"].get(
                                "asn_def_types", []
                            )
                        ),
                        nq,
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ linker_derived_cmake_hints failed (continuing without): %s", e
                    )
                logger.info(
                    "🏗️ CMake/build-system mode: skipping function retrieval; "
                    "using full log + CMake embeddings for fixes"
                )
            
            build_error_mode = bool(
                log_error_kind not in ("dependency",)
                and (log_error_kind in ("build", "cmake") or compiler_ground_truth_errors)
            )
            compiler_cluster_analysis = (
                self.log_parser.cluster_and_classify_compiler_errors(
                    compiler_ground_truth_errors
                )
                if compiler_ground_truth_errors
                else {}
            )
            if build_error_mode:
                deployment_context = deployment_context or {}
                deployment_context["compiler_cluster_analysis"] = compiler_cluster_analysis
                deployment_context["build_error_mode"] = True

            # Step 2.4: Candidate retrieval — CMake/build-system path OR legacy context-aware OR standard
            narrow_compiler_phase2 = False
            cmake_build_mode = bool(deployment_context.get("cmake_build_system_mode"))
            use_rich_context = log_was_parsed or pre_classify_had_context

            if deployment_context.get("dependency_advice_mode"):
                logger.info("\n🔍 STEP 2.4 - DEPENDENCY / TOOLCHAIN (retrieval skipped)")
                logger.info("-" * 50)
                phase2_results = {
                    "suspected_functions": [],
                    "suspected_configs": [],
                    "deployment_context": deployment_context,
                    "retrieval_method": "dependency_advice_skipped",
                }
            elif cmake_build_mode:
                logger.info("\n🔍 STEP 2.4 - CMAKE / BUILD-SYSTEM RETRIEVAL (config embeddings only)")
                logger.info("-" * 50)
                context_results = self.phase2_pipeline.retrieve_cmake_build_system_candidates(
                    error_text=error_message,
                    log_content=log_content,
                    deployment_context=deployment_context,
                    top_k=20,
                )
                phase2_results = {
                    "suspected_functions": [],
                    "suspected_configs": [
                        {
                            "param_name": config.get("candidate", config).get(
                                "param", config.get("candidate", config).get("param_name", "Unknown")
                            )
                            if isinstance(config.get("candidate", config), dict)
                            else "Unknown",
                            "file_path": config.get("candidate", config).get("file_path", "Unknown")
                            if isinstance(config.get("candidate", config), dict)
                            else "Unknown",
                            "param_value": config.get("candidate", config).get(
                                "value", config.get("candidate", config).get("param_value", "Unknown")
                            )
                            if isinstance(config.get("candidate", config), dict)
                            else "Unknown",
                            "relevance_score": config.get("relevance_score", config.get("score", 0.0)),
                            "source": self._extract_source(config),
                            "config_context": config.get("candidate", config).get("config_context", "")
                            if isinstance(config.get("candidate", config), dict)
                            else "",
                            "line_number": config.get("candidate", config).get("line_number", "Unknown")
                            if isinstance(config.get("candidate", config), dict)
                            else "Unknown",
                            "reason": config.get("candidate", config).get("reason", "No reason provided")
                            if isinstance(config.get("candidate", config), dict)
                            else "No reason provided",
                        }
                        for config in context_results.get("configs", [])
                        if isinstance(config, dict)
                    ],
                    "deployment_context": deployment_context,
                    "retrieval_method": "cmake_build_system_embeddings",
                }
                logger.info(
                    f"✅ CMake retrieval: {len(phase2_results['suspected_configs'])} config/build rows"
                )
            elif use_rich_context:
                logger.info(f"\n🔍 STEP 2.4 - CONTEXT-AWARE CANDIDATE RETRIEVAL")
                logger.info("-" * 50)

                if build_error_mode and not cmake_build_mode and log_error_kind == "build":
                    primary_funcs = self._build_candidates_from_primary_compiler_clusters(
                        compiler_cluster_analysis
                    )
                    if primary_funcs:
                        narrow_compiler_phase2 = True
                        logger.info(
                            "🏗️ Build-error mode: using primary compiler clusters only "
                            "(skipping semantic / CRAG retrieval for functions)."
                        )
                        phase2_results = {
                            "suspected_functions": primary_funcs,
                            "suspected_configs": [],
                            "deployment_context": deployment_context,
                            "retrieval_method": "compiler_ground_truth_primary_only",
                        }
                        logger.info(
                            f"✅ Primary-only candidates: {len(primary_funcs)} function(s)"
                        )
                    else:
                        logger.warning(
                            "🏗️ Build-error mode but no function-tagged primary errors; "
                            "falling back to context-aware retrieval."
                        )

                if not narrow_compiler_phase2:
                    context_results = self.phase2_pipeline.retrieve_candidates_with_context(
                        error_text=error_message,
                        deployment_context=deployment_context,
                        top_k=10,
                    )

                    logger.info(f"✅ Context-aware retrieval completed:")
                    logger.info(f"   Functions found: {len(context_results.get('functions', []))}")
                    logger.info(f"   Configs found: {len(context_results.get('configs', []))}")

                    phase2_results = {
                        "suspected_functions": [
                            {
                                "function_name": self._extract_function_name(func),
                                "file_path": self._extract_file_path(func),
                                "relevance_score": self._extract_relevance_score(func),
                                "source": self._extract_source(func),
                                "code_snippet": self._extract_code_snippet(func),
                                "reason": self._extract_grade_reason(func),
                            }
                            for func in context_results.get("functions", [])
                            if isinstance(func, dict)
                        ],
                        "suspected_configs": [
                            {
                                "param_name": config.get("candidate", config).get(
                                    "param", config.get("candidate", config).get("param_name", "Unknown")
                                )
                                if isinstance(config.get("candidate", config), dict)
                                else "Unknown",
                                "file_path": config.get("candidate", config).get("file_path", "Unknown")
                                if isinstance(config.get("candidate", config), dict)
                                else "Unknown",
                                "param_value": config.get("candidate", config).get(
                                    "value", config.get("candidate", config).get("param_value", "Unknown")
                                )
                                if isinstance(config.get("candidate", config), dict)
                                else "Unknown",
                                "relevance_score": config.get("relevance_score", config.get("score", 0.0)),
                                "source": self._extract_source(config),
                                "config_context": config.get("candidate", config).get("config_context", "")
                                if isinstance(config.get("candidate", config), dict)
                                else "",
                                "line_number": config.get("candidate", config).get("line_number", "Unknown")
                                if isinstance(config.get("candidate", config), dict)
                                else "Unknown",
                                "reason": config.get("candidate", config).get("reason", "No reason provided")
                                if isinstance(config.get("candidate", config), dict)
                                else "No reason provided",
                            }
                            for config in context_results.get("configs", [])
                            if isinstance(config, dict)
                        ],
                        "deployment_context": deployment_context,
                        "retrieval_method": "context_aware",
                    }
            else:
                logger.info(f"\n🔍 PHASE 2 - STANDARD ERROR ANALYSIS")
                logger.info("-" * 50)

                phase2_results = self.phase2_pipeline.process_error(error_message)
            
            logger.info(f"✅ Phase 2 completed:")
            logger.info(f"   Functions found: {len(phase2_results.get('suspected_functions', []))}")
            logger.info(f"   Configs found: {len(phase2_results.get('suspected_configs', []))}")
            
            # Extract candidate functions and configs
            candidate_functions = phase2_results.get('suspected_functions', [])
            candidate_configs = phase2_results.get('suspected_configs', [])

            # Highest-priority injection from direct compiler errors (legacy path only).
            if not narrow_compiler_phase2:
                candidate_functions = self._inject_ground_truth_priority(
                    candidate_functions,
                    compiler_ground_truth_errors
                )
                phase2_results['suspected_functions'] = candidate_functions

            phase2_results['ground_truth_errors'] = compiler_ground_truth_errors
            phase2_results['compiler_cluster_analysis'] = compiler_cluster_analysis
            phase2_results['build_error_mode'] = build_error_mode
            phase2_results['narrow_compiler_phase2'] = narrow_compiler_phase2
            phase2_results['log_error_kind'] = log_error_kind
            phase2_results['top_ground_truth_clusters'] = self._collect_ground_truth_clusters(compiler_ground_truth_errors)
            phase2_results['coverage_stats'] = {
                "total_compiler_errors_parsed": len(compiler_ground_truth_errors),
                "ground_truth_with_function_name": len([e for e in compiler_ground_truth_errors if isinstance(e, dict) and e.get("function_name")]),
                "ground_truth_with_file_path": len([e for e in compiler_ground_truth_errors if isinstance(e, dict) and e.get("file_path")])
            }
            
            # Load call graph context for the candidate functions
            call_graph_context = load_call_graph_context(candidate_functions)
            
            # Add downstream + KG expansion only when not using narrow primary-only Phase 2
            # or CMake/build-system mode (no function-based retrieval).
            if not narrow_compiler_phase2 and not deployment_context.get(
                "cmake_build_system_mode"
            ) and not deployment_context.get("dependency_advice_mode"):
                downstream_functions = self._extract_downstream_functions_from_call_graph(
                    candidate_functions, call_graph_context
                )
                existing_names = {f.get('function_name', '') for f in candidate_functions if isinstance(f, dict)}
                for downstream_func in downstream_functions:
                    if downstream_func.get('function_name', '') not in existing_names:
                        candidate_functions.append(downstream_func)
                        logger.info(f"✅ Added downstream function to Phase 2: {downstream_func.get('function_name', '')}")
                
                phase2_results['suspected_functions'] = candidate_functions

                try:
                    kg_functions = self._kg_expand_functions(
                        candidate_functions,
                        max_iters=2,
                        max_new=25,
                    )
                    if kg_functions:
                        kg_existing_names = {f.get('function_name', '') for f in candidate_functions if isinstance(f, dict)}
                        kg_added = 0
                        kg_added_sample = []
                        for kg_func in kg_functions:
                            if kg_func.get('function_name', '') not in kg_existing_names:
                                candidate_functions.append(kg_func)
                                kg_existing_names.add(kg_func.get('function_name', ''))
                                kg_added += 1
                                if len(kg_added_sample) < 5:
                                    kg_added_sample.append(kg_func.get('function_name', ''))
                                logger.info(f"✅ Added KG-expanded function to Phase 2: {kg_func.get('function_name', '')}")

                        phase2_results['suspected_functions'] = candidate_functions
                        call_graph_context = load_call_graph_context(candidate_functions)
                        logger.info(
                            f"📈 KG expansion summary: added {kg_added} new functions "
                            f"(sample: {', '.join([n for n in kg_added_sample if n])}{'...' if kg_added > len(kg_added_sample) else ''})"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ KG expansion failed (skipping KG enhancement): {e}")
            else:
                phase2_results['suspected_functions'] = candidate_functions
            
            # Save Phase 2 results (now includes downstream functions)
            phase2_output_file = "output/phase2_results.json"
            with open(phase2_output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "error_message": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "log_file": log_file_path,
                    "phase2_results": phase2_results,
                    "summary": {
                        "total_functions_analyzed": len(phase2_results.get('suspected_functions', [])),
                        "total_configs_analyzed": len(phase2_results.get('suspected_configs', [])),
                        "deployment_context_used": deployment_context is not None,
                        "retrieval_method": phase2_results.get('retrieval_method', 'standard')
                    }
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Phase 2 results saved to: {phase2_output_file}")
            
            # Phase 3: Fix Suggestions
            logger.info(f"\n🔧 PHASE 3 - FIX SUGGESTIONS")
            logger.info("-" * 50)
            
            phase3_error_text = error_message
            gt_append = self._phase3_compiler_ground_truth_block(
                compiler_ground_truth_errors,
                compiler_cluster_analysis,
                narrow_compiler_phase2,
            )
            if gt_append:
                phase3_error_text = f"{error_message}{gt_append}"

            batch_sz = _COMPILER_GROUND_TRUTH_PHASE3_BATCH_SIZE
            use_compiler_phase3_batches = (
                narrow_compiler_phase2
                and len(candidate_functions) > batch_sz
            )

            if use_compiler_phase3_batches:
                n_total = len(candidate_functions)
                n_batches = (n_total + batch_sz - 1) // batch_sz
                batch_results = []
                for bi in range(0, n_total, batch_sz):
                    batch_funcs = candidate_functions[bi : bi + batch_sz]
                    b_num = bi // batch_sz + 1
                    batch_suffix = (
                        f"\n\n[Compiler-ground-truth Phase 3: batch {b_num}/{n_batches} — "
                        f"generate code patches for every candidate function in this batch "
                        f"({len(batch_funcs)} function(s)); same compiler failure context as above.]"
                    )
                    fnames = {
                        f.get("function_name")
                        for f in batch_funcs
                        if isinstance(f, dict) and f.get("function_name")
                    }
                    cg_batch = self._call_graph_subset_for_functions(
                        call_graph_context, fnames
                    )
                    logger.info(
                        f"🔧 Phase 3 compiler batch {b_num}/{n_batches}: "
                        f"{len(batch_funcs)} function(s)"
                    )
                    batch_results.append(
                        self.phase3_pipeline.process_fix_request(
                            error=phase3_error_text + batch_suffix,
                            candidate_functions=batch_funcs,
                            candidate_configs=candidate_configs,
                            call_graph_context=cg_batch,
                            matched_pattern=None,
                            deployment_context=deployment_context,
                        )
                    )
                phase3_results = self._merge_phase3_compiler_batch_results(
                    batch_results,
                    phase3_error_text,
                    n_total,
                )
            else:
                phase3_results = self.phase3_pipeline.process_fix_request(
                    error=phase3_error_text,
                    candidate_functions=candidate_functions,
                    candidate_configs=candidate_configs,
                    call_graph_context=call_graph_context,
                    matched_pattern=None,
                    deployment_context=deployment_context,
                )

            logger.info("✅ Phase 3 completed:")
            fix_suggestion = phase3_results.get('fix_suggestion', {})
            pb = phase3_results.get("phase3_batching")
            if isinstance(pb, dict) and pb.get("enabled"):
                logger.info(
                    f"   Compiler-ground-truth batches: {pb.get('batch_count')} "
                    f"(≤{pb.get('batch_size')} functions per LLM call)"
                )
            logger.info(f"   Suspected functions: {len(fix_suggestion.get('suspected_functions', []))}")
            logger.info(f"   Suspected configs: {len(fix_suggestion.get('suspected_configs', []))}")
            
            # Combine results
            complete_results = {
                "error_message": error_message,
                "extracted_error": extracted_error is not None,
                "error_source": "extracted_from_log" if extracted_error else "user_provided",
                "timestamp": datetime.now().isoformat(),
                "log_file": log_file_path,
                "deployment_context": deployment_context,
                "log_error_kind": log_error_kind,
                "ground_truth_errors": compiler_ground_truth_errors,
                "phase2_analysis": phase2_results,
                "phase3_fixes": phase3_results,
                "summary": {
                    "log_error_kind": log_error_kind,
                    "dependency_advice_mode": bool(
                        deployment_context.get("dependency_advice_mode")
                    ),
                    "total_functions_analyzed": len(candidate_functions),
                    "total_configs_analyzed": len(candidate_configs),
                    "call_graph_entries": len(call_graph_context),
                    "fix_suggestions_generated": bool(fix_suggestion.get('reason')),
                    "context_aware": deployment_context is not None,
                    "missing_configs_resolved": len([c for c in phase2_results.get('suspected_configs', []) 
                                                   if isinstance(c, dict) and 'missing' in c]) if deployment_context else 0
                }
            }
            
            logger.info("\nCOMPLETE PIPELINE FINISHED")
            logger.info("=" * 70)
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ pipeline failed: {e}")
            return {
                "error_message": error_message,
                "log_file": log_file_path,
                "pipeline_error": str(e),
                "deployment_context": None,
                "phase2_analysis": {},
                "phase3_fixes": {},
                "summary": {"error": str(e)}
            }

def display_results(results: dict):
    """Display results in a user-friendly format"""
    print("\n" + "=" * 90)
    print("📊 COMPLETE ERROR FIXING RESULTS")
    print("=" * 90)
    
    error_msg = results.get('error_message', 'Unknown')
    log_file = results.get('log_file', 'None')
    print(f"🔥 Original Error: {error_msg}")
    print(f"📁 Log File: {log_file}")
    
    # Deployment Context Summary
    deployment_context = results.get('deployment_context')
    if deployment_context:
        print(f"\n🌐 DEPLOYMENT CONTEXT:")
        print(f"   🎭 Role: {deployment_context.get('role', 'Unknown')}")
        print(f"   📋 Active Configs: {len(deployment_context.get('active_configs', []))}")
        print(f"   🔗 Log Anchors: {len(deployment_context.get('log_anchors', []))}")
        network_params = deployment_context.get('network_params', {})
        print(f"   🌍 Network: gNB={network_params.get('gnb_ipv4')}, AMF={network_params.get('amf_ipv4')}")
        print(f"   📡 Association: {network_params.get('assoc_state', 'Unknown')}")
    else:
        print(f"\n⚠️  No deployment context available")
    
    # Phase 2 Summary
    phase2 = results.get('phase2_analysis', {})
    print(f"\n🔍 PHASE 2 - ERROR ANALYSIS:")
    print(f"   🔄 Retrieval Method: {phase2.get('retrieval_method', 'standard')}")
    print(f"   📊 Total Candidates: {phase2.get('total_candidates_retrieved', 0)}")
    
    functions = phase2.get('suspected_functions', [])
    configs = phase2.get('suspected_configs', [])
    print(f"   🔧 Functions Found: {len(functions)}")
    print(f"   ⚙️  Configs Found: {len(configs)}")
    
    # Show top functions
    if functions:
        print(f"\n   Top Functions:")
        for i, func in enumerate(functions[:3], 1):
            # Type safety: skip non-dict items
            if not isinstance(func, dict):
                continue
            name = func.get('function_name', 'Unknown')
            score = func.get('relevance_score', 0)
            source = func.get('source', 'unknown')
            print(f"     {i}. {name} (score: {score:.2f}, source: {source})")
    
    # Show top configs
    if configs:
        print(f"\n   Top Configs:")
        for i, config in enumerate(configs[:3], 1):
            # Type safety: skip non-dict items
            if not isinstance(config, dict):
                continue
            param = config.get('param_name', 'Unknown')
            value = config.get('param_value', 'Unknown')
            score = config.get('relevance_score', 0)
            source = config.get('source', 'unknown')
            print(f"     {i}. {param} = {value[:30]}... (score: {score:.2f}, source: {source})")
    
    # Phase 3 Summary
    phase3 = results.get('phase3_fixes', {})
    fix_suggestion = phase3.get('fix_suggestion', {})
    
    print(f"\n🔧 PHASE 3 - FIX SUGGESTIONS:")
    print(f"   🎯 Suspected Functions: {fix_suggestion.get('suspected_functions', [])}")
    print(f"   ⚙️  Suspected Configs: {fix_suggestion.get('suspected_configs', [])}")
    
    # Root Cause
    reason = fix_suggestion.get('reason', '')
    if reason:
        print(f"\n💡 Root Cause Analysis:")
        print(f"   {reason[:300]}{'...' if len(reason) > 300 else ''}")
    
    # Config Fix
    config_fix = fix_suggestion.get('config_fix', '')
    if config_fix:
        print(f"\n🔧 Configuration Fix:")
        print(f"   {config_fix[:400]}{'...' if len(config_fix) > 400 else ''}")
    
    # Code Patch
    code_patch = fix_suggestion.get('code_patch', '')
    if code_patch:
        print(f"\n💻 Code Patch:")
        print(f"   {code_patch[:400]}{'...' if len(code_patch) > 400 else ''}")
    
    # Investigation Steps
    steps = fix_suggestion.get('investigation_steps', [])
    if steps:
        print(f"\n📋 Investigation Steps:")
        for i, step in enumerate(steps[:5], 1):
            print(f"   {i}. {step}")

def generate_terminal_commands(error_message: str, investigation_steps: list, deployment_context: dict = None, troubleshooting_hints: list = None, openair_codebase_file_name: str = "openairinterface5g-develop") -> list:
    """
    Phase 4: Generate exact terminal commands for verification and implementation
    
    Args:
        error_message: The original error
        investigation_steps: List of investigation steps from Phase 3
        deployment_context: Deployment context with IP addresses
        troubleshooting_hints: Troubleshooting hints from error patterns
        openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        
    Returns:
        List of exact terminal commands
    """
    logger.info("🔧 PHASE 4 - GENERATING TERMINAL COMMANDS")
    logger.info("=" * 60)
    
    try:
        # Dependency / toolchain path: Phase 3 already lists concrete shell commands.
        if deployment_context and deployment_context.get("dependency_advice_mode"):
            cmds = []
            for step in investigation_steps or []:
                if not isinstance(step, str):
                    continue
                st = step.strip()
                if st.startswith("$"):
                    cmds.append(st[1:].strip())
                elif st.startswith(">"):
                    cmds.append(st[1:].strip())
            if cmds:
                logger.info(
                    "🔧 PHASE 4 - dependency-advice mode: using commands from Phase 3 (skipping LLM)"
                )
                return cmds

        # Prepare context for command generation
        context_parts = []
        
        # Error information
        context_parts.append(f"Error: {error_message}")
        
        # Deployment context (IP addresses, network info)
        # Always try to get from JSON file first (has real IP addresses), then fallback to deployment context
        network_config = {}
        try:
            with open('database/error_patterns_structured.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                deployment_context_json = data.get('deployment_context', {})
                network_config = {
                    'gNB IP': deployment_context_json.get('cu_ip_address', 'Unknown'),
                    'DU IP': deployment_context_json.get('du_ip_address', 'Unknown'),
                    'AMF IP': deployment_context_json.get('amf_ip_address', 'Unknown'),
                    'Core Network IP': deployment_context_json.get('core_network_machine_ip', 'Unknown'),
                    'Local SCTP Port': deployment_context_json.get('local_s_portc', 'Unknown'),
                    'Remote SCTP Port': deployment_context_json.get('remote_s_portc', 'Unknown')
                }
        except Exception as e:
            logger.warning(f"Could not load deployment context from JSON: {e}")
            # Fallback to deployment context from pipeline
            if deployment_context:
                network_params = deployment_context.get('network_params', {})
                network_config = {
                    'gNB IP': network_params.get('gnb_ipv4', 'Unknown'),
                    'AMF IP': network_params.get('amf_ipv4', 'Unknown'),
                    'Core Network IP': network_params.get('core_network_ip', 'Unknown'),
                    'Local SCTP Port': network_params.get('local_s_portc', 'Unknown'),
                    'Remote SCTP Port': network_params.get('remote_s_portc', 'Unknown')
                }
        
        if network_config:
            context_parts.append(f"Network Configuration:")
            for key, value in network_config.items():
                context_parts.append(f"- {key}: {value}")
        
        # Investigation steps
        context_parts.append(f"Investigation Steps to Convert:")
        for i, step in enumerate(investigation_steps, 1):
            context_parts.append(f"{i}. {step}")
        
        # Troubleshooting hints
        if troubleshooting_hints:
            context_parts.append(f"Troubleshooting Hints:")
            for hint in troubleshooting_hints:
                context_parts.append(f"- {hint}")
        
        context = "\n".join(context_parts)
        
        # Error-specific prompt for command generation
        prompt = f"""You are a 5G/LTE network troubleshooting expert. Given the error and context below, generate ONLY the ESSENTIAL terminal commands needed to verify and fix the issue.

{context}

IMPORTANT: Generate ONLY 2-3 essential commands, not a comprehensive list. 

For RA (Random Access) timer errors, focus on:
1. ONE command to check/verify the configuration file
2. ONE command to test network connectivity (if applicable)

For AMF association errors, focus on:
1. ONE connectivity test command (ping to AMF)
2. ONE routing command (if applicable)

Requirements:
- Provide EXACT commands that can be copy-pasted
- Use the specific IP addresses from the context (if available)
- Keep it minimal - only the most critical commands
- Format each command on a new line starting with "COMMAND: "
- Add brief explanation after each command
- If IP addresses are not available, use placeholder format like <AMF_IP> or <GNB_IP>

Example format for RA timer error:
COMMAND: grep -n "ra_ContentionResolutionTimer" du_gnb.conf
EXPLANATION: Check current value of contention resolution timer in gNB config

COMMAND: ping <GNB_IP>
EXPLANATION: Test connectivity to gNB (replace <GNB_IP> with actual IP)

Example format for AMF error:
COMMAND: ping <AMF_IP>
EXPLANATION: Test basic connectivity to AMF

COMMAND: ip route add <AMF_IP> via <CORE_NETWORK_IP> dev eth0
EXPLANATION: Add static route for AMF through core network machine
"""

        # Use the same Azure client from phase3_pipeline
        from .fix_suggestion_pipeline import FixSuggestionPipeline
        phase3_pipeline = FixSuggestionPipeline(openair_codebase_file_name=openair_codebase_file_name)
        
        response = phase3_pipeline.azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a network troubleshooting expert who generates exact terminal commands."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1,
            seed=33333  # Terminal command generation seed
        )
        
        commands_text = response.choices[0].message.content.strip()
        
        # Parse commands from response
        commands = []
        current_command = None
        current_explanation = None
        
        for line in commands_text.split('\n'):
            line = line.strip()
            if line.startswith('COMMAND:'):
                # Save previous command if exists
                if current_command:
                    commands.append({
                        "command": current_command,
                        "explanation": current_explanation or "No explanation provided"
                    })
                current_command = line.replace('COMMAND:', '').strip()
                current_explanation = None
            elif line.startswith('EXPLANATION:'):
                current_explanation = line.replace('EXPLANATION:', '').strip()
            elif line and not line.startswith('COMMAND:') and not line.startswith('EXPLANATION:'):
                # Only treat as command if it looks like a command (starts with common command words)
                if (current_command is None and line and 
                    any(line.startswith(cmd) for cmd in ['grep', 'ping', 'ip route', 'systemctl', 'cat', 'ls', 'cd', 'sudo', 'ssh', 'telnet', 'traceroute', 'netstat', 'ifconfig', 'iptables'])):
                    current_command = line
                elif current_explanation is None and line and not any(line.startswith(cmd) for cmd in ['grep', 'ping', 'ip route', 'systemctl', 'cat', 'ls', 'cd', 'sudo', 'ssh', 'telnet', 'traceroute', 'netstat', 'ifconfig', 'iptables']):
                    current_explanation = line
        
        # Add the last command
        if current_command:
            commands.append({
                "command": current_command,
                "explanation": current_explanation or "No explanation provided"
            })
        
        logger.info(f"✅ Generated {len(commands)} terminal commands")
        return commands
        
    except Exception as e:
        logger.error(f"❌ Failed to generate terminal commands: {e}")
        return []

def main():
    """Main function to run the complete pipeline"""
    print("🚀 Complete Error Fixing Pipeline")
    print("=" * 60)
    
    # Configuration variables - Update these as needed
    # error_message = "Contention resolution timer has expired, RA procedure has failed..."
    # log_file_path = "log_files/ue.log"  # Set to None if no log file available
    
    # error_message = "No AMF associated to the gNB"
    # log_file_path = "log_files/cu_ngap_failure.log"
    
    # Option 2: Let the pipeline extract error from log file automatically
    error_message = None  # Set to None to auto-extract from log
    log_file_path = "log_files/cu_rrc_segmentation_error.log"
    
    # 🔧 DYNAMIC FOLDER NAME - Change this to match your OAI codebase folder
    openair_codebase_file_name = "openairinterface5g-develop"
    
    # Option 3: Provide only error message (no log context)
    # error_message = "Contention resolution timer has expired, RA procedure has failed..."
    # log_file_path = None
    

    print(f"📝 Configuration:")
    print(f"   Error Message: {error_message if error_message else 'Auto-extract from log'}")
    print(f"   Log File: {log_file_path if log_file_path else 'None'}")
    print(f"   OAI Codebase Folder: {openair_codebase_file_name}")
    print()
    
    try:
        # Initialize and run pipeline with dynamic folder name
        pipeline = CompleteErrorFixingPipeline(openair_codebase_file_name=openair_codebase_file_name)
        results = pipeline.process_error_with_context(error_message, log_file_path)
        
        # Display results
        display_results(results)
        
        # Phase 4: Generate terminal commands
        print(f"\n🔧 PHASE 4 - GENERATING TERMINAL COMMANDS")
        print("=" * 60)
        
        # Extract data for command generation
        phase3_fixes = results.get('phase3_fixes', {})
        fix_suggestion = phase3_fixes.get('fix_suggestion', {})
        investigation_steps = fix_suggestion.get('investigation_steps', [])
        deployment_context = results.get('deployment_context')
        
        # Get troubleshooting hints from error patterns
        troubleshooting_hints = []
        try:
            with open('database/error_patterns_structured.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                patterns = data.get('patterns', {})
                error_lower = error_message.lower()
                
                # Find matching pattern
                pattern_found = False
                for pattern_name, pattern_data in patterns.items():
                    keywords = pattern_data.get('keywords', [])
                    if any(keyword in error_lower for keyword in keywords):
                        # Get troubleshooting hints from suggested_fixes
                        suggested_fixes = pattern_data.get('suggested_fixes', [])
                        troubleshooting_hints.extend(suggested_fixes)
                        pattern_found = True
                        break
                
                # If no pattern matches, generate dynamic pattern
                if not pattern_found:
                    logger.info(f"🔄 No pattern found for Phase 4, generating dynamic pattern for: {error_message}")
                    from .fix_suggestion_pipeline import FixSuggestionPipeline
                    fix_pipeline = FixSuggestionPipeline(openair_codebase_file_name=openair_codebase_file_name)
                    dynamic_pattern = fix_pipeline._generate_dynamic_error_pattern(error_message)
                    fix_pipeline._add_pattern_to_json(error_message, dynamic_pattern)
                    
                    # Use the generated pattern
                    suggested_fixes = dynamic_pattern.get('suggested_fixes', [])
                    troubleshooting_hints.extend(suggested_fixes)
                    
        except Exception as e:
            logger.warning(f"Could not load troubleshooting hints: {e}")
            # Fallback to default hints
            troubleshooting_hints = [
                "Validate network configuration and parameters in config files",
                "Check network reachability between endpoints",
                "Verify protocol-specific configuration settings",
                "Review error logs for additional context"
            ]
        
        # Generate terminal commands
        terminal_commands = generate_terminal_commands(
            error_message=error_message,
            investigation_steps=investigation_steps,
            deployment_context=deployment_context,
            troubleshooting_hints=troubleshooting_hints,
            openair_codebase_file_name=openair_codebase_file_name
        )
        
        # Add commands to results
        results['phase4_commands'] = {
            "terminal_commands": terminal_commands,
            "command_count": len(terminal_commands)
        }
        
        # Display commands
        if terminal_commands:
            print(f"\n💻 Generated Terminal Commands ({len(terminal_commands)} commands):")
            for i, cmd in enumerate(terminal_commands, 1):
                print(f"\n   {i}. {cmd['command']}")
                print(f"      💡 {cmd['explanation']}")
        else:
            print(f"\n⚠️  No terminal commands generated")
        
        # Save complete results
        output_file = "output/complete_error_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete results saved to: {output_file}")
        print(f"📋 Phase 2 results saved to: output/phase2_results.json")
        
        # Save fix suggestions separately (including terminal commands)
        fix_suggestions_file = "output/fix_suggestions.json"
        fix_suggestions_data = results.get('phase3_fixes', {}).copy()
        fix_suggestions_data['terminal_commands'] = results.get('phase4_commands', {})
        
        with open(fix_suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(fix_suggestions_data, f, indent=2, ensure_ascii=False)
        
        print(f"🔧 Fix suggestions saved to: {fix_suggestions_file}")
        
        # Save deployment context if available
        if results.get('deployment_context'):
            context_file = "output/deployment_context.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(results['deployment_context'], f, indent=2, ensure_ascii=False)
            print(f"🌐 Deployment context saved to: {context_file}")
        
        # Save summary report
        summary_file = "output/error_fix_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Error Fix Summary Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Error: {error_message}\n")
            f.write(f"Log File: {log_file_path or 'None'}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Deployment context
            if results.get('deployment_context'):
                ctx = results['deployment_context']
                f.write(f"Deployment Context:\n")
                f.write(f"- Role: {ctx.get('role', 'Unknown')}\n")
                f.write(f"- Active Configs: {len(ctx.get('active_configs', []))}\n")
                f.write(f"- Network: gNB={ctx.get('network_params', {}).get('gnb_ipv4')}, AMF={ctx.get('network_params', {}).get('amf_ipv4')}\n\n")
            
            # Phase 2
            phase2 = results.get('phase2_analysis', {})
            f.write(f"Phase 2 Results:\n")
            f.write(f"- Retrieval Method: {phase2.get('retrieval_method', 'standard')}\n")
            f.write(f"- Functions: {len(phase2.get('suspected_functions', []))}\n")
            f.write(f"- Configs: {len(phase2.get('suspected_configs', []))}\n\n")
            
            # Phase 3
            phase3 = results.get('phase3_fixes', {})
            fix_suggestion = phase3.get('fix_suggestion', {})
            f.write(f"Phase 3 Results:\n")
            f.write(f"- Root Cause: {fix_suggestion.get('reason', 'Not provided')[:200]}...\n")
            f.write(f"- Fix Available: {'Yes' if fix_suggestion.get('config_fix') or fix_suggestion.get('code_patch') else 'No'}\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
