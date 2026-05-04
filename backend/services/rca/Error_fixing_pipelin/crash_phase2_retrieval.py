#!/usr/bin/env python3
"""
Crash Analysis - Phase 2: Targeted Candidate Retrieval

This is a specialized Phase 2 for crash/segmentation fault analysis.
Unlike the regular Phase 2 (error_handling_pipeline.py), this:
- Uses backtrace functions as highest priority seeds
- Focuses call chain analysis from crash point
- Skips broad CRAG/FAISS search
- Focuses on validation/NULL check patterns
- Minimizes config search
- Searches git commits for similar crash fixes
- Reads full source code for backtrace functions

Author: AI Assistant
"""

import os
import json
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrashPhase2Retrieval:
    """Crash-specific Phase 2: Backtrace-driven candidate retrieval"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize crash-specific Phase 2 retrieval.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🔬 Initializing Crash Phase 2 Retrieval...")
        
        self.openair_codebase_file_name = openair_codebase_file_name
        self.codebase_path = f"Error_fixing_pipelin/{openair_codebase_file_name}"
        
        # Load function calls database for call chain analysis
        self._load_function_calls()
        
        # Load functions database for file path lookup
        self._load_functions_database()
        
        # Load git commit embeddings for crash pattern search
        self._load_git_embeddings()
        
        logger.info("✅ Crash Phase 2 Retrieval initialized")
    
    def _load_function_calls(self):
        """Load function calls database for call chain analysis"""
        try:
            if os.path.exists("database/function_calls.json"):
                with open("database/function_calls.json", 'r', encoding='utf-8') as f:
                    self.function_calls = json.load(f)
                logger.info(f"✅ Function calls loaded: {len(self.function_calls)} entries")
            else:
                logger.warning("⚠️  function_calls.json not found")
                self.function_calls = []
        except Exception as e:
            logger.error(f"Failed to load function calls: {e}")
            self.function_calls = []
    
    def _load_functions_database(self):
        """Load functions database for file path lookup"""
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
    
    def _lookup_function_file_path(self, function_name: str) -> Optional[str]:
        """
        Look up file path for a function by name from the functions database.
        
        Args:
            function_name: Name of the function to look up
            
        Returns:
            File path if found, None otherwise
        """
        if not self.functions_db:
            return None
        
        # Search for function by name (case-insensitive)
        for func_entry in self.functions_db:
            # Handle both dict and list formats
            if isinstance(func_entry, dict):
                entry_name = func_entry.get("function_name", "")
                if entry_name.lower() == function_name.lower():
                    file_path = func_entry.get("file_path", "")
                    if file_path:
                        # Normalize path separators (handle both / and \)
                        file_path = file_path.replace("\\", "/")
                        return file_path
        
        return None
    
    def _load_git_embeddings(self):
        """Load git commit embeddings for crash pattern search"""
        try:
            embeddings_dir = "resources/embeddings"
            
            if os.path.exists(os.path.join(embeddings_dir, 'git_commit_metadata.json')):
                with open(os.path.join(embeddings_dir, 'git_commit_metadata.json'), 'r') as f:
                    self.git_metadata = json.load(f)
                logger.info(f"✅ Git commit metadata loaded: {len(self.git_metadata)} commits")
            else:
                logger.warning("⚠️  Git commit metadata not found")
                self.git_metadata = []
        except Exception as e:
            logger.error(f"Failed to load git embeddings: {e}")
            self.git_metadata = []
    
    def process_crash_retrieval(self, crash_extraction_file: str) -> Dict[str, Any]:
        """
        Process Phase 2 retrieval for crash analysis.
        
        Args:
            crash_extraction_file: Path to Phase 1 extraction JSON file
            
        Returns:
            Prioritized candidates for crash analysis
        """
        logger.info("=" * 80)
        logger.info("🔬 CRASH ANALYSIS - PHASE 2: TARGETED CANDIDATE RETRIEVAL")
        logger.info("=" * 80)
        
        # Load Phase 1 extraction results
        if not os.path.exists(crash_extraction_file):
            logger.error(f"Phase 1 extraction file not found: {crash_extraction_file}")
            return {"error": "Phase 1 extraction file not found"}
        
        with open(crash_extraction_file, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
        
        crash_info = phase1_data.get("crash_info", {})
        
        logger.info(f"📥 Loaded Phase 1 data:")
        logger.info(f"   - Crash detected: {crash_info.get('crash_detected')}")
        logger.info(f"   - Backtrace frames: {len(crash_info.get('backtrace', []))}")
        logger.info(f"   - Faulting function: {crash_info.get('faulting_function')}")
        
        # Initialize result structure
        phase2_results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "crash_phase2_retrieval",
            "crash_info_summary": {
                "faulting_function": crash_info.get("faulting_function"),
                "fault_location": crash_info.get("fault_location"),
                "backtrace_frames": len(crash_info.get("backtrace", []))
            },
            "prioritized_functions": [],
            "call_chain_expansion": [],
            "validation_patterns": [],
            "similar_crash_fixes": [],
            "source_code_enrichment": {},
            "minimal_configs": []
        }
        
        # STEP 1: Backtrace Function Prioritization
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: BACKTRACE FUNCTION PRIORITIZATION")
        logger.info("=" * 60)
        backtrace_functions = self._prioritize_backtrace_functions(crash_info)
        phase2_results["prioritized_functions"] = backtrace_functions
        logger.info(f"✅ Prioritized {len(backtrace_functions)} backtrace functions")
        
        # STEP 2: Targeted Call Chain Expansion
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: TARGETED CALL CHAIN EXPANSION FROM CRASH POINT")
        logger.info("=" * 60)
        call_chain_expansion = self._expand_call_chain_from_crash(crash_info, backtrace_functions)
        phase2_results["call_chain_expansion"] = call_chain_expansion
        logger.info(f"✅ Expanded call chain: {len(call_chain_expansion)} additional functions")
        
        # STEP 3: Validation & Error Handling Search
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: VALIDATION & NULL CHECK PATTERN SEARCH")
        logger.info("=" * 60)
        validation_patterns = self._search_validation_patterns(crash_info, backtrace_functions)
        phase2_results["validation_patterns"] = validation_patterns
        logger.info(f"✅ Found {len(validation_patterns)} validation patterns")
        
        # STEP 4: Similar Crash Pattern Search in Git Commits
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: SIMILAR CRASH FIXES IN GIT COMMITS")
        logger.info("=" * 60)
        similar_crashes = self._search_similar_crash_fixes(crash_info)
        phase2_results["similar_crash_fixes"] = similar_crashes
        logger.info(f"✅ Found {len(similar_crashes)} similar crash fixes in git history")
        
        # STEP 5: Source Code Enrichment
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: SOURCE CODE ENRICHMENT FOR BACKTRACE FUNCTIONS")
        logger.info("=" * 60)
        source_enrichment = self._enrich_with_source_code(crash_info, backtrace_functions)
        phase2_results["source_code_enrichment"] = source_enrichment
        logger.info(f"✅ Enriched {len(source_enrichment)} functions with source code")
        
        # STEP 6: Minimal Config Search (only if initialization-related)
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: MINIMAL CONFIG SEARCH (OPTIONAL)")
        logger.info("=" * 60)
        minimal_configs = self._minimal_config_search(crash_info)
        phase2_results["minimal_configs"] = minimal_configs
        logger.info(f"✅ Found {len(minimal_configs)} potentially relevant configs")
        
        # Save Phase 2 results
        output_file = "output/crash_phase2_retrieval.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(phase2_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Phase 2 results saved to: {output_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ CRASH ANALYSIS PHASE 2 COMPLETED")
        logger.info("=" * 80)
        
        return phase2_results
    
    def _prioritize_backtrace_functions(self, crash_info: Dict) -> List[Dict]:
        """
        STEP 1: Assign priority scores to backtrace functions.
        Frame #0 (crash point) gets highest priority.
        """
        backtrace = crash_info.get("backtrace", [])
        prioritized = []
        
        for frame in backtrace:
            frame_num = frame.get("frame_number", 99)
            
            # Assign priority score based on frame number
            if frame_num == 0:
                priority_score = 1.00  # CRASH POINT - HIGHEST
            elif frame_num == 1:
                priority_score = 0.95  # IMMEDIATE CALLER
            elif frame_num == 2:
                priority_score = 0.90  # CALLER'S CALLER
            elif frame_num <= 5:
                priority_score = 0.85  # UPSTREAM CALLERS
            else:
                priority_score = 0.80  # DEEPER STACK
            
            # Normalize file path to relative format
            raw_file_path = frame.get("file")
            normalized_path = self._normalize_file_path(raw_file_path)
            
            prioritized.append({
                "function_name": frame.get("function"),
                "file_path": normalized_path,
                "line_number": frame.get("line"),
                "frame_number": frame_num,
                "priority_score": priority_score,
                "source": "backtrace",
                "role": self._classify_frame_role(frame_num)
            })
            
            logger.info(f"   Frame #{frame_num}: {frame.get('function')} (priority: {priority_score:.2f}) - {self._classify_frame_role(frame_num)}")
        
        return prioritized
    
    def _normalize_file_path(self, file_path: str) -> str:
        """
        Normalize file path to relative format.
        Converts absolute paths like /home/tcs/.../openairinterface5g/openair2/...
        to relative paths like openair2/...
        """
        if not file_path:
            return file_path
        
        # Extract relative path from absolute path
        oai_marker = "openairinterface5g/"
        if oai_marker in file_path:
            # Split at marker and take everything after
            relative_path = file_path.split(oai_marker, 1)[1]
            return relative_path
        
        # If already relative (no absolute prefix), return as is
        return file_path
    
    def _classify_frame_role(self, frame_num: int) -> str:
        """Classify the role of a stack frame"""
        if frame_num == 0:
            return "CRASH_POINT"
        elif frame_num == 1:
            return "IMMEDIATE_CALLER"
        elif frame_num == 2:
            return "UPSTREAM_CALLER"
        elif frame_num <= 5:
            return "CALL_CHAIN"
        else:
            return "DEEP_STACK"
    
    def _expand_call_chain_from_crash(self, crash_info: Dict, backtrace_functions: List[Dict]) -> List[Dict]:
        """
        STEP 2: Expand call chain from crash point (Frame #0 and Frame #1).
        Focus on what the crash function calls and who calls the caller.
        """
        expanded_functions = []
        
        # Get crash point function (Frame #0)
        crash_function = crash_info.get("faulting_function")
        
        if not crash_function:
            logger.warning("No crash function identified, skipping call chain expansion")
            return expanded_functions
        
        logger.info(f"🔍 Expanding call chain from crash function: {crash_function}")
        
        # Find what the crash function CALLS (might be null/invalid)
        downstream = self._find_downstream_calls(crash_function)
        for func in downstream:
            called_function = func.get("called_function")
            # Lookup file path from database
            file_path = self._lookup_function_file_path(called_function)
            
            expanded_functions.append({
                "function_name": called_function,
                "file_path": file_path if file_path else None,
                "priority_score": 0.85,
                "source": "crash_downstream",
                "reason": f"Called by crash function {crash_function} - potential null/invalid call"
            })
            
            if file_path:
                logger.debug(f"   ✓ Found file path for {called_function}: {file_path}")
            else:
                logger.debug(f"   ⚠ No file path found for {called_function}")
        
        logger.info(f"   ↓ Downstream: {len(downstream)} functions called by crash point")
        
        # Find who CALLS the crash function (for parameter validation)
        upstream = self._find_upstream_callers(crash_function)
        for func in upstream:
            caller_function = func.get("caller_function")
            # Skip if already in backtrace
            if not any(bt['function_name'] == caller_function for bt in backtrace_functions):
                # Lookup file path from database
                file_path = self._lookup_function_file_path(caller_function)
                
                expanded_functions.append({
                    "function_name": caller_function,
                    "file_path": file_path if file_path else None,
                    "priority_score": 0.80,
                    "source": "crash_upstream",
                    "reason": f"Calls crash function {crash_function} - check parameter passing"
                })
                
                if file_path:
                    logger.debug(f"   ✓ Found file path for {caller_function}: {file_path}")
                else:
                    logger.debug(f"   ⚠ No file path found for {caller_function}")
        
        logger.info(f"   ↑ Upstream: {len(upstream)} functions that call crash point")
        
        # Also expand from Frame #1 (immediate caller) if different from crash function
        if len(backtrace_functions) > 1:
            caller_function = backtrace_functions[1].get("function_name")
            if caller_function and caller_function != crash_function:
                logger.info(f"🔍 Expanding from immediate caller: {caller_function}")
                
                # Find what caller calls (to see parameter passing to crash function)
                caller_downstream = self._find_downstream_calls(caller_function)
                for func in caller_downstream:
                    called_function = func.get("called_function")
                    if called_function not in [crash_function, caller_function]:
                        # Lookup file path from database
                        file_path = self._lookup_function_file_path(called_function)
                        
                        expanded_functions.append({
                            "function_name": called_function,
                            "file_path": file_path if file_path else None,
                            "priority_score": 0.78,
                            "source": "caller_context",
                            "reason": f"Called by immediate caller {caller_function}"
                        })
                        
                        if file_path:
                            logger.debug(f"   ✓ Found file path for {called_function}: {file_path}")
                
                logger.info(f"   ↓ Caller downstream: {len(caller_downstream)} functions")
        
        return expanded_functions
    
    def _find_downstream_calls(self, function_name: str, depth: int = 1) -> List[Dict]:
        """Find functions called by the given function"""
        downstream = []
        
        for entry in self.function_calls:
            if entry.get("function") == function_name:
                calls = entry.get("calls", [])
                for called_func in calls[:10]:  # Limit to 10
                    downstream.append({
                        "called_function": called_func,
                        "caller": function_name
                    })
                break
        
        return downstream
    
    def _find_upstream_callers(self, function_name: str, depth: int = 1) -> List[Dict]:
        """Find functions that call the given function"""
        upstream = []
        
        for entry in self.function_calls:
            if entry.get("function") == function_name:
                called_by = entry.get("called_by", [])
                for caller_func in called_by[:10]:  # Limit to 10
                    upstream.append({
                        "caller_function": caller_func,
                        "called": function_name
                    })
                break
        
        return upstream
    
    def _search_validation_patterns(self, crash_info: Dict, backtrace_functions: List[Dict]) -> List[Dict]:
        """
        STEP 3: Search for validation and NULL check patterns in backtrace functions.
        """
        validation_patterns = []
        
        # Focus on top 3 backtrace functions
        top_functions = backtrace_functions[:3]
        
        for func_data in top_functions:
            function_name = func_data.get("function_name")
            file_path = func_data.get("file_path")
            
            logger.info(f"🔍 Searching validation patterns in: {function_name}")
            
            # Read the source file
            source_code = self._read_full_function_source(file_path, function_name)
            
            if not source_code:
                continue
            
            # Search for validation patterns
            patterns_found = {
                "function": function_name,
                "file": file_path,
                "has_null_checks": False,
                "null_check_count": 0,
                "has_error_handling": False,
                "error_handling_count": 0,
                "has_return_checks": False,
                "return_check_count": 0,
                "missing_validations": []
            }
            
            # Check for NULL pointer checks
            null_checks = re.findall(r'if\s*\(\s*(\w+)\s*==\s*NULL\s*\)|if\s*\(\s*!\s*(\w+)\s*\)', source_code)
            if null_checks:
                patterns_found["has_null_checks"] = True
                patterns_found["null_check_count"] = len(null_checks)
            
            # Check for error handling
            error_patterns = re.findall(r'if\s*\([^)]*<\s*0\)|if\s*\([^)]*==\s*-1\)|LOG_E\(|AssertFatal\(', source_code)
            if error_patterns:
                patterns_found["has_error_handling"] = True
                patterns_found["error_handling_count"] = len(error_patterns)
            
            # Check for return value checks
            return_checks = re.findall(r'if\s*\([^)]*\([^)]*\)\s*[!=]=\s*\w+\)', source_code)
            if return_checks:
                patterns_found["has_return_checks"] = True
                patterns_found["return_check_count"] = len(return_checks)
            
            # Identify missing validations (heuristic)
            # Look for pointer dereferences without prior NULL checks
            pointer_derefs = re.findall(r'(\w+)->(\w+)', source_code)
            if pointer_derefs:
                # Check if each pointer was validated
                for ptr_name, field in pointer_derefs[:5]:  # Check first 5
                    # Simple heuristic: if we don't see a NULL check for this pointer earlier
                    null_check_pattern = f"if.*{ptr_name}.*NULL|if.*!{ptr_name}"
                    if not re.search(null_check_pattern, source_code):
                        patterns_found["missing_validations"].append({
                            "pointer": ptr_name,
                            "field": field,
                            "issue": f"Pointer '{ptr_name}' dereferenced without NULL check"
                        })
            
            validation_patterns.append(patterns_found)
            
            logger.info(f"   ✓ NULL checks: {patterns_found['null_check_count']}")
            logger.info(f"   ✓ Error handling: {patterns_found['error_handling_count']}")
            logger.info(f"   ✓ Missing validations: {len(patterns_found['missing_validations'])}")
        
        return validation_patterns
    
    def _search_similar_crash_fixes(self, crash_info: Dict) -> List[Dict]:
        """
        STEP 4: Search git commits for similar crash fixes.
        """
        similar_fixes = []
        
        crash_function = crash_info.get("faulting_function")
        crash_signal = crash_info.get("signal", "SIGSEGV")
        
        if not crash_function:
            return similar_fixes
        
        logger.info(f"🔍 Searching git commits for similar crashes involving: {crash_function}")
        
        # Search git metadata for crash-related commits
        crash_keywords = ["crash", "segfault", "segmentation", "sigsegv", "null", "fix", crash_function.lower()]
        
        for commit in self.git_metadata:
            subject = commit.get("subject", "").lower()
            body = commit.get("body", "").lower()
            commit_keywords = commit.get("keywords", [])
            
            # Check if this commit is related to crashes
            is_crash_related = False
            match_score = 0
            
            # Check subject and body for crash keywords
            for keyword in crash_keywords:
                if keyword in subject:
                    is_crash_related = True
                    match_score += 2  # Subject match is more important
                if keyword in body:
                    is_crash_related = True
                    match_score += 1
            
            # Check if function name is mentioned
            if crash_function.lower() in subject or crash_function.lower() in body:
                is_crash_related = True
                match_score += 3  # Function match is highly relevant
            
            if is_crash_related:
                similar_fixes.append({
                    "commit_hash": commit.get("commit_hash_short", commit.get("commit_hash", "")[:10]),
                    "subject": commit.get("subject"),
                    "author": commit.get("author_name"),
                    "date": commit.get("date_iso", commit.get("date", "")),
                    "relevance_score": min(match_score / 10.0, 1.0),  # Normalize to 0-1
                    "is_rca": commit.get("is_rca_commit", False),
                    "files_changed": commit.get("files_changed", [])
                })
        
        # Sort by relevance score
        similar_fixes.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top 5 most relevant
        return similar_fixes[:5]
    
    def _enrich_with_source_code(self, crash_info: Dict, backtrace_functions: List[Dict]) -> Dict[str, Any]:
        """
        STEP 5: Read full source code for all backtrace functions.
        """
        enrichment = {}
        
        for func_data in backtrace_functions[:5]:  # Top 5 frames
            function_name = func_data.get("function_name")
            file_path = func_data.get("file_path")
            
            logger.info(f"📖 Reading source code for: {function_name}")
            
            # Read full function source
            source_code = self._read_full_function_source(file_path, function_name)
            
            if source_code:
                # Extract function signature
                signature = self._extract_function_signature(source_code)
                
                # Extract local variables
                local_vars = self._extract_local_variables(source_code)
                
                enrichment[function_name] = {
                    "file_path": file_path,
                    "source_code": source_code,
                    "source_code_length": len(source_code),
                    "function_signature": signature,
                    "local_variables": local_vars,
                    "has_source": True
                }
                
                logger.info(f"   ✓ Source code: {len(source_code)} chars")
                logger.info(f"   ✓ Local variables: {len(local_vars)}")
            else:
                enrichment[function_name] = {
                    "file_path": file_path,
                    "has_source": False,
                    "error": "Source code not found or not readable"
                }
                logger.warning(f"   ✗ Could not read source for {function_name}")
        
        return enrichment
    
    def _read_full_function_source(self, file_path: str, function_name: str) -> Optional[str]:
        """Read the complete source code of a function"""
        # Try to construct various possible paths
        possible_paths = [
            file_path,
            os.path.join(self.codebase_path, file_path),
            os.path.join(self.codebase_path, os.path.basename(file_path))
        ]
        
        # Also try relative path from common OAI directories
        for base_dir in ['openair2', 'openair3', 'common']:
            if base_dir in file_path:
                rel_path = file_path.split(base_dir)[1].lstrip('/')
                possible_paths.append(os.path.join(self.codebase_path, base_dir, rel_path))
        
        for full_path in possible_paths:
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract the specific function
                    function_code = self._extract_function_from_file(content, function_name)
                    return function_code if function_code else content  # Return full file if can't extract function
                    
                except Exception as e:
                    logger.warning(f"Error reading {full_path}: {e}")
                    continue
        
        return None
    
    def _extract_function_from_file(self, file_content: str, function_name: str) -> Optional[str]:
        """Extract a specific function from file content"""
        lines = file_content.split('\n')
        
        # Find function start
        function_pattern = rf'^\s*(?:static\s+)?\w+\s+\*?\s*{re.escape(function_name)}\s*\('
        
        start_line = None
        for i, line in enumerate(lines):
            if re.search(function_pattern, line):
                start_line = i
                break
        
        if start_line is None:
            return None
        
        # Find function end using brace counting
        brace_count = 0
        end_line = start_line
        found_opening = False
        
        for i in range(start_line, len(lines)):
            for char in lines[i]:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        end_line = i
                        function_code = '\n'.join(lines[start_line:end_line+1])
                        return function_code
        
        # Fallback: return reasonable chunk
        end_line = min(start_line + 150, len(lines))
        return '\n'.join(lines[start_line:end_line])
    
    def _extract_function_signature(self, source_code: str) -> str:
        """Extract function signature from source code"""
        lines = source_code.split('\n')
        
        # Function signature is usually in first few lines
        for i, line in enumerate(lines[:10]):
            if '(' in line and ('{' in line or (i < len(lines)-1 and '{' in lines[i+1])):
                # Found signature line
                signature = line.strip()
                # If signature spans multiple lines, collect them
                j = i + 1
                while j < len(lines) and '{' not in lines[j] and j < i + 5:
                    signature += " " + lines[j].strip()
                    j += 1
                return signature
        
        return "Signature not found"
    
    def _extract_local_variables(self, source_code: str) -> List[str]:
        """Extract local variable declarations from source code"""
        variables = []
        
        # Pattern for C variable declarations
        var_patterns = [
            r'^\s*(?:const\s+)?(?:static\s+)?(\w+)\s+\*?\s*(\w+)\s*[=;]',  # type var = or type var;
            r'^\s*(?:const\s+)?(?:static\s+)?(\w+)\s+\*\s*(\w+)\s*[=;]',  # type *var
        ]
        
        lines = source_code.split('\n')
        for line in lines:
            # Skip comments
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            for pattern in var_patterns:
                match = re.search(pattern, line)
                if match:
                    var_type = match.group(1)
                    var_name = match.group(2)
                    variables.append(f"{var_type} {var_name}")
        
        return list(set(variables))[:20]  # Return unique, limit to 20
    
    def _minimal_config_search(self, crash_info: Dict) -> List[Dict]:
        """
        STEP 6: Minimal config search - only for initialization-related crashes.
        """
        configs = []
        
        # Check if crash is initialization-related
        scenario_flow = crash_info.get("scenario_flow", [])
        faulting_function = crash_info.get("faulting_function", "")
        
        # Heuristic: if crash is in early stages or function name contains 'init'/'config'/'setup'
        is_init_related = (
            len(scenario_flow) < 5 or
            'init' in faulting_function.lower() or
            'config' in faulting_function.lower() or
            'setup' in faulting_function.lower()
        )
        
        if is_init_related:
            logger.info("🔍 Crash appears initialization-related, searching minimal configs...")
            
            # Search for configs mentioned in the function or file
            # This is a simplified search - in practice, you might want to use FAISS config index
            # For now, return empty since crashes are typically code issues
            logger.info("   Skipping config search (crashes are typically code issues)")
        else:
            logger.info("   Skipping config search (not initialization-related)")
        
        return configs


def main():
    """Test crash Phase 2 retrieval"""
    retrieval = CrashPhase2Retrieval()
    
    # Test with Phase 1 extraction results
    extraction_file = "output/segmentation_fault_extraction.json"
    
    if not os.path.exists(extraction_file):
        print(f"❌ Phase 1 extraction file not found: {extraction_file}")
        print(f"   Please run segmentation_fault_analyzer.py first")
        return
    
    results = retrieval.process_crash_retrieval(extraction_file)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 PHASE 2 RETRIEVAL SUMMARY")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"\n✅ Prioritized Functions: {len(results.get('prioritized_functions', []))}")
        print(f"✅ Call Chain Expansion: {len(results.get('call_chain_expansion', []))}")
        print(f"✅ Validation Patterns: {len(results.get('validation_patterns', []))}")
        print(f"✅ Similar Crash Fixes: {len(results.get('similar_crash_fixes', []))}")
        print(f"✅ Source Code Enrichment: {len(results.get('source_code_enrichment', {}))} functions")
        
        # Show prioritized functions
        if results.get('prioritized_functions'):
            print(f"\n🎯 Prioritized Backtrace Functions:")
            for func in results['prioritized_functions'][:5]:
                print(f"   {func['role']:20s} - {func['function_name']:40s} (priority: {func['priority_score']:.2f})")
        
        # Show similar crash fixes
        if results.get('similar_crash_fixes'):
            print(f"\n🔧 Similar Crash Fixes from Git:")
            for fix in results['similar_crash_fixes']:
                print(f"   [{fix['relevance_score']:.2f}] {fix['commit_hash']} - {fix['subject'][:60]}")
        
        print(f"\n💾 Results saved to: output/crash_phase2_retrieval.json")


if __name__ == "__main__":
    main()

