#!/usr/bin/env python3
"""
Error Handling Pipeline with CRAG (Corrective Retrieval-Augmented Generation)

This script implements a comprehensive error handling workflow using:
- Azure OpenAI GPT-4o-mini for query generation and grading
- FAISS indices for semantic search
- Pattern matching for known error templates
- Symbolic and semantic retrieval for comprehensive candidate collection

Author: AI Assistant
"""

import os
import json
import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ErrorContext:
    """Container for error context and processing state."""
    original_error: str
    matched_patterns: List[Dict]
    extracted_hints: Dict[str, List[str]]
    crag_queries: List[str]
    candidates: Dict[str, List[Dict]]
    graded_candidates: Dict[str, List[Dict]]
    final_results: Dict[str, Any]


def linker_derived_cmake_hints(error_message: str, log_content: str) -> Dict[str, Any]:
    """
    From linker / asn1 lines (often without naming a .cmake file), derive search queries
    and UI prompt bullets so retrieval and the LLM target ASN1/*.cmake bundles.
    """
    text = f"{error_message or ''}\n{log_content or ''}"
    text_norm = text.replace("\\", "/")
    hints: Dict[str, Any] = {
        "asn_def_types": [],
        "extra_search_queries": [],
        "prompt_bullets": [],
        "messages_lib_dir": None,
        "asn1_lib_short": None,
    }
    seen_types: set = set()

    for m in re.finditer(
        r"undefined\s+reference\s+to\s+[`']?asn_DEF_([A-Za-z0-9_]+)",
        text_norm,
        re.IGNORECASE,
    ):
        t = m.group(1)
        if t not in seen_types:
            seen_types.add(t)
            hints["asn_def_types"].append(t)
            hints["extra_search_queries"].append(
                f"{t} {t}.c asn1c generated source add_library cmake MESSAGES ASN1"
            )
            hints["prompt_bullets"].append(
                f"Linker needs `asn_DEF_{t}` → the defining **`{t}.c`** must be compiled into "
                f"`libasn1_*`. In OAI, the list of generated `.c` files is usually in "
                f"**`.../MESSAGES/ASN1/<protocol>-<release>.cmake`** (e.g. `f1ap-18.6.0.cmake`), "
                "not only top-level `CMakeLists.txt`."
            )

    for m in re.finditer(r"\basn_DEF_([A-Za-z0-9_]+)\b", text_norm):
        t = m.group(1)
        if t not in seen_types and len(hints["asn_def_types"]) < 10:
            seen_types.add(t)
            hints["asn_def_types"].append(t)
            hints["extra_search_queries"].append(
                f"asn_DEF_{t} cmake asn1 {t}.c target_sources"
            )

    m = re.search(
        r"(openair\d+/[A-Za-z0-9_./-]+/MESSAGES)/libasn1_([A-Za-z0-9_]+)\.a",
        text_norm,
        re.IGNORECASE,
    )
    if m:
        rel, lib_short = m.group(1), m.group(2)
        hints["messages_lib_dir"] = rel
        hints["asn1_lib_short"] = lib_short
        proto = lib_short.lower().lstrip("_")
        hints["extra_search_queries"].append(
            f"{rel}/ASN1 {proto} cmake f1ap ngap s1ap generated C sources list"
        )
        hints["prompt_bullets"].append(
            f"Log points to **`{rel}/libasn1_{lib_short}.a`** → inspect **`{rel}/ASN1/`** for "
            f"**`{proto}-*.cmake`** (or similarly named) that list generated ASN.1 `.c` files "
            "for that library; patch the file list there if a `.c` is missing."
        )

    uq: List[str] = []
    for q in hints["extra_search_queries"]:
        q = (q or "").strip()
        if q and q not in uq:
            uq.append(q)
    hints["extra_search_queries"] = uq[:18]
    hints["prompt_bullets"] = (hints["prompt_bullets"] or [])[:10]
    return hints


class ErrorHandlingPipeline:
    """
    Main pipeline for error handling using CRAG methodology.
    """
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the pipeline with all required components.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        self.openair_codebase_file_name = openair_codebase_file_name
        self.azure_client = None
        self.embedding_model = None
        self.functions_index = None
        self.config_index = None
        self.functions_mapping = []
        self.config_mapping = []
        self.error_patterns = []
        self.function_calls = {}
        
        # Initialize components
        self._setup_azure_client()
        self._load_embedding_model()
        self._load_faiss_indices()
        self._load_data_sources()
    
    def _setup_azure_client(self):
        """Setup Azure OpenAI client."""
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
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            logger.info("Please set the following in your .env file:")
            logger.info("AZURE_OPENAI_API_KEY=your-api-key")
            logger.info("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        try:
            self.azure_client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-05-01-preview",
                azure_endpoint=endpoint
            )
            logger.info("✅ Azure OpenAI client initialized successfully")
            logger.info("📦 Using deployment: gpt-4o-mini")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load the same embedding model used for index creation."""
        logger.info("📥 Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _load_faiss_indices(self):
        """Load FAISS indices and mappings."""
        logger.info("📊 Loading FAISS indices...")
        
        try:
            # Load FAISS indices
            if os.path.exists("faiss_indices/functions_index.faiss"):
                self.functions_index = faiss.read_index("faiss_indices/functions_index.faiss")
                logger.info(f"✅ Functions index loaded: {self.functions_index.ntotal} vectors")
                # Set nprobe for IVF indices to search more clusters for better recall
                if hasattr(self.functions_index, 'nprobe'):
                    self.functions_index.nprobe = 20  # Search 20 clusters instead of default 1
                    logger.info(f"✅ Set functions_index nprobe=20 for better search coverage")
            else:
                logger.warning("⚠️  faiss_indices/functions_index.faiss not found")
            
            if os.path.exists("faiss_indices/config_index.faiss"):
                self.config_index = faiss.read_index("faiss_indices/config_index.faiss")
                logger.info(f"✅ Config index loaded: {self.config_index.ntotal} vectors")
                # Set nprobe for IVF indices to search more clusters for better recall
                if hasattr(self.config_index, 'nprobe'):
                    self.config_index.nprobe = 20  # Search 20 clusters instead of default 1
                    logger.info(f"✅ Set config_index nprobe=20 for better search coverage")
            else:
                logger.warning("⚠️  faiss_indices/config_index.faiss not found")
            
            # Load mappings
            if os.path.exists("faiss_indices/functions_mapping.json"):
                with open("faiss_indices/functions_mapping.json", 'r', encoding='utf-8') as f:
                    self.functions_mapping = json.load(f)
                logger.info(f"✅ Functions mapping loaded: {len(self.functions_mapping)} entries")
            
            if os.path.exists("faiss_indices/config_mapping.json"):
                with open("faiss_indices/config_mapping.json", 'r', encoding='utf-8') as f:
                    self.config_mapping = json.load(f)
                logger.info(f"✅ Config mapping loaded: {len(self.config_mapping)} entries")
                
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS indices: {e}")
            raise
    
    def _load_data_sources(self):
        """Load error patterns and function calls data."""
        logger.info("📂 Loading data sources...")
        
        try:
            # Load error patterns
            if os.path.exists("database/error_patterns_enhanced.json"):
                with open("database/error_patterns_enhanced.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle nested structure: extract patterns dict
                    if isinstance(data, dict) and "patterns" in data:
                        self.error_patterns = data["patterns"]  # Keep as dict
                        logger.info(f"✅ Error patterns loaded: {len(self.error_patterns)} patterns")
                    else:
                        self.error_patterns = data if isinstance(data, dict) else {}
                        logger.info(f"✅ Error patterns loaded: {len(self.error_patterns)} patterns")
            else:
                logger.warning("⚠️  error_patterns_enhanced.json not found")
                self.error_patterns = {}
            
            # Load function calls
            if os.path.exists("database/function_calls.json"):
                with open("database/function_calls.json", 'r', encoding='utf-8') as f:
                    self.function_calls = json.load(f)
                logger.info(f"✅ Function calls loaded: {len(self.function_calls)} entries")
            else:
                logger.warning("⚠️  function_calls.json not found")
                self.function_calls = {}
                
        except Exception as e:
            logger.error(f"❌ Failed to load data sources: {e}")
            raise
    
    def error_ingest(self, error_message: str) -> ErrorContext:
        """
        Step 2.1 - Error Ingestion
        
        Args:
            error_message: Runtime error, crash dump, or error message
            
        Returns:
            ErrorContext: Initial error context
        """
        logger.info("=" * 60)
        logger.info("🔥 PHASE 2.1 - ERROR INGESTION")
        logger.info("=" * 60)
        logger.info(f"📥 Processing error: {error_message}")
        
        context = ErrorContext(
            original_error=error_message,
            matched_patterns=[],
            extracted_hints={"keywords": [], "config_params": [], "functions": []},
            crag_queries=[],
            candidates={"functions": [], "configs": []},
            graded_candidates={"functions": [], "configs": []},
            final_results={}
        )
        
        logger.info("✅ Error ingestion completed")
        return context
    
    def pattern_match(self, context: ErrorContext) -> ErrorContext:
        """
        Step 2.2 - Pattern Match (Hint Path)
        
        Args:
            context: Error context from ingestion
            
        Returns:
            Updated ErrorContext with matched patterns
        """
        logger.info("=" * 60)
        logger.info("🔍 PHASE 2.2 - PATTERN MATCHING")
        logger.info("=" * 60)
        
        error_text = context.original_error.lower()
        matched_patterns = []
        
        for pattern_key, pattern_data in self.error_patterns.items():
            if not isinstance(pattern_data, dict):
                continue
                
            # Check patterns array (both string matches and regex)
            if "patterns" in pattern_data:
                patterns_list = pattern_data["patterns"] if isinstance(pattern_data["patterns"], list) else [pattern_data["patterns"]]
                for pattern_str in patterns_list:
                    # Try exact substring match first
                    if pattern_str.lower() in error_text or error_text in pattern_str.lower():
                        matched_patterns.append({
                            "pattern_id": pattern_key,
                            "matched_text": pattern_str,
                            "pattern_data": pattern_data
                        })
                        logger.info(f"✅ Pattern match: {pattern_str}")
                        break
                    
                    # Try flexible word-based matching for similar phrases
                    error_words = set(error_text.replace(".", "").replace(",", "").split())
                    pattern_words = set(pattern_str.lower().replace(".", "").replace(",", "").split())
                    
                    # Calculate word overlap - if 80% of error words are in pattern, it's a match
                    if error_words and pattern_words:
                        overlap = len(error_words.intersection(pattern_words))
                        error_word_ratio = overlap / len(error_words)
                        
                        if error_word_ratio >= 0.8:  # 80% of error words must be in pattern
                            matched_patterns.append({
                                "pattern_id": pattern_key,
                                "matched_text": pattern_str,
                                "pattern_data": pattern_data
                            })
                            logger.info(f"✅ Flexible match: {pattern_str} (overlap: {error_word_ratio:.2f})")
                            break
                    
                    # Also try regex match
                    try:
                        if re.search(pattern_str, error_text, re.IGNORECASE):
                            matched_patterns.append({
                                "pattern_id": pattern_key,
                                "matched_text": pattern_str,
                                "pattern_data": pattern_data
                            })
                            logger.info(f"✅ Regex match: {pattern_str}")
                            break
                    except re.error:
                        pass  # Not a valid regex, skip
            
            # Check keywords array
            if "keywords" in pattern_data:
                keywords = pattern_data["keywords"] if isinstance(pattern_data["keywords"], list) else [pattern_data["keywords"]]
                for keyword in keywords:
                    if keyword.lower() in error_text:
                        matched_patterns.append({
                            "pattern_id": pattern_key,
                            "matched_text": keyword,
                            "pattern_data": pattern_data
                        })
                        logger.info(f"✅ Keyword match: {keyword}")
                        break
        
        context.matched_patterns = matched_patterns
        logger.info(f"📊 Found {len(matched_patterns)} pattern matches")
        
        # Extract hints from matched patterns
        extracted_hints = {"keywords": [], "config_params": [], "functions": []}
        for pattern_match in matched_patterns:
            pattern_data = pattern_match.get("pattern_data", {})
            if "suggested_keywords" in pattern_data:
                extracted_hints["keywords"].extend(pattern_data["suggested_keywords"])
            if "related_configs" in pattern_data:
                extracted_hints["config_params"].extend(pattern_data["related_configs"])
            if "related_functions" in pattern_data:
                extracted_hints["functions"].extend(pattern_data["related_functions"])
        
        # Store extracted hints in context for direct use
        context.extracted_hints = extracted_hints
        logger.info(f"💡 Extracted hints: {extracted_hints}")
        return context
    
    def crag_query_generation(self, context: ErrorContext) -> ErrorContext:
        """
        Step 2.3 - CRAG Query Generation
        
        Args:
            context: Error context with pattern matches
            
        Returns:
            Updated ErrorContext with CRAG queries
        """
        logger.info("=" * 60)
        logger.info("🧠 PHASE 2.3 - CRAG QUERY GENERATION")
        logger.info("=" * 60)
        
        # Prepare context for query generation
        pattern_hints = ""
        if context.matched_patterns:
            pattern_hints = "\\nKnown pattern hints: " + str([p.get("description", "") for p in context.matched_patterns])
        
        prompt = f"""You are an expert system for analyzing 5G/LTE telecommunications errors in the OpenAirInterface codebase.

Given this error message:
"{context.original_error}"{pattern_hints}

Generate 5-7 specific search queries to find relevant code functions and configuration parameters. Each query should target:
1. Function names (e.g., "amf_connect", "setup_ngap")
2. Configuration parameters (e.g., "amf_ip_address", "ngap_port")  
3. Error handling routines
4. Protocol-specific keywords

Focus on 5G/LTE terms: AMF, gNB, NGAP, SCTP, RRC, PDCP, etc.

Return ONLY the queries, one per line, no explanations:"""

        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a telecommunications error analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3,
                seed=45
            )
            
            queries_text = response.choices[0].message.content.strip()
            queries = [q.strip() for q in queries_text.split('\\n') if q.strip()]
            
            context.crag_queries = queries
            logger.info(f"🎯 Generated {len(queries)} CRAG queries:")
            for i, query in enumerate(queries, 1):
                logger.info(f"  {i}. {query}")
                
        except Exception as e:
            logger.error(f"❌ Failed to generate CRAG queries: {e}")
            # Fallback to basic queries
            context.crag_queries = [
                f"error {context.original_error}",
                "connection setup function",
                "configuration parameter"
            ]
            logger.info("🔄 Using fallback queries")
        
        return context
    
    def retrieve_candidates(self, context: ErrorContext, top_k: int = 100) -> ErrorContext:
        """
        Step 2.4 - Candidate Retrieval (Symbolic + Semantic)
        
        Args:
            context: Error context with CRAG queries
            top_k: Number of top candidates to retrieve per query
            
        Returns:
            Updated ErrorContext with retrieved candidates
        """
        logger.info("=" * 60)
        logger.info("🔍 PHASE 2.4 - CANDIDATE RETRIEVAL")
        logger.info("=" * 60)
        
        all_function_candidates = []
        all_config_candidates = []
        
        # 🎯 DIRECT HINT LOOKUP - Highest Priority (NEW!)
        if context.extracted_hints and context.extracted_hints.get("functions"):
            logger.info("🎯 Performing direct hint lookup...")
            hint_function_candidates = self._direct_hint_lookup(context.extracted_hints)
            all_function_candidates.extend(hint_function_candidates["functions"])
            all_config_candidates.extend(hint_function_candidates["configs"])
            logger.info(f"📊 Direct hints: {len(hint_function_candidates['functions'])} functions, {len(hint_function_candidates['configs'])} configs")
        
        # Enhanced Symbolic Retrieval with Call Chain Analysis
        logger.info("🔗 Performing enhanced symbolic retrieval...")
        symbolic_functions = self._symbolic_retrieval(context.original_error, context)
        all_function_candidates.extend(symbolic_functions)
        logger.info(f"📊 Enhanced symbolic retrieval: {len(symbolic_functions)} functions")
        
        # Error Handling Pattern Search - NEW!
        logger.info("🔍 Performing error handling pattern search...")
        error_handling_candidates = self._search_error_handling_functions(context.original_error)
        all_function_candidates.extend(error_handling_candidates)
        logger.info(f"📊 Error handling patterns: {len(error_handling_candidates)} candidates")

        # Semantic Retrieval using FAISS
        logger.info("🧠 Performing semantic retrieval...")
        for i, query in enumerate(context.crag_queries, 1):
            logger.info(f"  Query {i}: {query}")
            
            # Search functions
            if self.functions_index and self.functions_mapping:
                func_candidates = self._semantic_search(
                    query, self.functions_index, self.functions_mapping, top_k
                )
                all_function_candidates.extend(func_candidates)
                logger.info(f"    Functions: {len(func_candidates)} candidates")
            
            # Search configs
            if self.config_index and self.config_mapping:
                config_candidates = self._semantic_search(
                    query, self.config_index, self.config_mapping, top_k
                )
                all_config_candidates.extend(config_candidates)
                logger.info(f"    Configs: {len(config_candidates)} candidates")
        
        # Remove duplicates and rank
        context.candidates = {
            "functions": self._deduplicate_candidates(all_function_candidates, "function_name"),
            "configs": self._deduplicate_candidates(all_config_candidates, "param_name")
        }
        
        logger.info(f"📊 Total candidates - Functions: {len(context.candidates['functions'])}, Configs: {len(context.candidates['configs'])}")
        return context
    
    def _symbolic_retrieval(self, error_message: str, context: ErrorContext = None) -> List[Dict]:
        """Enhanced symbolic retrieval with bidirectional call chain analysis"""
        candidates = []
        
        # If we have extracted hints, use them as seed functions for call chain analysis
        if context and context.extracted_hints.get("functions"):
            candidates = self._enhanced_call_chain_analysis(context)
        else:
            # Fallback to basic keyword-based retrieval
            candidates = self._basic_keyword_retrieval(error_message)
        
        return candidates
    
    def _enhanced_call_chain_analysis(self, context: ErrorContext) -> List[Dict]:
        """Walk call chains bidirectionally from hint functions"""
        candidates = []
        
        # Get seed functions from hints
        seed_functions = []
        for hint_func_name in context.extracted_hints.get("functions", []):
            func_data = self._find_function_by_name(hint_func_name)
            if func_data:
                seed_functions.append(func_data)
        
        # For each seed function, walk call chain in both directions
        for seed_func in seed_functions:
            # Add the seed function itself (highest priority)
            candidates.append({
                "candidate": seed_func,
                "score": 0.95,
                "source": "call_chain_seed",
                "chain_position": "origin"
            })
            
            # Walk upstream (who calls this function?)
            upstream_candidates = self._walk_call_chain_upstream(seed_func, depth=2)
            candidates.extend(upstream_candidates)
            
            # Walk downstream (what does this function call?)
            downstream_candidates = self._walk_call_chain_downstream(seed_func, depth=2)
            candidates.extend(downstream_candidates)
        
        return self._deduplicate_call_chain_candidates(candidates)
    
    def _walk_call_chain_upstream(self, function_data: Dict, depth: int) -> List[Dict]:
        """Find functions that call this function (error propagation path)"""
        candidates = []
        target_func_name = function_data.get("function_name", "")
        
        if depth <= 0:
            return candidates
        
        # Search function_calls.json for functions that call our target
        for func_entry in self.function_calls:
            calls_list = func_entry.get("calls", [])
            
            if target_func_name in calls_list:
                caller_name = func_entry.get("function", "")
                caller_data = self._find_function_by_name(caller_name)
                
                if caller_data:
                    candidates.append({
                        "candidate": caller_data,
                        "score": 0.7 - (2 - depth) * 0.1,  # Score decreases with distance
                        "source": "call_chain_upstream",
                        "chain_position": f"upstream_depth_{2-depth+1}",
                        "relationship": f"calls {target_func_name}"
                    })
                    
                    # Recursively walk further upstream
                    if depth > 1:
                        upstream_candidates = self._walk_call_chain_upstream(caller_data, depth - 1)
                        candidates.extend(upstream_candidates)
        
        return candidates
    
    def _walk_call_chain_downstream(self, function_data: Dict, depth: int) -> List[Dict]:
        """Find functions called by this function (potential root causes)"""
        candidates = []
        source_func_name = function_data.get("function_name", "")
        
        if depth <= 0:
            return candidates
        
        # Find the function_calls entry for this function
        func_entry = next((entry for entry in self.function_calls 
                          if entry.get("function") == source_func_name), None)
        
        if func_entry:
            calls_list = func_entry.get("calls", [])
            
            for called_func_name in calls_list:
                called_func_data = self._find_function_by_name(called_func_name)
                
                if called_func_data:
                    candidates.append({
                        "candidate": called_func_data,
                        "score": 0.6 - (2 - depth) * 0.1,  # Score decreases with distance
                        "source": "call_chain_downstream",
                        "chain_position": f"downstream_depth_{2-depth+1}",
                        "relationship": f"called by {source_func_name}"
                    })
                    
                    # Recursively walk further downstream
                    if depth > 1:
                        downstream_candidates = self._walk_call_chain_downstream(called_func_data, depth - 1)
                        candidates.extend(downstream_candidates)
        
        return candidates
    
    def _find_function_by_name(self, function_name: str) -> Dict:
        """Helper to find function data by name"""
        if isinstance(self.functions_mapping, dict):
            for func_data in self.functions_mapping.values():
                if func_data and func_data.get("function_name") == function_name:
                    return func_data
        else:
            for func_data in self.functions_mapping:
                if func_data and func_data.get("function_name") == function_name:
                    return func_data
        return None
    
    def _deduplicate_call_chain_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicates and keep highest scoring candidates"""
        seen_functions = {}
        
        for candidate in candidates:
            func_name = candidate["candidate"].get("function_name", "")
            
            if func_name not in seen_functions:
                seen_functions[func_name] = candidate
            else:
                # Keep the highest scoring occurrence
                if candidate["score"] > seen_functions[func_name]["score"]:
                    seen_functions[func_name] = candidate
        
        return sorted(seen_functions.values(), key=lambda x: x["score"], reverse=True)
    
    def _basic_keyword_retrieval(self, error_message: str) -> List[Dict]:
        """Fallback basic keyword-based retrieval"""
        candidates = []
        
        # Extract domain-specific keywords only
        domain_keywords = ['amf', 'gnb', 'ngap', 'nas', 'rrc', 'sctp']
        error_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', error_message.lower())
        relevant_words = [word for word in error_words if word in domain_keywords]
        
        # Search for functions containing these keywords
        for func_entry in self.function_calls:
            function_name = func_entry.get("function", "").lower()
            
            for keyword in relevant_words:
                if keyword in function_name:
                    func_data = self._find_function_by_name(func_entry.get("function"))
                    if func_data:
                        candidates.append({
                            "candidate": func_data,
                            "score": 0.6,
                            "source": "symbolic_keyword",
                            "matched_keyword": keyword
                        })
                        break  # Avoid duplicate entries for same function
        
        return candidates
    
    def _search_error_handling_functions(self, error_text: str) -> List[Dict]:
        """Search for error handling functions using name patterns and domain knowledge."""
        candidates = []
        
        # Define error handling function patterns
        error_patterns = [
            r'.*[Rr]eject.*',
            r'.*[Ee]rror.*',
            r'.*[Ff]ailure.*',
            r'.*[Hh]andle.*[Ee]rror.*',
            r'.*[Pp]rocess.*[Ee]rror.*',
            r'.*[Ss]etup.*[Ff]ailure.*',
            r'.*[Cc]leanup.*',
            r'.*[Rr]emove.*[Uu][Ee].*',
            r'.*[Gg]enerate.*[Rr]eject.*',
            r'.*[Ss]end.*[Rr]eject.*'
        ]
        
        if isinstance(self.functions_mapping, (dict, list)):
            # Handle both dict and list formats
            if isinstance(self.functions_mapping, dict):
                func_list = self.functions_mapping.values()
            else:  # list
                func_list = self.functions_mapping
                
            for func_data in func_list:
                if func_data:
                    func_name = func_data.get("function_name", "")
                    
                    # Check if function name matches error handling patterns
                    for pattern in error_patterns:
                        if re.match(pattern, func_name):
                            # Extract function body
                            current_function_body = self._extract_current_function_body(
                                func_data.get("file_path", ""), func_name
                            )
                            
                            # Special boost for critical RRC rejection functions
                            score = 0.9  # Default high confidence for pattern matches
                            if "rrc_gNB_generate_RRCReject" in func_name or "rrc_gNB_send_RRCReject" in func_name:
                                score = 0.95  # Maximum priority for RRC rejection functions
                            
                            candidates.append({
                                "candidate": {
                                    "function_name": func_name,
                                    "file_path": func_data.get("file_path", ""),
                                    "code_body": current_function_body or func_data.get("code_body", ""),
                                    "code_snippet": current_function_body or func_data.get("code_snippet", ""),
                                    "signature": func_data.get("signature", ""),
                                    "return_type": func_data.get("return_type", "")
                                },
                                "score": score,
                                "source": "error_handling_pattern",
                                "query": "error_handling_patterns"
                            })
                            break
        
        logger.info(f"🔍 Error handling pattern search found {len(candidates)} functions")
        return candidates

    def _semantic_search(self, query: str, index: faiss.Index, mapping: List[Dict], top_k: int) -> List[Dict]:
        """Perform semantic search using FAISS index"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(
                [query], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            
            # Search more broadly to find candidates from important files
            search_k = min(top_k * 20, index.ntotal)  # Search 20x more broadly
            scores, indices = index.search(query_embedding.astype(np.float32), search_k)
            
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                # Check if index is valid and convert score properly
                if idx != -1 and idx < len(mapping):
                    # For IndexFlatIP, scores are inner products
                    # Since embeddings are normalized, this is cosine similarity
                    # Convert to a reasonable threshold (cosine similarity ranges -1 to 1)
                    cosine_score = float(score)
                    
                    # Use a lower threshold for cosine similarity (0.3 instead of 0.5)
                    if cosine_score > 0.3:
                                                # Handle both list and dict mapping formats
                        if isinstance(mapping, dict):
                            # Mapping is a dict with string keys - convert idx to string
                            mapping_item = mapping.get(str(int(idx)))
                        else:
                            # Mapping is a list with integer indices
                            mapping_item = mapping[idx] if idx < len(mapping) else None
                        
                        if mapping_item:
                            # Apply file-specific boosting and update current values
                            boosted_score = cosine_score
                            
                            if "param_name" in mapping_item:  # This is a config candidate
                                file_path = mapping_item.get('file_path', '')
                                param_name = mapping_item.get('param_name', '')
                                line_number = mapping_item.get('line_number', None)
                                
                                # Extract current parameter value from the actual file
                                current_param_value = self._extract_current_param_value(file_path, param_name, line_number)
                                if current_param_value:
                                    # Update the mapping item with current value
                                    mapping_item = mapping_item.copy()
                                    mapping_item['param_value'] = current_param_value
                                
                                if file_path:
                                    file_path_lower = file_path.lower()
                                    # Boost cu_gnb.conf and du_gnb.conf files as they are primary config files
                                    if "cu_gnb.conf" in file_path_lower or "du_gnb.conf" in file_path_lower:
                                        boosted_score += 0.3  # Significant boost for primary config files
                                        logger.debug(f"Boosted config score by 0.3 for primary config file: {file_path}")
                                    # Boost other important config files
                                    elif "gnb" in file_path_lower and ".conf" in file_path_lower:
                                        boosted_score += 0.1  # Small boost for other gNB config files
                                        logger.debug(f"Boosted config score by 0.1 for gNB config file: {file_path}")
                            
                            elif "function_name" in mapping_item:  # This is a function candidate
                                file_path = mapping_item.get('file_path', '')
                                function_name = mapping_item.get('function_name', '')
                                
                                # Extract current function body from the actual file
                                current_function_body = self._extract_current_function_body(file_path, function_name)
                                if current_function_body:
                                    # Update the mapping item with current function body
                                    mapping_item = mapping_item.copy()
                                    mapping_item['code_snippet'] = current_function_body
                                    mapping_item['code_body'] = current_function_body
                            
                            candidates.append({
                                "candidate": mapping_item,
                                "score": min(1.0, boosted_score),  # Cap at 1.0
                                "source": "semantic",
                                "query": query
                            })
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed for query '{query}': {e}")
            return []
    
    def _deduplicate_candidates(self, candidates: List[Dict], key_field: str) -> List[Dict]:
        """Remove duplicate candidates and keep highest scoring ones"""
        seen = {}
        for candidate in candidates:
            item_key = candidate["candidate"].get(key_field, "")
            if item_key and (item_key not in seen or candidate["score"] > seen[item_key]["score"]):
                seen[item_key] = candidate
        
        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    
    def _direct_hint_lookup(self, extracted_hints: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """
        🎯 NEW: Direct lookup of functions and configs from extracted hints
        
        This bypasses semantic search and directly finds exact matches from functions.json
        and config.json based on the hints extracted from error patterns.
        
        Args:
            extracted_hints: Dictionary with 'functions' and 'config_params' lists
            
        Returns:
            Dictionary with 'functions' and 'configs' candidate lists
        """
        hint_candidates = {"functions": [], "configs": []}
        
        # Direct function lookup from hints
        if extracted_hints.get("functions"):
            logger.info(f"🔍 Looking up {len(extracted_hints['functions'])} hinted functions...")
            for hint_function in extracted_hints["functions"]:
                # Search through functions_mapping for exact matches
                if isinstance(self.functions_mapping, dict):
                    # Mapping is a dictionary
                    for func_data in self.functions_mapping.values():
                        if func_data and func_data.get("function_name") == hint_function:
                            hint_candidates["functions"].append({
                                "candidate": func_data,
                                "score": 0.98,  # Very high confidence from pattern hint
                                "source": "pattern_hint_direct",
                                "hint": hint_function
                            })
                            logger.info(f"  ✅ Found hint function: {hint_function}")
                            break
                else:
                    # Mapping is a list
                    for func_data in self.functions_mapping:
                        if func_data and func_data.get("function_name") == hint_function:
                            hint_candidates["functions"].append({
                                "candidate": func_data,
                                "score": 0.98,  # Very high confidence from pattern hint
                                "source": "pattern_hint_direct",
                                "hint": hint_function
                            })
                            logger.info(f"  ✅ Found hint function: {hint_function}")
                            break
        
        # Direct config lookup from hints
        if extracted_hints.get("config_params"):
            logger.info(f"🔍 Looking up {len(extracted_hints['config_params'])} hinted configs...")
            for hint_config in extracted_hints["config_params"]:
                # Search through config_mapping for matches
                if isinstance(self.config_mapping, dict):
                    # Mapping is a dictionary
                    for config_data in self.config_mapping.values():
                        if config_data and hint_config.lower() in config_data.get("param_name", "").lower():
                            hint_candidates["configs"].append({
                                "candidate": config_data,
                                "score": 0.98,  # Very high confidence from pattern hint
                                "source": "pattern_hint_direct",
                                "hint": hint_config
                            })
                            logger.info(f"  ✅ Found hint config: {hint_config}")
                            break
                else:
                    # Mapping is a list
                    for config_data in self.config_mapping:
                        if config_data and hint_config.lower() in config_data.get("param_name", "").lower():
                            hint_candidates["configs"].append({
                                "candidate": config_data,
                                "score": 0.98,  # Very high confidence from pattern hint
                                "source": "pattern_hint_direct",
                                "hint": hint_config
                            })
                            logger.info(f"  ✅ Found hint config: {hint_config}")
                            break
        
        return hint_candidates
    
    def grade_candidates(self, context: ErrorContext, max_iterations: int = 2) -> ErrorContext:
        """
        Step 2.5 - Self-Reflective Grading
        
        Args:
            context: Error context with candidates
            max_iterations: Maximum number of CRAG refinement iterations
            
        Returns:
            Updated ErrorContext with graded candidates
        """
        logger.info("=" * 60)
        logger.info("🎓 PHASE 2.5 - SELF-REFLECTIVE GRADING")
        logger.info("=" * 60)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Grading iteration {iteration}")
            
            # Grade function candidates
            if context.candidates["functions"]:
                graded_functions = self._grade_candidate_set(
                    context.original_error, 
                    context.candidates["functions"][:100],  # Increased limit to process more candidates
                    "functions"
                )
                context.graded_candidates["functions"] = graded_functions
            
            # Grade config candidates  
            if context.candidates["configs"]:
                graded_configs = self._grade_candidate_set(
                    context.original_error,
                    context.candidates["configs"][:100],  # Increased limit to process more candidates
                    "configs"
                )
                context.graded_candidates["configs"] = graded_configs
            
            # Check if we have sufficient relevant candidates (lowered threshold)
            relevant_functions = [c for c in context.graded_candidates["functions"] if c.get("relevance_score", 0) > 0.4]
            relevant_configs = [c for c in context.graded_candidates["configs"] if c.get("relevance_score", 0) > 0.4]
            
            logger.info(f"📊 Iteration {iteration} results:")
            logger.info(f"  Relevant functions: {len(relevant_functions)}")
            logger.info(f"  Relevant configs: {len(relevant_configs)}")
            
            # If we have sufficient relevant candidates, break
            if len(relevant_functions) >= 3 or len(relevant_configs) >= 3:
                logger.info("✅ Sufficient relevant candidates found")
                break
            
            # If not sufficient and not last iteration, regenerate queries
            if iteration < max_iterations:
                logger.info("🔄 Insufficient relevant candidates, regenerating queries...")
                context = self._regenerate_crag_queries(context)
                context = self.retrieve_candidates(context)
        
        return context
    
    def _grade_candidate_set(self, error_message: str, candidates: List[Dict], candidate_type: str) -> List[Dict]:
        """Grade a set of candidates for relevance to the error"""
        
        # Normalize candidate structure - some have nested "candidate" key, others are flat
        normalized_candidates = []
        for candidate in candidates:
            if "candidate" in candidate:
                # Already has nested structure
                normalized_candidates.append(candidate)
            else:
                # Flat structure, wrap it
                normalized_candidates.append({
                    "candidate": candidate,
                    "score": candidate.get("relevance_score", 0.5),
                    "source": candidate.get("source", "context_aware_retrieval")
                })
        
        # Prepare detailed candidate information for grading
        candidate_details = []
        for i, candidate in enumerate(normalized_candidates[:100]):  # Increased limit to process more candidates
            item = candidate["candidate"]
            if candidate_type == "functions":
                # Include complete function information
                function_name = item.get('function_name', 'Unknown')
                file_path = item.get('file_path', 'Unknown')
                code_snippet = item.get('code_snippet', item.get('code_body', ''))
                relevance_score = item.get('relevance_score', 0)
                
                detail = f"{i+1}. **Function:** {function_name}\n"
                detail += f"   **File:** {file_path}\n"
                detail += f"   **Current Score:** {relevance_score:.2f}\n"
                
                if code_snippet:
                    # Use smart truncation for very long code
                    if len(code_snippet) > 4000:
                        code_snippet = self._smart_truncate_code(code_snippet)
                    detail += f"   **Code:**\n```c\n{code_snippet}\n```\n"
                else:
                    detail += f"   **Code:** No code available\n"
                    
            else:
                # Include complete config information
                param_name = item.get('param_name', 'Unknown')
                param_value = item.get('param_value', 'Unknown')
                file_path = item.get('file_path', 'Unknown')
                line_number = item.get('line_number', 'Unknown')
                config_context = item.get('config_context', '')
                relevance_score = item.get('relevance_score', 0)
                
                detail = f"{i+1}. **Config:** {param_name}\n"
                detail += f"   **File:** {file_path}\n"
                detail += f"   **Current Value:** {param_value}\n"
                detail += f"   **Line Number:** {line_number}\n"
                detail += f"   **Current Score:** {relevance_score:.2f}\n"
                
                if config_context:
                    detail += f"   **Config Context:**\n```\n{config_context}\n```\n"
                else:
                    detail += f"   **Config Context:** No context available\n"
                    
            candidate_details.append(detail)
        
        prompt = f"""You are an expert in 5G/LTE telecommunications debugging. You should be generous with relevance scores for items that could plausibly help debug or fix the error.

Error: "{error_message}"

Evaluate these {candidate_type} for relevance to the error. You have access to the complete code/context for each candidate:

{chr(10).join(candidate_details)}

For each item, provide a relevance score (0.0-1.0) and brief reason based on the actual code/context.
Format: "1: 0.8 - reason, 2: 0.3 - reason, ..."

**CRITICAL FOR RRC ERRORS**: For RRC-related errors, be especially generous with:
- RRC rejection functions (rrc_gNB_generate_RRCReject, rrc_gNB_send_RRCReject)
- RRC error handling functions (handle_error, process_error)
- RRC setup failure functions (setup_failure, handle_failure)
- RRC context management functions (remove_ue, cleanup_context)
- Functions that handle RRCSetupRequest processing
- Functions that generate RRC messages (Reject, Failure, etc.)

Score generously - consider relevant:
- Any function containing keywords from the error (AMF, gNB, NGAP, RRC, etc.)
- Configuration parameters related to the error components
- Protocol handling functions even if indirectly related
- Error handling and setup functions
- Network interface and connection functions
- Functions that could be involved in the error scenario based on their actual implementation
- Config parameters that could affect the error based on their current values and context

**For RRC segmentation faults, ANY RRC error handling function should score 0.7+**

Score 0.5+ if there's ANY reasonable connection to the error scenario based on the actual code/context."""

        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a telecommunications debugging expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,  # Increased for more detailed responses
                temperature=0.2,
                seed=54321  # Self-reflective grading seed
            )
            
            grades_text = response.choices[0].message.content.strip()
            graded_candidates = self._parse_grades(normalized_candidates, grades_text)
            
            return graded_candidates
            
        except Exception as e:
            logger.error(f"❌ Failed to grade {candidate_type}: {e}")
            # Return candidates with default scores
            for candidate in normalized_candidates:
                candidate["relevance_score"] = 0.5
                candidate["grade_reason"] = "Grading failed, default score"
            return normalized_candidates
    
    def _parse_grades(self, candidates: List[Dict], grades_text: str) -> List[Dict]:
        """Parse GPT grading response and assign scores"""
        graded = []
        
        # Parse all numbered items from the LLM response
        # Look for patterns like "1: 0.8 - reason" throughout the text
        # Use non-greedy matching to avoid consuming too much text
        grade_matches = re.findall(r'(\d+):\s*([0-9.]+)\s*-\s*(.*?)(?=\n\d+:|$)', grades_text, re.DOTALL)
        
        # Create a mapping of item number to score and reason
        grade_map = {}
        for item_num, score_str, reason in grade_matches:
            try:
                item_num = int(item_num)
                score = min(float(score_str), 1.0)
                grade_map[item_num] = {
                    'score': score,
                    'reason': reason.strip()
                }
            except (ValueError, TypeError):
                continue
        
        # Assign grades to candidates
        for i, candidate in enumerate(candidates):
            score = 0.3  # Default score
            reason = "No grade provided"
            
            # Look up the grade for this candidate (1-indexed)
            if i + 1 in grade_map:
                score = grade_map[i + 1]['score']
                reason = grade_map[i + 1]['reason']
            
            # Set the score on the candidate object (which is wrapped in {"candidate": func})
            candidate["relevance_score"] = score
            candidate["grade_reason"] = reason
            
            # Also set the score on the inner candidate object if it exists
            if "candidate" in candidate:
                candidate["candidate"]["relevance_score"] = score
                candidate["candidate"]["grade_reason"] = reason
            
            graded.append(candidate)
        
        return sorted(graded, key=lambda x: x["relevance_score"], reverse=True)
    
    def _regenerate_crag_queries(self, context: ErrorContext) -> ErrorContext:
        """Regenerate CRAG queries based on grading feedback"""
        logger.info("🔄 Regenerating CRAG queries with feedback...")
        
        # Analyze what was found to be irrelevant
        irrelevant_functions = [c for c in context.graded_candidates.get("functions", []) if c.get("relevance_score", 0) < 0.4]
        irrelevant_configs = [c for c in context.graded_candidates.get("configs", []) if c.get("relevance_score", 0) < 0.4]
        
        feedback = f"Previous searches found these irrelevant: {[c['candidate'].get('function_name', '') for c in irrelevant_functions[:3]]}"
        
        prompt = f"""Given this error: "{context.original_error}"

Previous search was too broad. {feedback}

Generate 3-5 MORE SPECIFIC search queries focusing on:
- Exact error keywords and symbols
- Core protocol functions (not generic utilities)
- Critical configuration parameters
- Error handling and validation functions

Return ONLY the queries, one per line:"""

        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate specific technical search queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4,
                seed=98765  # Semantic search query generation seed
            )
            
            queries_text = response.choices[0].message.content.strip()
            refined_queries = [q.strip() for q in queries_text.split('\\n') if q.strip()]
            
            context.crag_queries = refined_queries
            logger.info(f"🎯 Regenerated {len(refined_queries)} refined queries")
            
        except Exception as e:
            logger.error(f"❌ Failed to regenerate queries: {e}")
        
        return context
    
    def finalize_results(self, context: ErrorContext) -> ErrorContext:
        """
        Finalize and format the results for output with comprehensive debugging
        
        Args:
            context: Error context with graded candidates
            
        Returns:
            Updated ErrorContext with final results
        """
        logger.info("=" * 60)
        logger.info("📋 FINALIZING RESULTS")
        logger.info("=" * 60)
        
        # Create comprehensive debugging information
        debug_info = self._create_debug_analysis(context)
        
        # Select only LLM-approved candidates (no limit)
        top_functions = [c for c in context.graded_candidates.get("functions", []) if c.get("relevance_score", 0) > 0.4]  # Only LLM-approved + no limit
        top_configs = [c for c in context.graded_candidates.get("configs", []) if c.get("relevance_score", 0) > 0.4]  # Only LLM-approved + no limit
        
        # Generate error analysis
        error_analysis = self._generate_error_analysis(context.original_error, top_functions, top_configs)
        
        context.final_results = {
            "error_message": context.original_error,
            "suspected_functions": [
                {
                    "function_name": c["candidate"].get("function_name", ""),
                    "file_path": c["candidate"].get("file_path", ""),
                    "relevance_score": c.get("relevance_score", 0),
                    "reason": c.get("grade_reason", ""),
                    "code_snippet": self._smart_truncate_code(c["candidate"].get("code_body", ""))
                }
                for c in top_functions
            ],
            "suspected_configs": [
                {
                    "param_name": c["candidate"].get("param_name", ""),
                    "param_value": c["candidate"].get("param_value", ""),
                    "file_path": c["candidate"].get("file_path", ""),
                    "relevance_score": c.get("relevance_score", 0),
                    "reason": c.get("grade_reason", "")
                }
                for c in top_configs
            ],
            "error_analysis": error_analysis,
            "matched_patterns": len(context.matched_patterns),
            "queries_generated": len(context.crag_queries),
            "total_candidates_retrieved": len(context.candidates.get("functions", [])) + len(context.candidates.get("configs", []))
        }
        
        # Save comprehensive debug information
        self._save_debug_analysis(debug_info, context.original_error)
        
        return context
    
    def _create_debug_analysis(self, context: ErrorContext) -> dict:
        """Create comprehensive debugging analysis"""
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "error_text": context.original_error,
            "retrieval_methods": {
                "pattern_matching": {
                    "found": len(context.matched_patterns) > 0,
                    "patterns_matched": len(context.matched_patterns),
                    "candidates_from_patterns": []
                },
                "semantic_search": {
                    "queries_generated": len(context.crag_queries),
                    "queries": context.crag_queries,
                    "candidates_per_query": {},
                    "total_candidates": 0
                },
                "symbolic_retrieval": {
                    "keywords_used": [],
                    "direct_matches": [],
                    "call_chain_expansions": [],
                    "total_candidates": 0
                },
                "llm_grading": {
                    "functions_graded": len(context.graded_candidates.get("functions", [])),
                    "configs_graded": len(context.graded_candidates.get("configs", [])),
                    "grade_distribution": {}
                }
            },
            "candidate_analysis": {
                "functions": {
                    "total_before_filtering": len(context.candidates.get("functions", [])),
                    "by_source": {},
                    "by_grade_range": {},
                    "final_selected": []
                },
                "configs": {
                    "total_before_filtering": len(context.candidates.get("configs", [])),
                    "by_source": {},
                    "by_grade_range": {},
                    "final_selected": []
                }
            },
            "cutoff_analysis": {
                "function_threshold": 0.4,
                "config_threshold": 0.4,
                "functions_excluded": [],
                "configs_excluded": []
            }
        }
        
        # Analyze function candidates by source
        for candidate in context.candidates.get("functions", []):
            source = candidate.get("source", "unknown")
            if source not in debug_info["candidate_analysis"]["functions"]["by_source"]:
                debug_info["candidate_analysis"]["functions"]["by_source"][source] = []
            debug_info["candidate_analysis"]["functions"]["by_source"][source].append({
                "function_name": candidate["candidate"]["function_name"],
                "score": candidate.get("score", 0),
                "file_path": candidate["candidate"]["file_path"]
            })
        
        # Analyze config candidates by source
        for candidate in context.candidates.get("configs", []):
            source = candidate.get("source", "unknown")
            if source not in debug_info["candidate_analysis"]["configs"]["by_source"]:
                debug_info["candidate_analysis"]["configs"]["by_source"][source] = []
            debug_info["candidate_analysis"]["configs"]["by_source"][source].append({
                "param_name": candidate["candidate"]["param_name"],
                "score": candidate.get("score", 0),
                "file_path": candidate["candidate"]["file_path"]
            })
        
        # Analyze grade distribution for functions
        function_grades = [c.get("relevance_score", 0) for c in context.graded_candidates.get("functions", [])]
        config_grades = [c.get("relevance_score", 0) for c in context.graded_candidates.get("configs", [])]
        
        debug_info["candidate_analysis"]["functions"]["by_grade_range"] = {
            "0-0.2": len([g for g in function_grades if 0 <= g < 0.2]),
            "0.2-0.4": len([g for g in function_grades if 0.2 <= g < 0.4]),
            "0.4-0.6": len([g for g in function_grades if 0.4 <= g < 0.6]),
            "0.6-0.8": len([g for g in function_grades if 0.6 <= g < 0.8]),
            "0.8-1.0": len([g for g in function_grades if 0.8 <= g <= 1.0])
        }
        
        debug_info["candidate_analysis"]["configs"]["by_grade_range"] = {
            "0-0.2": len([g for g in config_grades if 0 <= g < 0.2]),
            "0.2-0.4": len([g for g in config_grades if 0.2 <= g < 0.4]),
            "0.4-0.6": len([g for g in config_grades if 0.4 <= g < 0.6]),
            "0.6-0.8": len([g for g in config_grades if 0.6 <= g < 0.8]),
            "0.8-1.0": len([g for g in config_grades if 0.8 <= g <= 1.0])
        }
        
        # Track excluded candidates
        excluded_functions = [c for c in context.graded_candidates.get("functions", []) if c.get("relevance_score", 0) <= 0.4]
        excluded_configs = [c for c in context.graded_candidates.get("configs", []) if c.get("relevance_score", 0) <= 0.4]
        
        debug_info["cutoff_analysis"]["functions_excluded"] = [
            {
                "function_name": c["candidate"]["function_name"],
                "relevance_score": c.get("relevance_score", 0),
                "reason": f"Score {c.get('relevance_score', 0):.2f} below threshold 0.4"
            } for c in excluded_functions
        ]
        
        debug_info["cutoff_analysis"]["configs_excluded"] = [
            {
                "param_name": c["candidate"]["param_name"],
                "relevance_score": c.get("relevance_score", 0),
                "reason": f"Score {c.get('relevance_score', 0):.2f} below threshold 0.4"
            } for c in excluded_configs
        ]
        
        # Record final selections
        final_functions = [c for c in context.graded_candidates.get("functions", []) if c.get("relevance_score", 0) > 0.4]
        final_configs = [c for c in context.graded_candidates.get("configs", []) if c.get("relevance_score", 0) > 0.4]
        
        debug_info["candidate_analysis"]["functions"]["final_selected"] = [
            {
                "function_name": c["candidate"]["function_name"],
                "relevance_score": c.get("relevance_score", 0),
                "source": c.get("source", "unknown"),
                "file_path": c["candidate"]["file_path"]
            } for c in final_functions
        ]
        
        debug_info["candidate_analysis"]["configs"]["final_selected"] = [
            {
                "param_name": c["candidate"]["param_name"],
                "relevance_score": c.get("relevance_score", 0),
                "source": c.get("source", "unknown"),
                "file_path": c["candidate"]["file_path"]
            } for c in final_configs
        ]
        
        return debug_info
    
    def _save_debug_analysis(self, debug_info: dict, error_text: str):
        """Save comprehensive debug analysis to file"""
        # Create safe filename from error text
        safe_error = "".join(c for c in error_text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_error = safe_error.replace(' ', '_')[:50]  # Limit length
        
        debug_file = f"output/debug_retrieval_analysis_{safe_error}.json"
        
        with open(debug_file, "w") as f:
            json.dump(debug_info, f, indent=2)
        
        print(f"🔍 Comprehensive retrieval analysis saved to: {debug_file}")
        
        # Also create a summary file for quick reference
        summary_file = f"output/debug_summary_{safe_error}.txt"
        with open(summary_file, "w") as f:
            f.write(f"RETRIEVAL ANALYSIS SUMMARY\n")
            f.write(f"========================\n")
            f.write(f"Error: {error_text}\n")
            f.write(f"Timestamp: {debug_info['timestamp']}\n\n")
            
            f.write(f"RETRIEVAL METHODS:\n")
            f.write(f"- Pattern Matching: {'Found' if debug_info['retrieval_methods']['pattern_matching']['found'] else 'Not Found'}\n")
            f.write(f"- Semantic Search: {debug_info['retrieval_methods']['semantic_search']['queries_generated']} queries\n")
            f.write(f"- LLM Grading: {debug_info['retrieval_methods']['llm_grading']['functions_graded']} functions, {debug_info['retrieval_methods']['llm_grading']['configs_graded']} configs\n\n")
            
            f.write(f"CANDIDATE ANALYSIS:\n")
            f.write(f"- Functions: {debug_info['candidate_analysis']['functions']['total_before_filtering']} total, {len(debug_info['candidate_analysis']['functions']['final_selected'])} selected\n")
            f.write(f"- Configs: {debug_info['candidate_analysis']['configs']['total_before_filtering']} total, {len(debug_info['candidate_analysis']['configs']['final_selected'])} selected\n\n")
            
            f.write(f"FUNCTIONS BY SOURCE:\n")
            for source, candidates in debug_info['candidate_analysis']['functions']['by_source'].items():
                f.write(f"- {source}: {len(candidates)} candidates\n")
            
            f.write(f"\nCONFIGS BY SOURCE:\n")
            for source, candidates in debug_info['candidate_analysis']['configs']['by_source'].items():
                f.write(f"- {source}: {len(candidates)} candidates\n")
            
            f.write(f"\nFINAL SELECTED FUNCTIONS:\n")
            for func in debug_info['candidate_analysis']['functions']['final_selected']:
                f.write(f"- {func['function_name']} (score: {func['relevance_score']:.2f}, source: {func['source']})\n")
            
            f.write(f"\nFINAL SELECTED CONFIGS:\n")
            for config in debug_info['candidate_analysis']['configs']['final_selected']:
                f.write(f"- {config['param_name']} (score: {config['relevance_score']:.2f}, source: {config['source']})\n")
        
        print(f"📋 Quick summary saved to: {summary_file}")
    
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
        
        # Take signature + first 100 lines + last 20 lines
        preserved_lines = lines[:signature_end + 1]  # Include signature
        preserved_lines.extend(lines[signature_end + 1:signature_end + 101])  # First 100 lines after signature
        
        if len(lines) > 120:  # Only add ending if function is long enough
            preserved_lines.append("... [middle of function truncated] ...")
            preserved_lines.extend(lines[-20:])  # Last 20 lines
        
        result = '\n'.join(preserved_lines)
        
        # Still truncate if result is too long
        if len(result) > 8000:
            result = result[:8000] + "\n... [further truncated]"
        
        return result
    
    def _generate_error_analysis(self, error_message: str, functions: List[Dict], configs: List[Dict]) -> str:
        """Generate comprehensive error analysis using GPT"""
        
        func_summaries = [f"- {f['candidate'].get('function_name', '')} (score: {f.get('relevance_score', 0):.2f})" for f in functions[:3]]
        config_summaries = [f"- {c['candidate'].get('param_name', '')} = {c['candidate'].get('param_value', '')} (score: {c.get('relevance_score', 0):.2f})" for c in configs[:3]]
        
        prompt = f"""Provide a technical analysis of this 5G/LTE error:

Error: "{error_message}"

Top relevant functions:
{chr(10).join(func_summaries) if func_summaries else "None found"}

Top relevant configurations:
{chr(10).join(config_summaries) if config_summaries else "None found"}

Provide:
1. Likely root cause
2. Impact on system operation
3. Recommended investigation steps
4. Potential solutions

Keep analysis concise and technical."""

        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a 5G/LTE telecommunications expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3,
                seed=11111  # Error analysis seed
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to generate error analysis: {e}")
            return f"Error analysis failed. Manual investigation recommended for: {error_message}"
    
    def process_error(self, error_message: str) -> Dict[str, Any]:
        """
        Main entry point for error processing pipeline
        
        Args:
            error_message: Runtime error, crash dump, or error message
            
        Returns:
            Dict containing final results and analysis
        """
        logger.info("🚀 Starting Error Handling Pipeline")
        logger.info(f"📥 Input: {error_message}")
        
        try:
            # Execute pipeline phases
            context = self.error_ingest(error_message)
            context = self.pattern_match(context)
            context = self.crag_query_generation(context)
            context = self.retrieve_candidates(context)
            context = self.grade_candidates(context)
            context = self.finalize_results(context)
            
            logger.info("=" * 60)
            logger.info("🎉 ERROR HANDLING PIPELINE COMPLETED")
            logger.info("=" * 60)
            
            return context.final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                "error_message": error_message,
                "pipeline_error": str(e),
                "suspected_functions": [],
                "suspected_configs": [],
                "error_analysis": f"Pipeline processing failed: {e}"
            }

    def retrieve_candidates_with_context(self, error_text: str, deployment_context: Dict, 
                                       top_k: int = 50) -> Dict:
        """
        Enhanced context-aware retrieval with all search strategies:
        1. Direct hint lookup (from error patterns)
        2. Symbolic retrieval (call chain analysis)
        3. Semantic search (FAISS embeddings)
        
        Args:
            error_text: Raw error string
            deployment_context: Parsed deployment context from logs
            top_k: Number of top candidates to return
            
        Returns:
            Dictionary with filtered and boosted candidates from all methods
        """
        logger.info(f"🔍 Enhanced context-aware retrieval for error: {error_text}")
        
        # Extract deployment context
        role = deployment_context.get('role', 'Unknown')
        active_configs = deployment_context.get('active_configs', [])
        log_anchors = deployment_context.get('log_anchors', [])
        
        logger.info(f"📋 Deployment context - Role: {role}, Active configs: {len(active_configs)}, Log anchors: {len(log_anchors)}")
        
        # Initialize result containers
        all_function_candidates = []
        all_config_candidates = []
        
        # 1. 🎯 DIRECT HINT LOOKUP - Highest Priority
        logger.info("🎯 Performing context-aware direct hint lookup...")
        extracted_hints = self._extract_hints_from_error_patterns(error_text)
        if extracted_hints and (extracted_hints.get("functions") or extracted_hints.get("config_params")):
            hint_candidates = self._direct_hint_lookup_with_context(
                extracted_hints, role, active_configs, log_anchors, deployment_context
            )
            all_function_candidates.extend(hint_candidates["functions"])
            all_config_candidates.extend(hint_candidates["configs"])
            logger.info(f"📊 Direct hints: {len(hint_candidates['functions'])} functions, {len(hint_candidates['configs'])} configs")
        
        # 2. 🔗 SYMBOLIC RETRIEVAL - Call Chain Analysis
        logger.info("🔗 Performing context-aware symbolic retrieval...")
        symbolic_candidates = self._symbolic_retrieval_with_context(
            error_text, role, log_anchors, extracted_hints, deployment_context
        )
        all_function_candidates.extend(symbolic_candidates)
        logger.info(f"📊 Symbolic retrieval: {len(symbolic_candidates)} functions")
        
        # 3. 🔍 ERROR HANDLING PATTERN SEARCH
        logger.info("🔍 Performing error handling pattern search...")
        error_handling_candidates = self._search_error_handling_functions(error_text)
        all_function_candidates.extend(error_handling_candidates)
        logger.info(f"📊 Error handling patterns: {len(error_handling_candidates)} candidates")
        
        print("error handling",[error['candidate']['function_name'] for error in error_handling_candidates])
        # 4. 🧠 CRAG QUERY GENERATION + SEMANTIC SEARCH
        logger.info("🧠 Generating CRAG queries and performing semantic search...")
        
        # Generate CRAG queries for enhanced semantic search
        crag_queries = self._generate_crag_queries_for_error(error_text, extracted_hints)
        logger.info(f"📝 Generated {len(crag_queries)} CRAG queries: {crag_queries}")
        
        # Use CRAG queries for semantic search (combine with original error)
        search_queries = [error_text] + crag_queries
        error_embedding = self.embedding_model.encode(search_queries)
        
        # Retrieve function candidates with semantic search
        semantic_function_candidates = self._retrieve_functions_with_context(
            error_embedding, role, log_anchors, top_k
        )
        all_function_candidates.extend(semantic_function_candidates)
        
        # Retrieve config candidates with semantic search
        semantic_config_candidates = self._retrieve_configs_with_context(
            error_embedding, active_configs, log_anchors, top_k, deployment_context
        )
        all_config_candidates.extend(semantic_config_candidates)
        
        logger.info(f"📊 Semantic search: {len(semantic_function_candidates)} functions, {len(semantic_config_candidates)} configs")
        
        # with open("all_function_candidates.json", "w") as f:
        #     json.dump(all_function_candidates, f, indent=2)
        # with open("all_config_candidates.json", "w") as f:
        #     json.dump(all_config_candidates, f, indent=2)
        
        # 4. 🎯 DEDUPLICATE AND RANK
        logger.info("🎯 Deduplicating and ranking candidates...")
        final_functions = self._deduplicate_and_rank_functions(all_function_candidates, top_k)
        final_configs = self._deduplicate_and_rank_configs(all_config_candidates, top_k)
        
        # with open("final_functions.json", "w") as f:
        #     json.dump(final_functions, f, indent=2)
        # with open("final_configs.json", "w") as f:
        #     json.dump(final_configs, f, indent=2)

        # 5. 🎓 SELF-REFLECTIVE GRADING
        logger.info("🎓 Performing self-reflective grading...")
        
        # Wrap candidates in expected format for grading - preserve source field
        wrapped_functions = [{"candidate": func, "source": func.get("source", "context_aware_retrieval")} for func in final_functions]
        wrapped_configs = [{"candidate": config, "source": config.get("source", "context_aware_retrieval")} for config in final_configs]
        
        graded_functions = self._grade_candidate_set(error_text, wrapped_functions, "functions")
        graded_configs = self._grade_candidate_set(error_text, wrapped_configs, "configs")

        # with open("graded_functions.json", "w") as f:
        #     json.dump(graded_functions, f)
        # with open("graded_configs.json", "w") as f:
        #     json.dump(graded_configs, f)
        
        logger.info(f"✅ Enhanced context-aware retrieval with CRAG and grading completed:")
        logger.info(f"   Total functions found: {len(graded_functions)}")
        logger.info(f"   Total configs found: {len(graded_configs)}")
        
        return {
            "functions": graded_functions,
            "configs": graded_configs
        }

    @staticmethod
    def _is_cmake_or_build_script_path(file_path: str) -> bool:
        """True for CMakeLists.txt and *.cmake (repo-relative or absolute)."""
        if not file_path:
            return False
        p = file_path.replace("\\", "/").lower()
        return p.endswith("cmakelists.txt") or p.endswith(".cmake")

    def _is_active_config_cmake_build_system(
        self,
        file_path: str,
        active_configs: List,
        deployment_context: Dict,
        param_name: str,
    ) -> bool:
        """
        For CMake/build-system retrieval: always allow CMake script paths so logs that only
        reference .conf paths do not filter out CMake index rows.
        """
        if self._is_cmake_or_build_script_path(file_path):
            return True
        return self._is_active_config(file_path, active_configs, deployment_context, param_name)

    def _retrieve_configs_cmake_build_system(
        self,
        error_embedding: np.ndarray,
        active_configs: List[str],
        log_anchors: List[str],
        top_k: int,
        deployment_context: Dict,
    ) -> List[Dict]:
        """FAISS config search with relaxed file filtering for CMake/build logs."""
        if not self.config_index or not self.config_mapping:
            logger.warning("⚠️  Config index/mapping missing; CMake retrieval empty")
            return []

        search_k = min(max(top_k * 80, top_k), self.config_index.ntotal)
        scores, indices = self.config_index.search(error_embedding, search_k)

        cmake_first: List[Dict] = []
        fallback: List[Dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            config_info = self.config_mapping.get(str(int(idx)), {})
            if not config_info:
                continue
            param_name = config_info.get("param_name", "")
            file_path = config_info.get("file_path", "")
            if not self._is_active_config_cmake_build_system(
                file_path, active_configs, deployment_context, param_name
            ):
                continue
            param_value = config_info.get("param_value", "")
            line_number = config_info.get("line_number", None)
            config_context = self._extract_config_context(file_path, param_name, line_number)
            current_param_value = self._extract_current_param_value(file_path, param_name, line_number)
            if current_param_value:
                param_value = current_param_value
            config_text = f"{param_name} {param_value}"
            boosted_score = self._calculate_boosted_score(
                float(score), config_text, log_anchors, file_path
            )
            row = {
                "param_name": param_name,
                "param": param_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score),
                "param_value": param_value,
                "value": param_value,
                "line_number": line_number,
                "config_context": config_context,
                "source": "cmake_build_system_retrieval",
            }
            if self._is_cmake_or_build_script_path(file_path):
                cmake_first.append(row)
            else:
                fallback.append(row)

        cmake_first.sort(key=lambda x: x["score"], reverse=True)
        fallback.sort(key=lambda x: x["score"], reverse=True)
        merged = cmake_first + fallback
        return merged[:top_k]

    def _retrieve_configs_cmake_merged_faiss(
        self,
        query_texts: List[str],
        active_configs: List[str],
        log_anchors: List[str],
        top_k: int,
        deployment_context: Dict,
        asn_type_hints: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run several embedding queries, merge by best FAISS score per vector id, then
        rank rows. Boosts ASN1/*.cmake rows when linker hints name ASN.1 types.
        """
        if not self.config_index or not self.config_mapping:
            logger.warning("⚠️  Config index/mapping missing; CMake retrieval empty")
            return []

        texts = [t.strip() for t in query_texts if t and str(t).strip()]
        if not texts:
            return []

        asn_type_hints = asn_type_hints or []
        ht_parts = [x.lower() for x in asn_type_hints if x]

        emb = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        search_k = min(max(top_k * 60, 240), self.config_index.ntotal)
        best: Dict[int, float] = {}
        for i in range(len(emb)):
            q = emb[i : i + 1].astype(np.float32)
            scores, indices = self.config_index.search(q, search_k)
            for s, ix in zip(scores[0], indices[0]):
                if ix == -1:
                    continue
                ix = int(ix)
                s = float(s)
                if s > best.get(ix, -1e9):
                    best[ix] = s

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[
            : min(len(best), max(400, top_k * 45))
        ]

        cmake_first: List[Dict] = []
        fallback: List[Dict] = []

        for idx, score in ranked:
            config_info = self.config_mapping.get(str(idx), {})
            if not config_info:
                continue
            param_name = config_info.get("param_name", "")
            file_path = config_info.get("file_path", "")
            if not self._is_active_config_cmake_build_system(
                file_path, active_configs, deployment_context, param_name
            ):
                continue
            param_value = config_info.get("param_value", "")
            line_number = config_info.get("line_number", None)
            config_context = self._extract_config_context(file_path, param_name, line_number)
            current_param_value = self._extract_current_param_value(
                file_path, param_name, line_number
            )
            if current_param_value:
                param_value = current_param_value
            config_text = f"{param_name} {param_value}"
            boosted_score = self._calculate_boosted_score(
                float(score), config_text, log_anchors, file_path
            )
            fp_low = file_path.replace("\\", "/").lower()
            blob = f"{param_name} {param_value} {config_context}".lower()
            for frag in ht_parts:
                if frag and (frag in fp_low or frag in blob):
                    boosted_score += 0.14
                    break
            if "/asn1/" in fp_low and fp_low.endswith(".cmake"):
                boosted_score += 0.1
            row = {
                "param_name": param_name,
                "param": param_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score),
                "param_value": param_value,
                "value": param_value,
                "line_number": line_number,
                "config_context": config_context,
                "source": "cmake_build_system_retrieval",
            }
            if self._is_cmake_or_build_script_path(file_path):
                cmake_first.append(row)
            else:
                fallback.append(row)

        cmake_first.sort(key=lambda x: x["score"], reverse=True)
        fallback.sort(key=lambda x: x["score"], reverse=True)
        merged = cmake_first + fallback
        return merged[: max(top_k, 28)]

    def retrieve_cmake_build_system_candidates(
        self,
        error_text: str,
        log_content: str,
        deployment_context: Dict,
        top_k: int = 20,
    ) -> Dict[str, List]:
        """
        Retrieval for CMake / build-system failures: no function FAISS, no error-handling
        name scan, no CRAG. Multi-query embeddings + linker-derived hints → config index.
        """
        logger.info("🏗️ CMake/build-system mode: retrieving CMake + build config candidates only")
        active_configs = (deployment_context or {}).get("active_configs", [])
        log_anchors = (deployment_context or {}).get("log_anchors", [])

        hints = (deployment_context or {}).get("linker_derived_hints") or {}
        extra_q = hints.get("extra_search_queries") or []

        log_tail = (log_content or "")[-8000:]
        queries = [
            error_text,
            f"CMake build linker ASN.1 makefile add_library target_sources MESSAGES ASN1:\n{log_tail}",
        ]
        queries.extend(extra_q)

        raw_configs = self._retrieve_configs_cmake_merged_faiss(
            queries,
            active_configs,
            log_anchors,
            max(top_k, 28),
            deployment_context or {},
            asn_type_hints=hints.get("asn_def_types") or [],
        )

        wrapped = [
            {"candidate": c, "source": c.get("source", "cmake_build_system_retrieval")}
            for c in raw_configs
        ]
        graded_configs = self._grade_candidate_set(
            error_text, wrapped, "configs"
        )
        return {"functions": [], "configs": graded_configs}
    
    def _generate_crag_queries_for_error(self, error_text: str, extracted_hints: Dict) -> List[str]:
        """
        Generate CRAG queries for enhanced semantic search.
        
        Args:
            error_text: Original error message
            extracted_hints: Hints extracted from error patterns
            
        Returns:
            List of CRAG query strings
        """
        try:
            # Prepare context for query generation
            pattern_hints = ""
            if extracted_hints:
                functions = extracted_hints.get("functions", [])
                configs = extracted_hints.get("config_params", [])
                if functions or configs:
                    pattern_hints = f"\nKnown pattern hints - Functions: {functions}, Configs: {configs}"
            print(error_text,pattern_hints)
            prompt = f"""You are an expert system for analyzing 5G/LTE telecommunications errors in the OpenAirInterface codebase.

Given this error message:
"{error_text}"{pattern_hints}

Generate 7-10 specific search queries to find relevant code functions and configuration parameters. Each query should target:

1. **Error Handling Functions**: Look for functions that handle errors, rejections, failures
2. **Protocol-Specific Functions**: RRC, NGAP, S1AP, F1AP error handling
3. **Rejection Functions**: Functions with names containing "Reject", "Failure", "Error"
4. **Setup/Request Functions**: Functions handling setup requests and responses
5. **Configuration Parameters**: Network configuration, SCTP, AMF, gNB settings
6. **Logging and Debugging**: Functions for error logging and debugging
7. **Context Management**: UE context, association management functions

**CRITICAL**: For segmentation faults and RRC errors, ALWAYS include queries for:

- Error handling functions (handle_error, process_error, error_indication)
- Setup failure functions (setup_failure, handle_failure)
- Context cleanup functions (remove_ue, cleanup_context)

Focus on 5G/LTE/OpenAirInterface specific terms and error handling patterns.

Return ONLY the search queries, one per line, without numbering or explanations:"""
# - RRC rejection functions (rrc_gNB_generate_RRCReject, rrc_gNB_send_RRCReject)
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a telecommunications error analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                seed=22222  # Dynamic pattern generation seed
            )
            response = response.choices[0].message.content
            
            # Parse the response to extract queries
            queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '-', '*')):
                    queries.append(line)
            
            # Ensure we have at least 3 queries, fallback to basic ones if needed
            if len(queries) < 3:
                fallback_queries = [
                    f"function handling {error_text.split()[0] if error_text.split() else 'error'}",
                    f"configuration parameter for {error_text.split()[0] if error_text.split() else 'error'}",
                    f"error handling {error_text.split()[0] if error_text.split() else 'error'}"
                ]
                queries.extend(fallback_queries[:3-len(queries)])
            print(queries)
            return queries[:7]  # Limit to 7 queries max
            
        except Exception as e:
            logger.warning(f"Failed to generate CRAG queries: {e}")
            # Fallback to basic queries based on error text
            return [
                f"function handling {error_text.split()[0] if error_text.split() else 'error'}",
                f"configuration parameter for {error_text.split()[0] if error_text.split() else 'error'}",
                f"error handling {error_text.split()[0] if error_text.split() else 'error'}"
            ]
    
    def _extract_hints_from_error_patterns(self, error_text: str) -> Dict[str, List[str]]:
        """Extract hints from error patterns for direct lookup."""
        try:
            # Use error_patterns_enhanced.json which contains related_functions
            with open('database/error_patterns_enhanced.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                patterns = data.get('patterns', {})
                error_lower = error_text.lower()
                
                # Find matching pattern using flexible word-based matching
                for pattern_name, pattern_data in patterns.items():
                    pattern_list = pattern_data.get('patterns', [])
                    
                    # Try each pattern with flexible matching
                    pattern_matched = False
                    for pattern_str in pattern_list:
                        # Try exact substring match first
                        if pattern_str.lower() in error_lower or error_lower in pattern_str.lower():
                            pattern_matched = True
                            logger.info(f"✅ Pattern match (exact): {pattern_str}")
                            break
                        
                        # Try flexible word-based matching for similar phrases
                        error_words = set(error_lower.replace(".", "").replace(",", "").split())
                        pattern_words = set(pattern_str.lower().replace(".", "").replace(",", "").split())
                        
                        # Calculate word overlap - if 80% of error words are in pattern, it's a match
                        if error_words and pattern_words:
                            overlap = len(error_words.intersection(pattern_words))
                            error_word_ratio = overlap / len(error_words)
                            
                            if error_word_ratio >= 0.8:  # 80% of error words must be in pattern
                                pattern_matched = True
                                logger.info(f"✅ Flexible pattern match: {pattern_str} (overlap: {error_word_ratio:.2f})")
                                break
                    
                    if pattern_matched:
                        # Extract hints from the pattern
                        extracted_hints = {"keywords": [], "config_params": [], "functions": []}
                        
                        # Extract function hints from related_functions field
                        if "related_functions" in pattern_data:
                            extracted_hints["functions"].extend(pattern_data["related_functions"])
                            logger.info(f"✅ Extracted {len(pattern_data['related_functions'])} related functions from pattern {pattern_name}")
                        
                        # Extract config hints from related_configs field
                        if "related_configs" in pattern_data:
                            extracted_hints["config_params"].extend(pattern_data["related_configs"])
                            logger.info(f"✅ Extracted {len(pattern_data['related_configs'])} related configs from pattern {pattern_name}")
                        
                        # Extract config hints from fix_recommendations
                        if "fix_recommendations" in pattern_data:
                            for recommendation in pattern_data["fix_recommendations"]:
                                # Look for config parameter patterns
                                if 'config' in recommendation.lower() or 'parameter' in recommendation.lower():
                                    words = recommendation.split()
                                    for word in words:
                                        if '_' in word and word.replace('_', '').isalnum():
                                            extracted_hints["config_params"].append(word)
                        
                        logger.info(f"💡 Extracted hints for pattern {pattern_name}: {extracted_hints}")
                        return extracted_hints
                
                logger.warning(f"No matching pattern found for error: {error_text}")
                return {"keywords": [], "config_params": [], "functions": []}
        except Exception as e:
            logger.warning(f"Could not extract hints from error patterns: {e}")
            return {"keywords": [], "config_params": [], "functions": []}
    
    def _direct_hint_lookup_with_context(self, extracted_hints: Dict[str, List[str]], 
                                       role: str, active_configs: List[str], 
                                       log_anchors: List[str], deployment_context: Dict) -> Dict[str, List[Dict]]:
        """Context-aware direct hint lookup with role and config filtering."""
        hint_candidates = {"functions": [], "configs": []}
        
        # Direct function lookup from hints with context filtering
        if extracted_hints.get("functions"):
            logger.info(f"🔍 Looking up {len(extracted_hints['functions'])} hinted functions with context...")
            for hint_function in extracted_hints["functions"]:
                # Search through functions_mapping for exact matches
                if isinstance(self.functions_mapping, dict):
                    for func_data in self.functions_mapping.values():
                        if func_data and func_data.get("function_name") == hint_function:
                            # Apply context filtering
                            if self._is_active_function(func_data.get("file_path", ""), role, log_anchors):
                                # Extract current function body
                                current_function_body = self._extract_current_function_body(
                                    func_data.get("file_path", ""), hint_function
                                )
                                
                                hint_candidates["functions"].append({
                                    "name": hint_function,
                                    "file_path": func_data.get("file_path", ""),
                                    "score": 0.98,  # Very high confidence from pattern hint
                                    "source": "pattern_hint_direct",
                                    "code_snippet": current_function_body or func_data.get("code_snippet", ""),
                                    "hint": hint_function
                                })
                                break
        
        # Direct config lookup from hints with context filtering
        if extracted_hints.get("config_params"):
            logger.info(f"🔍 Looking up {len(extracted_hints['config_params'])} hinted configs with context...")
            for hint_config in extracted_hints["config_params"]:
                # Search through config_mapping for matches
                if isinstance(self.config_mapping, dict):
                    for config_data in self.config_mapping.values():
                        if config_data and config_data.get("param_name") == hint_config:
                            # Apply context filtering
                            if self._is_active_config(
                                config_data.get("file_path", ""), active_configs, deployment_context, hint_config
                            ):
                                # Extract current config value
                                current_value = self._extract_current_param_value(
                                    config_data.get("file_path", ""), hint_config
                                )
                                
                                hint_candidates["configs"].append({
                                    "param": hint_config,
                                    "file_path": config_data.get("file_path", ""),
                                    "value": current_value or config_data.get("param_value", ""),
                                    "score": 0.98,  # Very high confidence from pattern hint
                                    "source": "pattern_hint_direct",
                                    "hint": hint_config
                                })
                                break
        
        return hint_candidates
    
    def _symbolic_retrieval_with_context(self, error_text: str, role: str, 
                                       log_anchors: List[str], extracted_hints: Dict[str, List[str]], 
                                       deployment_context: Dict) -> List[Dict]:
        """Context-aware symbolic retrieval with call chain analysis."""
        candidates = []
        
        # If we have extracted hints, use them as seed functions for call chain analysis
        if extracted_hints and extracted_hints.get("functions"):
            candidates = self._enhanced_call_chain_analysis_with_context(
                extracted_hints, role, log_anchors, deployment_context
            )
        else:
            # Fallback to basic keyword-based retrieval with context
            candidates = self._basic_keyword_retrieval_with_context(
                error_text, role, log_anchors, deployment_context
            )
        
        return candidates
    
    def _enhanced_call_chain_analysis_with_context(self, extracted_hints: Dict[str, List[str]], 
                                                 role: str, log_anchors: List[str], 
                                                 deployment_context: Dict) -> List[Dict]:
        """Context-aware call chain analysis from hint functions."""
        candidates = []
        
        # Get seed functions from hints
        seed_functions = []
        for hint_func_name in extracted_hints.get("functions", []):
            func_data = self._find_function_by_name(hint_func_name)
            if func_data:
                seed_functions.append(func_data)
        
        # For each seed function, walk call chain in both directions with context filtering
        for seed_func in seed_functions:
            # Add the seed function itself (highest priority) if context allows
            if self._is_active_function(seed_func.get("file_path", ""), role, log_anchors):
                current_function_body = self._extract_current_function_body(
                    seed_func.get("file_path", ""), seed_func.get("function_name", "")
                )
                
                candidates.append({
                    "name": seed_func.get("function_name", ""),
                    "file_path": seed_func.get("file_path", ""),
                    "score": 0.95,
                    "source": "call_chain_seed",
                    "code_snippet": current_function_body or seed_func.get("code_snippet", ""),
                    "call_chain_depth": 0
                })
            
            # Walk call chain in both directions with context filtering
            # Walk upstream (who calls this function?)
            upstream_candidates = self._walk_call_chain_upstream_with_context(seed_func, role, log_anchors, depth=2)
            candidates.extend(upstream_candidates)
            
            # Walk downstream (what does this function call?)
            downstream_candidates = self._walk_call_chain_downstream_with_context(seed_func, role, log_anchors, depth=2)
            candidates.extend(downstream_candidates)
        
        return candidates
    
    def _walk_call_chain_upstream_with_context(self, function_data: Dict, role: str, log_anchors: List[str], depth: int) -> List[Dict]:
        """Walk call chain upstream (who calls this function) with context filtering."""
        candidates = []
        
        if depth <= 0:
            return candidates
            
        function_name = function_data.get("function_name", "")
        
        # Look for functions that call this function in function_calls.json
        for call_entry in self.function_calls:
            if function_name in call_entry.get("called_by", []):
                caller_name = call_entry.get("function")
                caller_data = self._find_function_by_name(caller_name)
                
                if caller_data and self._is_active_function(caller_data.get("file_path", ""), role, log_anchors):
                    current_function_body = self._extract_current_function_body(
                        caller_data.get("file_path", ""), caller_name
                    )
                    
                    candidates.append({
                        "name": caller_name,
                        "file_path": caller_data.get("file_path", ""),
                        "score": 0.8 - (3 - depth) * 0.1,  # Decreasing score with depth
                        "source": "call_chain_upstream",
                        "code_snippet": current_function_body or caller_data.get("code_snippet", ""),
                        "call_chain_depth": 3 - depth
                    })
                    
                    # Recursively walk upstream
                    if depth > 1:
                        upstream_candidates = self._walk_call_chain_upstream_with_context(caller_data, role, log_anchors, depth - 1)
                        candidates.extend(upstream_candidates)
        
        return candidates
    
    def _walk_call_chain_downstream_with_context(self, function_data: Dict, role: str, log_anchors: List[str], depth: int) -> List[Dict]:
        """Walk call chain downstream (what does this function call) with context filtering."""
        candidates = []
        
        if depth <= 0:
            return candidates
            
        function_name = function_data.get("function_name", "")
        
        # Look for functions called by this function in function_calls.json
        for call_entry in self.function_calls:
            if call_entry.get("function") == function_name:
                for called_func_name in call_entry.get("calls", []):
                    called_func_data = self._find_function_by_name(called_func_name)
                    
                    if called_func_data and self._is_active_function(called_func_data.get("file_path", ""), role, log_anchors):
                        current_function_body = self._extract_current_function_body(
                            called_func_data.get("file_path", ""), called_func_name
                        )
                        
                        candidates.append({
                            "name": called_func_name,
                            "file_path": called_func_data.get("file_path", ""),
                            "score": 0.8 - (3 - depth) * 0.1,  # Decreasing score with depth
                            "source": "call_chain_downstream",
                            "code_snippet": current_function_body or called_func_data.get("code_snippet", ""),
                            "call_chain_depth": 3 - depth
                        })
                        
                        # Recursively walk downstream
                        if depth > 1:
                            downstream_candidates = self._walk_call_chain_downstream_with_context(called_func_data, role, log_anchors, depth - 1)
                            candidates.extend(downstream_candidates)
        
        return candidates
    
    def _extract_functions_from_error_text(self, error_text: str) -> List[Dict]:
        """Extract specific function names mentioned in error messages or stack traces."""
        candidates = []
        
        # Common patterns for function names in error messages and stack traces
        function_patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # function_name(
            r'in\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+at',  # in function_name at
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # function function_name
            r'crash\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # crash in function_name
            r'fault\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # fault in function_name
            r'0x[0-9a-fA-F]+\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # 0x... in function_name
        ]
        
        import re
        extracted_functions = set()
        
        for pattern in function_patterns:
            matches = re.finditer(pattern, error_text)
            for match in matches:
                func_name = match.group(1)
                # Filter out common non-function words
                if (len(func_name) > 3 and 
                    func_name.lower() not in ['main', 'void', 'int', 'char', 'long', 'short', 'bool', 'true', 'false', 'null']):
                    extracted_functions.add(func_name)
        
        # Look up each extracted function in our database
        if isinstance(self.functions_mapping, dict):
            for func_data in self.functions_mapping.values():
                if func_data and func_data.get("function_name") in extracted_functions:
                    current_function_body = self._extract_current_function_body(
                        func_data.get("file_path", ""), func_data.get("function_name", "")
                    )
                    
                    candidates.append({
                        "name": func_data.get("function_name", ""),
                        "file_path": func_data.get("file_path", ""),
                        "score": 0.95,  # High confidence for direct function name matches
                        "source": "direct_function_extraction",
                        "code_snippet": current_function_body or func_data.get("code_snippet", ""),
                        "extracted_from": "error_message"
                    })
                    
                    logger.info(f"🎯 Direct function extraction found: {func_data.get('function_name', '')}")
        
        return candidates
    
    def _basic_keyword_retrieval_with_context(self, error_text: str, role: str, 
                                            log_anchors: List[str], deployment_context: Dict) -> List[Dict]:
        """Context-aware basic keyword-based retrieval."""
        candidates = []
        
        # First, try to extract specific function names mentioned in the error
        function_candidates = self._extract_functions_from_error_text(error_text)
        candidates.extend(function_candidates)
        
        # Extract keywords from error text
        keywords = [word.lower() for word in error_text.split() if len(word) > 3]
        
        # Search through functions for keyword matches with context filtering
        if isinstance(self.functions_mapping, dict):
            for func_data in self.functions_mapping.values():
                if func_data:
                    func_name = func_data.get("function_name", "").lower()
                    code_snippet = func_data.get("code_snippet", "").lower()
                    
                    # Check for keyword matches
                    if any(keyword in func_name or keyword in code_snippet for keyword in keywords):
                        # Apply context filtering
                        if self._is_active_function(func_data.get("file_path", ""), role, log_anchors):
                            current_function_body = self._extract_current_function_body(
                                func_data.get("file_path", ""), func_data.get("function_name", "")
                            )
                            
                            candidates.append({
                                "name": func_data.get("function_name", ""),
                                "file_path": func_data.get("file_path", ""),
                                "score": 0.7,  # Lower confidence for keyword matches
                                "source": "keyword_match",
                                "code_snippet": current_function_body or func_data.get("code_snippet", ""),
                                "matched_keywords": [kw for kw in keywords if kw in func_name or kw in code_snippet]
                            })
        
        return candidates
    
    def _deduplicate_and_rank_functions(self, all_candidates: List[Dict], top_k: int) -> List[Dict]:
        """Deduplicate function candidates and rank by score, preserving original source."""
        seen = {}
        
        for candidate in all_candidates:
            # Handle both wrapped and direct candidate formats
            item = candidate.get("candidate", candidate)
            func_name = item.get("function_name", item.get("name", ""))
            file_path = item.get("file_path", "")
            key = f"{func_name}:{file_path}"
            
            if key not in seen:
                # First time seeing this function - store it
                seen[key] = candidate
            elif candidate.get("score", 0) > seen[key].get("score", 0):
                # Found higher-scoring duplicate - keep higher score but preserve original source
                original_source = seen[key].get("source", "unknown")
                seen[key] = candidate
                # Restore the original source
                seen[key]["source"] = original_source
        
        # Sort by score descending and return top_k
        sorted_candidates = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]
    
    def _deduplicate_and_rank_configs(self, all_candidates: List[Dict], top_k: int) -> List[Dict]:
        """Deduplicate config candidates and rank by score, preserving original source."""
        seen = {}
        
        for candidate in all_candidates:
            # Handle both wrapped and direct candidate formats
            item = candidate.get("candidate", candidate)
            param_name = item.get("param_name", item.get("param", ""))
            file_path = item.get("file_path", "")
            key = f"{param_name}:{file_path}"
            
            if key not in seen:
                # First time seeing this config - store it
                seen[key] = candidate
            elif candidate.get("score", 0) > seen[key].get("score", 0):
                # Found higher-scoring duplicate - keep higher score but preserve original source
                original_source = seen[key].get("source", "unknown")
                seen[key] = candidate
                # Restore the original source
                seen[key]["source"] = original_source
        
        # Sort by score descending and return top_k
        sorted_candidates = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]
    
    def _find_function_by_name(self, function_name: str) -> Dict:
        """Find function data by name in the functions mapping."""
        if isinstance(self.functions_mapping, dict):
            for func_data in self.functions_mapping.values():
                if func_data and func_data.get("function_name") == function_name:
                    return func_data
        return {}
    
    def _is_active_function(self, file_path: str, role: str, log_anchors: List[str]) -> bool:
        """Check if function is relevant for the current role and context."""
        if not file_path:
            return False
        
        # Normalize file path for comparison
        normalized_path = file_path.lower().replace('\\', '/')
        
        # Role-based filtering
        if role == 'UE':
            # UE functions should be in UE-related files
            ue_indicators = ['ue', 'user', 'terminal', 'mobile']
            if any(indicator in normalized_path for indicator in ue_indicators):
                return True
            # Also allow gNB functions for RA-related issues (as we implemented before)
            if any(keyword in normalized_path for keyword in ['gnb', 'cu', 'du', 'mac', 'ra']):
                return True
        elif role == 'CU':
            # CU functions should be in CU-related files
            cu_indicators = ['cu', 'central', 'control']
            if any(indicator in normalized_path for indicator in cu_indicators):
                return True
        elif role == 'DU':
            # DU functions should be in DU-related files
            du_indicators = ['du', 'distributed', 'radio']
            if any(indicator in normalized_path for indicator in du_indicators):
                return True
        elif role == 'gNB':
            # gNB functions should be in gNB-related files
            gnb_indicators = ['gnb', 'gnodeb', 'base']
            if any(indicator in normalized_path for indicator in gnb_indicators):
                return True
        elif role == 'AMF':
            # AMF functions should be in AMF-related files
            amf_indicators = ['amf', 'access', 'mobility']
            if any(indicator in normalized_path for indicator in amf_indicators):
                return True
        
        # If no specific role match, allow all functions (fallback)
        return True
    
    def _retrieve_functions_with_context(self, error_embedding: np.ndarray, role: str, 
                                       log_anchors: List[str], top_k: int) -> List[Dict]:
        """Retrieve function candidates with role-based filtering and log anchoring."""
        logger.info("🔍 Retrieving function candidates with context...")
        
        # Perform FAISS search (get more candidates for filtering)
        search_k = min(top_k * 3, self.functions_index.ntotal)
        scores, indices = self.functions_index.search(error_embedding, search_k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Get function info from mapping
            function_info = self.functions_mapping.get(str(idx), {})
            if not function_info:
                continue
            
            function_name = function_info.get('function_name', '')
            file_path = function_info.get('file_path', '')
            code_snippet = function_info.get('code_snippet', function_info.get('code_body', ''))
            
            # Extract current function body from the actual file
            current_function_body = self._extract_current_function_body(file_path, function_name)
            if current_function_body:
                code_snippet = current_function_body  # Use current function body from file
            
            # Apply role-based filtering
            if not self._is_role_appropriate(function_name, file_path, role):
                continue
            
            # Calculate boosted score
            boosted_score = self._calculate_boosted_score(score, code_snippet, log_anchors, file_path)
            
            candidates.append({
                "name": function_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score),
                "code_snippet": code_snippet,
                "source": "context_aware_retrieval"
            })
        
        # Sort by boosted score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _retrieve_configs_with_context(self, error_embedding: np.ndarray, active_configs: List[str],
                                     log_anchors: List[str], top_k: int, deployment_context: Dict = None) -> List[Dict]:
        """Retrieve config candidates with active config filtering and log anchoring."""
        logger.info("🔍 Retrieving config candidates with context...")

        # Perform FAISS search (get more candidates for filtering)
        # Increase search range to ensure we find configs from active files
        search_k = min(top_k * 50, self.config_index.ntotal)  # Search more broadly
        scores, indices = self.config_index.search(error_embedding, search_k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Get config info from mapping
            config_info = self.config_mapping.get(str(idx), {})
            if not config_info:
                continue
            
            param_name = config_info.get('param_name', '')
            file_path = config_info.get('file_path', '')
            param_value = config_info.get('param_value', '')
            line_number = config_info.get('line_number', None)
            
            # Apply active config filtering with special handling for UE RA errors
            if not self._is_active_config(file_path, active_configs, deployment_context, param_name):
                continue
            
            # Extract config context from the actual file
            config_context = self._extract_config_context(file_path, param_name, line_number)
            
            # Extract current parameter value from the actual file
            current_param_value = self._extract_current_param_value(file_path, param_name, line_number)
            if current_param_value:
                param_value = current_param_value  # Use current value from file
            
            # Calculate boosted score
            config_text = f"{param_name} {param_value}"
            boosted_score = self._calculate_boosted_score(score, config_text, log_anchors, file_path)
            
            candidates.append({
                "param": param_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score),
                "value": param_value,
                "line_number": line_number,
                "config_context": config_context,
                "source": "context_aware_retrieval"
            })
        
        # Sort by boosted score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _extract_config_context(self, file_path: str, param_name: str, line_number: int = None) -> str:
        """Extract configuration context from the actual config file."""
        try:
            # Handle file path - check if it already contains the prefix
            if file_path.startswith(self.openair_codebase_file_name):
                full_path = file_path.replace("\\", "/")  # Normalize path separators
            else:
                full_path = os.path.join(self.openair_codebase_file_name, file_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Config file not found: {full_path}")
                return ""
            
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # If we have a line number, use it; otherwise search for the parameter
            target_line = None
            if line_number and 0 < line_number <= len(lines):
                target_line = line_number - 1  # Convert to 0-based index
            else:
                # Search for the parameter in the file
                for i, line in enumerate(lines):
                    if param_name in line and ('=' in line or ':' in line):
                        target_line = i
                        break
            
            if target_line is None:
                return ""
            
            # Extract context around the parameter (5 lines before and after)
            context_start = max(0, target_line - 5)
            context_end = min(len(lines), target_line + 6)
            
            context_lines = []
            for i in range(context_start, context_end):
                line_content = lines[i].rstrip()
                if i == target_line:
                    # Mark the target parameter
                    context_lines.append(f"  {i+1:3d}: {line_content}  <-- TARGET PARAMETER")
                else:
                    context_lines.append(f"  {i+1:3d}: {line_content}")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.warning(f"Could not extract config context from {file_path}: {e}")
            return ""
    
    def _extract_current_param_value(self, file_path: str, param_name: str, line_number: int = None) -> str:
        """Extract the current parameter value from the actual config file."""
        try:
            # Handle file path - check if it already contains the prefix
            if file_path.startswith(self.openair_codebase_file_name):
                full_path = file_path.replace("\\", "/")  # Normalize path separators
            else:
                full_path = os.path.join(self.openair_codebase_file_name, file_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Config file not found: {full_path}")
                return ""
            
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the parameter line - prioritize search by name over line number
            target_line = None
            
            # First, try to find by parameter name (more reliable)
            for i, line in enumerate(lines):
                if param_name in line and ('=' in line or ':' in line):
                    target_line = i
                    break
            
            # If not found by name and line number is provided, try line number as fallback
            if target_line is None and line_number is not None:
                if 0 <= line_number - 1 < len(lines):
                    target_line = line_number - 1
            
            if target_line is None:
                return ""
            
            # Extract the parameter value from the line
            line_content = lines[target_line].strip()
            
            # Handle different parameter formats
            if '=' in line_content:
                # Extract value after '='
                value_part = line_content.split('=', 1)[1].strip()
                # Remove trailing semicolon if present
                if value_part.endswith(';'):
                    value_part = value_part[:-1].strip()
                return value_part
            elif ':' in line_content:
                # Extract value after ':'
                value_part = line_content.split(':', 1)[1].strip()
                return value_part
            
            return ""
            
        except Exception as e:
            logger.warning(f"Could not extract current param value from {file_path}: {e}")
            return ""
    
    def _extract_current_function_body(self, file_path: str, function_name: str) -> str:
        """Extract the current function body from the actual source file."""
        try:
            # Handle file path - check if it already contains the prefix
            if file_path.startswith(self.openair_codebase_file_name):
                full_path = file_path.replace("\\", "/")  # Normalize path separators
            else:
                full_path = os.path.join(self.openair_codebase_file_name, file_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"File not found: {full_path}")
                return ""
            
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the function definition
            function_start = None
            for i, line in enumerate(lines):
                # Look for function definition patterns
                if (function_name in line and 
                    ('(' in line or ' ' + function_name + '(' in line or 
                     function_name + '(' in line or function_name + ' (' in line)):
                    function_start = i
                    break
            
            if function_start is None:
                return ""
            
            # Find the function body boundaries by counting braces
            brace_count = 0
            function_end = None
            in_function = False
            
            for i in range(function_start, len(lines)):
                line = lines[i]
                
                # Count opening and closing braces
                for char in line:
                    if char == '{':
                        brace_count += 1
                        in_function = True
                    elif char == '}':
                        brace_count -= 1
                        if in_function and brace_count == 0:
                            function_end = i
                            break
                
                if function_end is not None:
                    break
            
            if function_end is None:
                # If we can't find the end, return a reasonable amount of lines
                function_end = min(function_start + 50, len(lines) - 1)
            
            # Extract the function body
            function_lines = lines[function_start:function_end + 1]
            function_body = ''.join(function_lines)
            
            return function_body.strip()
            
        except Exception as e:
            logger.warning(f"Could not extract current function body from {file_path}: {e}")
            return ""
    
    def _is_role_appropriate(self, function_name: str, file_path: str, role: str) -> bool:
        """Check if function is appropriate for the given role."""
        if role == 'CU':
            # For CU: keep only CU-related functions, exclude DU/UE
            if any(exclude in file_path.lower() for exclude in ['/du/', '/ue/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['du_', 'ue_']):
                return False
            return True
        
        elif role == 'DU':
            # For DU: keep only DU-related functions, exclude CU/UE
            if any(exclude in file_path.lower() for exclude in ['/cu/', '/ue/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['cu_', 'ue_']):
                return False
            return True
        
        elif role == 'UE':
            # For UE: keep only UE-related functions, exclude CU/DU
            if any(exclude in file_path.lower() for exclude in ['/cu/', '/du/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['cu_', 'du_']):
                return False
            return True
        
        elif role == 'gNB':
            # For gNB: keep gNB-related functions
            if any(keyword in file_path.lower() for keyword in ['gnb', 'ngap']):
                return True
            if any(function_name.lower().startswith(prefix) for prefix in ['gnb_', 'ngap_']):
                return True
            return False
        
        # For unknown role, keep all functions
        return True
    
    def _is_active_config(self, file_path: str, active_configs: List, deployment_context: Dict = None, param_name: str = "") -> bool:
        """Check if config file is in the active configs list."""
        if not active_configs:
            return True  # If no active configs specified, keep all
        
        # Normalize file path for comparison
        normalized_file_path = file_path.replace('\\', '/').lower()
        
        # Special handling for UE RA errors - include gNB configs for RA-related parameters
        if deployment_context and deployment_context.get('role') == 'UE':
            # Check if this is an RA-related parameter
            ra_keywords = ['ra_', 'random_access', 'contention', 'preamble', 'timer', 'backoff']
            if any(keyword in param_name.lower() for keyword in ra_keywords):
                # For RA-related parameters, also include gNB config files
                if 'gnb' in normalized_file_path or 'cu' in normalized_file_path or 'du' in normalized_file_path:
                    return True
        
        # Check if any active config matches the file path
        for active_config in active_configs:
            # Handle new format with missing/used structure
            if isinstance(active_config, dict):
                # Check both missing and used configs
                configs_to_check = []
                
                # Add used config if exists
                used_config = active_config.get('used')
                if used_config:
                    configs_to_check.append(used_config)
                
                # Add missing config if exists (important for 5g_sa_ue.conf!)
                missing_config = active_config.get('missing')
                if missing_config:
                    configs_to_check.append(missing_config)
                
                # Check both paths
                for config_path in configs_to_check:
                    # Normalize the config path
                    normalized_config = config_path.replace('\\', '/').lower()
                    
                    # Extract just the filename and relative path for comparison
                    config_filename = os.path.basename(normalized_config)
                    config_rel_path = normalized_config.split('targets/')[-1] if 'targets/' in normalized_config else normalized_config
                    
                    # Also extract relative path from openair3/ if present
                    if 'openair3/' in normalized_config:
                        config_openair3_path = 'openair3/' + normalized_config.split('openair3/')[-1]
                    else:
                        config_openair3_path = None
                    
                    # Check multiple matching strategies
                    if (config_path in file_path or file_path in config_path or
                        config_filename in normalized_file_path or
                        config_rel_path in normalized_file_path or
                        normalized_file_path.endswith(config_rel_path) or
                        (config_openair3_path and config_openair3_path in normalized_file_path)):
                        return True
            else:
                # Handle old string format
                normalized_active = active_config.replace('\\', '/').lower()
                if (active_config in file_path or file_path in active_config or
                    normalized_active in normalized_file_path or
                    normalized_file_path in normalized_active):
                    return True
        
        return False
    
    def _calculate_boosted_score(self, original_score: float, text: str, log_anchors: List[str], file_path: str = "") -> float:
        """Calculate boosted score based on log anchor matches and file importance."""
        boosted_score = original_score
        
        # File-specific boosting for important config files
        if file_path:
            file_path_lower = file_path.lower()
            # Boost cu_gnb.conf and du_gnb.conf files as they are primary config files
            if "cu_gnb.conf" in file_path_lower or "du_gnb.conf" in file_path_lower:
                boosted_score += 0.5  # Very significant boost for primary config files
                logger.debug(f"Boosted score by 0.5 for primary config file: {file_path}")
            # Penalize generic/example config files
            elif any(generic in file_path_lower for generic in ['ci-scripts', 'example', 'template', 'band78', 'aw2s', 'ddsuu']):
                boosted_score -= 0.3  # Penalty for generic config files
                logger.debug(f"Penalized score by 0.3 for generic config file: {file_path}")
            # Small boost for other gNB config files
            elif "gnb" in file_path_lower and ".conf" in file_path_lower:
                boosted_score += 0.1  # Small boost for other gNB config files
                logger.debug(f"Boosted score by 0.1 for gNB config file: {file_path}")
        
        if not log_anchors:
            return min(1.0, boosted_score)  # Cap at 1.0
        
        # Check for log anchor matches in text
        text_lower = text.lower()
        anchor_matches = 0
        
        for anchor in log_anchors:
            anchor_lower = anchor.lower()
            # Extract key terms from anchor (remove brackets, timestamps, etc.)
            anchor_terms = [term.strip() for term in anchor_lower.split() if len(term) > 3]
            
            for term in anchor_terms:
                if term in text_lower:
                    anchor_matches += 1
                    break  # Count each anchor only once
        
        # Boost score based on anchor matches
        if anchor_matches > 0:
            boost = min(0.2, anchor_matches * 0.1)  # Max boost of 0.2
            boosted_score += boost
            logger.debug(f"Boosted score by {boost} for {anchor_matches} anchor matches")
        
        return min(1.0, boosted_score)  # Cap at 1.0

def main():
    """Demo the error handling pipeline"""
    print("🔧 Error Handling Pipeline with CRAG")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        pipeline = ErrorHandlingPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Example error messages
    test_errors = [
        "No AMF associated to gNB",
        "SCTP connection failed to AMF",
        "RRC setup failure: UE context not found",
        "NGAP setup failed: invalid configuration"
    ]
    
    print("\\n🧪 Testing with example errors...")
    
    for error_msg in test_errors:
        print(f"\\n{'='*80}")
        print(f"🔍 Processing: {error_msg}")
        print('='*80)
        
        results = pipeline.process_error(error_msg)
        
        print("\\n📊 RESULTS:")
        print(f"🔧 Suspected Functions: {len(results.get('suspected_functions', []))}")
        for func in results.get('suspected_functions', [])[:3]:
            print(f"  - {func.get('function_name', '')} (score: {func.get('relevance_score', 0):.2f})")
        
        print(f"⚙️  Suspected Configs: {len(results.get('suspected_configs', []))}")
        for config in results.get('suspected_configs', [])[:3]:
            print(f"  - {config.get('param_name', '')} = {config.get('param_value', '')} (score: {config.get('relevance_score', 0):.2f})")
        
        print(f"\\n💡 Analysis: {results.get('error_analysis', 'No analysis available')[:200]}...")
        
        # For demo, only process first error unless user wants to continue
        user_input = input("\\nContinue with next error? (y/n): ")
        if user_input.lower() != 'y':
            break
    
    print("\\n✅ Demo completed!")

if __name__ == "__main__":
    main()
