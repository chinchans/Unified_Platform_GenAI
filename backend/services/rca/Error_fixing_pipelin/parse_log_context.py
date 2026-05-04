#!/usr/bin/env python3
"""
Log Context Parser for Telecom Error Fixing Pipeline
Extracts deployment context from runtime log files to enhance error analysis.
"""

import re
import json
import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogContextParser:
    """Parser for extracting deployment context from telecom log files."""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the log context parser.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        self.openair_codebase_file_name = openair_codebase_file_name
        
        # Regex patterns for different types of information
        self.patterns = {
            'role': [
                r'\[CU\]', r'\[DU\]', r'\[UE\]', r'\[gNB\]', r'\[AMF\]',
                r'Starting.*CU', r'Starting.*DU', r'Starting.*UE',
                r'CU Task', r'DU Task', r'UE Task',
                r'F1AP.*CU', r'F1AP.*DU', r'NGAP.*gNB'
            ],
            'config_files': [
                r'Reading.*\.conf', r'Loading.*\.conf', r'Config.*\.conf',
                r'\.conf', r'\.yaml', r'\.json',
                r'CONF/.*\.conf', r'conf_files/.*\.conf'
            ],
            'ipv4': [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'ipv4\s*=\s*["\']?([0-9]{1,3}\.){3}[0-9]{1,3}["\']?',
                r'address\s*:\s*([0-9]{1,3}\.){3}[0-9]{1,3}',
                r'GTPu address\s*:\s*([0-9]{1,3}\.){3}[0-9]{1,3}'
            ],
            'ports': [
                r'port\s*[=:]\s*(\d+)',
                r'ngap_port\s*[=:]\s*(\d+)',
                r'sctp_port\s*[=:]\s*(\d+)',
                r'port\s+(\d+)',
                r':(\d+)\s'
            ],
            'association_state': [
                r'assoc.*CONNECTED', r'assoc.*DISCONNECTED',
                r'SCTP.*CONNECTED', r'SCTP.*DISCONNECTED',
                r'NGAP.*CONNECTED', r'NGAP.*DISCONNECTED',
                r'association.*established', r'association.*failed',
                r'No AMF.*associated', r'AMF.*registered'
            ],
            'log_anchors': [
                r'ERROR', r'WARN', r'failed', r'No AMF',
                r'NGAP', r'SCTP', r'F1AP', r'registration',
                r'association', r'connection', r'timeout'
            ]
        }
        
        # Keywords for role detection
        self.role_keywords = {
            'CU': ['CU', 'Central Unit', 'F1AP.*CU', 'CU Task'],
            'DU': ['DU', 'Distributed Unit', 'F1AP.*DU', 'DU Task'],
            'UE': ['UE', 'User Equipment', 'UE Task', 'rnti'],
            'gNB': ['gNB', 'gnb', 'NGAP.*gNB', 'gNB_APP'],
            'AMF': ['AMF', 'amf', 'Access.*Mobility.*Function']
        }
    
    def parse_log_file(self, log_file_path: str) -> Dict:
        """
        Parse log file and extract deployment context.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Dictionary containing extracted deployment context
        """
        logger.info(f"Parsing log file: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            logger.error(f"Log file not found: {log_file_path}")
            return self._get_empty_context()
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                log_lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return self._get_empty_context()
        
        # Extract context from log content
        context = {
            "role": self._detect_role(log_content),
            "active_configs": self._extract_config_files(log_content),
            "network_params": self._extract_network_params(log_content),
            "log_anchors": self._extract_log_anchors(log_content)
        }
        
        # Apply missing config resolution
        context["active_configs"] = self._resolve_missing_configs(
            context["active_configs"], 
            context["role"]
        )
        
        logger.info(f"Successfully parsed log file. Role: {context['role']}")
        return context
    
    def _detect_role(self, log_content: str) -> str:
        """Detect the role (CU/DU/UE/gNB/AMF) from log content."""
        log_content_lower = log_content.lower()
        
        # Priority-based role detection (most specific first)
        role_indicators = [
            # CU indicators (highest priority)
            (['f1ap.*cu', 'cu task', 'central unit', 'nr-softmodem.*cu'], 'CU'),
            # DU indicators
            (['f1ap.*du', 'du task', 'distributed unit'], 'DU'),
            # UE indicators (higher priority than gNB to avoid false matches)
            (['task_phy_ue', '\\[ue\\d+\\]', 'ue0', 'ue1', 'ue task', 'user equipment', 'rnti.*ue'], 'UE'),
            # gNB indicators (lower priority to avoid false matches with UE logs)
            (['gnb_app', 'gNB_APP', 'ngap.*gnb', 'gnb.*registration'], 'gNB'),
            # AMF indicators
            (['amf.*app', 'amf.*task', 'access.*mobility.*function'], 'AMF')
        ]
        
        # Check for specific role indicators
        for indicators, role in role_indicators:
            for indicator in indicators:
                if re.search(indicator, log_content, re.IGNORECASE):
                    return role
        
        # Fallback: count occurrences of each role
        role_scores = {}
        for role, keywords in self.role_keywords.items():
            score = 0
            for keyword in keywords:
                # Case-insensitive search
                pattern = re.compile(re.escape(keyword.lower()), re.IGNORECASE)
                matches = pattern.findall(log_content_lower)
                score += len(matches)
            role_scores[role] = score
        
        # Find role with highest score
        if role_scores:
            detected_role = max(role_scores, key=role_scores.get)
            if role_scores[detected_role] > 0:
                return detected_role
        
        return 'Unknown'
    
    def _extract_config_files(self, log_content: str) -> List[str]:
        """Extract active configuration file paths from log content."""
        config_files = set()
        
        # Look for specific configuration loading patterns
        config_patterns = [
            r'Reading.*?([^\s]+\.conf)',
            r'Loading.*?([^\s]+\.conf)',
            r'Config.*?([^\s]+\.conf)',
            r'CONF/([^\s]+\.conf)',
            r'conf_files/([^\s]+\.conf)',
            r'([^\s]+\.conf)',
            r'([^\s]+\.yaml)',
            r'([^\s]+\.json)'
        ]
        
        for pattern in config_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                # Clean up the path
                config_path = match.strip()
                if config_path and (config_path.endswith('.conf') or 
                                  config_path.endswith('.yaml') or 
                                  config_path.endswith('.json')):
                    # Remove any leading/trailing quotes or brackets
                    config_path = re.sub(r'^["\'\[\(]+|["\'\]\)]+$', '', config_path)
                    config_files.add(config_path)
        
        # Filter out generic patterns and keep only actual file paths
        filtered_configs = []
        for config in config_files:
            # Skip generic patterns
            if config in ['.conf', '.yaml', '.json']:
                continue
            # Keep paths that look like actual config files
            if ('/' in config or '\\' in config or config.endswith('.conf') or 
                config.endswith('.yaml') or config.endswith('.json')):
                filtered_configs.append(config)
        
        return filtered_configs
    
    def _extract_network_params(self, log_content: str) -> Dict:
        """Extract network parameters from log content."""
        network_params = {
            "gnb_ipv4": None,
            "amf_ipv4": None,
            "ngap_port": None,
            "assoc_state": "UNKNOWN"
        }
        
        # Extract IPv4 addresses
        ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ip_addresses = re.findall(ipv4_pattern, log_content)
        
        # Try to identify gNB and AMF IPs based on context
        for ip in ip_addresses:
            # Look for context around the IP address
            ip_context = self._get_ip_context(log_content, ip)
            
            if any(keyword in ip_context.lower() for keyword in ['gnb', 'gNB', 'gtpu', 'local']):
                if not network_params["gnb_ipv4"]:
                    network_params["gnb_ipv4"] = ip
            elif any(keyword in ip_context.lower() for keyword in ['amf', 'remote']):
                if not network_params["amf_ipv4"]:
                    network_params["amf_ipv4"] = ip
        
        # Extract port numbers
        port_patterns = [
            r'ngap_port\s*[=:]\s*(\d+)',
            r'port\s*[=:]\s*(\d+)',
            r':(\d+)\s'
        ]
        
        for pattern in port_patterns:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            for match in matches:
                if match and match.isdigit():
                    port = int(match)
                    if 1000 <= port <= 65535:  # Valid port range
                        network_params["ngap_port"] = port
                        break
        
        # Detect association state
        assoc_patterns = [
            (r'assoc.*CONNECTED|SCTP.*CONNECTED|NGAP.*CONNECTED', 'CONNECTED'),
            (r'assoc.*DISCONNECTED|SCTP.*DISCONNECTED|NGAP.*DISCONNECTED', 'DISCONNECTED'),
            (r'No AMF.*associated|AMF.*not.*registered', 'DISCONNECTED'),
            (r'AMF.*registered|association.*established', 'CONNECTED')
        ]
        
        for pattern, state in assoc_patterns:
            if re.search(pattern, log_content, re.IGNORECASE):
                network_params["assoc_state"] = state
                break
        
        return network_params
    
    def _get_ip_context(self, log_content: str, ip: str) -> str:
        """Get context around an IP address in the log."""
        # Find the line containing the IP
        lines = log_content.split('\n')
        for line in lines:
            if ip in line:
                return line
        return ""
    
    def _extract_log_anchors(self, log_content: str) -> List[str]:
        """Extract important log lines (errors, warnings, key events)."""
        anchors = []
        lines = log_content.split('\n')
        
        # Priority keywords (most important first)
        priority_keywords = [
            'No AMF.*associated', 'ERROR', 'WARN', 'failed', 'NGAP', 'SCTP', 
            'F1AP', 'registration', 'association', 'connection', 'timeout',
            # RRC-specific debug information
            'NR_InitialUE_Identity_PR', 'rrcSetupRequest.*ue_Identity', 'present.*value',
            'randomValue', 'ng_5G_S_TMSI', 'RA.*Contention.*Resolution', 'segmentation.*fault',
            'Decoding.*CCCH', 'RNTI.*payload', 'UE.*context'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Clean up the line (remove ANSI color codes)
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            # Check for priority keywords
            for keyword in priority_keywords:
                if re.search(keyword, clean_line, re.IGNORECASE):
                    if clean_line not in anchors:
                        anchors.append(clean_line)
                        break
        
        # Sort by importance (errors first, then warnings, then info)
        def sort_priority(line):
            line_lower = line.lower()
            if 'error' in line_lower or 'failed' in line_lower or 'no amf' in line_lower:
                return 0
            elif 'warn' in line_lower:
                return 1
            else:
                return 2
        
        anchors.sort(key=sort_priority)
        
        # Limit to most important anchors (max 20)
        return anchors[:20]
    
    def extract_detailed_log_context(self, log_content: str, error_message: str = None) -> Dict[str, Any]:
        """Extract detailed log context including error sequences and debug information."""
        lines = log_content.split('\n')
        context = {
            "total_lines": len(lines),
            "error_sequences": [],
            "debug_values": {},
            "relevant_sections": []
        }
        
        # Extract debug values (like enum constants)
        debug_patterns = {
            "NR_InitialUE_Identity_PR_randomValue": r'NR_InitialUE_Identity_PR_randomValue.*?=\s*(\d+)',
            "ue_Identity_present": r'rrcSetupRequest.*ue_Identity\.present.*?value.*?=\s*(\d+)',
            "NR_InitialUE_Identity_PR_ng_5G_S_TMSI_Part1": r'NR_InitialUE_Identity_PR_ng_5G_S_TMSI_Part1.*?value.*?=\s*(\d+)',
            "RNTI_values": r'RNTI\s+([0-9a-fA-Fx]+)',
            "UE_IDs": r'UE\s+([0-9a-fA-Fx]+)'
        }
        
        for line in lines:
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
            for key, pattern in debug_patterns.items():
                match = re.search(pattern, clean_line, re.IGNORECASE)
                if match:
                    if key not in context["debug_values"]:
                        context["debug_values"][key] = []
                    context["debug_values"][key].append({
                        "value": match.group(1),
                        "line": clean_line
                    })
        
        # Extract error sequences (lines around errors)
        error_keywords = ['failed', 'error', 'segmentation', 'fault', 'expired', 'timeout']
        for i, line in enumerate(lines):
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
            if any(keyword in clean_line.lower() for keyword in error_keywords):
                # Get context around error (5 lines before and after)
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                sequence = []
                for j in range(start, end):
                    marker = " >>> " if j == i else "     "
                    clean_line_j = re.sub(r'\x1b\[[0-9;]*m', '', lines[j].strip())
                    sequence.append(f"{marker}{clean_line_j}")
                
                context["error_sequences"].append({
                    "error_line": clean_line,
                    "line_number": i + 1,
                    "context": sequence
                })
        
        return context

    def extract_compiler_ground_truth_errors(self, log_content: str) -> List[Dict[str, Any]]:
        """
        Extract compiler ground-truth errors directly from build logs.

        This parser is intentionally conservative and only captures explicit
        compiler errors with file/line coordinates.
        """
        lines = log_content.split('\n')
        results: List[Dict[str, Any]] = []
        current_function = None
        current_function_file = None

        function_ctx_re = re.compile(r"^(?P<file>.+?):\s*In function [‘'](?P<func>[^’']+)[’']:")
        # GCC/clang with column; also accept `file:line: error:` (no column) — common in some toolchains.
        error_re_with_col = re.compile(
            r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.+)$"
        )
        error_re_line_only = re.compile(
            r"^(?P<file>.+?):(?P<line>\d+):\s*error:\s*(?P<msg>.+)$"
        )
        quoted_symbol_re = re.compile(r"[‘']([^’']+)[’']")

        def _categorize_error(message: str) -> str:
            m = message.lower()
            if "unknown type name" in m:
                return "unknown_type_name"
            if "not a structure or union" in m:
                return "member_access_on_non_struct"
            if "incompatible type for argument" in m:
                return "incompatible_argument_type"
            if "undeclared" in m:
                return "undeclared_identifier"
            return "compiler_error"

        for raw_line in lines:
            line = re.sub(r'\x1b\[[0-9;]*m', '', raw_line.strip())
            if not line:
                continue

            fn_match = function_ctx_re.match(line)
            if fn_match:
                current_function = fn_match.group("func")
                current_function_file = fn_match.group("file")
                continue

            err_match = error_re_with_col.match(line) or error_re_line_only.match(line)
            if not err_match:
                continue

            file_path = err_match.group("file")
            line_no = int(err_match.group("line"))
            _col = err_match.groupdict().get("col")
            col_no = int(_col) if _col is not None else 0
            message = err_match.group("msg").strip()
            symbols = quoted_symbol_re.findall(message)

            # Prefer function context only when file appears to match.
            fn_name = None
            if current_function:
                if (current_function_file and current_function_file == file_path) or (not current_function_file):
                    fn_name = current_function

            entry = {
                "file_path": file_path,
                "line_number": line_no,
                "column_number": col_no,
                "function_name": fn_name,
                "error_text": message,
                "error_code_category": _categorize_error(message),
                "symbol_candidates": symbols[:8],
                "confidence": 1.0,
                "priority_score": 1.0,
                "source": "compiler_ground_truth"
            }
            results.append(entry)

        return results

    def cluster_and_classify_compiler_errors(
        self, errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Group parsed compiler errors into clusters and mark primary vs secondary.

        Primary: root causes (unknown type, undeclared id, incompatible args, generic).
        Secondary: member_access_on_non_struct in the same file+function after an
        earlier unknown_type / undeclared_identifier in that scope (cascade).
        """
        empty: Dict[str, Any] = {
            "clusters": [],
            "primary_cluster_ids": [],
            "secondary_cluster_ids": [],
        }
        if not errors:
            return empty

        def _norm_path(p: Optional[str]) -> str:
            if not p:
                return ""
            return str(p).replace("\\", "/").lower()

        member_member_re = re.compile(
            r"request for member [‘']([^’']+)[’']", re.IGNORECASE
        )
        incompatible_callee_re = re.compile(
            r"argument\s+\d+\s+of\s+[‘']([^’']+)[’']", re.IGNORECASE
        )

        def _cluster_key(err: Dict[str, Any]) -> Tuple[Any, ...]:
            cat = err.get("error_code_category") or "compiler_error"
            fp = _norm_path(err.get("file_path"))
            fn = err.get("function_name") or "_"
            msg = err.get("error_text") or ""
            syms = err.get("symbol_candidates") or []

            if cat == "unknown_type_name":
                sym = syms[0] if syms else "unknown"
                return (cat, fp, sym)
            if cat == "undeclared_identifier":
                sym = syms[0] if syms else (msg[:80] if msg else "unknown")
                return (cat, fp, sym)
            if cat == "member_access_on_non_struct":
                m = member_member_re.search(msg)
                mem = m.group(1) if m else ""
                return (cat, fp, fn, mem)
            if cat == "incompatible_argument_type":
                m = incompatible_callee_re.search(msg)
                callee = m.group(1) if m else ""
                return (cat, fp, fn, callee)
            ln = err.get("line_number")
            return (cat, fp, fn, ln if ln is not None else 0)

        buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for err in errors:
            if not isinstance(err, dict):
                continue
            k = _cluster_key(err)
            buckets.setdefault(k, []).append(err)

        clusters: List[Dict[str, Any]] = []
        for idx, (_key, items) in enumerate(buckets.items()):
            lines = [
                e.get("line_number")
                for e in items
                if isinstance(e.get("line_number"), int)
            ]
            fns_set = set()
            for e in items:
                n = e.get("function_name")
                if n:
                    fns_set.add(n)
            file_path = items[0].get("file_path", "") if items else ""
            cat = items[0].get("error_code_category", "compiler_error") if items else "compiler_error"
            sample = items[:6]
            clusters.append(
                {
                    "cluster_id": f"c{idx}",
                    "cluster_key": "|".join(str(x) for x in _key),
                    "category": cat,
                    "file_path": file_path,
                    "function_names": sorted(fns_set),
                    "error_count": len(items),
                    "min_line": min(lines) if lines else 0,
                    "errors": sample,
                }
            )

        for c in clusters:
            if c["category"] == "member_access_on_non_struct":
                c["role"] = None
                c["likely_resolved_by"] = []
            else:
                c["role"] = "primary"
                c["likely_resolved_by"] = []

        earliest_unknown: Dict[Tuple[str, str], Tuple[int, str]] = {}
        for c in clusters:
            if c["category"] not in (
                "unknown_type_name",
                "undeclared_identifier",
            ):
                continue
            fps = _norm_path(c["file_path"])
            for fn in c["function_names"] or ["_"]:
                k = (fps, fn)
                ml = c["min_line"]
                if k not in earliest_unknown or ml < earliest_unknown[k][0]:
                    earliest_unknown[k] = (ml, c["cluster_id"])

        for c in clusters:
            if c.get("role") is not None:
                continue
            fps = _norm_path(c["file_path"])
            fns = c["function_names"] or ["_"]
            linked: List[str] = []
            is_secondary = False
            for fn in fns:
                ek = earliest_unknown.get((fps, fn))
                if ek and c["min_line"] > ek[0]:
                    is_secondary = True
                    if ek[1] not in linked:
                        linked.append(ek[1])
            if is_secondary:
                c["role"] = "secondary"
                c["likely_resolved_by"] = linked
            else:
                c["role"] = "primary"

        primary_ids = [c["cluster_id"] for c in clusters if c.get("role") == "primary"]
        secondary_ids = [
            c["cluster_id"] for c in clusters if c.get("role") == "secondary"
        ]

        return {
            "clusters": clusters,
            "primary_cluster_ids": primary_ids,
            "secondary_cluster_ids": secondary_ids,
        }

    def extract_primary_compiler_errors(
        self, errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract ALL compiler ground-truth error entries classified as primary.

        Unlike `cluster_and_classify_compiler_errors()` this does not truncate
        cluster errors to a small sample; it returns every original parsed
        diagnostic line that belongs to a primary cluster.
        """
        if not errors:
            return []

        def _norm_path(p: Optional[str]) -> str:
            if not p:
                return ""
            return str(p).replace("\\", "/").lower()

        member_member_re = re.compile(
            r"request for member [‘']([^’']+)[’']", re.IGNORECASE
        )
        incompatible_callee_re = re.compile(
            r"argument\s+\d+\s+of\s+[‘']([^’']+)[’']", re.IGNORECASE
        )

        def _cluster_key(err: Dict[str, Any]) -> Tuple[Any, ...]:
            cat = err.get("error_code_category") or "compiler_error"
            fp = _norm_path(err.get("file_path"))
            fn = err.get("function_name") or "_"
            msg = err.get("error_text") or ""
            syms = err.get("symbol_candidates") or []

            if cat == "unknown_type_name":
                sym = syms[0] if syms else "unknown"
                return (cat, fp, sym)
            if cat == "undeclared_identifier":
                sym = syms[0] if syms else (msg[:80] if msg else "unknown")
                return (cat, fp, sym)
            if cat == "member_access_on_non_struct":
                m = member_member_re.search(msg)
                mem = m.group(1) if m else ""
                return (cat, fp, fn, mem)
            if cat == "incompatible_argument_type":
                m = incompatible_callee_re.search(msg)
                callee = m.group(1) if m else ""
                return (cat, fp, fn, callee)
            ln = err.get("line_number")
            return (cat, fp, fn, ln if ln is not None else 0)

        # Bucket errors exactly like the main classifier.
        buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for err in errors:
            if not isinstance(err, dict):
                continue
            k = _cluster_key(err)
            buckets.setdefault(k, []).append(err)

        clusters: List[Dict[str, Any]] = []
        for idx, (_key, items) in enumerate(buckets.items()):
            lines = [
                e.get("line_number")
                for e in items
                if isinstance(e.get("line_number"), int)
            ]
            fns_set = set()
            for e in items:
                n = e.get("function_name")
                if n:
                    fns_set.add(n)
            file_path = items[0].get("file_path", "") if items else ""
            cat = (
                items[0].get("error_code_category", "compiler_error")
                if items
                else "compiler_error"
            )
            min_line = min(lines) if lines else 0
            clusters.append(
                {
                    "cluster_id": f"c{idx}",
                    "category": cat,
                    "file_path": file_path,
                    "function_names": sorted(fns_set),
                    "min_line": min_line,
                    "role": None,  # to be filled later
                    "errors_all": items,
                }
            )

        for c in clusters:
            if c["category"] == "member_access_on_non_struct":
                c["role"] = None
            else:
                c["role"] = "primary"

        # Determine earliest unknown/undeclared cluster per (file,function).
        earliest_unknown: Dict[Tuple[str, str], Tuple[int, str]] = {}
        for c in clusters:
            if c["category"] not in ("unknown_type_name", "undeclared_identifier"):
                continue
            fps = _norm_path(c["file_path"])
            for fn in c["function_names"] or ["_"]:
                k = (fps, fn)
                ml = c["min_line"]
                if k not in earliest_unknown or ml < earliest_unknown[k][0]:
                    earliest_unknown[k] = (ml, c["cluster_id"])

        # Resolve secondary member_access clusters.
        for c in clusters:
            if c.get("role") is not None:
                continue
            fps = _norm_path(c["file_path"])
            fns = c["function_names"] or ["_"]
            linked: List[str] = []
            is_secondary = False
            for fn in fns:
                ek = earliest_unknown.get((fps, fn))
                if ek and c["min_line"] > ek[0]:
                    is_secondary = True
                    if ek[1] not in linked:
                        linked.append(ek[1])
            c["role"] = "secondary" if is_secondary else "primary"

        primary_errors: List[Dict[str, Any]] = []
        for c in clusters:
            if c.get("role") == "primary":
                # Return every original parsed compiler diagnostic line.
                primary_errors.extend(c.get("errors_all") or [])

        return primary_errors

    def _resolve_missing_configs(self, config_files: List[str], role: str) -> List[Union[str, Dict]]:
        """
        Resolve missing configuration files by checking existence and suggesting alternatives.
        
        Args:
            config_files: List of config file paths from log
            role: Detected role (CU/DU/UE/gNB/AMF)
            
        Returns:
            List of config files with missing ones replaced by alternatives
        """
        logger.info(f"🔍 Resolving missing configs for role: {role}")
        
        resolved_configs = []
        codebase_root = self._find_codebase_root()
        
        for config_path in config_files:
            # Check if config file exists
            if self._config_file_exists(config_path, codebase_root):
                logger.info(f"✅ Config exists: {config_path}")
                resolved_configs.append(config_path)
            else:
                logger.warning(f"❌ Config missing: {config_path}")
                
                # Find alternative config
                alternative = self._find_alternative_config(config_path, role, codebase_root)
                
                if alternative:
                    logger.info(f"🔄 Using alternative: {alternative}")
                    resolved_configs.append({
                        "missing": config_path,
                        "used": alternative,
                        "reason": f"Original file not found, using {os.path.basename(alternative)}"
                    })
                else:
                    logger.warning(f"⚠️  No alternative found for: {config_path}")
                    resolved_configs.append({
                        "missing": config_path,
                        "used": None,
                        "reason": "No suitable alternative config found"
                    })
        
        return resolved_configs
    
    def _find_codebase_root(self) -> Optional[str]:
        """Find the root of the OpenAirInterface codebase."""
        # Look for common OAI indicators
        current_dir = os.getcwd()
        
        # Check current directory and parent directories
        for _ in range(5):  # Check up to 5 levels up
            if self._is_oai_root(current_dir):
                return current_dir
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # Reached root
                break
            current_dir = parent
        
        # Fallback: look for the configured OAI directory name
        for root, dirs, files in os.walk('.'):
            if self.openair_codebase_file_name in dirs:
                return os.path.join(root, self.openair_codebase_file_name)
        
        return None
    
    def _is_oai_root(self, path: str) -> bool:
        """Check if a directory is the OAI codebase root."""
        oai_indicators = [
            'CMakeLists.txt',
            'openair1',
            'openair2', 
            'openair3',
            'targets'
        ]
        
        for indicator in oai_indicators:
            if os.path.exists(os.path.join(path, indicator)):
                return True
        return False
    
    def _config_file_exists(self, config_path: str, codebase_root: Optional[str]) -> bool:
        """Check if a configuration file exists in the codebase."""
        if not codebase_root:
            return False
        
        # Extract just the filename for fallback searches
        config_filename = os.path.basename(config_path)
        
        # Try different path variations (in priority order)
        possible_paths = [
            config_path,  # Exact path from log
            os.path.join(codebase_root, config_path),  # Relative to codebase root
            # Standard OAI config directories
            os.path.join(codebase_root, 'targets', 'PROJECTS', 'GENERIC-NR-5GC', 'CONF', config_filename),
            os.path.join(codebase_root, 'openair3', 'NAS', 'TOOLS', config_filename),  # NEW: UE security configs location
            os.path.join(codebase_root, 'targets', 'CONF', config_filename),
            os.path.join(codebase_root, 'targets', config_filename)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return True
        
        return False
    
    def extract_error_message(self, log_file_path: str) -> Optional[str]:
        """
        Extract the most relevant error message from a log file using LLM analysis
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            The most relevant error message or None if no clear error found
        """
        logger.info(f"🔍 Extracting error message from log file: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            logger.error(f"Log file not found: {log_file_path}")
            return None
        
        try:
            # Read the log file
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            # Limit log content size for LLM processing (last 10000 characters for recent errors)
            if len(log_content) > 10000:
                log_content = "... (earlier logs truncated) ...\n" + log_content[-10000:]
            
            # Create prompt for error extraction
            prompt = f"""You are an expert 5G/LTE telecommunications system analyzer. Analyze the following log file and extract the MOST CRITICAL ERROR MESSAGE along with any relevant stack trace or crash information.

INSTRUCTIONS:
1. Look for ERROR, FATAL, CRITICAL, FAILED, or similar severity indicators
2. Also look for stack traces, segmentation faults, crashes, or function call information
3. Focus on errors that would prevent system operation (AMF association, timer expiration, connection failures, crashes, etc.)
4. If you find both an error message AND stack trace/crash information, combine them contextually
5. Extract the EXACT error message text, and if available, mention the specific function/location
6. If multiple errors exist, choose the most recent and critical one.
7. If no clear error exists, return "No critical error found"

LOG CONTENT:
{log_content}

RESPONSE FORMAT:
Return the error message, and if available, include function/location context. Examples:
- "No AMF associated to the gNB"
- "Contention resolution timer has expired, RA procedure failed"
- "Segmentation fault in rrc_handle_RRCSetupRequest at line 1065"
- "NGAP setup failed: Connection refused"
- "RA procedure failed - crash in rrc_handle_RRCSetupRequest function"
- "No critical error found"

ERROR MESSAGE:"""

            # Import here to avoid circular imports
            from .fix_suggestion_pipeline import FixSuggestionPipeline
            
            # Use the same Azure client setup as other components
            fix_pipeline = FixSuggestionPipeline()
            
            response = fix_pipeline.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a telecommunications log analysis expert who extracts critical error messages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,
                seed=99999  # Error extraction seed for consistency
            )
            
            error_message = response.choices[0].message.content.strip()
            
            # Clean up the response
            error_message = error_message.strip('"\'')  # Remove quotes
            
            if error_message.lower() in ['no critical error found', 'no error found', 'none']:
                logger.info("📋 No critical error found in log file")
                return None
            
            logger.info(f"✅ Extracted error message: {error_message}")
            return error_message
            
        except Exception as e:
            logger.error(f"❌ Failed to extract error message: {e}")
            return None
    
    def _find_alternative_config(self, missing_config: str, role: str, codebase_root: Optional[str]) -> Optional[str]:
        """Find an alternative configuration file based on role and patterns."""
        if not codebase_root:
            return None
        
        # Extract filename from missing config to check for specific files
        missing_filename = os.path.basename(missing_config).lower()
        
        # Define role-based config patterns
        role_patterns = {
            'CU': ['*cu*.conf', 'gnb-cu*.conf', 'cu_gnb.conf'],
            'DU': ['*du*.conf', 'gnb-du*.conf', 'du_gnb.conf'],
            'UE': ['ue*.conf', '*ue*.conf', '5g_sa_ue.conf'],  # Added explicit 5g_sa_ue.conf
            'gNB': ['gnb*.conf', '*gnb*.conf'],
            'AMF': ['amf*.conf', '*amf*.conf']
        }
        
        # Get patterns for the detected role
        patterns = role_patterns.get(role, ['*.conf'])
        
        # Define search directories (in priority order)
        search_dirs = [
            os.path.join(codebase_root, 'targets', 'PROJECTS', 'GENERIC-NR-5GC', 'CONF'),
            os.path.join(codebase_root, 'openair3', 'NAS', 'TOOLS'),  # NEW: Search for UE security configs
            os.path.join(codebase_root, 'targets', 'CONF'),
            os.path.join(codebase_root, 'targets')
        ]
        
        alternatives = []
        
        # Search in all directories
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            for pattern in patterns:
                search_pattern = os.path.join(search_dir, pattern)
                matches = glob.glob(search_pattern)
                alternatives.extend(matches)
        
        # Remove duplicates and sort
        alternatives = list(set(alternatives))
        alternatives.sort()
        
        # Return the first suitable alternative with priority matching
        if alternatives:
            # PRIORITY 1: Exact filename match (e.g., looking for 5g_sa_ue.conf, found 5g_sa_ue.conf)
            for alt in alternatives:
                alt_name = os.path.basename(alt).lower()
                if alt_name == missing_filename:
                    logger.info(f"Found exact filename match: {alt}")
                    return alt
            
            # PRIORITY 2: Prefer configs with role in name
            for alt in alternatives:
                alt_name = os.path.basename(alt).lower()
                if role.lower() in alt_name:
                    return alt
            
            # PRIORITY 3: Fall back to any alternative
            return alternatives[0]
        
        return None
    
    def _get_empty_context(self) -> Dict:
        """Return empty context structure."""
        return {
            "role": "Unknown",
            "active_configs": [],
            "network_params": {
                "gnb_ipv4": None,
                "amf_ipv4": None,
                "ngap_port": None,
                "assoc_state": "UNKNOWN"
            },
            "log_anchors": []
        }
    
    def save_context(self, context: Dict, output_file: str = "deployment_context.json"):
        """Save deployment context to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
            logger.info(f"Deployment context saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving context: {e}")

def main():
    """Main function to run the log context parser."""
    if len(sys.argv) != 2:
        print("Usage: python parse_log_context.py <log_file_path>")
        print("Example: python parse_log_context.py cu_usrp.log")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    # Initialize parser
    parser = LogContextParser()
    
    # Parse log file
    context = parser.parse_log_file(log_file_path)
    
    # Save results
    parser.save_context(context)
    
    # Print summary
    print("\n" + "="*50)
    print("DEPLOYMENT CONTEXT EXTRACTED")
    print("="*50)
    print(f"Role: {context['role']}")
    print(f"Active Configs: {len(context['active_configs'])} files")
    print(f"gNB IP: {context['network_params']['gnb_ipv4']}")
    print(f"AMF IP: {context['network_params']['amf_ipv4']}")
    print(f"NGAP Port: {context['network_params']['ngap_port']}")
    print(f"Association State: {context['network_params']['assoc_state']}")
    print(f"Log Anchors: {len(context['log_anchors'])} important lines")
    print("="*50)
    
    # Show some log anchors
    if context['log_anchors']:
        print("\nKey Log Lines:")
        for i, anchor in enumerate(context['log_anchors'][:5], 1):
            print(f"{i}. {anchor}")
        if len(context['log_anchors']) > 5:
            print(f"... and {len(context['log_anchors']) - 5} more")

if __name__ == "__main__":
    main()
