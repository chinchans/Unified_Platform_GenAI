#!/usr/bin/env python3
"""
Crash Analysis - Phase 2.5: Intelligent LLM-Based Function Grading

Takes all functions from Phase 2 retrieval and uses LLM to intelligently select
the TOP 10 most crash-relevant functions.

Features:
- Fixes path resolution to read actual source code
- Sends all functions to LLM with crash context
- LLM selects top 10 most relevant functions
- Formats output compatible with Phase 3 (like phase2_results.json)

Author: AI Assistant
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class CrashPhase2Grading:
    """Intelligent LLM-based function grading for crash analysis"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize crash Phase 2.5 grading.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🧠 Initializing Crash Phase 2.5 Grading...")
        
        self.openair_codebase_file_name = openair_codebase_file_name
        self.codebase_path = f"{openair_codebase_file_name}"
        
        # Setup Azure OpenAI client
        self._setup_azure_client()
        
        logger.info("✅ Crash Phase 2.5 Grading initialized")
    
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
        
        self.azure_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint
        )
        
        self.model = "gpt-4o-mini"
        logger.info("✅ Azure OpenAI client initialized")
    
    def _extract_relative_path(self, full_path: str) -> str:
        """
        Extract relative path from full log path.
        
        Example:
        Input:  "/home/tcs/jayasvi/cu/openairinterface5g/openair2/RRC/NR/rrc_gNB_NGAP.c"
        Output: "openair2/RRC/NR/rrc_gNB_NGAP.c"
        """
        # Common OAI directory markers
        markers = ['openair1', 'openair2', 'openair3', 'common', 'targets', 'executables']
        
        for marker in markers:
            if marker in full_path:
                # Split at marker and take everything from marker onwards
                parts = full_path.split(marker)
                if len(parts) > 1:
                    relative_path = marker + parts[1]
                    # Clean up any leading slashes
                    return relative_path.lstrip('/')
        
        # If no marker found, try to get just the filename and search for it
        return os.path.basename(full_path)
    
    def _read_source_code(self, file_path: str, function_name: str = None) -> Optional[str]:
        """
        Read source code with improved path resolution.
        
        Args:
            file_path: Original file path from log
            function_name: Optional function name to extract
            
        Returns:
            Source code or None
        """
        # Extract relative path
        relative_path = self._extract_relative_path(file_path)
        
        # Try multiple path combinations
        possible_paths = [
            os.path.join(self.codebase_path, relative_path),
            os.path.join(self.codebase_path, file_path),
            os.path.join(self.codebase_path, os.path.basename(file_path)),
            relative_path,
            file_path
        ]
        
        # Try each path
        for full_path in possible_paths:
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # If function name provided, try to extract just that function
                    if function_name:
                        function_code = self._extract_function_from_file(content, function_name)
                        if function_code:
                            logger.info(f"   ✓ Read function {function_name} from {full_path}")
                            return function_code
                    
                    # Return full file content
                    logger.info(f"   ✓ Read full file from {full_path}")
                    return content
                    
                except Exception as e:
                    logger.warning(f"   ✗ Error reading {full_path}: {e}")
                    continue
        
        logger.warning(f"   ✗ Could not find source file: {file_path}")
        return None
    
    def _extract_function_from_file(self, file_content: str, function_name: str) -> Optional[str]:
        """Extract a specific function from file content"""
        lines = file_content.split('\n')
        
        # Find function start
        function_pattern = rf'^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?\w+\s+\*?\s*{re.escape(function_name)}\s*\('
        
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
        end_line = min(start_line + 200, len(lines))
        return '\n'.join(lines[start_line:end_line])
    
    def _truncate_code_smartly(self, code: str, max_length: int = 3000) -> str:
        """Smart truncation preserving important parts"""
        if len(code) <= max_length:
            return code
        
        lines = code.split('\n')
        
        # Keep first 40 lines and last 20 lines
        if len(lines) > 60:
            truncated = lines[:40] + ["... [middle truncated] ..."] + lines[-20:]
            return '\n'.join(truncated)
        
        return code[:max_length] + "\n... [truncated]"
    
    def process_grading(self, phase2_file: str, phase1_file: str) -> Dict[str, Any]:
        """
        Process Phase 2.5: Intelligent function grading.
        
        Args:
            phase2_file: Path to Phase 2 retrieval JSON
            phase1_file: Path to Phase 1 extraction JSON
            
        Returns:
            Graded and filtered results
        """
        logger.info("=" * 80)
        logger.info("🧠 CRASH ANALYSIS - PHASE 2.5: INTELLIGENT FUNCTION GRADING")
        logger.info("=" * 80)
        
        # Load Phase 2 retrieval results
        if not os.path.exists(phase2_file):
            logger.error(f"Phase 2 file not found: {phase2_file}")
            return {"error": "Phase 2 file not found"}
        
        with open(phase2_file, 'r', encoding='utf-8') as f:
            phase2_data = json.load(f)
        
        # Load Phase 1 extraction for crash context
        if not os.path.exists(phase1_file):
            logger.error(f"Phase 1 file not found: {phase1_file}")
            return {"error": "Phase 1 file not found"}
        
        with open(phase1_file, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
        
        crash_info = phase1_data.get("crash_info", {})
        
        # STEP 1: Collect all functions
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: COLLECTING ALL FUNCTIONS")
        logger.info("=" * 60)
        
        all_functions = []
        
        # Add prioritized backtrace functions
        prioritized = phase2_data.get("prioritized_functions", [])
        for func in prioritized:
            all_functions.append({
                "function_name": func.get("function_name"),
                "file_path": func.get("file_path"),
                "priority_score": func.get("priority_score"),
                "source": func.get("source"),
                "role": func.get("role", ""),
                "frame_number": func.get("frame_number")
            })
        
        # Add call chain expansion functions
        call_chain = phase2_data.get("call_chain_expansion", [])
        for func in call_chain:
            # Check if not already in list
            if not any(f["function_name"] == func.get("function_name") for f in all_functions):
                all_functions.append({
                    "function_name": func.get("function_name"),
                    "file_path": func.get("file_path"),  # ✅ Get from Phase 2 JSON (now populated!)
                    "priority_score": func.get("priority_score"),
                    "source": func.get("source"),
                    "role": func.get("reason", ""),
                    "frame_number": None
                })
        
        logger.info(f"✅ Collected {len(all_functions)} total functions")
        logger.info(f"   - Backtrace functions: {len(prioritized)}")
        logger.info(f"   - Call chain functions: {len(call_chain)}")
        
        # STEP 2: Read source code for all functions
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: READING SOURCE CODE WITH FIXED PATH RESOLUTION")
        logger.info("=" * 60)
        
        functions_with_code = []
        for func in all_functions:
            function_name = func.get("function_name")
            file_path = func.get("file_path")
            
            if not file_path or file_path in ["unknown", "Unknown"]:
                logger.warning(f"⚠️  {function_name}: No file path available, skipping")
                continue
            
            # Skip system libraries (pthread, libc, etc.)
            if any(sys_lib in file_path for sys_lib in ['pthread', 'libc', 'sysdeps', 'nptl']):
                logger.info(f"⚠️  {function_name}: System library, skipping")
                continue
            
            logger.info(f"📖 Reading: {function_name}")
            
            # Read source code with fixed path resolution
            source_code = self._read_source_code(file_path, function_name)
            
            if source_code:
                # Truncate if too long
                truncated_code = self._truncate_code_smartly(source_code, max_length=3000)
                
                func["code_snippet"] = truncated_code
                func["has_source"] = True
                func["source_code_length"] = len(source_code)
                functions_with_code.append(func)
                
                logger.info(f"   ✓ Read {len(source_code)} chars (truncated to {len(truncated_code)})")
            else:
                func["has_source"] = False
                func["code_snippet"] = "// Source code not available"
                # Still add to list for LLM to see (LLM might still grade based on function name)
                functions_with_code.append(func)
                logger.warning(f"   ✗ Failed to read source")
        
        logger.info(f"✅ Successfully read source code for {sum(1 for f in functions_with_code if f.get('has_source'))} functions")
        
        # STEP 3: LLM Intelligent Grading
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: LLM INTELLIGENT FUNCTION SELECTION (TOP 10)")
        logger.info("=" * 60)
        
        top_functions = self._llm_select_top_functions(crash_info, functions_with_code)
        
        # STEP 4: Format as phase2_results.json compatible structure
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: FORMATTING AS PHASE2 RESULTS")
        logger.info("=" * 60)
        
        graded_results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "crash_phase2_graded",
            "crash_analysis": True,
            "crash_info_summary": {
                "faulting_function": crash_info.get("faulting_function"),
                "fault_location": crash_info.get("fault_location"),
                "signal": crash_info.get("signal")
            },
            "suspected_functions": top_functions,
            "suspected_configs": [],  # Crashes typically don't involve configs
            "total_candidates_analyzed": len(functions_with_code),
            "top_functions_selected": len(top_functions)
        }
        
        # Save results
        output_file = "output/crash_phase2_graded.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graded_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Graded results saved to: {output_file}")
        logger.info(f"✅ Selected {len(top_functions)} out of {len(functions_with_code)} functions")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ CRASH ANALYSIS PHASE 2.5 COMPLETED")
        logger.info("=" * 80)
        
        return graded_results
    
    def _llm_select_top_functions(self, crash_info: Dict, all_functions: List[Dict]) -> List[Dict]:
        """
        Use LLM to intelligently select top 10 most crash-relevant functions.
        """
        logger.info(f"🤖 Sending {len(all_functions)} functions to LLM for intelligent selection...")
        
        # Build crash context
        context_parts = []
        context_parts.append("# SEGMENTATION FAULT - FUNCTION SELECTION")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Crash summary
        context_parts.append("## 🔥 CRASH SUMMARY")
        context_parts.append(f"Signal: {crash_info.get('signal', 'SIGSEGV')}")
        context_parts.append(f"Fault Location: {crash_info.get('fault_location', 'Unknown')}")
        context_parts.append(f"Faulting Function: {crash_info.get('faulting_function', 'Unknown')}")
        context_parts.append(f"Thread: {crash_info.get('crash_thread', {}).get('name', 'Unknown')}")
        context_parts.append("")
        
        # Backtrace
        if crash_info.get("backtrace"):
            context_parts.append("## 📚 BACKTRACE")
            for frame in crash_info["backtrace"][:10]:
                context_parts.append(f"#{frame['frame_number']}: {frame['function']} at {frame['file']}:{frame['line']}")
            context_parts.append("")
        
        # Scenario flow
        if crash_info.get("scenario_flow"):
            context_parts.append("## 🔄 SCENARIO FLOW (Steps before crash)")
            for i, step in enumerate(crash_info["scenario_flow"], 1):
                context_parts.append(f"{i}. {step}")
            context_parts.append("")
        
        # All candidate functions
        context_parts.append("## 🔍 ALL CANDIDATE FUNCTIONS")
        context_parts.append(f"Total: {len(all_functions)} functions")
        context_parts.append("")
        
        for i, func in enumerate(all_functions, 1):
            context_parts.append(f"### Function {i}: {func.get('function_name')}")
            context_parts.append(f"**Priority Score:** {func.get('priority_score', 0):.2f}")
            context_parts.append(f"**Source:** {func.get('source', 'unknown')}")
            
            if func.get('role'):
                context_parts.append(f"**Role:** {func.get('role')}")
            
            if func.get('frame_number') is not None:
                context_parts.append(f"**Backtrace Frame:** #{func.get('frame_number')}")
            
            if func.get('has_source') and func.get('code_snippet'):
                context_parts.append(f"**Source Code:**")
                context_parts.append("```c")
                context_parts.append(func.get('code_snippet', '// Not available'))
                context_parts.append("```")
            else:
                context_parts.append(f"**Source Code:** Not available")
            
            context_parts.append("")
        
        assembled_context = "\n".join(context_parts)
        
        # Create LLM prompt
        prompt = f"""{assembled_context}

## TASK: INTELLIGENT FUNCTION SELECTION

You are analyzing a segmentation fault crash. Review ALL {len(all_functions)} candidate functions above and select the **TOP 10 functions** most likely involved in causing this crash.

### SELECTION CRITERIA (in priority order):

1. **MUST INCLUDE:**
   - Frame #0 (CRASH_POINT) - where the actual crash occurred
   - Frame #1 (IMMEDIATE_CALLER) - the function that called the crash point

2. **HIGH PRIORITY:**
   - Functions with missing NULL pointer checks
   - Functions that pass parameters to the crash function
   - Functions with pointer dereferences before validation
   - Functions in the critical execution path leading to crash

3. **MEDIUM PRIORITY:**
   - Upstream callers that might pass invalid data
   - Downstream functions that might be called with invalid pointers
   - Functions with error handling issues

4. **CONSIDER:**
   - Function names and their typical responsibilities
   - Code patterns that suggest validation issues
   - Execution flow context

### OUTPUT FORMAT:

Return a JSON array with EXACTLY 10 functions (or fewer if less than 10 available):

```json
{{
  "selected_functions": [
    {{
      "function_name": "rrc_gNB_send_NGAP_NAS_FIRST_REQ",
      "relevance_score": 1.0,
      "reason": "Frame #0 crash point at line 173. Null pointer dereference on rrcSetupComplete parameter. Missing NULL check before accessing rrcSetupComplete->dedicatedNAS_Message.size."
    }},
    {{
      "function_name": "rrc_gNB_decode_dcch",
      "relevance_score": 0.95,
      "reason": "Frame #1 immediate caller. Likely passes NULL or invalid rrcSetupComplete to crash function. Should validate before calling."
    }},
    ...
  ]
}}
```

### IMPORTANT:
- Return ONLY valid JSON
- Select EXACTLY 10 functions (or fewer if total < 10)
- Relevance scores: 1.0 (most critical) to 0.5 (still relevant)
- Provide detailed reasons explaining why each function is selected
- Frame #0 and Frame #1 are MANDATORY in selection

Return your analysis:"""

        try:
            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert crash analyzer specializing in C/C++ segmentation faults and debugging."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up JSON formatting
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            result_json = json.loads(result_text.strip())
            selected = result_json.get("selected_functions", [])
            
            logger.info(f"✅ LLM selected {len(selected)} functions")
            
            # Enrich selected functions with original data
            enriched_functions = []
            for sel_func in selected:
                # Find the original function data
                orig_func = next((f for f in all_functions if f["function_name"] == sel_func["function_name"]), None)
                
                if orig_func:
                    enriched_functions.append({
                        "function_name": sel_func["function_name"],
                        "file_path": orig_func.get("file_path", "Unknown"),
                        "relevance_score": sel_func.get("relevance_score", 0.5),
                        "source": orig_func.get("source", "llm_selected"),
                        "code_snippet": orig_func.get("code_snippet", "// Source not available"),
                        "reason": sel_func.get("reason", "Selected by LLM for crash analysis"),
                        "role": orig_func.get("role", ""),
                        "frame_number": orig_func.get("frame_number"),
                        "priority_score": orig_func.get("priority_score"),
                        "has_source": orig_func.get("has_source", False)
                    })
            
            return enriched_functions
            
        except Exception as e:
            logger.error(f"❌ LLM grading failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: return top 10 by priority score
            logger.warning("⚠️  Falling back to priority score sorting")
            sorted_funcs = sorted(
                [f for f in all_functions if f.get('has_source')],
                key=lambda x: x.get('priority_score', 0),
                reverse=True
            )[:10]
            
            # Format fallback results
            return [
                {
                    "function_name": f["function_name"],
                    "file_path": f.get("file_path", "Unknown"),
                    "relevance_score": f.get("priority_score", 0.5),
                    "source": f.get("source", "fallback"),
                    "code_snippet": f.get("code_snippet", "// Source not available"),
                    "reason": f"Priority score: {f.get('priority_score', 0):.2f} - {f.get('role', 'No description')}",
                    "role": f.get("role", ""),
                    "frame_number": f.get("frame_number")
                }
                for f in sorted_funcs
            ]


def main():
    """Test Phase 2.5 grading"""
    grading = CrashPhase2Grading()
    
    phase2_file = "output/crash_phase2_retrieval.json"
    phase1_file = "output/segmentation_fault_extraction.json"
    
    if not os.path.exists(phase2_file):
        print(f"❌ Phase 2 file not found: {phase2_file}")
        print(f"   Please run Phase 2 first")
        return
    
    if not os.path.exists(phase1_file):
        print(f"❌ Phase 1 file not found: {phase1_file}")
        print(f"   Please run Phase 1 first")
        return
    
    results = grading.process_grading(phase2_file, phase1_file)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 PHASE 2.5 GRADING SUMMARY")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"\n✅ Phase 2.5 Completed!")
        print(f"   Total candidates analyzed: {results.get('total_candidates_analyzed', 0)}")
        print(f"   Top functions selected: {results.get('top_functions_selected', 0)}")
        
        suspected_functions = results.get("suspected_functions", [])
        if suspected_functions:
            print(f"\n🎯 TOP 10 SELECTED FUNCTIONS:")
            for i, func in enumerate(suspected_functions, 1):
                has_code = "✓" if func.get("has_source") else "✗"
                print(f"   {i}. [{func.get('relevance_score', 0):.2f}] {has_code} {func['function_name']}")
                print(f"      Role: {func.get('role', 'N/A')}")
                print(f"      Reason: {func.get('reason', 'N/A')[:100]}...")
                print()
        
        print(f"💾 Results saved to: output/crash_phase2_graded.json")


if __name__ == "__main__":
    main()

