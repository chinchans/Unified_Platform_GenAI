#!/usr/bin/env python3
"""
Segmentation Fault Analyzer
Specialized module for analyzing segmentation faults, crashes, and core dumps

This module provides:
- Backtrace parsing from GDB logs
- Stack frame analysis
- Null pointer dereference detection
- Memory access violation analysis
- Crash-specific fix suggestions

Author: AI Assistant
"""

import os
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Import crash-specific Phase 2 and Phase 2.5
try:
    from .crash_phase2_retrieval import CrashPhase2Retrieval
except ImportError:
    logger.warning("crash_phase2_retrieval module not found, Phase 2 will be unavailable")
    CrashPhase2Retrieval = None

try:
    from .crash_phase2_grading import CrashPhase2Grading
except ImportError:
    logger.warning("crash_phase2_grading module not found, Phase 2.5 will be unavailable")
    CrashPhase2Grading = None

try:
    from .crash_phase3_fix_generation import CrashPhase3FixGeneration
except ImportError:
    logger.warning("crash_phase3_fix_generation module not found, Phase 3 will be unavailable")
    CrashPhase3FixGeneration = None


class SegmentationFaultAnalyzer:
    """Analyzer for segmentation faults and crash dumps"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the segmentation fault analyzer.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🔬 Initializing Segmentation Fault Analyzer...")
        
        self.openair_codebase_file_name = openair_codebase_file_name
        self.codebase_path = f"Error_fixing_pipelin/{openair_codebase_file_name}"
        
        # Setup Azure OpenAI client
        self._setup_azure_client()
        
        # Initialize Phase 2 retrieval if available
        self.phase2_retrieval = None
        if CrashPhase2Retrieval:
            self.phase2_retrieval = CrashPhase2Retrieval(openair_codebase_file_name=openair_codebase_file_name)
        
        # Initialize Phase 2.5 grading if available
        self.phase2_grading = None
        if CrashPhase2Grading:
            self.phase2_grading = CrashPhase2Grading(openair_codebase_file_name=openair_codebase_file_name)
        
        # Initialize Phase 3 fix generation if available
        self.phase3_fix_gen = None
        if CrashPhase3FixGeneration:
            self.phase3_fix_gen = CrashPhase3FixGeneration(openair_codebase_file_name=openair_codebase_file_name)
        
        logger.info("✅ Segmentation Fault Analyzer initialized successfully")
    
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
        logger.info("✅ Azure OpenAI client initialized successfully")
    
    def parse_segmentation_fault_log(self, log_file_path: str) -> Dict[str, Any]:
        """
        Parse segmentation fault log to extract crash information.
        
        Args:
            log_file_path: Path to the segmentation fault log file
            
        Returns:
            Dictionary containing parsed crash information
        """
        logger.info(f"📄 Parsing segmentation fault log: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            logger.error(f"Log file not found: {log_file_path}")
            return {"error": "Log file not found"}
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            crash_info = {
                "log_file": log_file_path,
                "timestamp": datetime.now().isoformat(),
                "crash_detected": False,
                "fault_location": None,
                "backtrace": [],
                "crash_thread": None,
                "signal": None,
                "faulting_function": None,
                "faulting_file": None,
                "faulting_line": None,
                "scenario_flow": [],
                "pre_crash_logs": []
            }
            
            # Extract segmentation fault signal
            signal_match = re.search(r'received signal (\w+), (.+)', log_content)
            if signal_match:
                crash_info["crash_detected"] = True
                crash_info["signal"] = signal_match.group(1)
                crash_info["crash_type"] = signal_match.group(2)
                logger.info(f"🔥 Crash detected: {signal_match.group(1)} - {signal_match.group(2)}")
            
            # Extract thread information
            thread_match = re.search(r'Thread (\d+) "([^"]+)" received signal', log_content)
            if thread_match:
                crash_info["crash_thread"] = {
                    "id": thread_match.group(1),
                    "name": thread_match.group(2)
                }
                logger.info(f"📍 Crash thread: {thread_match.group(2)} (ID: {thread_match.group(1)})")
            
            # Extract fault location (immediate crash line)
            fault_match = re.search(r'at ([^:]+):(\d+)', log_content, re.MULTILINE)
            if fault_match:
                crash_info["faulting_file"] = fault_match.group(1)
                crash_info["faulting_line"] = fault_match.group(2)
                
                # Extract function name from the line before the fault location
                lines = log_content.split('\n')
                for i, line in enumerate(lines):
                    if f"at {fault_match.group(1)}:{fault_match.group(2)}" in line:
                        # The function is usually on the same line or previous line
                        func_match = re.search(r'(\w+)\s*\([^)]*\)\s*at', line)
                        if func_match:
                            crash_info["faulting_function"] = func_match.group(1)
                        elif i > 0:
                            prev_line = lines[i-1]
                            func_match = re.search(r'(\w+)\s*\([^)]*\)', prev_line)
                            if func_match:
                                crash_info["faulting_function"] = func_match.group(1)
                        break
                
                crash_info["fault_location"] = f"{crash_info['faulting_file']}:{crash_info['faulting_line']}"
                logger.info(f"💥 Fault location: {crash_info['fault_location']}")
                if crash_info["faulting_function"]:
                    logger.info(f"⚙️  Faulting function: {crash_info['faulting_function']}")
            
            # Extract backtrace
            backtrace_section = re.search(r'\(gdb\)\s*bt\s*\n(.*?)(?:\(gdb\)|$)', log_content, re.DOTALL)
            if backtrace_section:
                backtrace_text = backtrace_section.group(1)
                
                # Parse each frame in the backtrace
                frame_pattern = r'#(\d+)\s+(?:0x[0-9a-f]+\s+in\s+)?([^\s(]+)\s*\([^)]*\)\s*(?:at\s+([^:]+):(\d+))?'
                for match in re.finditer(frame_pattern, backtrace_text):
                    frame = {
                        "frame_number": int(match.group(1)),
                        "function": match.group(2),
                        "file": match.group(3) if match.group(3) else "unknown",
                        "line": match.group(4) if match.group(4) else "unknown"
                    }
                    crash_info["backtrace"].append(frame)
                
                logger.info(f"📚 Backtrace extracted: {len(crash_info['backtrace'])} frames")
            
            # Extract scenario flow (pre-crash successful operations)
            crash_info["scenario_flow"] = self._extract_scenario_flow(log_content)
            logger.info(f"🔄 Scenario flow: {len(crash_info['scenario_flow'])} steps before crash")
            
            # Extract relevant pre-crash logs (last 30 lines before crash)
            lines = log_content.split('\n')
            crash_line_idx = None
            for i, line in enumerate(lines):
                if 'received signal SIGSEGV' in line or 'Segmentation fault' in line:
                    crash_line_idx = i
                    break
            
            if crash_line_idx:
                start_idx = max(0, crash_line_idx - 30)
                crash_info["pre_crash_logs"] = [
                    line.strip() for line in lines[start_idx:crash_line_idx]
                    if line.strip() and not line.strip().startswith('#')
                ]
            
            return crash_info
            
        except Exception as e:
            logger.error(f"❌ Failed to parse segmentation fault log: {e}")
            return {"error": str(e)}
    
    def _extract_scenario_flow(self, log_content: str) -> List[str]:
        """
        Extract the scenario flow from log - what was successfully executed before crash.
        
        Args:
            log_content: Full log content
            
        Returns:
            List of scenario steps
        """
        scenario_steps = []
        
        # Key milestones in 5G gNB operation
        milestones = [
            (r'Starting NGAP layer', 'NGAP layer started'),
            (r'Registered new gNB', 'gNB registered with AMF'),
            (r'F1 Setup Response', 'F1 interface setup completed'),
            (r'Activating RA procedure', 'Random Access procedure initiated'),
            (r'Send RAR to RA-RNTI', 'Random Access Response sent'),
            (r'Received SDU for CCCH', 'CCCH SDU received from UE'),
            (r'Decoding CCCH', 'Decoding CCCH message'),
            (r'Create UE context', 'UE context created'),
            (r'Send RRC Setup', 'RRC Setup message sent'),
            (r'Received RRCSetupComplete', 'RRC Setup Complete received'),
            (r'RRC_CONNECTED reached', 'UE reached RRC_CONNECTED state'),
        ]
        
        for pattern, description in milestones:
            if re.search(pattern, log_content, re.IGNORECASE):
                scenario_steps.append(description)
        
        return scenario_steps
    
    def read_source_code_at_fault(self, file_path: str, line_number: int, context_lines: int = 20) -> Optional[str]:
        """
        Read source code at the fault location with context.
        
        Args:
            file_path: Source file path
            line_number: Line number of the fault
            context_lines: Number of lines before and after to include
            
        Returns:
            Source code snippet with context
        """
        # Construct full path
        full_path = os.path.join(self.codebase_path, file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"Source file not found: {full_path}")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            target_line = int(line_number) - 1  # Convert to 0-indexed
            start_line = max(0, target_line - context_lines)
            end_line = min(len(lines), target_line + context_lines + 1)
            
            code_snippet = ""
            for i in range(start_line, end_line):
                marker = " >>> " if i == target_line else "     "
                code_snippet += f"{marker}{i+1:4d}: {lines[i]}"
            
            return code_snippet
            
        except Exception as e:
            logger.error(f"Error reading source code: {e}")
            return None
    
    def analyze_crash(self, crash_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the crash using LLM with specialized segmentation fault prompt.
        
        Args:
            crash_info: Parsed crash information
            
        Returns:
            Analysis results with fix suggestions
        """
        logger.info("🧠 Analyzing crash with LLM...")
        
        if not crash_info.get("crash_detected"):
            return {
                "error": "No crash detected in log",
                "analysis": None
            }
        
        # Read source code at fault location
        source_code = None
        if crash_info.get("faulting_file") and crash_info.get("faulting_line"):
            source_code = self.read_source_code_at_fault(
                crash_info["faulting_file"],
                int(crash_info["faulting_line"]),
                context_lines=25
            )
        
        # Build comprehensive context
        context_parts = []
        context_parts.append("# SEGMENTATION FAULT ANALYSIS")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Crash summary
        context_parts.append("## 🔥 CRASH SUMMARY")
        context_parts.append(f"**Signal:** {crash_info.get('signal', 'UNKNOWN')}")
        context_parts.append(f"**Type:** {crash_info.get('crash_type', 'Segmentation fault')}")
        context_parts.append(f"**Thread:** {crash_info.get('crash_thread', {}).get('name', 'Unknown')} (ID: {crash_info.get('crash_thread', {}).get('id', 'N/A')})")
        context_parts.append(f"**Fault Location:** {crash_info.get('fault_location', 'Unknown')}")
        if crash_info.get("faulting_function"):
            context_parts.append(f"**Faulting Function:** {crash_info['faulting_function']}")
        context_parts.append("")
        
        # Backtrace
        context_parts.append("## 📚 BACKTRACE (Call Stack)")
        if crash_info.get("backtrace"):
            for frame in crash_info["backtrace"][:10]:  # Limit to top 10 frames
                context_parts.append(f"#{frame['frame_number']} {frame['function']} at {frame['file']}:{frame['line']}")
        else:
            context_parts.append("No backtrace available")
        context_parts.append("")
        
        # Scenario flow
        context_parts.append("## 🔄 SCENARIO FLOW (Successful steps before crash)")
        if crash_info.get("scenario_flow"):
            for i, step in enumerate(crash_info["scenario_flow"], 1):
                context_parts.append(f"{i}. ✅ {step}")
            context_parts.append(f"\n💥 **CRASH OCCURRED AFTER STEP {len(crash_info['scenario_flow'])}**")
        else:
            context_parts.append("No scenario flow detected")
        context_parts.append("")
        
        # Source code at fault
        if source_code:
            context_parts.append("## 💻 SOURCE CODE AT FAULT LOCATION")
            context_parts.append("```c")
            context_parts.append(source_code)
            context_parts.append("```")
            context_parts.append("")
        
        # Pre-crash logs
        if crash_info.get("pre_crash_logs"):
            context_parts.append("## 📋 PRE-CRASH LOG CONTEXT (Last 30 lines)")
            for log_line in crash_info["pre_crash_logs"][-30:]:
                context_parts.append(f"   {log_line}")
            context_parts.append("")
        
        assembled_context = "\n".join(context_parts)
        
        # Create the specialized segmentation fault analysis prompt
        analysis_prompt = f"""Objective: Analyze the following segmentation fault log and backtrace. Identify the root cause in the codebase, locate the exact line of failure, and suggest appropriate fixes. The codebase is available locally and includes the file paths mentioned in the log.

{assembled_context}

Instructions:

1. Parse the log to identify the function and line where the segmentation fault occurred.

2. Analyze the code at the fault location to determine the cause (e.g., null pointer dereference, invalid memory access).

3. Check whether proper validation or error handling is missing before accessing the faulty variable or pointer.

4. Trace the call stack to understand how the function was invoked and whether the caller should have ensured valid inputs.

5. Suggest code fixes or defensive programming practices to prevent this crash.

6. Optionally, recommend runtime checks, logging improvements, or unit tests to catch such issues earlier.

7. Review the log to understand the scenario progression. Identify which functional or test scenarios were successfully executed before the crash, and determine the exact scenario or step during which the segmentation fault occurred. Provide a summary of the scenario flow leading up to the fault to help contextualize the issue.

Please provide your analysis in the following JSON format:

{{
  "root_cause": "Detailed explanation of the root cause",
  "fault_analysis": "Analysis of the fault location and what went wrong",
  "null_pointer_check": "Whether this is a null pointer dereference (yes/no/likely)",
  "missing_validation": "What validation is missing",
  "call_stack_analysis": "Analysis of how the function was called",
  "crash_scenario": "What scenario was being executed when crash occurred",
  "code_fixes": [
    {{
      "function_name": "function name",
      "file_path": "file path",
      "line_number": "line number",
      "original_code": "original code snippet",
      "fixed_code": "fixed code with validation",
      "description": "explanation of the fix"
    }}
  ],
  "defensive_programming": [
    "List of defensive programming recommendations"
  ],
  "runtime_checks": [
    "List of runtime checks to add"
  ],
  "investigation_steps": [
    "Step-by-step investigation procedure"
  ]
}}

Return ONLY valid JSON, no other text."""

        try:
            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert C/C++ crash analyzer specializing in telecommunications software and segmentation fault debugging."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Clean up JSON formatting
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:]
            if analysis_text.endswith('```'):
                analysis_text = analysis_text[:-3]
            
            analysis_result = json.loads(analysis_text)
            
            logger.info("✅ Crash analysis completed successfully")
            
            return {
                "crash_info": crash_info,
                "analysis": analysis_result,
                "context_used": assembled_context
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze crash: {e}")
            return {
                "crash_info": crash_info,
                "error": str(e),
                "analysis": None
            }
    
    def process_segmentation_fault(self, log_file_path: str, extract_only: bool = True) -> Dict[str, Any]:
        """
        Phase 1: Extract error and traceback from segmentation fault log.
        
        Args:
            log_file_path: Path to the segmentation fault log file
            extract_only: If True, only extract info without LLM analysis
            
        Returns:
            Extracted crash information
        """
        logger.info("=" * 80)
        logger.info("🔬 PHASE 1: SEGMENTATION FAULT ERROR & TRACEBACK EXTRACTION")
        logger.info("=" * 80)
        
        # Step 1: Parse the log to extract crash information
        logger.info("\n📄 Parsing segmentation fault log...")
        crash_info = self.parse_segmentation_fault_log(log_file_path)
        
        if "error" in crash_info:
            logger.error(f"Failed to parse log: {crash_info['error']}")
            return crash_info
        
        if not crash_info.get("crash_detected"):
            logger.warning("No crash detected in log file")
            return {
                "error": "No segmentation fault detected in log file",
                "crash_info": crash_info
            }
        
        # Step 2: Read source code at fault location if available
        source_code = None
        if crash_info.get("faulting_file") and crash_info.get("faulting_line"):
            source_code = self.read_source_code_at_fault(
                crash_info["faulting_file"],
                int(crash_info["faulting_line"]),
                context_lines=25
            )
            if source_code:
                crash_info["source_code_at_fault"] = source_code
        
        # Create extraction result
        extraction_result = {
            "phase": "extraction",
            "crash_info": crash_info,
            "log_file": log_file_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # If extract_only is False, perform full analysis
        if not extract_only:
            logger.info("\n🧠 Performing full crash analysis with LLM...")
            analysis_result = self.analyze_crash(crash_info)
            extraction_result["analysis"] = analysis_result.get("analysis")
            extraction_result["phase"] = "full_analysis"
        
        # Save extraction results
        logger.info("\n💾 Saving extraction results...")
        output_file = "output/segmentation_fault_extraction.json"
        os.makedirs("output", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ PHASE 1 EXTRACTION COMPLETED")
        logger.info("=" * 80)
        
        return extraction_result
    
    def run_phase2_retrieval(self, phase1_extraction_file: str = "output/segmentation_fault_extraction.json") -> Optional[Dict]:
        """
        Run Phase 2: Targeted candidate retrieval for crash analysis.
        
        Args:
            phase1_extraction_file: Path to Phase 1 extraction results
            
        Returns:
            Phase 2 retrieval results or None if Phase 2 not available
        """
        if not self.phase2_retrieval:
            logger.error("❌ Phase 2 retrieval not available (CrashPhase2Retrieval not loaded)")
            return None
        
        logger.info("\n🔗 Running Phase 2: Targeted Candidate Retrieval...")
        return self.phase2_retrieval.process_crash_retrieval(phase1_extraction_file)
    
    def run_phase2_grading(self, 
                          phase2_file: str = "output/crash_phase2_retrieval.json",
                          phase1_file: str = "output/segmentation_fault_extraction.json") -> Optional[Dict]:
        """
        Run Phase 2.5: Intelligent LLM-based function grading.
        
        Args:
            phase2_file: Path to Phase 2 retrieval results
            phase1_file: Path to Phase 1 extraction results
            
        Returns:
            Graded and filtered results (top 10 functions)
        """
        if not self.phase2_grading:
            logger.error("❌ Phase 2.5 grading not available (CrashPhase2Grading not loaded)")
            return None
        
        logger.info("\n🧠 Running Phase 2.5: Intelligent Function Grading...")
        return self.phase2_grading.process_grading(phase2_file, phase1_file)
    
    def run_phase3_fix_generation(self,
                                  phase2_graded_file: str = "output/crash_phase2_graded.json",
                                  phase1_file: str = "output/segmentation_fault_extraction.json") -> Optional[Dict]:
        """
        Run Phase 3: Generate crash fixes.
        
        Args:
            phase2_graded_file: Path to Phase 2.5 graded results
            phase1_file: Path to Phase 1 extraction results
            
        Returns:
            Fix suggestions in fix_suggestions.json format
        """
        if not self.phase3_fix_gen:
            logger.error("❌ Phase 3 fix generation not available (CrashPhase3FixGeneration not loaded)")
            return None
        
        logger.info("\n🔧 Running Phase 3: Fix Generation...")
        return self.phase3_fix_gen.process_crash_fix_generation(phase2_graded_file, phase1_file)


def main():
    """Test function for segmentation fault analyzer - Runs Phase 1, Phase 2, Phase 2.5, and Phase 3"""
    import sys
    
    # Check command line arguments
    run_phase2 = "--phase2" in sys.argv or "--all" in sys.argv
    run_phase25 = "--phase2.5" in sys.argv or "--all" in sys.argv
    run_phase3 = "--phase3" in sys.argv or "--all" in sys.argv
    
    analyzer = SegmentationFaultAnalyzer()
    
    # Test with the provided log file
    log_file = "log_files/segmentation_fault.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please run this script from the Error_fixing_pipelin directory")
        return
    
    # Run Phase 1
    print("=" * 80)
    print("🔬 RUNNING PHASE 1: ERROR & TRACEBACK EXTRACTION")
    print("=" * 80)
    
    results = analyzer.process_segmentation_fault(log_file, extract_only=True)
    
    # Display Phase 1 summary
    print("\n" + "=" * 80)
    print("📊 PHASE 1 EXTRACTION SUMMARY")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    crash_info = results.get("crash_info", {})
    print(f"\n✅ Crash Detected: {crash_info.get('crash_detected', False)}")
    print(f"🔥 Signal: {crash_info.get('signal', 'N/A')}")
    print(f"💥 Fault Location: {crash_info.get('fault_location', 'N/A')}")
    print(f"⚙️  Faulting Function: {crash_info.get('faulting_function', 'N/A')}")
    print(f"🧵 Thread: {crash_info.get('crash_thread', {}).get('name', 'N/A')}")
    print(f"📚 Backtrace Frames: {len(crash_info.get('backtrace', []))}")
    print(f"🔄 Scenario Steps Before Crash: {len(crash_info.get('scenario_flow', []))}")
    
    if crash_info.get('scenario_flow'):
        print(f"\n📋 Scenario Flow:")
        for i, step in enumerate(crash_info['scenario_flow'], 1):
            print(f"   {i}. {step}")
    
    if crash_info.get('backtrace'):
        print(f"\n🔗 Backtrace:")
        for frame in crash_info['backtrace'][:5]:
            print(f"   #{frame['frame_number']}: {frame['function']} at {frame['file']}:{frame['line']}")
    
    print(f"\n💾 Phase 1 results saved to: output/segmentation_fault_extraction.json")
    
    # Run Phase 2 if requested
    if run_phase2:
        print("\n" + "=" * 80)
        print("🔗 RUNNING PHASE 2: TARGETED CANDIDATE RETRIEVAL")
        print("=" * 80)
        
        phase2_results = analyzer.run_phase2_retrieval()
        
        if phase2_results:
            print(f"\n✅ Phase 2 Completed!")
            total_functions = len(phase2_results.get('prioritized_functions', [])) + len(phase2_results.get('call_chain_expansion', []))
            print(f"   Total Functions Collected: {total_functions}")
            print(f"   - Prioritized (Backtrace): {len(phase2_results.get('prioritized_functions', []))}")
            print(f"   - Call Chain Expansion: {len(phase2_results.get('call_chain_expansion', []))}")
            print(f"   - Similar Crash Fixes: {len(phase2_results.get('similar_crash_fixes', []))}")
            print(f"\n💾 Phase 2 results saved to: output/crash_phase2_retrieval.json")
            
            # Automatically run Phase 2.5 if Phase 2 succeeded
            if run_phase25 or "--all" in sys.argv:
                print("\n" + "=" * 80)
                print("🧠 RUNNING PHASE 2.5: INTELLIGENT FUNCTION GRADING")
                print("=" * 80)
                
                phase25_results = analyzer.run_phase2_grading()
                
                if phase25_results:
                    print(f"\n✅ Phase 2.5 Completed!")
                    print(f"   Total Candidates Analyzed: {phase25_results.get('total_candidates_analyzed', 0)}")
                    print(f"   Top Functions Selected: {phase25_results.get('top_functions_selected', 0)}")
                    
                    selected = phase25_results.get('suspected_functions', [])
                    if selected:
                        print(f"\n🎯 TOP {len(selected)} SELECTED FUNCTIONS:")
                        for i, func in enumerate(selected[:10], 1):
                            has_code = "✓" if func.get("has_source") else "✗"
                            role = func.get("role", "N/A")
                            if func.get("frame_number") is not None:
                                role = f"Frame #{func.get('frame_number')} - {role}"
                            print(f"   {i}. [{func.get('relevance_score', 0):.2f}] {has_code} {func['function_name']}")
                            print(f"      {role}")
                    
                    print(f"\n💾 Phase 2.5 results saved to: output/crash_phase2_graded.json")
                    
                    # Automatically run Phase 3 if Phase 2.5 succeeded
                    if run_phase3 or "--all" in sys.argv:
                        print("\n" + "=" * 80)
                        print("🔧 RUNNING PHASE 3: CRASH FIX GENERATION")
                        print("=" * 80)
                        
                        phase3_results = analyzer.run_phase3_fix_generation()
                        
                        if phase3_results and "error" not in phase3_results:
                            print(f"\n✅ Phase 3 Completed!")
                            fix_suggestion = phase3_results.get('fix_suggestion', {})
                            print(f"   Suspected Functions: {len(fix_suggestion.get('suspected_functions', []))}")
                            print(f"   Code Patches Generated: {len(fix_suggestion.get('code_patches', []))}")
                            print(f"   Investigation Steps: {len(fix_suggestion.get('investigation_steps', []))}")
                            
                            # Show generated patches
                            if fix_suggestion.get('code_patches'):
                                print(f"\n🔧 GENERATED CODE PATCHES:")
                                for i, patch in enumerate(fix_suggestion['code_patches'], 1):
                                    print(f"   {i}. {patch.get('function_name')} ({patch.get('patch_type')})")
                                    print(f"      File: {patch.get('file_path')}")
                                    print(f"      {patch.get('description', 'N/A')[:120]}")
                            
                            print(f"\n💾 Phase 3 fixes saved to: output/crash_phase3_fixes.json")
                        else:
                            print(f"\n❌ Phase 3 failed")
                else:
                    print(f"\n❌ Phase 2.5 failed")
        else:
            print(f"\n❌ Phase 2 failed or not available")
    else:
        print(f"\n💡 Available options:")
        print(f"   --phase2     : Run Phase 1 + Phase 2")
        print(f"   --phase2.5   : Run Phase 1 + Phase 2 + Phase 2.5")
        print(f"   --phase3     : Run Phase 1 + Phase 2 + Phase 2.5 + Phase 3")
        print(f"   --all        : Run all phases (1 + 2 + 2.5 + 3)")
        print(f"\n   Example: python segmentation_fault_analyzer.py --all")


if __name__ == "__main__":
    main()

