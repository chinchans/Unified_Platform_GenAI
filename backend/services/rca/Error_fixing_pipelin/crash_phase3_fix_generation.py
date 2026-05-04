#!/usr/bin/env python3
"""
Crash Analysis - Phase 3: Fix Generation for Segmentation Faults

This module generates specific code fixes for segmentation faults,
focusing on NULL pointer checks, parameter validation, and defensive programming.

Author: AI Assistant
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AzureOpenAI

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrashPhase3FixGeneration:
    """Crash-specific Phase 3: Generate fixes for segmentation faults"""
    
    def __init__(self, openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize crash fix generation.
        
        Args:
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        logger.info("🔧 Initializing Crash Phase 3 Fix Generation...")
        
        self.openair_codebase_file_name = openair_codebase_file_name
        self.codebase_path = f"Error_fixing_pipelin/{openair_codebase_file_name}"
        
        # Setup Azure OpenAI client
        self._setup_azure_client()
        
        logger.info("✅ Crash Phase 3 Fix Generation initialized")
    
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
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize Azure OpenAI client
        self.azure_client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint
        )
        
        logger.info("✅ Azure OpenAI client initialized")
    
    def process_crash_fix_generation(self, 
                                     phase2_graded_file: str,
                                     phase1_extraction_file: str) -> Dict[str, Any]:
        """
        Generate crash fixes from Phase 2.5 graded functions.
        
        Args:
            phase2_graded_file: Path to crash_phase2_graded.json
            phase1_extraction_file: Path to segmentation_fault_extraction.json
            
        Returns:
            Fix suggestions in the same format as fix_suggestions.json
        """
        logger.info("=" * 80)
        logger.info("🔧 CRASH ANALYSIS - PHASE 3: FIX GENERATION")
        logger.info("=" * 80)
        
        # Load Phase 2.5 graded results
        if not os.path.exists(phase2_graded_file):
            logger.error(f"Phase 2.5 file not found: {phase2_graded_file}")
            return {"error": "Phase 2.5 file not found"}
        
        with open(phase2_graded_file, 'r', encoding='utf-8') as f:
            phase2_data = json.load(f)
        
        # Load Phase 1 extraction for crash context
        if not os.path.exists(phase1_extraction_file):
            logger.error(f"Phase 1 file not found: {phase1_extraction_file}")
            return {"error": "Phase 1 file not found"}
        
        with open(phase1_extraction_file, 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
        
        crash_info = phase1_data.get("crash_info", {})
        suspected_functions = phase2_data.get("suspected_functions", [])
        
        logger.info(f"📥 Loaded crash data:")
        logger.info(f"   - Crash signal: {crash_info.get('signal')}")
        logger.info(f"   - Faulting function: {crash_info.get('faulting_function')}")
        logger.info(f"   - Suspected functions: {len(suspected_functions)}")
        
        # Generate error text summary
        error_text = self._generate_error_text(crash_info)
        
        # Generate fixes using LLM
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: GENERATING CODE FIXES")
        logger.info("=" * 60)
        
        fix_suggestions = self._generate_llm_fix_suggestions(
            error_text=error_text,
            crash_info=crash_info,
            suspected_functions=suspected_functions
        )
        
        # Format in fix_suggestions.json style
        formatted_results = {
            "error_text": error_text,
            "fix_suggestion": fix_suggestions,
            "context_summary": {
                "candidate_functions_count": len(suspected_functions),
                "candidate_configs_count": 0,  # Crashes don't typically involve configs
                "call_graph_entries": 0,
                "pattern_matched": False
            }
        }
        
        # Save results
        output_file = "output/crash_phase3_fixes.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Phase 3 fixes saved to: {output_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ CRASH ANALYSIS PHASE 3 COMPLETED")
        logger.info("=" * 80)
        
        return formatted_results
    
    def _generate_error_text(self, crash_info: Dict) -> str:
        """Generate error text summary from crash info"""
        signal = crash_info.get("signal", "SIGSEGV")
        fault_location = crash_info.get("fault_location", "Unknown")
        faulting_function = crash_info.get("faulting_function", "Unknown")
        thread = crash_info.get("crash_thread", {}).get("name", "Unknown")
        
        error_text = (
            f"{signal} - Segmentation fault in {faulting_function} "
            f"at {fault_location} (Thread: {thread})"
        )
        
        return error_text
    
    def _generate_llm_fix_suggestions(self,
                                      error_text: str,
                                      crash_info: Dict,
                                      suspected_functions: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to generate crash fix suggestions.
        
        Returns:
            Dictionary matching fix_suggestion structure
        """
        logger.info(f"🤖 Generating crash fixes for {len(suspected_functions)} functions...")
        
        # Build comprehensive crash context
        context = self._build_crash_context(error_text, crash_info, suspected_functions)
        
        # Create LLM prompt  
        # Note: LLM should use relative paths (like openair2/RRC/NR/file.c), we'll add codebase prefix later
        # Use regular string, not f-string, since we have many braces in JSON examples
        prompt = f"""{context}

## TASK: GENERATE CRASH FIX SUGGESTIONS

You are a C/C++ crash debugging expert. Analyze the segmentation fault above and generate SPECIFIC code patches to prevent this crash.

### FIX STRATEGY FOR SEGMENTATION FAULTS:

**CRITICAL ANALYSIS STEPS:**

1. **STEP 1: Search ALL suspected functions for the SMOKING GUN:**
   
   🔍 **Check EACH function's source code for NULL assignment patterns:**
   - Search for: `variableName=NULL;` or `variableName = NULL;`
   - Check if this happens BEFORE the variable is used
   - **The bug might be in Frame #0 OR in a CALLER function!**
   
   **Example A - Bug in Frame #0 (crash point):**
   ```
   void crashFunc(Struct *param) {{
     param=NULL;        // ← Bug here
     param->field = x;  // ← Crash here
   }}
   ```
   
   **Example B - Bug in CALLER (more common!):**
   ```
   void callerFunc(..., Struct *param) {{
     param=NULL;           // ← Bug here (sets to NULL)
     crashFunc(param);     // ← Passes NULL → crash in crashFunc!
   }}
   ```
   
   **Fix for both:** Remove the `param=NULL;` line in whichever function has it!

2. **STEP 2: Check which function actually HAS the bug:**
   - Review the source code of ALL suspected functions
   - Look for which function contains the NULL assignment
   - Generate the patch for THAT function, not always Frame #0!

3. **STEP 3: If NO null assignment found in any function, check for other patterns:**
   - Pointer used WITHOUT validation (missing `if (!ptr)` check)
   - Use-after-free (accessing freed memory)
   - Uninitialized pointer dereference
   - Array index out of bounds
   - Missing return value checks (e.g., `malloc` returns NULL)

2. **FIX TYPES by Bug Pattern:**
   
   **Pattern A: Pointer set to NULL before use**
   - Fix: Remove the NULL assignment 
   - Patch type: `remove_null_assignment` or `move_statement`
   
   **Pattern B: Missing NULL check**
   - Fix: Add NULL check before dereferencing
   - Patch type: `null_check_insertion`
   
   **Pattern C: Uninitialized pointer**
   - Fix: Initialize to NULL or valid value
   - Patch type: `initialization_fix`
   
   **Pattern D: Missing allocation check**
   - Fix: Check if malloc/alloc returned NULL
   - Patch type: `allocation_check`

3. **SECONDARY FIXES (Frame #1 - IMMEDIATE_CALLER):**
   - Validate data BEFORE passing to crash function
   - Check return values and handle errors
   - Ensure proper initialization

4. **TERTIARY FIXES (Upstream/Downstream):**
   - Add validation in parameter-passing functions
   - Ensure proper error handling throughout call chain

### OUTPUT FORMAT (JSON):

Generate **MULTIPLE** specific code patches addressing the SPECIFIC bug pattern you identified.

**EXAMPLES for Different Bug Patterns:**

**Example A - Pointer set to NULL before use (MOST IMPORTANT - Check for this FIRST!):**

⚠️ CRITICAL: If the Frame #0 source code contains a line like `rrcSetupComplete=NULL;` or `param=NULL;` 
and then LATER the code tries to use that pointer (like `param->field`), 
then the bug is that NULL assignment line! Your patch should REMOVE it!

Example pattern in source code:
```
void func(SomeType *param) 
  memset(req, 0, sizeof(*req));
  param=NULL;  // BUG: Setting param to NULL here
  // Later in the code...
  param->field = value;  // CRASH: Using NULL pointer
```

Correct patch to generate:
```json
PATCH_TYPE: "remove_null_assignment"
ORIGINAL_CODE: Show the context around param=NULL line
PATCHED_CODE: Same code but with param=NULL line removed or commented out
DESCRIPTION: "Remove the param=NULL assignment that was invalidating the parameter before use"
```

**Example B - Missing NULL check (ONLY if NO null assignment found in code):**
```
PATCH_TYPE: "null_check_insertion"
Add NULL validation at function entry if parameter is used without checking
```

**Example C - Missing allocation check:**
```
PATCH_TYPE: "allocation_check"
Check if malloc/alloc returned NULL before using
```

**NOW ANALYZE THE CODE ABOVE:**

Step 1: Look at ALL suspected functions source code - search for `variableName=NULL;`
Step 2: Identify WHICH function has the NULL assignment (could be Frame #0, caller, or any other function)
Step 3: Generate patch for the CORRECT function that contains the bug
Step 4: If bug is in caller function, fix THAT function, not the crash point!
Step 5: If NO null assignment found in any function, then add defensive NULL checks

### REQUIRED JSON OUTPUT FORMAT (Follow EXACTLY):

```json
{{
  "suspected_functions": ["function1", "function2", "function3"],
  "suspected_configs": [],
  "reason": "Brief summary of root cause",
  "config_fix": "",
  "code_patches": [
    {{
      "function_name": "rrc_gNB_send_NGAP_NAS_FIRST_REQ",
      "file_path": "openair2/RRC/NR/rrc_gNB_NGAP.c",
      "patch_type": "remove_null_assignment",
      "original_code": "  memset(req, 0, sizeof(*req));\\n  rrcSetupComplete=NULL;\\n  // RAN UE NGAP ID",
      "patched_code": "  memset(req, 0, sizeof(*req));\\n  // REMOVED: rrcSetupComplete=NULL;\\n  // RAN UE NGAP ID",
      "line_numbers": "241",
      "description": "Remove the rrcSetupComplete=NULL statement that was setting parameter to NULL before use"
    }},
    {{
      "function_name": "rrc_gNB_process_RRCSetupComplete",
      "file_path": "openair2/RRC/NR/rrc_gNB.c",
      "patch_type": "validation_before_call",
      "original_code": "  rrc_gNB_send_NGAP_NAS_FIRST_REQ(rrc, UE, rrcSetupComplete);",
      "patched_code": "  if (!rrcSetupComplete) {{\\n    LOG_E(RRC, \\"Invalid param\\\\n\\");\\n    return;\\n  }}\\n  rrc_gNB_send_NGAP_NAS_FIRST_REQ(rrc, UE, rrcSetupComplete);",
      "line_numbers": "1965",
      "description": "Validate parameter before calling crash function"
    }}
  ],
  "config_patches": [],
  "root_cause_analysis": "Technical explanation of crash and fix",
  "investigation_steps": ["Step 1", "Step 2", "Step 3"],
  "specification_context": ""
}}
```

### CRITICAL REQUIREMENTS:
1. Use LOWERCASE field names: patch_type, code_patches, function_name (NOT PATCH_TYPE, PATCHES)
2. Include ALL required fields for each patch
3. Generate 2-4 patches for top functions
4. First check for null assignment bug, then add defensive checks
5. Return ONLY the JSON, no extra text

Generate now:"""

        # Call LLM
        response = self.azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert C/C++ crash debugger who generates precise NULL pointer check and validation fixes for segmentation faults. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3,
            seed=44444  # Crash fix generation seed
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean up markdown if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        # Parse JSON
        try:
            fix_suggestion = json.loads(response_text)
            
            # Add codebase folder prefix to all file paths
            # LLM generates relative paths like "openair2/RRC/NR/file.c"
            # We need "openairinterface5g-develop/openair2/RRC/NR/file.c"
            for patch in fix_suggestion.get('code_patches', []):
                if 'file_path' in patch:
                    original_path = patch['file_path']
                    # Only add prefix if path doesn't already start with the codebase folder
                    if not original_path.startswith(self.openair_codebase_file_name):
                        patch['file_path'] = f"{self.openair_codebase_file_name}/{original_path}"
                        logger.debug(f"   Prefixed path: {original_path} → {patch['file_path']}")
            
            for patch in fix_suggestion.get('config_patches', []):
                if 'file_path' in patch:
                    original_path = patch['file_path']
                    if not original_path.startswith(self.openair_codebase_file_name):
                        patch['file_path'] = f"{self.openair_codebase_file_name}/{original_path}"
                        logger.debug(f"   Prefixed path: {original_path} → {patch['file_path']}")
            
            logger.info(f"✅ Generated {len(fix_suggestion.get('code_patches', []))} code patches")
            return fix_suggestion
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Return minimal structure
            return {
                "suspected_functions": [f.get("function_name") for f in suspected_functions[:3]],
                "suspected_configs": [],
                "reason": "Failed to generate fix suggestion - JSON parse error",
                "config_fix": "",
                "code_patches": [],
                "config_patches": [],
                "root_cause_analysis": "",
                "investigation_steps": [],
                "specification_context": ""
            }
    
    def _build_crash_context(self, 
                            error_text: str, 
                            crash_info: Dict, 
                            suspected_functions: List[Dict]) -> str:
        """Build comprehensive crash context for LLM"""
        
        context_parts = []
        context_parts.append("# SEGMENTATION FAULT ANALYSIS")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Crash Summary
        context_parts.append("## 🔥 CRASH SUMMARY")
        context_parts.append(f"**Error:** {error_text}")
        context_parts.append(f"**Signal:** {crash_info.get('signal', 'SIGSEGV')}")
        context_parts.append(f"**Fault Location:** {crash_info.get('fault_location', 'Unknown')}")
        context_parts.append(f"**Faulting Function:** {crash_info.get('faulting_function', 'Unknown')}")
        context_parts.append(f"**Thread:** {crash_info.get('crash_thread', {}).get('name', 'Unknown')}")
        context_parts.append("")
        
        # Backtrace
        if crash_info.get("backtrace"):
            context_parts.append("## 📚 BACKTRACE (Call Stack)")
            for frame in crash_info["backtrace"]:
                context_parts.append(
                    f"#{frame['frame_number']}: {frame['function']} "
                    f"at {frame['file']}:{frame['line']}"
                )
            context_parts.append("")
        
        # Scenario Flow
        if crash_info.get("scenario_flow"):
            context_parts.append("## 🔄 SCENARIO FLOW (What happened before crash)")
            for i, step in enumerate(crash_info["scenario_flow"], 1):
                context_parts.append(f"{i}. {step}")
            context_parts.append("")
        
        # Suspected Functions with Source Code
        context_parts.append("## 🎯 SUSPECTED FUNCTIONS (From Phase 2.5 LLM Grading)")
        context_parts.append(f"Total: {len(suspected_functions)} functions")
        context_parts.append("")
        
        for i, func in enumerate(suspected_functions, 1):
            context_parts.append(f"### Function {i}: {func.get('function_name')}")
            context_parts.append(f"**File:** {func.get('file_path')}")
            context_parts.append(f"**Relevance Score:** {func.get('relevance_score', 0):.2f}")
            context_parts.append(f"**Role:** {func.get('role', 'Unknown')}")
            
            if func.get('frame_number') is not None:
                context_parts.append(f"**Backtrace Frame:** #{func.get('frame_number')}")
            
            context_parts.append(f"**Reason:** {func.get('reason', 'N/A')}")
            
            if func.get('has_source') and func.get('code_snippet'):
                context_parts.append(f"\n**Source Code:**")
                context_parts.append("```c")
                context_parts.append(func.get('code_snippet', '// Not available'))
                context_parts.append("```")
            else:
                context_parts.append(f"\n**Source Code:** Not available")
            
            context_parts.append("")
        
        return "\n".join(context_parts)


def main():
    """Test crash Phase 3 fix generation"""
    fix_gen = CrashPhase3FixGeneration()
    
    # Test with Phase 2.5 and Phase 1 results
    phase2_file = "output/crash_phase2_graded.json"
    phase1_file = "output/segmentation_fault_extraction.json"
    
    if not os.path.exists(phase2_file):
        print(f"❌ Phase 2.5 file not found: {phase2_file}")
        print(f"   Please run Phase 2.5 first")
        return
    
    if not os.path.exists(phase1_file):
        print(f"❌ Phase 1 file not found: {phase1_file}")
        print(f"   Please run Phase 1 first")
        return
    
    results = fix_gen.process_crash_fix_generation(phase2_file, phase1_file)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 PHASE 3 FIX GENERATION SUMMARY")
    print("=" * 80)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        fix_suggestion = results.get('fix_suggestion', {})
        print(f"\n✅ Suspected Functions: {len(fix_suggestion.get('suspected_functions', []))}")
        print(f"✅ Code Patches Generated: {len(fix_suggestion.get('code_patches', []))}")
        print(f"✅ Investigation Steps: {len(fix_suggestion.get('investigation_steps', []))}")
        
        # Show code patches
        if fix_suggestion.get('code_patches'):
            print(f"\n🔧 Code Patches:")
            for i, patch in enumerate(fix_suggestion['code_patches'], 1):
                print(f"   {i}. {patch.get('function_name')} - {patch.get('patch_type')}")
                print(f"      {patch.get('description', 'N/A')[:100]}")
        
        print(f"\n💾 Results saved to: output/crash_phase3_fixes.json")


if __name__ == "__main__":
    main()

