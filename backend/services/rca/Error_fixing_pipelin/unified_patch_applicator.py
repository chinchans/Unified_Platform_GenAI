#!/usr/bin/env python3
"""
Unified Patch Applicator

This script applies both code patches and configuration patches from AI suggestions,
combining smart function detection with targeted configuration fixes.

Features:
- Auto-detects function locations regardless of AI-provided line numbers
- Applies configuration patches only to AI-identified files
- Smart brace counting for accurate function boundaries
- Clean focused output showing only relevant context
- Comprehensive logging and validation

Usage:
    python unified_patch_applicator.py [--dry-run] [--backup]

Author: AI Assistant
"""

import json
import os
import shutil
import argparse
import logging
import re
import textwrap
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Configure minimal logging for errors only
logging.basicConfig(level=logging.ERROR)

class UnifiedPatchApplicator:
    """Unified applicator for both code and configuration patches"""
    
    def __init__(self, suggestions_file: str = "output/crash_phase3_fixes.json"):
        self.suggestions_file = suggestions_file
        self.applied_code_patches = []
        self.failed_code_patches = []
        self.applied_config_patches = []
        self.failed_config_patches = []
    
    def _resolve_file_path(self, file_path: str) -> str:
        """Resolve file path relative to current working directory.
        
        For Git fixes, paths may include 'Error_fixing_pipelin/' prefix.
        If we're running from within Error_fixing_pipelin directory, strip it.
        For regular RCA analysis, paths don't have this prefix, so leave them unchanged.
        """
        if not file_path:
            return file_path
        
        # Normalize path separators
        normalized_path = file_path.replace('\\', '/')
        
        # Only strip Error_fixing_pipelin/ prefix if:
        # 1. Path starts with it (Git fix paths)
        # 2. We're actually in the Error_fixing_pipelin directory
        current_dir = os.path.basename(os.getcwd())
        if normalized_path.startswith('Error_fixing_pipelin/') and current_dir == 'Error_fixing_pipelin':
            # Strip the prefix since we're already in that directory
            normalized_path = normalized_path[len('Error_fixing_pipelin/'):]
        
        # Convert back to OS-specific path
        return os.path.normpath(normalized_path)
    
    def load_suggestions(self) -> Dict:
        """Load fix suggestions from JSON file"""
        try:
            with open(self.suggestions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Fix suggestions file not found: {self.suggestions_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {self.suggestions_file}: {e}")
            return {}
    
    # ============================================================================
    # CODE PATCH METHODS (from smart_patch_applicator.py)
    # ============================================================================
    
    def _find_function_start(self, file_path: str, function_name: str) -> Optional[int]:
        """Find the starting line of a function using multiple patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            return None
        
        # Enhanced patterns for function detection
        patterns = [
            # Standard function definitions
            rf'^\s*static\s+\w+\s*\*?\s*{re.escape(function_name)}\s*\(',
            rf'^\s*\w+\s*\*?\s*{re.escape(function_name)}\s*\(',
            rf'^\s*{re.escape(function_name)}\s*\(',
            # Complex return types
            rf'\s+{re.escape(function_name)}\s*\(',
            # Inline and other variations
            rf'^\s*inline\s+\w+\s*{re.escape(function_name)}\s*\(',
            rf'^\s*extern\s+\w+\s*{re.escape(function_name)}\s*\(',
            rf'^\s*__attribute__.*{re.escape(function_name)}\s*\(',
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Found function signature - now find the opening brace
                    return self._find_opening_brace(lines, i)
        
        # Fallback search
        return self._fallback_function_search(lines, function_name)
    
    def _find_context_based_insertion(self, file_path: str, context_description: str) -> Optional[int]:
        """Find insertion point based on specific context description"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            return None
        
        # Extract search terms from context description
        if "containing" in context_description:
            # Extract text between quotes or after "containing"
            import re
            match = re.search(r'containing\s*["\']([^"\']+)["\']', context_description)
            if match:
                search_text = match.group(1)
                for i, line in enumerate(lines):
                    if search_text in line:
                        # Check if this is a function signature - if so, find the opening brace
                        if self._is_function_signature(lines, i):
                            return self._find_opening_brace_after_signature(lines, i)
                        else:
                            return i + 2  # Insert after the line containing the text
        elif "after" in context_description and "line" in context_description:
            # Extract line number
            import re
            match = re.search(r'line\s+(\d+)', context_description)
            if match:
                line_num = int(match.group(1))
                return line_num + 1  # Insert after the specified line
        
        return None
    
    def _is_function_signature(self, lines: List[str], line_idx: int) -> bool:
        """Check if a line is a function signature"""
        line = lines[line_idx]
        # Look for function signature patterns
        function_patterns = [
            r'^\s*\w+\s+\w+\s*\(',  # return_type function_name(
            r'^\s*static\s+\w+\s+\w+\s*\(',  # static return_type function_name(
            r'^\s*\w+\s*\*?\s*\w+\s*\(',  # return_type* function_name(
        ]
        
        for pattern in function_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _find_opening_brace_after_signature(self, lines: List[str], signature_line: int) -> int:
        """Find the opening brace after a function signature"""
        # Look for opening brace in the same line or next few lines
        for i in range(signature_line, min(signature_line + 10, len(lines))):
            line = lines[i]
            if '{' in line:
                # Found opening brace - return the line number after it
                return i + 2  # +2 because we want to insert after the brace line
        
        # If no brace found, return line after signature
        return signature_line + 2
    
    def _fallback_function_search(self, lines: List[str], function_name: str) -> Optional[int]:
        """Fallback search with more relaxed criteria"""
        # Silent fallback search
        for i, line in enumerate(lines):
            if function_name in line and ('(' in line and ')' in line):
                if self._has_function_context(lines, i):
                    return i + 1
        
        return None
    
    def _find_opening_brace(self, lines: List[str], signature_line: int) -> int:
        """Find the opening brace of a function after its signature"""
        # Look for opening brace in the same line or next few lines
        for i in range(signature_line, min(signature_line + 5, len(lines))):
            line = lines[i]
            if '{' in line:
                # Found opening brace - return the line number after it
                return i + 2  # +2 because we want to insert after the brace line
        
        # If no brace found, return line after signature
        return signature_line + 2
    
    def _has_function_context(self, lines: List[str], line_idx: int) -> bool:
        """Check if the line is in a function context"""
        # Check previous lines for function-like patterns
        start = max(0, line_idx - 3)
        end = min(len(lines), line_idx + 3)
        
        context_lines = lines[start:end]
        context_text = ''.join(context_lines)
        
        # Look for function indicators
        indicators = ['{', 'return', 'static', 'void', 'int', 'char', 'struct']
        return any(indicator in context_text for indicator in indicators)
    
    def _find_function_end(self, file_path: str, start_line: int) -> int:
        """Find the end of a function using brace counting"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            return start_line
        
        brace_count = 0
        found_opening = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
                    if found_opening and brace_count == 0:
                        return i + 1
        
        # Fallback: assume function is reasonably sized
        estimated_end = min(start_line + 100, len(lines))
        return estimated_end
    
    def apply_code_patch(self, patch: Dict, dry_run: bool = False, backup: bool = True) -> bool:
        """Apply a code patch using smart function detection"""
        function_name = patch.get('function_name', 'Unknown')
        file_path = patch.get('file_path', '')
        # Resolve file path relative to current working directory
        file_path = self._resolve_file_path(file_path)
        patched_code = patch.get('patched_code', '')
        original_code = patch.get('original_code', '')
        patch_type = patch.get('patch_type', 'insertion')
        description = patch.get('description', '')
        
        # Store function name for clean display
        self.current_function_name = function_name
        
        print(f"\nAPPLYING CODE PATCH")
        print(f"Description: {description}")
        print(f"Patch Type: {patch_type}")
        print()
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        if not dry_run:
            # Apply the actual patch based on patch type
            if patch_type in ('targeted_insertion_or_adjustment', 'targeted_replacement') and original_code and patched_code:
                return self._apply_code_replacement(file_path, original_code, patched_code, backup)
            else:
                # Fallback to insertion method
                line_numbers = patch.get('line_numbers', '')
                if line_numbers and ('containing' in line_numbers or 'after line' in line_numbers):
                    start_line = self._find_context_based_insertion(file_path, line_numbers)
                    if start_line is not None:
                        print(f"Found context-based insertion point at line {start_line}")
                    else:
                        print(f"Could not find context-based insertion point, falling back to function start")
                        start_line = self._find_function_start(file_path, function_name)
                else:
                    # Find the function start
                    start_line = self._find_function_start(file_path, function_name)
                
                if not start_line:
                    print(f"Function '{function_name}' not found in {file_path}")
                    return False
                
                # Show focused context
                end_line = self._find_function_end(file_path, start_line)
                self._show_focused_context(file_path, start_line, end_line, patched_code, dry_run)
                
                return self._apply_code_insertion(file_path, start_line, patched_code, backup)
        else:
            # For dry run, show what would be changed
            if patch_type in ('targeted_insertion_or_adjustment', 'targeted_replacement') and original_code and patched_code:
                self._show_replacement_preview(file_path, original_code, patched_code)
            else:
                # Fallback to insertion preview
                line_numbers = patch.get('line_numbers', '')
                if line_numbers and ('containing' in line_numbers or 'after line' in line_numbers):
                    start_line = self._find_context_based_insertion(file_path, line_numbers)
                else:
                    start_line = self._find_function_start(file_path, function_name)
                
                if start_line:
                    end_line = self._find_function_end(file_path, start_line)
                    self._show_focused_context(file_path, start_line, end_line, patched_code, dry_run)
        
        return True
    
    @staticmethod
    def _find_loose_original_line_span(content: str, original_block: str) -> Optional[Tuple[int, int]]:
        """Find start/end line indices (0-based, inclusive) where each line matches after .strip()."""
        orig_lines = original_block.strip().splitlines()
        if not orig_lines:
            return None
        file_lines = content.splitlines()
        n = len(orig_lines)
        for i in range(len(file_lines) - n + 1):
            if all(
                file_lines[i + j].strip() == orig_lines[j].strip()
                for j in range(n)
            ):
                return (i, i + n - 1)
        return None
    
    @staticmethod
    def _reindent_patched_to_match_file(
        patched_block: str, base_lead: str, old_snippet: str, full_content: str
    ) -> str:
        """Dedents LLM snippet and prefixes each non-empty line with the file's leading indent."""
        ded = textwrap.dedent(patched_block.strip()).rstrip('\n')
        sep = '\r\n' if '\r\n' in old_snippet else ('\n' if '\n' in old_snippet else ('\r\n' if '\r\n' in full_content else '\n'))
        out: List[str] = []
        for line in ded.splitlines():
            out.append(base_lead + line if line.strip() else '')
        result = sep.join(out)
        if old_snippet.endswith('\n') or old_snippet.endswith('\r\n'):
            if not result.endswith(sep):
                result += sep
        return result
    
    def _apply_code_replacement(self, file_path: str, original_code: str, patched_code: str, backup: bool) -> bool:
        """Apply code replacement by finding and replacing the original code with patched code"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if backup:
                backup_path = f"{file_path}.backup"
                shutil.copy2(file_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            # Clean up the original code (remove leading/trailing whitespace, normalize line endings)
            original_clean = original_code.strip()
            patched_clean = patched_code.strip()
            
            new_content: Optional[str] = None
            applied_loose = False
            
            # Exact substring match (fast path)
            if original_clean in content:
                new_content = content.replace(original_clean, patched_clean)
            else:
                span = self._find_loose_original_line_span(content, original_clean)
                if span is not None:
                    start, end = span
                    chunks = content.splitlines(keepends=True)
                    old_snippet = ''.join(chunks[start : end + 1])
                    first = chunks[start]
                    m = re.match(r'^([ \t]*)', first)
                    base_lead = m.group(1) if m else ''
                    new_block = self._reindent_patched_to_match_file(
                        patched_clean, base_lead, old_snippet, content
                    )
                    new_content = content.replace(old_snippet, new_block, 1)
                    applied_loose = True
            
            if new_content is not None:
                # Write the updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"File Name: {file_path}")
                print(f"Function: {self.current_function_name}")
                print("Original Code:")
                print("=" * 50)
                print(original_clean)
                print("=" * 50)
                print("\nPatched Code:")
                print("=" * 50)
                print(patched_clean)
                print("=" * 50)
                if applied_loose:
                    print("   [Code replacement applied successfully — matched ignoring leading indentation]")
                else:
                    print("   [Code replacement applied successfully]")
                print("-" * 80)
                return True
            
            print(f"Original code block not found in {file_path}")
            print("Original code block:")
            print("=" * 50)
            print(original_clean)
            print("=" * 50)
            print("-" * 80)
            return False
                
        except Exception as e:
            print(f"Failed to apply code replacement: {e}")
            print("-" * 80)
            return False
    
    def _show_replacement_preview(self, file_path: str, original_code: str, patched_code: str):
        """Show preview of what would be replaced"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_clean = original_code.strip()
            patched_clean = patched_code.strip()
            
            print(f"File Name: {file_path}")
            print(f"Function: {self.current_function_name}")
            print("Original Code:")
            print("=" * 50)
            print(original_clean)
            print("=" * 50)
            print("\nPatched Code:")
            print("=" * 50)
            print(patched_clean)
            print("=" * 50)
            
            if original_clean in content:
                print("   [DRY RUN - Code replacement would be applied]")
            elif self._find_loose_original_line_span(content, original_clean) is not None:
                print("   [DRY RUN - Code replacement would be applied (indentation-insensitive line match)]")
            else:
                print("   [DRY RUN - Original code block not found, replacement would fail]")
            print("-" * 80)
                
        except Exception as e:
            print(f"Failed to show replacement preview: {e}")
            print("-" * 80)
    
    def _show_focused_context(self, file_path: str, start_line: int, end_line: int, 
                             patched_code: str, dry_run: bool):
        """Show clean focused context around the patch location"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read file: {e}")
            return
        
        # Show clean format without timestamps
        print(f"File Name to be changed: {file_path}")
        print(f"In function: {self.current_function_name}")
        print(f"Line no. {start_line} to {end_line}")
        print("Code:")
        
        # Show original function code (focused area)
        context_start = max(0, start_line - 3)
        context_end = min(len(lines), start_line + 8)
        
        for i in range(context_start, context_end):
            line_num = i + 1
            print(f"  {line_num:3d}: {lines[i].rstrip()}")
        
        print()  # Add spacing before "Updated code" section
        
        print("\nUpdated code:")
        print(f"Line no {start_line + 1} (after line {start_line}):")
        print("Code:")
        
        # Clean up and format the patched code properly
        clean_patched_code = patched_code.strip()
        if clean_patched_code.startswith('//'):
            # Handle comment + code format
            lines_to_insert = clean_patched_code.split('\n')
            for i, line in enumerate(lines_to_insert):
                print(f"  {start_line + 1 + i:3d}: {line}")
        else:
            # Handle single line or multi-line code
            lines_to_insert = clean_patched_code.split('\n')
            for i, line in enumerate(lines_to_insert):
                print(f"  {start_line + 1 + i:3d}: {line}")
        
        # Show the next original line for context
        if start_line < len(lines):
            print(f"  {start_line + 1 + len(lines_to_insert):3d}: {lines[start_line].rstrip()}")
        
        if dry_run:
            print("   [DRY RUN - Would insert the above code]")
        else:
            print("   [Code will be inserted]")
        print("-" * 80)
    
    def _apply_code_insertion(self, file_path: str, line_number: int, 
                             patched_code: str, backup: bool) -> bool:
        """Apply code insertion at specified line"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if backup:
                backup_path = f"{file_path}.backup"
                shutil.copy2(file_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            # Insert the new code (simple insertion after the line)
            insert_line = f"    {patched_code}\n"  # Add proper indentation
            lines.insert(line_number, insert_line)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print("   [Code patch applied successfully]")
            print("-" * 80)
            return True
            
        except Exception as e:
            print(f"Failed to apply code patch: {e}")
            print("-" * 80)
            return False
    
    # ============================================================================
    # CONFIG PATCH METHODS (from targeted_config_patch_applicator.py)
    # ============================================================================
    
    def apply_config_patch(self, patch: Dict, dry_run: bool = False, backup: bool = True) -> bool:
        """Apply a configuration patch"""
        config_name = patch.get('config_name', '')
        file_path = patch.get('file_path', '')
        # Resolve file path relative to current working directory
        file_path = self._resolve_file_path(file_path)
        patch_type = patch.get('patch_type', 'set_value')
        current_value = patch.get('current_value', '')
        new_value = patch.get('new_value', '')
        description = patch.get('description', '')
        relevance_score = patch.get('relevance_score', 0)
        
        print(f"\nAPPLYING CONFIG PATCH")
        print(f"Description: {description}")
        print(f"Relevance Score: {relevance_score}")
        print()
        
        if not os.path.exists(file_path):
            print(f"Config file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            full_text = ''.join(lines)

            # CMake/structured block replacement mode:
            # use grounded snippet replacement instead of key=value parameter rewrite.
            if patch_type in ('targeted_insertion', 'targeted_replacement', 'targeted_insertion_or_adjustment'):
                if current_value and new_value and current_value in full_text:
                    print(f"File Name to be changed: {file_path}")
                    print(f"Patch Type: {patch_type}")
                    if config_name:
                        print(f"Config parameter/block: {config_name}")
                    print(f"Expected current value snippet found: yes")

                    updated_text = full_text.replace(current_value, new_value, 1)

                    if dry_run:
                        print("   [DRY RUN - Would replace targeted config snippet]")
                        print("-" * 80)
                        return True

                    if backup:
                        backup_path = f"{file_path}.backup"
                        shutil.copy2(file_path, backup_path)
                        print(f"Backup created: {backup_path}")

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_text)

                    print("   [Targeted config snippet replaced successfully]")
                    print("-" * 80)
                    return True
                elif patch_type in ('targeted_insertion', 'targeted_replacement', 'targeted_insertion_or_adjustment'):
                    print(f"Target snippet not found for patch_type={patch_type} in {file_path}")
                    print("-" * 80)
                    return False
            
            # Find the configuration parameter (legacy key/value config mode)
            found_line = None
            for i, line in enumerate(lines):
                if line.strip().startswith('#') or line.strip() == '':
                    continue
                
                if config_name in line and ('=' in line or ':' in line):
                    found_line = i
                    break
            
            if found_line is not None:
                old_line = lines[found_line].strip()
                
                # Create new line based on existing format
                if '({' in old_line and '})' in old_line:
                    # Handle complex format like amf_ip_address = ({ ipv4 = "192.168.70.132"; });
                    if not new_value.endswith('}'):
                        # Extract the complex format structure and replace just the IP
                        if 'ipv4' in old_line:
                            # Use regex to find and replace the IP address in the complex format
                            import re
                            # Extract just the IP address from new_value if it contains format info
                            if 'ipv4' in new_value:
                                # Extract IP from: { ipv4 = "192.168.70.132"
                                ip_match = re.search(r'"([0-9.]+)"', new_value)
                                if ip_match:
                                    new_ip = ip_match.group(1)
                                else:
                                    new_ip = new_value.strip('"')
                            else:
                                new_ip = new_value.strip('"')
                            
                            # Replace the IP in the existing format
                            new_line = re.sub(r'ipv4 = "[0-9.]+"', f'ipv4 = "{new_ip}"', old_line) + '\n'
                        else:
                            new_line = f"{config_name} = ({{ ipv4 = \"{new_value}\"; }});\n"
                    else:
                        new_line = f"{config_name} = {new_value};\n"
                elif '=' in old_line:
                    # Handle simple format like GNB_IPV4_ADDRESS_FOR_NG_AMF = "192.168.18.207";
                    # Special handling for timer values - they should not have quotes
                    if config_name.lower().endswith('timer') and new_value.isdigit():
                        new_line = f'{config_name} = {new_value};\n'
                    elif not new_value.startswith('"') and not new_value.startswith('{'):
                        new_line = f'{config_name} = "{new_value}";\n'
                    else:
                        new_line = f"{config_name} = {new_value};\n"
                else:
                    new_line = f"{config_name}: {new_value}\n"
                
                print(f"File Name to be changed: {file_path}")
                print(f"Config parameter: {config_name}")
                print(f"Line no. {found_line + 1}")
                print(f"Expected current value: {current_value}")
                print(f"New value to set: {new_value}")
                print("Code:")
                print(f"  {found_line + 1:3d}: {old_line}")
                
                print("\nUpdated code:")
                print(f"Line no {found_line + 1}:")
                print("Code:")
                print(f"  {found_line + 1:3d}: {new_line.strip()}")
                
                if dry_run:
                    print("   [DRY RUN - Would update configuration]")
                    print("-" * 80)
                    return True
                
                if backup:
                    backup_path = f"{file_path}.backup"
                    shutil.copy2(file_path, backup_path)
                    print(f"Backup created: {backup_path}")
                
                lines[found_line] = new_line
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print("   [Configuration updated successfully]")
                print("-" * 80)
                return True
            else:
                print(f"Parameter {config_name} not found in {file_path}")
                print("-" * 80)
                return False
                
        except Exception as e:
            print(f"Failed to apply config patch: {e}")
            print("-" * 80)
            return False
    
    # ============================================================================
    # UNIFIED APPLICATION METHODS
    # ============================================================================
    
    def apply_all_patches(self, dry_run: bool = False, backup: bool = True) -> Dict:
        """Apply all patches (both code and config) from fix suggestions"""
        print("UNIFIED PATCH APPLICATION")
        print("=" * 60)
        
        # Load suggestions
        suggestions = self.load_suggestions()
        if not suggestions:
            return {"success": False, "error": "Failed to load suggestions"}
        
        fix_suggestion = suggestions.get('fix_suggestion', {})
        code_patches = fix_suggestion.get('code_patches', [])
        config_patches = fix_suggestion.get('config_patches', [])
        
        print(f"Found {len(code_patches)} code patches and {len(config_patches)} config patches")
        
        if dry_run:
            print("DRY RUN MODE - No files will be modified")
        
        # Apply code patches
        if code_patches:
            print(f"\nCODE PATCHES")
            print("=" * 40)
            
            for i, patch in enumerate(code_patches, 1):
                print(f"\nCODE PATCH {i}/{len(code_patches)}")
                if self.apply_code_patch(patch, dry_run, backup):
                    self.applied_code_patches.append(patch)
                    print(f"Code patch {i} applied successfully\n")
                else:
                    self.failed_code_patches.append(patch)
                    print(f"Code patch {i} failed\n")
        
        # Apply config patches
        if config_patches:
            print(f"\nCONFIG PATCHES")
            print("=" * 40)
            
            for i, patch in enumerate(config_patches, 1):
                print(f"\nCONFIG PATCH {i}/{len(config_patches)}")
                if self.apply_config_patch(patch, dry_run, backup):
                    self.applied_config_patches.append(patch)
                    print(f"Config patch {i} applied successfully\n")
                else:
                    self.failed_config_patches.append(patch)
                    print(f"Config patch {i} failed\n")
        
        # Summary
        total_applied = len(self.applied_code_patches) + len(self.applied_config_patches)
        total_failed = len(self.failed_code_patches) + len(self.failed_config_patches)
        
        print("\n" + "=" * 60)
        print("PATCH APPLICATION SUMMARY")
        print("=" * 60)
        print(f"Code patches: {len(self.applied_code_patches)} applied, {len(self.failed_code_patches)} failed")
        print(f"Config patches: {len(self.applied_config_patches)} applied, {len(self.failed_config_patches)} failed")
        print(f"Total applied: {total_applied}")
        print(f"Total failed: {total_failed}")
        
        if self.applied_code_patches:
            print("\nSuccessfully applied code patches:")
            for patch in self.applied_code_patches:
                print(f"   - {patch['function_name']} in {patch['file_path']}")
        
        if self.applied_config_patches:
            print("\nSuccessfully applied config patches:")
            for patch in self.applied_config_patches:
                print(f"   - {patch['config_name']} in {patch['file_path']}")
        
        return {
            "success": total_failed == 0,
            "code_applied": len(self.applied_code_patches),
            "code_failed": len(self.failed_code_patches),
            "config_applied": len(self.applied_config_patches),
            "config_failed": len(self.failed_config_patches),
            "total_applied": total_applied,
            "total_failed": total_failed
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Apply unified code and configuration patches")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup files (default: True)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")
    parser.add_argument("--suggestions", default="output/crash_phase3_fixes.json", help="Path to fix suggestions file")
    
    args = parser.parse_args()
    
    # Handle backup flag
    backup = args.backup and not args.no_backup
    
    try:
        applicator = UnifiedPatchApplicator(args.suggestions)
        result = applicator.apply_all_patches(dry_run=args.dry_run, backup=backup)
        
        if result["success"]:
            print(f"\nAll patches applied successfully!")
            print(f"Applied: {result['total_applied']} patches")
            exit(0)
        else:
            print(f"\nSome patches failed. Check logs for details.")
            print(f"Applied: {result['total_applied']}, Failed: {result['total_failed']}")
            exit(1)
            
    except Exception as e:
        print(f"Unified patch application failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
