#!/usr/bin/env python3
"""
Improved Function Extraction Script

This script extracts function definitions from C and C++ source code files using
an improved approach that handles complex function signatures and multi-line definitions.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedFunctionExtractor:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.functions = []
        self.processed_files = 0
        self.skipped_files = 0
        self.errors = []
        
    def _get_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content, checking file size first."""
        try:
            file_size = file_path.stat().st_size
            if file_size > 5 * 1024 * 1024:  # 5 MB limit (increased from 1MB)
                logger.warning(f"Skipping {file_path}: file too large ({file_size / 1024 / 1024:.1f} MB)")
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def _preprocess_content(self, content: str) -> str:
        """Remove comments and preprocess content to make parsing easier."""
        # Remove single-line comments (but preserve line breaks)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove excessive whitespace but preserve line structure
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content

    def _get_line_number(self, content: str, start_pos: int) -> int:
        """Convert character position to line number."""
        return content[:start_pos].count('\n') + 1

    def _find_function_end(self, content: str, start_pos: int) -> int:
        """Find the end of a function by tracking braces."""
        brace_count = 0
        in_string = False
        in_char = False
        escape_next = False
        
        i = start_pos
        while i < len(content):
            char = content[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
                
            if char == '\\':
                escape_next = True
                i += 1
                continue
                
            if char == '"' and not in_char:
                in_string = not in_string
            elif char == "'" and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return i + 1
            
            i += 1
        
        return -1

    def _extract_function_signature(self, content: str, match_start: int) -> Tuple[str, str, str]:
        """Extract function signature, return type, and name."""
        # Look backwards to find the start of the function signature
        lines = content[:match_start].split('\n')
        signature_lines = []
        
        # Start from the line containing the opening brace and work backwards
        current_line_idx = len(lines) - 1
        
        while current_line_idx >= 0:
            line = lines[current_line_idx].strip()
            if not line:
                current_line_idx -= 1
                continue
                
            signature_lines.insert(0, line)
            
            # Check if this looks like the start of a function signature
            # Look for typical patterns like return_type function_name or modifiers
            combined = ' '.join(signature_lines)
            if re.search(r'\b(static|inline|extern|const|volatile|typedef|struct|enum|union)\b', combined) and '(' in combined:
                break
            if re.match(r'^[a-zA-Z_].*\(', combined):
                break
            if current_line_idx == 0:
                break
                
            current_line_idx -= 1
            
            # Safety limit - don't go back more than 10 lines for a signature
            if len(signature_lines) > 10:
                break
        
        full_signature = ' '.join(signature_lines)
        full_signature = re.sub(r'\s+', ' ', full_signature).strip()
        
        # Extract function name and return type
        # Pattern to match: [modifiers] return_type function_name(parameters)
        pattern = r'^\s*((?:static|inline|extern|const|volatile|\w+\s+)*?)\s*(\w+(?:\s*\*+)?)\s+(\w+)\s*\('
        match = re.search(pattern, full_signature)
        
        if match:
            modifiers = match.group(1).strip()
            return_type = match.group(2).strip()
            function_name = match.group(3).strip()
            
            # Combine modifiers with return type if present
            if modifiers:
                return_type = f"{modifiers} {return_type}".strip()
                
            return function_name, return_type, full_signature
        
        # Fallback pattern for complex cases
        # Try to find function_name( pattern
        pattern2 = r'(\w+)\s*\('
        match2 = re.search(pattern2, full_signature)
        if match2:
            function_name = match2.group(1)
            # Try to extract return type (everything before function name)
            return_type_match = re.search(r'^(.*?)\s+' + re.escape(function_name) + r'\s*\(', full_signature)
            return_type = return_type_match.group(1).strip() if return_type_match else "unknown"
            return function_name, return_type, full_signature
        
        return "unknown", "unknown", full_signature

    def _extract_functions_from_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract functions using improved parsing approach."""
        functions = []
        original_content = content
        content = self._preprocess_content(content)
        
        # Find all opening braces that are likely function starts
        # Look for patterns like:
        # function_name(...) {
        # ) {
        brace_pattern = r'[)\w]\s*\n?\s*\{'
        
        for match in re.finditer(brace_pattern, content):
            try:
                brace_pos = match.end() - 1  # Position of the opening brace
                
                # Find the end of the function
                function_end = self._find_function_end(content, brace_pos)
                if function_end == -1:
                    continue
                
                # Extract function signature
                function_name, return_type, signature = self._extract_function_signature(content, brace_pos)
                
                # Skip if we couldn't extract a valid function name
                if function_name == "unknown" or not function_name.replace('_', '').isalnum():
                    continue
                    
                # Skip common non-function patterns
                if function_name in ['if', 'for', 'while', 'switch', 'do', 'else', 'case', 'default']:
                    continue
                
                # Extract the full function text
                # Find the actual start of the function in original content
                signature_in_original = signature.replace(' ', r'\s+')
                signature_pattern = re.escape(signature_in_original).replace(r'\\ ', r'\s*').replace(r'\\s\+', r'\s+')
                
                sig_match = re.search(signature_pattern, original_content, re.MULTILINE)
                if sig_match:
                    func_start_pos = sig_match.start()
                    # Find the end position in original content
                    brace_pos_orig = original_content.find('{', func_start_pos)
                    if brace_pos_orig != -1:
                        func_end_pos = self._find_function_end(original_content, brace_pos_orig)
                        if func_end_pos != -1:
                            function_text = original_content[func_start_pos:func_end_pos]
                            start_line = self._get_line_number(original_content, func_start_pos)
                            end_line = self._get_line_number(original_content, func_end_pos)
                        else:
                            continue
                    else:
                        continue
                else:
                    # Fallback - use processed content positions
                    func_start_pos = max(0, match.start() - 200)  # Go back a bit to capture signature
                    function_text = content[func_start_pos:function_end]
                    start_line = self._get_line_number(content, func_start_pos)
                    end_line = self._get_line_number(content, function_end)
                
                # Clean up function text
                function_text = re.sub(r'\n\s*\n', '\n', function_text.strip())
                
                function_info = {
                    "function_name": function_name,
                    "file_path": str(file_path),
                    "start_line": start_line,
                    "end_line": end_line,
                    "code_body": function_text,
                    "return_type": return_type,
                    "signature": signature
                }
                
                # Check for duplicates
                duplicate = False
                for existing in functions:
                    if (existing["function_name"] == function_name and 
                        existing["file_path"] == str(file_path) and
                        abs(existing["start_line"] - start_line) < 3):
                        duplicate = True
                        break
                
                if not duplicate:
                    functions.append(function_info)
                    logger.debug(f"Found function: {function_name} at lines {start_line}-{end_line}")
                    
            except Exception as e:
                logger.warning(f"Error processing function at position {match.start()} in {file_path}: {e}")
                continue
        
        return functions

    def _parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single file and extract functions."""
        content = self._get_file_content(file_path)
        if content is None:
            return []
        
        try:
            functions = self._extract_functions_from_content(content, file_path)
            return functions
            
        except Exception as e:
            error_msg = f"Error parsing {file_path}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return []

    def scan_repository(self):
        """Scan the repository recursively for C and C++ files."""
        logger.info(f"Starting scan of repository: {self.repo_path}")
        
        # Supported file extensions
        supported_extensions = {'.c', '.cpp', '.cc', '.cxx', '.h'}
        
        # Skip certain directories that are unlikely to contain relevant functions
        skip_dirs = {'.git', '__pycache__', 'build', 'dist', '.vscode', '.idea'}
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Skip files in skip directories
                if any(skip_dir in str(file_path) for skip_dir in skip_dirs):
                    continue
                    
                logger.info(f"Processing: {file_path}")
                
                functions = self._parse_file(file_path)
                if functions:
                    self.functions.extend(functions)
                    logger.info(f"  Found {len(functions)} functions")
                
                self.processed_files += 1
        
        logger.info(f"Repository scan completed. Processed {self.processed_files} files.")

    def save_functions(self, output_file: str = 'database/functions.json'):
        """Save extracted functions to JSON file."""
        try:
            # Ensure directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.functions, f, indent=2, ensure_ascii=False)
            logger.info(f"Functions saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving functions: {e}")
            raise

    def print_summary(self):
        """Print a summary of the extraction process."""
        print(f"\n{'='*60}")
        print("IMPROVED FUNCTION EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Extracted {len(self.functions)} functions from {self.processed_files} files")
        print(f"📁 Repository: {self.repo_path}")
        print(f"📄 Output file: database/functions.json")
        
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} errors encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more errors")
        
        # Show some example functions found
        if self.functions:
            print(f"\n📝 Example functions found:")
            for i, func in enumerate(self.functions[:5]):
                print(f"   {i+1}. {func['function_name']} in {func['file_path']}")
        
        print(f"{'='*60}")


def main():
    """Main function to run the function extraction."""
    import sys
    
    # Get repository path from command line or use current directory
    repo_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    try:
        # Create extractor and scan repository
        extractor = ImprovedFunctionExtractor(repo_path)
        extractor.scan_repository()
        
        # Save results
        extractor.save_functions()
        
        # Print summary
        extractor.print_summary()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
