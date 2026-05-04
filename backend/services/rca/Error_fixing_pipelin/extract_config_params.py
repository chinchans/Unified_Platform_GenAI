#!/usr/bin/env python3
"""
Configuration Parameter Extraction Script

This script extracts configuration parameters from .conf and .json files
throughout a repository and saves them to config.json.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigParameterExtractor:
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
        self.config_params = []
        self.processed_files = 0
        self.skipped_files = 0
        self.errors = []
        
        # Regex patterns for .conf file parsing
        self.conf_patterns = [
            # key=value format
            re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_\-\.]*)\s*=\s*(.*)$'),
            # key : value format
            re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_\-\.]*)\s*:\s*(.*)$')
        ]
        
        # Comment patterns
        self.comment_patterns = [
            re.compile(r'^\s*#'),  # Lines starting with #
            re.compile(r'^\s*;'),  # Lines starting with ;
            re.compile(r'^\s*$')   # Blank lines
        ]

    def _is_comment_or_blank(self, line: str) -> bool:
        """Check if a line is a comment or blank."""
        return any(pattern.match(line) for pattern in self.comment_patterns)

    def _clean_value(self, value: str) -> str:
        """Clean configuration value by removing quotes and inline comments."""
        # Remove inline comments
        value = re.sub(r'\s*[#;].*$', '', value)
        
        # Remove surrounding quotes
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        return value.strip()

    def _get_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with size check and error handling."""
        try:
            file_size = file_path.stat().st_size
            if file_size > 1024 * 1024:  # 1 MB limit
                logger.warning(f"Skipping {file_path}: file too large ({file_size / 1024 / 1024:.1f} MB)")
                self.skipped_files += 1
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            error_msg = f"Error reading {file_path}: {e}"
            logger.warning(error_msg)
            self.errors.append(error_msg)
            self.skipped_files += 1
            return None

    def _parse_conf_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse .conf file and extract key-value pairs."""
        content = self._get_file_content(file_path)
        if content is None:
            return []
        
        params = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if self._is_comment_or_blank(line):
                continue
            
            # Try to match key=value or key:value patterns
            for pattern in self.conf_patterns:
                match = pattern.match(line)
                if match:
                    param_name = match.group(1).strip()
                    param_value = self._clean_value(match.group(2))
                    
                    param_entry = {
                        "param_name": param_name,
                        "param_value": param_value,
                        "file_path": str(file_path.relative_to(self.repo_path)),
                        "line_number": line_num
                    }
                    params.append(param_entry)
                    break
        
        return params

    def _flatten_json_object(self, obj: Any, prefix: str = "", file_path: str = "") -> List[Dict[str, Any]]:
        """Recursively flatten JSON object into key-value pairs."""
        params = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                params.extend(self._flatten_json_object(value, new_key, file_path))
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{prefix}[{i}]"
                params.extend(self._flatten_json_object(item, new_key, file_path))
        
        else:
            # Leaf value
            param_entry = {
                "param_name": prefix,
                "param_value": str(obj),
                "file_path": file_path,
                "line_number": None  # JSON doesn't provide line numbers easily
            }
            params.append(param_entry)
        
        return params

    def _parse_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse .json file and extract all key-value pairs."""
        content = self._get_file_content(file_path)
        if content is None:
            return []
        
        try:
            json_data = json.loads(content)
            relative_path = str(file_path.relative_to(self.repo_path))
            params = self._flatten_json_object(json_data, "", relative_path)
            return params
        
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {file_path}: {e}"
            logger.warning(error_msg)
            self.errors.append(error_msg)
            return []

    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single configuration file."""
        if file_path.suffix.lower() == '.conf':
            return self._parse_conf_file(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._parse_json_file(file_path)
        else:
            return []

    def scan_repository(self):
        """Scan repository recursively for configuration files."""
        logger.info(f"Starting configuration scan of repository: {self.repo_path}")
        
        # Supported file extensions
        supported_extensions = {'.conf', '.json'}
        
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'build', 'dist', 'database'}
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Skip files in skip directories
                if any(skip_dir in str(file_path) for skip_dir in skip_dirs):
                    continue
                
                # Skip certain JSON files that are not configuration files
                if file_path.suffix.lower() == '.json':
                    skip_json_files = {
                        'error_patterns_enhanced.json',
                        'functions.json', 
                        'function_calls.json',
                        'function_calls_enhanced.json',
                        'functions_mapping.json',
                        'config_mapping.json',
                        'package.json',
                        'tsconfig.json',
                        'webpack.config.json'
                    }
                    if file_path.name in skip_json_files:
                        logger.info(f"Skipping non-config JSON file: {file_path}")
                        continue
                
                logger.info(f"Processing: {file_path}")
                
                params = self._process_file(file_path)
                if params:
                    self.config_params.extend(params)
                    logger.info(f"  Found {len(params)} config parameters")
                
                self.processed_files += 1
        
        logger.info(f"Repository scan completed. Processed {self.processed_files} files, skipped {self.skipped_files} files.")

    def save_config_params(self, output_file: str = 'config.json'):
        """Save extracted configuration parameters to JSON file."""
        try:
            # Ensure directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_params, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration parameters saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving configuration parameters: {e}")
            raise

    def print_summary(self):
        """Print a summary of the extraction process."""
        print(f"\n{'='*60}")
        print("CONFIGURATION PARAMETER EXTRACTION SUMMARY")
        print(f"{'='*60}")
        
        total_params = len(self.config_params)
        total_files = self.processed_files
        
        if total_params > 0:
            print(f"✅ Extracted {total_params} config parameters from {total_files} files. Saved to config.json")
        else:
            print("❌ No configuration parameters found")
        
        print(f"📁 Repository: {self.repo_path}")
        print(f"📄 Processed files: {self.processed_files}")
        print(f"⏭️  Skipped files: {self.skipped_files}")
        
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} errors encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more errors")
        
        # Show some example parameters
        if self.config_params:
            print(f"\n📝 Example config parameters found:")
            for i, param in enumerate(self.config_params[:5]):
                print(f"   {i+1}. {param['param_name']} = {param['param_value'][:50]}{'...' if len(param['param_value']) > 50 else ''}")
                print(f"      📁 {param['file_path']}:{param['line_number'] or 'N/A'}")
        
        print(f"{'='*60}")

    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        conf_params = sum(1 for p in self.config_params if p['file_path'].endswith('.conf'))
        json_params = sum(1 for p in self.config_params if p['file_path'].endswith('.json'))
        
        return {
            'total_params': len(self.config_params),
            'conf_params': conf_params,
            'json_params': json_params,
            'processed_files': self.processed_files,
            'skipped_files': self.skipped_files,
            'errors': len(self.errors)
        }


def main():
    """Main function to run the configuration parameter extraction."""
    import sys
    
    # Get repository path from command line or use current directory
    repo_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    try:
        # Create extractor and scan repository
        extractor = ConfigParameterExtractor(repo_path)
        extractor.scan_repository()
        
        # Save results
        extractor.save_config_params()
        
        # Print summary
        extractor.print_summary()
        
        # Print statistics
        stats = extractor.get_statistics()
        logger.info(f"Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
