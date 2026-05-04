#!/usr/bin/env python3
"""
Enhanced Function Call Graph Generator

This script analyzes the codebase to extract function call relationships 
specifically for high-level telecommunications functions, creating a 
comprehensive call graph that will power the enhanced symbolic retrieval.

Features:
1. Focuses on protocol functions (NGAP, RRC, NAS, SCTP, etc.)
2. Extracts actual function calls from code bodies
3. Builds bidirectional call relationships
4. Filters out noise (system calls, logging)
5. Creates rich metadata for call chain analysis
"""

import os
import json
import re
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFunctionCallAnalyzer:
    def __init__(self):
        self.functions_data = {}  # function_name -> function_info
        self.call_relationships = defaultdict(set)  # caller -> set(callees)
        self.reverse_calls = defaultdict(set)  # callee -> set(callers)
        
        # High-value telecommunications function patterns
        self.priority_patterns = [
            # NGAP (Next Generation Application Protocol)
            r'ngap_\w+',
            # RRC (Radio Resource Control)
            r'rrc_\w+',
            # NAS (Non-Access Stratum)
            r'nas_\w+',
            # AMF (Access and Mobility Management Function)
            r'\w*amf\w*',
            # gNB (Next Generation NodeB)
            r'\w*gnb\w*', r'\w*gNB\w*',
            # SCTP (Stream Control Transmission Protocol)
            r'sctp_\w+',
            # Protocol handlers
            r'\w*_handler\w*', r'\w*_handle_\w*',
            # Setup and configuration
            r'\w*_setup\w*', r'\w*_config\w*',
            # Registration and association
            r'\w*_register\w*', r'\w*_associate\w*',
            # Connection and establishment
            r'\w*_connect\w*', r'\w*_establish\w*',
            # Selection functions
            r'\w*_select\w*', r'\w*_choose\w*',
            # Error handling
            r'\w*_error\w*', r'\w*_fail\w*',
        ]
        
        # Noise patterns to exclude
        self.noise_patterns = [
            r'LOG_[DIWEA]', r'printf', r'sprintf', r'fprintf',
            r'malloc', r'free', r'calloc', r'realloc',
            r'memcpy', r'memset', r'strlen', r'strcpy',
            r'getenv', r'setenv', r'exit', r'abort',
            r'assert\w*', r'AssertFatal',
            r'pthread_\w+', r'mutex_\w+',
            r'sleep', r'usleep', r'nanosleep',
        ]

    def load_functions_data(self):
        """Load function data from functions.json"""
        logger.info("📊 Loading functions data...")
        
        try:
            with open('database/functions.json', 'r', encoding='utf-8') as f:
                functions_list = json.load(f)
            
            # Convert list to dictionary for faster lookup
            for func in functions_list:
                func_name = func.get('function_name', '')
                if func_name:
                    self.functions_data[func_name] = func
            
            logger.info(f"✅ Loaded {len(self.functions_data)} functions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load functions data: {e}")
            return False
        
        return True

    def is_priority_function(self, func_name: str) -> bool:
        """Check if function matches high-value patterns"""
        func_lower = func_name.lower()
        
        # Check priority patterns
        for pattern in self.priority_patterns:
            if re.search(pattern, func_lower):
                return True
        
        return False

    def is_noise_function(self, func_name: str) -> bool:
        """Check if function is noise (system calls, logging, etc.)"""
        for pattern in self.noise_patterns:
            if re.search(pattern, func_name, re.IGNORECASE):
                return True
        
        return False

    def extract_function_calls_from_code(self, code_body: str) -> Set[str]:
        """Extract function calls from code body using multiple patterns"""
        calls = set()
        
        if not code_body:
            return calls
        
        # Pattern 1: Standard function calls func_name(...)
        pattern1 = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches1 = re.findall(pattern1, code_body)
        calls.update(matches1)
        
        # Pattern 2: Function pointer calls (*func_ptr)(...)
        pattern2 = r'\(\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*\('
        matches2 = re.findall(pattern2, code_body)
        calls.update(matches2)
        
        # Pattern 3: Struct member function calls struct.func(...)
        pattern3 = r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches3 = re.findall(pattern3, code_body)
        calls.update(matches3)
        
        # Pattern 4: Arrow operator calls struct->func(...)
        pattern4 = r'->([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches4 = re.findall(pattern4, code_body)
        calls.update(matches4)
        
        # Filter out noise and keep only valid function names
        valid_calls = set()
        for call in calls:
            if (len(call) >= 3 and  # Minimum length
                not call.isdigit() and  # Not a number
                not self.is_noise_function(call) and  # Not noise
                call in self.functions_data):  # Exists in our function database
                valid_calls.add(call)
        
        return valid_calls

    def analyze_call_relationships(self):
        """Analyze all functions and build call relationship graph"""
        logger.info("🔍 Analyzing function call relationships...")
        
        priority_functions = []
        analyzed_count = 0
        
        # First pass: identify priority functions
        for func_name, func_data in self.functions_data.items():
            if self.is_priority_function(func_name):
                priority_functions.append((func_name, func_data))
        
        logger.info(f"🎯 Found {len(priority_functions)} priority functions to analyze")
        
        # Second pass: analyze call relationships for priority functions
        for func_name, func_data in priority_functions:
            code_body = func_data.get('code_body', '')
            if code_body:
                # Extract function calls from this function's code
                called_functions = self.extract_function_calls_from_code(code_body)
                
                # Store the relationships
                self.call_relationships[func_name].update(called_functions)
                
                # Build reverse mapping (who calls this function)
                for called_func in called_functions:
                    self.reverse_calls[called_func].add(func_name)
                
                analyzed_count += 1
                if analyzed_count % 100 == 0:
                    logger.info(f"  📊 Analyzed {analyzed_count}/{len(priority_functions)} functions...")
        
        logger.info(f"✅ Analysis complete! {analyzed_count} functions analyzed")
        logger.info(f"📊 Found {len(self.call_relationships)} functions with outgoing calls")
        logger.info(f"📊 Found {len(self.reverse_calls)} functions with incoming calls")

    def build_enhanced_function_calls_json(self) -> List[Dict]:
        """Build the enhanced function_calls.json structure"""
        logger.info("🏗️ Building enhanced function_calls.json...")
        
        enhanced_calls = []
        
        for func_name in sorted(self.call_relationships.keys()):
            func_data = self.functions_data.get(func_name, {})
            called_functions = list(self.call_relationships[func_name])
            calling_functions = list(self.reverse_calls.get(func_name, set()))
            
            # Calculate priority score
            priority_score = self.calculate_priority_score(func_name, called_functions, calling_functions)
            
            entry = {
                "function": func_name,
                "file": func_data.get('file_path', ''),
                "start_line": func_data.get('start_line', 0),
                "end_line": func_data.get('end_line', 0),
                "calls": sorted(called_functions),  # Functions this function calls
                "called_by": sorted(calling_functions),  # Functions that call this function
                "priority_score": priority_score,
                "call_count": len(called_functions),
                "caller_count": len(calling_functions),
                "categories": self.categorize_function(func_name)
            }
            
            enhanced_calls.append(entry)
        
        # Sort by priority score (highest first)
        enhanced_calls.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"✅ Built enhanced function calls for {len(enhanced_calls)} functions")
        return enhanced_calls

    def calculate_priority_score(self, func_name: str, called_functions: List[str], calling_functions: List[str]) -> float:
        """Calculate priority score based on function characteristics"""
        score = 0.0
        
        # Base score for telecommunications functions
        func_lower = func_name.lower()
        if any(keyword in func_lower for keyword in ['amf', 'gnb', 'ngap', 'rrc', 'nas']):
            score += 10.0
        
        # Score for error/handler functions
        if any(keyword in func_lower for keyword in ['error', 'handle', 'fail', 'setup', 'register']):
            score += 5.0
        
        # Score based on connectivity (more connected = higher priority)
        connectivity_score = len(called_functions) + len(calling_functions)
        score += min(connectivity_score * 0.5, 10.0)  # Cap at 10
        
        # Bonus for functions that are called by many others (central functions)
        if len(calling_functions) > 5:
            score += 3.0
        
        # Bonus for functions that call many others (orchestrator functions)
        if len(called_functions) > 10:
            score += 2.0
        
        return round(score, 2)

    def categorize_function(self, func_name: str) -> List[str]:
        """Categorize function based on name patterns"""
        categories = []
        func_lower = func_name.lower()
        
        if 'amf' in func_lower:
            categories.append('amf')
        if any(gnb in func_lower for gnb in ['gnb', 'gnodeb']):
            categories.append('gnb')
        if 'ngap' in func_lower:
            categories.append('ngap')
        if 'rrc' in func_lower:
            categories.append('rrc')
        if 'nas' in func_lower:
            categories.append('nas')
        if 'sctp' in func_lower:
            categories.append('sctp')
        if any(keyword in func_lower for keyword in ['handle', 'handler']):
            categories.append('handler')
        if any(keyword in func_lower for keyword in ['setup', 'init', 'initialize']):
            categories.append('setup')
        if any(keyword in func_lower for keyword in ['register', 'associate']):
            categories.append('registration')
        if any(keyword in func_lower for keyword in ['connect', 'establish']):
            categories.append('connection')
        if any(keyword in func_lower for keyword in ['select', 'choose']):
            categories.append('selection')
        if any(keyword in func_lower for keyword in ['error', 'fail', 'failure']):
            categories.append('error_handling')
        
        return categories if categories else ['general']

    def save_enhanced_function_calls(self, enhanced_calls: List[Dict]):
        """Save the enhanced function calls to JSON file"""
        logger.info("💾 Saving enhanced function_calls.json...")
        
        try:
            # Create backup of existing file
            if os.path.exists('database/function_calls.json'):
                os.rename('database/function_calls.json', 'database/function_calls_backup.json')
                logger.info("📋 Backed up existing function_calls.json")
            
            # Save enhanced version
            with open('database/function_calls_enhanced.json', 'w', encoding='utf-8') as f:
                json.dump(enhanced_calls, f, indent=2, ensure_ascii=False)
            
            # Also save as the main function_calls.json
            with open('database/function_calls.json', 'w', encoding='utf-8') as f:
                json.dump(enhanced_calls, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved enhanced function_calls.json with {len(enhanced_calls)} entries")
            
            # Generate summary report
            self.generate_summary_report(enhanced_calls)
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced function calls: {e}")

    def generate_summary_report(self, enhanced_calls: List[Dict]):
        """Generate a summary report of the enhanced function calls"""
        logger.info("📊 Generating summary report...")
        
        # Statistics
        total_functions = len(enhanced_calls)
        total_calls = sum(len(entry['calls']) for entry in enhanced_calls)
        avg_calls_per_function = total_calls / total_functions if total_functions > 0 else 0
        
        # Category distribution
        category_counts = defaultdict(int)
        for entry in enhanced_calls:
            for category in entry['categories']:
                category_counts[category] += 1
        
        # Top priority functions
        top_priority = enhanced_calls[:10]  # Already sorted by priority
        
        report = f"""
# Enhanced Function Calls Analysis Report

## Summary Statistics
- **Total Functions Analyzed**: {total_functions:,}
- **Total Function Calls**: {total_calls:,}
- **Average Calls per Function**: {avg_calls_per_function:.2f}

## Category Distribution
"""
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_functions) * 100
            report += f"- **{category.title()}**: {count:,} functions ({percentage:.1f}%)\n"
        
        report += "\n## Top Priority Functions\n"
        for i, entry in enumerate(top_priority, 1):
            report += f"{i}. **{entry['function']}** (Score: {entry['priority_score']})\n"
            report += f"   - File: `{entry['file']}`\n"
            report += f"   - Categories: {', '.join(entry['categories'])}\n"
            report += f"   - Calls {entry['call_count']} functions, Called by {entry['caller_count']} functions\n\n"
        
        # Save report
        with open('database/function_calls_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ Summary report saved to database/function_calls_report.md")

def main():
    """Main execution function"""
    logger.info("🚀 Starting Enhanced Function Call Graph Generation")
    logger.info("=" * 60)
    
    analyzer = EnhancedFunctionCallAnalyzer()
    
    # Step 1: Load functions data
    if not analyzer.load_functions_data():
        logger.error("❌ Failed to load functions data. Exiting.")
        return
    
    # Step 2: Analyze call relationships
    analyzer.analyze_call_relationships()
    
    # Step 3: Build enhanced function calls JSON
    enhanced_calls = analyzer.build_enhanced_function_calls_json()
    
    # Step 4: Save results
    analyzer.save_enhanced_function_calls(enhanced_calls)
    
    logger.info("=" * 60)
    logger.info("🎉 Enhanced Function Call Graph Generation Complete!")
    logger.info("📂 Output files:")
    logger.info("   - database/function_calls.json (main file)")
    logger.info("   - database/function_calls_enhanced.json (backup)")
    logger.info("   - database/function_calls_report.md (analysis report)")
    logger.info("   - database/function_calls_backup.json (original backup)")

if __name__ == "__main__":
    main()
