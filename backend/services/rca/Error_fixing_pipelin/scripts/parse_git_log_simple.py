#!/usr/bin/env python3
"""
Simple Git Log Parser - Converts dev_log.txt to structured JSON
Author: AgenticRAN
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any

def parse_git_log(input_file: str, output_file: str):
    """
    Parse git log text file and convert to JSON format
    """
    
    print(f"Reading git log from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split content by commit entries
    # Each commit starts with "commit" followed by a 40-character hash
    commit_blocks = re.split(r'(?=^commit [a-f0-9]{40})', content, flags=re.MULTILINE)
    
    commits = []
    
    for block in commit_blocks:
        if not block.strip() or not block.startswith('commit'):
            continue
        
        commit_data = parse_commit_block(block)
        if commit_data:
            commits.append(commit_data)
    
    # Create final JSON structure
    output_data = {
        "metadata": {
            "total_commits": len(commits),
            "generated_at": datetime.now().isoformat(),
            "source_file": input_file,
            "description": "Git commit history for OpenAirInterface5G repository"
        },
        "commits": commits
    }
    
    # Save to JSON file
    print(f"Writing JSON output to: {output_file}")
    
    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully parsed {len(commits)} commits")
    return output_data

def parse_commit_block(block: str) -> Dict[str, Any]:
    """
    Parse a single commit block and extract all information
    """
    
    lines = block.split('\n')
    
    commit_data = {
        "commit_hash": "",
        "commit_hash_short": "",
        "merge": None,
        "author_name": "",
        "author_email": "",
        "date": "",
        "date_iso": "",
        "subject": "",
        "body": "",
        "message_full": "",
        "rca_patches": {
            "is_rca_commit": False,
            "code_patches": [],
            "config_patches": [],
            "code_patch_count": 0,
            "config_patch_count": 0
        },
        "files_changed": [],
        "keywords": []
    }
    
    # Extract commit hash
    commit_match = re.match(r'^commit ([a-f0-9]{40})$', lines[0])
    if commit_match:
        full_hash = commit_match.group(1)
        commit_data["commit_hash"] = full_hash
        commit_data["commit_hash_short"] = full_hash[:10]
    
    # Extract merge information (if present)
    merge_match = re.match(r'^Merge:\s+([a-f0-9]+)\s+([a-f0-9]+)', lines[1]) if len(lines) > 1 else None
    if merge_match:
        commit_data["merge"] = {
            "parent1": merge_match.group(1),
            "parent2": merge_match.group(2)
        }
    
    # Extract author information
    for line in lines:
        author_match = re.match(r'^Author:\s+(.+?)\s+<(.+?)>$', line)
        if author_match:
            commit_data["author_name"] = author_match.group(1).strip()
            commit_data["author_email"] = author_match.group(2).strip()
            break
    
    # Extract date
    for line in lines:
        date_match = re.match(r'^Date:\s+(.+)$', line)
        if date_match:
            date_str = date_match.group(1).strip()
            commit_data["date"] = date_str
            # Try to convert to ISO format
            try:
                # Parse various git date formats
                for fmt in ['%a %b %d %H:%M:%S %Y %z', '%a %b %d %H:%M:%S %Y %Z']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        commit_data["date_iso"] = dt.isoformat()
                        break
                    except:
                        continue
            except:
                commit_data["date_iso"] = date_str
            break
    
    # Extract commit message (subject and body)
    message_lines = []
    in_message = False
    
    for i, line in enumerate(lines):
        # Skip header lines
        if line.startswith('commit') or line.startswith('Merge:') or \
           line.startswith('Author:') or line.startswith('Date:'):
            continue
        
        # Message lines are indented with spaces
        if line.startswith('    '):
            message_lines.append(line[4:])  # Remove 4 spaces indentation
            in_message = True
        elif in_message and line.strip() == '':
            message_lines.append('')
    
    # Parse message components
    if message_lines:
        # First non-empty line is the subject
        subject = ''
        body_lines = []
        found_subject = False
        
        for line in message_lines:
            if not found_subject and line.strip():
                subject = line.strip()
                found_subject = True
            elif found_subject:
                body_lines.append(line)
        
        commit_data["subject"] = subject
        commit_data["body"] = '\n'.join(body_lines).strip()
        commit_data["message_full"] = '\n'.join(message_lines).strip()
    
    # Check if this is an RCA commit and parse patches
    if "Generated by AgenticRAN RCA Analysis" in commit_data["message_full"]:
        commit_data["rca_patches"]["is_rca_commit"] = True
        parse_rca_patches(commit_data, message_lines)
    
    # Extract keywords for searching
    commit_data["keywords"] = extract_keywords(commit_data["subject"], commit_data["body"])
    
    return commit_data

def parse_rca_patches(commit_data: Dict, message_lines: List[str]) -> None:
    """
    Parse RCA patch information from commit message
    """
    
    in_code_patches = False
    in_config_patches = False
    
    for line in message_lines:
        line_stripped = line.strip()
        
        # Parse patch counts
        code_count_match = re.match(r'- Applied (\d+) code patches?', line_stripped)
        if code_count_match:
            commit_data["rca_patches"]["code_patch_count"] = int(code_count_match.group(1))
            continue
        
        config_count_match = re.match(r'- Applied (\d+) config patches?', line_stripped)
        if config_count_match:
            commit_data["rca_patches"]["config_patch_count"] = int(config_count_match.group(1))
            continue
        
        # Section markers
        if line_stripped == "Code patches:":
            in_code_patches = True
            in_config_patches = False
            continue
        elif line_stripped == "Config patches:":
            in_config_patches = True
            in_code_patches = False
            continue
        elif line_stripped == "Generated by AgenticRAN RCA Analysis":
            break
        
        # Parse patch entries
        if in_code_patches and line_stripped.startswith('- '):
            patch_info = line_stripped[2:].strip()
            # Format: "function_name (file.c)"
            patch_match = re.match(r'(.+?)\s+\((.+?)\)$', patch_info)
            if patch_match:
                commit_data["rca_patches"]["code_patches"].append({
                    "function": patch_match.group(1).strip(),
                    "file": patch_match.group(2).strip()
                })
                commit_data["files_changed"].append(patch_match.group(2).strip())
            else:
                commit_data["rca_patches"]["code_patches"].append({
                    "function": patch_info,
                    "file": "unknown"
                })
        
        elif in_config_patches and line_stripped.startswith('- '):
            patch_info = line_stripped[2:].strip()
            # Format: "parameter_name (config.conf)"
            patch_match = re.match(r'(.+?)\s+\((.+?)\)$', patch_info)
            if patch_match:
                commit_data["rca_patches"]["config_patches"].append({
                    "parameter": patch_match.group(1).strip(),
                    "file": patch_match.group(2).strip()
                })
                commit_data["files_changed"].append(patch_match.group(2).strip())
            else:
                commit_data["rca_patches"]["config_patches"].append({
                    "parameter": patch_info,
                    "file": "unknown"
                })

def extract_keywords(subject: str, body: str) -> List[str]:
    """
    Extract important keywords from commit message for error matching
    """
    
    text = (subject + " " + body).lower()
    
    # Define important keywords for 5G RAN
    keyword_patterns = [
        r'\bamf\b', r'\brrc\b', r'\bngap\b', r'\bf1ap\b', r'\be1ap\b',
        r'\bpdcp\b', r'\brlc\b', r'\bmac\b', r'\bphy\b',
        r'\btimeout\b', r'\bfail', r'\berror\b', r'\bfix\b', r'\bissue\b',
        r'\bcrash\b', r'\bexception\b', r'\bwarning\b',
        r'\bhandover\b', r'\bsetup\b', r'\brelease\b', r'\bcontext\b',
        r'\bregistration\b', r'\bassociation\b', r'\bconnection\b',
        r'\battach\b', r'\bdetach\b', r'\bpdu\b', r'\bsession\b',
        r'\bconfiguration\b', r'\bconfig\b', r'\bparameter\b'
    ]
    
    keywords = []
    for pattern in keyword_patterns:
        if re.search(pattern, text):
            # Extract the keyword without regex markers
            keyword = pattern.replace(r'\b', '').replace('\\', '')
            keywords.append(keyword)
    
    return list(set(keywords))

def main():
    """
    Main function
    """
    print("=" * 60)
    print("Git Log to JSON Converter")
    print("=" * 60)
    
    input_file = "openairinterface5g-develop/dev_log.txt"
    output_file = "resources/git_log_commits.json"
    
    try:
        result = parse_git_log(input_file, output_file)
        
        print("\n" + "=" * 60)
        print("📊 Summary:")
        print("=" * 60)
        print(f"Total commits parsed: {result['metadata']['total_commits']}")
        
        # Count RCA commits
        rca_commits = [c for c in result['commits'] if c['rca_patches']['is_rca_commit']]
        print(f"RCA commits: {len(rca_commits)}")
        print(f"Regular commits: {len(result['commits']) - len(rca_commits)}")
        
        if rca_commits:
            print("\n🔧 RCA Commits Details:")
            for commit in rca_commits:
                print(f"  - {commit['commit_hash_short']}: {commit['subject']}")
                print(f"    Code patches: {commit['rca_patches']['code_patch_count']}, " +
                      f"Config patches: {commit['rca_patches']['config_patch_count']}")
        
        print("\n✅ JSON file created successfully!")
        print(f"📁 Location: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

