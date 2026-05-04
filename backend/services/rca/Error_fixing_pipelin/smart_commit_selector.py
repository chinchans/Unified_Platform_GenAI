#!/usr/bin/env python3
"""
Smart Commit Selector - Two-Stage Selection with Optional LLM Verification
Usage: Modify ERROR_QUERY and run: python smart_commit_selector.py
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

class CommitSearcher:
    """Search for similar commits using embeddings"""
    
    def __init__(self, embeddings_dir='resources/embeddings', validate_commits=True, 
                 openair_codebase_file_name='openairinterface5g-develop'):
        """
        Initialize commit searcher.
        
        Args:
            embeddings_dir: Directory containing embeddings
            validate_commits: Whether to validate commits exist in git
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        self.embeddings_dir = embeddings_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.validate_commits = validate_commits
        self.invalid_commits_cache = set()  # Cache deleted commits
        # Construct git repo path dynamically
        self.git_repo_path = f'Error_fixing_pipelin/{openair_codebase_file_name}'
        
        print(f"Loading search engine...")
        
        with open(os.path.join(embeddings_dir, 'embedding_config.json'), 'r') as f:
            self.config = json.load(f)
        
        self.model = SentenceTransformer(self.config['model_name'], device=self.device)
        self.embeddings = np.load(os.path.join(embeddings_dir, 'git_commit_embeddings.npy'))
        
        with open(os.path.join(embeddings_dir, 'git_commit_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded {len(self.metadata)} commits")
        if validate_commits:
            print(f"   Commit validation: ENABLED")
        print()
    
    def commit_exists_in_git(self, commit_hash):
        """Check if commit still exists in git repository"""
        
        # Check cache first
        if commit_hash in self.invalid_commits_cache:
            return False
        
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'cat-file', '-e', commit_hash],
                cwd=self.git_repo_path,
                capture_output=True,
                timeout=2
            )
            
            exists = result.returncode == 0
            
            if not exists:
                self.invalid_commits_cache.add(commit_hash)
            
            return exists
            
        except Exception as e:
            # If validation fails, assume commit exists
            print(f"⚠️ Validation error for {commit_hash[:10]}: {e}")
            return True
    
    def search(self, query, top_k=10):
        """Search with optional commit validation"""
        
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get more results than needed for validation filtering
        fetch_count = top_k + 20 if self.validate_commits else top_k
        sorted_indices = np.argsort(similarities)[::-1][:fetch_count]
        
        results = []
        skipped_count = 0
        
        for idx in sorted_indices:
            meta = self.metadata[idx]
            commit_hash = meta.get('commit_hash', '')
            
            # Validate commit if enabled
            if self.validate_commits and commit_hash:
                if not self.commit_exists_in_git(commit_hash):
                    skipped_count += 1
                    continue
            
            results.append({'similarity': float(similarities[idx]), **meta})
            
            # Stop when we have enough valid results
            if len(results) >= top_k:
                break
        
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} deleted commits")
        
        return results


class SmartSelector:
    def __init__(self, use_llm=False, llm_config=None):
        self.use_llm = use_llm
        self.llm_config = llm_config or {}
    
    def extract_keywords_from_error(self, error_description):
        keyword_patterns = [
            r'\bamf\b', r'\brrc\b', r'\bngap\b', r'\bf1ap\b', r'\btimeout\b',
            r'\bfail', r'\berror\b', r'\bassociation\b', r'\bconnection\b',
            r'\bsetup\b', r'\bconfig\b', r'\bue\b', r'\bgnb\b'
        ]
        
        text_lower = error_description.lower()
        keywords = []
        
        for pattern in keyword_patterns:
            if re.search(pattern, text_lower):
                keyword = pattern.replace(r'\b', '').replace('\\', '')
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def calculate_recency_score(self, date_iso):
        if not date_iso:
            return 0.0
        try:
            commit_date = datetime.fromisoformat(date_iso.replace('Z', '+00:00'))
            current_date = datetime.now(commit_date.tzinfo)
            days_ago = (current_date - commit_date).days
            
            if days_ago < 30:
                return 0.05
            elif days_ago < 90:
                return 0.03
            elif days_ago < 365:
                return 0.01
            return 0.0
        except:
            return 0.0
    
    def calculate_boosted_score(self, result, error_keywords):
        base_similarity = result['similarity']
        rca_boost = 0.2 if result.get('is_rca_commit', False) else 0.0
        
        commit_keywords = set(k.lower() for k in result.get('keywords', []))
        error_keywords_set = set(k.lower() for k in error_keywords)
        overlap = len(commit_keywords & error_keywords_set)
        keyword_score = min(overlap * 0.05, 0.15)
        
        recency_score = self.calculate_recency_score(result.get('date_iso', ''))
        
        final_score = base_similarity * 0.6 + rca_boost + keyword_score + recency_score
        
        result['score_breakdown'] = {
            'base_similarity': base_similarity,
            'rca_boost': rca_boost,
            'keyword_overlap': keyword_score,
            'recency': recency_score,
            'keyword_matches': list(commit_keywords & error_keywords_set)
        }
        
        return final_score
    
    def stage1_fast_selection(self, results, error_description):
        print("🔍 Stage 1: Fast heuristic selection...")
        
        error_keywords = self.extract_keywords_from_error(error_description)
        print(f"   Error keywords: {error_keywords}")
        
        for result in results:
            result['boosted_score'] = self.calculate_boosted_score(result, error_keywords)
        
        results_sorted = sorted(results, key=lambda x: x['boosted_score'], reverse=True)
        
        if not results_sorted:
            return {'status': 'no_fix_found', 'confidence': 'none', 'commit': None}
        
        best = results_sorted[0]
        print(f"   Best match: {best['commit_hash_short']} (score: {best['boosted_score']:.4f})")
        
        # High confidence RCA
        if best['is_rca_commit'] and best['boosted_score'] >= 0.75:
            print(f"   ✅ AUTO-SELECT (high confidence)")
            return {
                'status': 'auto_selected',
                'confidence': 'high',
                'commit': best,
                'alternatives': results_sorted[1:3],
                'reasoning': f"RCA fix with {best['boosted_score']:.0%} confidence",
                'should_apply': True
            }
        
        # Medium confidence RCA
        if best['is_rca_commit'] and best['boosted_score'] >= 0.60:
            print(f"   ✅ SUGGEST (medium-high confidence)")
            return {
                'status': 'suggested',
                'confidence': 'medium-high',
                'commit': best,
                'alternatives': results_sorted[1:3],
                'reasoning': f"RCA fix with {best['boosted_score']:.0%} confidence",
                'should_apply': True
            }
        
        # Check for ambiguous cases (multiple close matches)
        if len(results_sorted) >= 3:
            top3_scores = [r['boosted_score'] for r in results_sorted[:3]]
            score_diff = max(top3_scores) - min(top3_scores)
            
            if score_diff < 0.15 and best['boosted_score'] >= 0.50:
                print(f"   🤔 AMBIGUOUS (multiple close matches)")
                return {
                    'status': 'needs_llm',
                    'confidence': 'uncertain',
                    'candidates': results_sorted[:5],
                    'reasoning': f"Multiple candidates with similar scores (diff: {score_diff:.2f})"
                }
        
        # Low score - no fix
        if best['boosted_score'] < 0.50:
            print(f"   ❌ NO FIX FOUND (score too low)")
            return {
                'status': 'no_fix_found',
                'confidence': 'none',
                'commit': None,
                'reasoning': f"No relevant fixes (best: {best['boosted_score']:.0%})"
            }
        
        # Default: medium-low confidence
        print(f"   ⚠️ SUGGEST (medium-low confidence)")
        return {
            'status': 'suggested',
            'confidence': 'medium-low',
            'commit': best,
            'alternatives': results_sorted[1:3],
            'reasoning': f"Best match ({best['boosted_score']:.0%})",
            'should_apply': False
        }
    
    def prepare_llm_prompt(self, error_description, candidates):
        """Prepare LLM prompt"""
        
        prompt = f"""You are an expert 5G RAN engineer. Analyze which commit fix is most likely to solve this error.

Error Description:
"{error_description}"

Available Commit Fixes:
"""
        
        for i, candidate in enumerate(candidates, 1):
            rca_tag = "[RCA - Tested Fix]" if candidate['is_rca_commit'] else "[Community Fix]"
            prompt += f"\n{i}. {rca_tag} (Similarity: {candidate['similarity']:.2%})\n"
            prompt += f"   Commit: {candidate['commit_hash_short']}\n"
            prompt += f"   Message: {candidate['subject']}\n"
            
            if candidate.get('keywords'):
                prompt += f"   Keywords: {', '.join(candidate['keywords'][:6])}\n"
            
            if candidate.get('files_changed'):
                files = ', '.join(candidate['files_changed'][:3])
                prompt += f"   Files: {files}\n"
            
            if candidate['is_rca_commit']:
                if candidate.get('code_patches'):
                    prompt += f"   Code Patches: {candidate['code_patch_count']}\n"
                if candidate.get('config_patches'):
                    prompt += f"   Config Patches: {candidate['config_patch_count']}\n"
        
        prompt += """

Task: Select the commit that is MOST LIKELY to fix this error.

Respond in JSON format:
{
    "selected_commit_index": 1-5 or null,
    "confidence": "high" | "medium" | "low",
    "reasoning": "Brief explanation"
}

Response:"""
        
        return prompt
    
    def call_llm(self, prompt):
        """Call LLM API"""
        
        if not self.use_llm:
            print("   ⚠️ LLM disabled")
            return None
        
        try:
            provider = self.llm_config.get('provider')
            
            if provider == 'openai':
                import openai
                openai.api_key = self.llm_config.get('api_key')
                
                response = openai.ChatCompletion.create(
                    model=self.llm_config.get('model', 'gpt-4'),
                    messages=[
                        {"role": "system", "content": "You are an expert 5G RAN engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
                
            elif provider == 'azure':
                import openai
                openai.api_type = "azure"
                openai.api_key = self.llm_config.get('api_key')
                openai.api_base = self.llm_config.get('endpoint')
                openai.api_version = self.llm_config.get('api_version', '2023-05-15')
                
                response = openai.ChatCompletion.create(
                    engine=self.llm_config.get('deployment_name'),
                    messages=[
                        {"role": "system", "content": "You are an expert 5G RAN engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
            else:
                print("   ⚠️ No LLM provider configured")
                return None
            
            # Parse JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"   ❌ LLM error: {str(e)}")
            return None
    
    def stage2_llm_verification(self, error_description, candidates):
        """Stage 2: LLM verification"""
        
        print("\n🤖 Stage 2: LLM verification...")
        
        prompt = self.prepare_llm_prompt(error_description, candidates)
        llm_response = self.call_llm(prompt)
        
        if llm_response is None:
            print("   ❌ LLM failed, using best candidate")
            return {
                'status': 'suggested',
                'confidence': 'medium-low',
                'commit': candidates[0],
                'alternatives': candidates[1:3],
                'reasoning': 'LLM unavailable, using highest scored match'
            }
        
        selected_idx = llm_response.get('selected_commit_index')
        
        if selected_idx and 1 <= selected_idx <= len(candidates):
            selected_commit = candidates[selected_idx - 1]
            print(f"   ✅ LLM selected: {selected_commit['commit_hash_short']}")
            
            return {
                'status': 'llm_verified',
                'confidence': llm_response.get('confidence', 'medium'),
                'commit': selected_commit,
                'alternatives': [c for c in candidates if c != selected_commit][:2],
                'reasoning': llm_response.get('reasoning', 'LLM verified fix'),
                'should_apply': llm_response.get('confidence') == 'high' and selected_commit['is_rca_commit']
            }
        else:
            print("   ❌ LLM found no suitable fix")
            return {
                'status': 'no_fix_found',
                'confidence': 'none',
                'commit': None,
                'reasoning': 'LLM found no suitable fix'
            }
    
    def select_best_fix(self, error_description, search_results):
        """Main selection logic"""
        
        print(f"\n{'='*80}")
        print(f"SMART FIX SELECTION")
        print(f"{'='*80}\n")
        
        # Stage 1
        stage1_result = self.stage1_fast_selection(search_results, error_description)
        
        # Stage 2 if needed
        if stage1_result['status'] == 'needs_llm' and self.use_llm:
            candidates = stage1_result['candidates']
            return self.stage2_llm_verification(error_description, candidates)
        
        return stage1_result


def print_selection_result(result):
    print(f"\n{'='*80}")
    print(f"SELECTION RESULT")
    print(f"{'='*80}\n")
    
    status_icons = {'auto_selected': '✅', 'suggested': '💡', 'llm_verified': '🤖', 'no_fix_found': '❌'}
    icon = status_icons.get(result['status'], '📌')
    print(f"Status: {icon} {result['status'].upper().replace('_', ' ')}")
    print(f"Confidence: {result['confidence'].upper()}")
    
    if result['commit']:
        commit = result['commit']
        rca_badge = "🟢 RCA" if commit['is_rca_commit'] else "🔵 REG"
        
        print(f"\n{rca_badge} Selected Fix:")
        print(f"{'─'*80}")
        print(f"Commit: {commit['commit_hash_short']}")
        print(f"Subject: {commit['subject']}")
        print(f"Author: {commit['author_name']}")
        print(f"Date: {commit['date_iso'][:10]}")
        
        print(f"\nScores:")
        print(f"  Base Similarity: {commit['similarity']:.2%}")
        if commit.get('boosted_score'):
            print(f"  Boosted Score: {commit['boosted_score']:.2%}")
            if commit.get('score_breakdown'):
                bd = commit['score_breakdown']
                print(f"  RCA Boost: +{bd['rca_boost']:.2%}")
                print(f"  Keyword Match: +{bd['keyword_overlap']:.2%} ({bd['keyword_matches']})")
        
        if commit.get('keywords'):
            print(f"\nKeywords: {', '.join(commit['keywords'][:8])}")
        
        if commit['is_rca_commit']:
            print(f"\n🔧 Available Patches:")
            if commit.get('code_patches'):
                print(f"  Code Patches ({commit['code_patch_count']}):")
                for p in commit['code_patches']:
                    print(f"    - {p['function']} in {p['file']}")
            if commit.get('config_patches'):
                print(f"  Config Patches ({commit['config_patch_count']}):")
                for p in commit['config_patches']:
                    print(f"    - {p['parameter']} in {p['file']}")
    
    print(f"\nReasoning: {result['reasoning']}")
    if result.get('should_apply') is not None:
        text = "✅ YES" if result['should_apply'] else "⚠️  MANUAL REVIEW"
        print(f"Auto-apply: {text}")
    
    if result.get('alternatives'):
        print(f"\n{'─'*80}")
        print(f"Alternatives:")
        for i, alt in enumerate(result['alternatives'], 1):
            rca = "🟢" if alt['is_rca_commit'] else "🔵"
            score = alt.get('boosted_score', alt['similarity'])
            print(f"{i}. {rca} [{score:.4f}] {alt['subject'][:60]}")
    
    print(f"\n{'='*80}\n")


def main():
    # ============= CONFIGURATION =============
    ERROR_QUERY = "No AMF associated to the gNB"
    TOP_K_RESULTS = 10
    
    # LLM Configuration (optional)
    USE_LLM = False  # Set True to enable
    LLM_CONFIG = {
        'provider': 'azure',  # 'openai' or 'azure'
        'api_key': 'your-api-key',
        'endpoint': 'https://your-endpoint.openai.azure.com/',  # For Azure
        'deployment_name': 'gpt-4',  # For Azure
        'model': 'gpt-4'  # For OpenAI
    }
    # =========================================
    
    print(f"{'='*80}")
    print(f"SMART GIT COMMIT SELECTOR")
    print(f"{'='*80}\n")
    print(f"Error: {ERROR_QUERY}\n")
    
    searcher = CommitSearcher()
    search_results = searcher.search(ERROR_QUERY, top_k=TOP_K_RESULTS)
    print(f"✅ Found {len(search_results)} similar commits\n")
    
    selector = SmartSelector(use_llm=USE_LLM, llm_config=LLM_CONFIG)
    selection_result = selector.select_best_fix(ERROR_QUERY, search_results)
    
    print_selection_result(selection_result)
    return selection_result


if __name__ == "__main__":
    main()