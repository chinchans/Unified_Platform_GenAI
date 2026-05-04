#!/usr/bin/env python3
"""
Context-Aware Candidate Retrieval Module
Step 2.4 of the Enhanced Error Fixing Pipeline
"""

import json
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextAwareRetrieval:
    """Context-aware candidate retrieval with role-based filtering and log anchoring."""
    
    def __init__(self, functions_index_path: str, config_index_path: str, 
                 functions_mapping_path: str, config_mapping_path: str,
                 openair_codebase_file_name: str = "openairinterface5g-develop"):
        """
        Initialize the context-aware retrieval system.
        
        Args:
            functions_index_path: Path to FAISS functions index
            config_index_path: Path to FAISS config index
            functions_mapping_path: Path to functions mapping JSON
            config_mapping_path: Path to config mapping JSON
            openair_codebase_file_name: Name of the OpenAirInterface5G codebase folder
        """
        self.openair_codebase_file_name = openair_codebase_file_name
        self.functions_index_path = functions_index_path
        self.config_index_path = config_index_path
        self.functions_mapping_path = functions_mapping_path
        self.config_mapping_path = config_mapping_path
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load FAISS indices
        self._load_indices()
        
        # Load mappings
        self._load_mappings()
        
        logger.info("Context-aware retrieval system initialized")
    
    def _load_indices(self):
        """Load FAISS indices for functions and configs."""
        logger.info("Loading FAISS indices...")
        
        # Load functions index
        if os.path.exists(self.functions_index_path):
            self.functions_index = faiss.read_index(self.functions_index_path)
            logger.info(f"✅ Functions index loaded: {self.functions_index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Functions index not found: {self.functions_index_path}")
        
        # Load config index
        if os.path.exists(self.config_index_path):
            self.config_index = faiss.read_index(self.config_index_path)
            logger.info(f"✅ Config index loaded: {self.config_index.ntotal} vectors")
        else:
            raise FileNotFoundError(f"Config index not found: {self.config_index_path}")
    
    def _load_mappings(self):
        """Load function and config mappings."""
        logger.info("Loading mappings...")
        
        # Load functions mapping
        if os.path.exists(self.functions_mapping_path):
            with open(self.functions_mapping_path, 'r', encoding='utf-8') as f:
                self.functions_mapping = json.load(f)
            logger.info(f"✅ Functions mapping loaded: {len(self.functions_mapping)} entries")
        else:
            raise FileNotFoundError(f"Functions mapping not found: {self.functions_mapping_path}")
        
        # Load config mapping
        if os.path.exists(self.config_mapping_path):
            with open(self.config_mapping_path, 'r', encoding='utf-8') as f:
                self.config_mapping = json.load(f)
            logger.info(f"✅ Config mapping loaded: {len(self.config_mapping)} entries")
        else:
            raise FileNotFoundError(f"Config mapping not found: {self.config_mapping_path}")
    
    def retrieve_candidates_with_context(self, error_text: str, deployment_context: Dict, 
                                       top_k: int = 10) -> Dict:
        """
        Retrieve candidates with context-aware filtering and boosting.
        
        Args:
            error_text: Raw error string
            deployment_context: Parsed deployment context from logs
            top_k: Number of top candidates to return
            
        Returns:
            Dictionary with filtered and boosted candidates
        """
        logger.info(f"🔍 Context-aware retrieval for error: {error_text}")
        
        # Extract deployment context
        role = deployment_context.get('role', 'Unknown')
        active_configs = deployment_context.get('active_configs', [])
        log_anchors = deployment_context.get('log_anchors', [])
        
        logger.info(f"📋 Deployment context - Role: {role}, Active configs: {len(active_configs)}, Log anchors: {len(log_anchors)}")
        
        # Generate embedding for error text
        error_embedding = self.embedding_model.encode([error_text])
        
        # Retrieve function candidates
        function_candidates = self._retrieve_functions_with_context(
            error_embedding, role, log_anchors, top_k
        )
        
        # Retrieve config candidates
        config_candidates = self._retrieve_configs_with_context(
            error_embedding, active_configs, log_anchors, top_k
        )
        
        return {
            "functions": function_candidates,
            "configs": config_candidates
        }
    
    def _retrieve_functions_with_context(self, error_embedding: np.ndarray, role: str, 
                                       log_anchors: List[str], top_k: int) -> List[Dict]:
        """Retrieve function candidates with role-based filtering and log anchoring."""
        logger.info("🔍 Retrieving function candidates with context...")
        
        # Perform FAISS search (get more candidates for filtering)
        search_k = min(top_k * 3, self.functions_index.ntotal)
        scores, indices = self.functions_index.search(error_embedding, search_k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Get function info from mapping
            function_info = self.functions_mapping.get(str(idx), {})
            if not function_info:
                continue
            
            function_name = function_info.get('function_name', '')
            file_path = function_info.get('file_path', '')
            code_snippet = function_info.get('code_snippet', '')
            
            # Apply role-based filtering
            if not self._is_role_appropriate(function_name, file_path, role):
                continue
            
            # Calculate boosted score
            boosted_score = self._calculate_boosted_score(score, code_snippet, log_anchors)
            
            candidates.append({
                "name": function_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score)
            })
        
        # Sort by boosted score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _retrieve_configs_with_context(self, error_embedding: np.ndarray, active_configs: List[str], 
                                     log_anchors: List[str], top_k: int) -> List[Dict]:
        """Retrieve config candidates with active config filtering and log anchoring."""
        logger.info("🔍 Retrieving config candidates with context...")
        
        # Perform FAISS search (get more candidates for filtering)
        search_k = min(top_k * 3, self.config_index.ntotal)
        scores, indices = self.config_index.search(error_embedding, search_k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Get config info from mapping
            config_info = self.config_mapping.get(str(idx), {})
            if not config_info:
                continue
            
            param_name = config_info.get('param_name', '')
            file_path = config_info.get('file_path', '')
            param_value = config_info.get('param_value', '')
            
            # Apply active config filtering
            if not self._is_active_config(file_path, active_configs):
                continue
            
            # Calculate boosted score
            config_text = f"{param_name} {param_value}"
            boosted_score = self._calculate_boosted_score(score, config_text, log_anchors)
            
            candidates.append({
                "param": param_name,
                "file_path": file_path,
                "score": float(boosted_score),
                "original_score": float(score),
                "value": param_value
            })
        
        # Sort by boosted score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _is_role_appropriate(self, function_name: str, file_path: str, role: str) -> bool:
        """Check if function is appropriate for the given role."""
        if role == 'CU':
            # For CU: keep only CU-related functions, exclude DU/UE
            if any(exclude in file_path.lower() for exclude in ['/du/', '/ue/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['du_', 'ue_']):
                return False
            return True
        
        elif role == 'DU':
            # For DU: keep only DU-related functions, exclude CU/UE
            if any(exclude in file_path.lower() for exclude in ['/cu/', '/ue/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['cu_', 'ue_']):
                return False
            return True
        
        elif role == 'UE':
            # For UE: keep only UE-related functions, exclude CU/DU
            if any(exclude in file_path.lower() for exclude in ['/cu/', '/du/']):
                return False
            if any(function_name.lower().startswith(prefix) for prefix in ['cu_', 'du_']):
                return False
            return True
        
        elif role == 'gNB':
            # For gNB: keep gNB-related functions
            if any(keyword in file_path.lower() for keyword in ['gnb', 'ngap']):
                return True
            if any(function_name.lower().startswith(prefix) for prefix in ['gnb_', 'ngap_']):
                return True
            return False
        
        # For unknown role, keep all functions
        return True
    
    def _is_active_config(self, file_path: str, active_configs: List[str]) -> bool:
        """Check if config file is in the active configs list."""
        if not active_configs:
            return True  # If no active configs specified, keep all
        
        # Check if any active config matches the file path
        for active_config in active_configs:
            if active_config in file_path or file_path in active_config:
                return True
        
        return False
    
    def _calculate_boosted_score(self, original_score: float, text: str, log_anchors: List[str]) -> float:
        """Calculate boosted score based on log anchor matches."""
        boosted_score = original_score
        
        if not log_anchors:
            return boosted_score
        
        # Check for log anchor matches in text
        text_lower = text.lower()
        anchor_matches = 0
        
        for anchor in log_anchors:
            anchor_lower = anchor.lower()
            # Extract key terms from anchor (remove brackets, timestamps, etc.)
            anchor_terms = [term.strip() for term in anchor_lower.split() if len(term) > 3]
            
            for term in anchor_terms:
                if term in text_lower:
                    anchor_matches += 1
                    break  # Count each anchor only once
        
        # Boost score based on anchor matches
        if anchor_matches > 0:
            boost = min(0.2, anchor_matches * 0.1)  # Max boost of 0.2
            boosted_score += boost
            logger.debug(f"Boosted score by {boost} for {anchor_matches} anchor matches")
        
        return min(1.0, boosted_score)  # Cap at 1.0

def main():
    """Test the context-aware retrieval system."""
    # Example usage
    error_text = "No AMF associated to gNB"
    
    # Load deployment context
    with open('deployment_context.json', 'r') as f:
        deployment_context = json.load(f)
    
    # Initialize retrieval system
    retrieval = ContextAwareRetrieval(
        functions_index_path='faiss_indices/functions_index.faiss',
        config_index_path='faiss_indices/config_index.faiss',
        functions_mapping_path='faiss_indices/functions_mapping.json',
        config_mapping_path='faiss_indices/config_mapping.json'
    )
    
    # Retrieve candidates
    results = retrieval.retrieve_candidates_with_context(error_text, deployment_context, top_k=10)
    
    # Save results
    with open('output/context_aware_candidates.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Context-aware retrieval completed!")
    print(f"Functions found: {len(results['functions'])}")
    print(f"Configs found: {len(results['configs'])}")

if __name__ == "__main__":
    main()
