#!/usr/bin/env python3
"""
FAISS Index Builder for Functions and Configuration Parameters

This script builds FAISS indices from functions.json and config.json using
the jinaai/jina-embeddings-v2-base-code model for semantic search capabilities.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import faiss
except ImportError:
    print("❌ FAISS not found. Please install with: pip install faiss-cpu")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not found. Please install with: pip install sentence-transformers")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSIndexBuilder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.functions_data = []
        self.config_data = []
        self.embedding_dim = None
        
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension by encoding a test string
            test_embedding = self.model.encode(["test"], normalize_embeddings=True)
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and validate JSON file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error(f"Expected list in {file_path}, got {type(data)}")
                return []
            
            logger.info(f"Loaded {len(data)} items from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def load_data(self, functions_file: str = "../database/functions.json", 
                                      config_file: str = "../database/config.json"):
        """Load functions and config data from JSON files."""
        logger.info("Loading input data files...")
        
        # Load functions data
        self.functions_data = self._load_json_file(functions_file)
        
        # Load config data
        self.config_data = self._load_json_file(config_file)
        
        if not self.functions_data and not self.config_data:
            raise ValueError("No valid data found in input files")
        
        logger.info(f"Data loaded: {len(self.functions_data)} functions, {len(self.config_data)} config parameters")

    def _prepare_function_text(self, func_obj: Dict[str, Any]) -> str:
        """Prepare text for function embedding: [function_name] + [code_body] + [file_path]"""
        function_name = func_obj.get('function_name', '')
        code_body = func_obj.get('code_body', '')
        file_path = func_obj.get('file_path', '')
        
        # Clean and truncate code body if too long (to avoid token limits)
        code_body = code_body.strip()
        if len(code_body) > 2000:  # Truncate very long functions
            code_body = code_body[:2000] + "..."
        
        # Combine fields
        text = f"{function_name}\n{code_body}\n{file_path}"
        return text

    def _prepare_config_text(self, config_obj: Dict[str, Any]) -> str:
        """Prepare text for config embedding: [param_name] + [param_value] + [file_path]"""
        param_name = config_obj.get('param_name', '')
        param_value = config_obj.get('param_value', '')
        file_path = config_obj.get('file_path', '')
        
        # Clean param value
        param_value = str(param_value).strip()
        if len(param_value) > 500:  # Truncate very long values
            param_value = param_value[:500] + "..."
        
        # Combine fields
        text = f"{param_name}\n{param_value}\n{file_path}"
        return text

    def _create_embeddings(self, texts: List[str], item_type: str) -> np.ndarray:
        """Create embeddings for a list of texts."""
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Creating embeddings for {len(texts)} {item_type} items...")
        
        try:
            # Create embeddings in batches to manage memory
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings with normalization
                batch_embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
                
                if (i + batch_size) % 320 == 0:  # Progress every 10 batches
                    logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} {item_type} items...")
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Created {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings for {item_type}: {e}")
            raise

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index with cosine similarity."""
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to index")
            return None
        
        # Create FAISS index for cosine similarity (inner product with normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index

    def _save_index(self, index: faiss.Index, filename: str):
        """Save FAISS index to file."""
        try:
            faiss.write_index(index, filename)
            logger.info(f"Saved FAISS index to {filename}")
        except Exception as e:
            logger.error(f"Error saving index to {filename}: {e}")
            raise

    def _save_mapping(self, data: List[Dict[str, Any]], filename: str):
        """Save mapping from vector ID to original object."""
        try:
            # Create mapping where index is vector_id and value is original object
            mapping = {i: obj for i, obj in enumerate(data)}
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved mapping for {len(mapping)} items to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving mapping to {filename}: {e}")
            raise

    def build_functions_index(self):
        """Build FAISS index for functions."""
        if not self.functions_data:
            logger.warning("No functions data to index")
            return None, 0
        
        logger.info("Building functions index...")
        
        # Prepare texts for embedding
        function_texts = []
        for func_obj in self.functions_data:
            text = self._prepare_function_text(func_obj)
            function_texts.append(text)
        
        # Create embeddings
        embeddings = self._create_embeddings(function_texts, "function")
        
        # Build FAISS index
        index = self._build_faiss_index(embeddings)
        
        # Save index and mapping
        if index is not None:
            self._save_index(index, "functions_index_codebert.faiss")
            self._save_mapping(self.functions_data, "functions_mapping_codebert.json")
        
        return index, len(self.functions_data)

    def build_config_index(self):
        """Build FAISS index for configuration parameters."""
        if not self.config_data:
            logger.warning("No config data to index")
            return None, 0
        
        logger.info("Building config index...")
        
        # Prepare texts for embedding
        config_texts = []
        for config_obj in self.config_data:
            text = self._prepare_config_text(config_obj)
            config_texts.append(text)
        
        # Create embeddings
        embeddings = self._create_embeddings(config_texts, "config")
        
        # Build FAISS index
        index = self._build_faiss_index(embeddings)
        
        # Save index and mapping
        if index is not None:
            self._save_index(index, "config_index_codebert.faiss")
            self._save_mapping(self.config_data, "config_mapping_codebert.json")
        
        return index, len(self.config_data)

    def build_indices(self):
        """Build both FAISS indices."""
        logger.info("Starting FAISS index building process...")
        
        # Load the embedding model
        self._load_model()
        
        # Build functions index
        functions_index, functions_count = self.build_functions_index()
        
        # Build config index
        config_index, config_count = self.build_config_index()
        
        return functions_count, config_count

    def print_summary(self, functions_count: int, config_count: int):
        """Print summary of the indexing process."""
        print(f"\n{'='*60}")
        print("FAISS INDEX BUILDING SUMMARY")
        print(f"{'='*60}")
        
        if functions_count > 0 or config_count > 0:
            print(f"✅ Built FAISS indices: functions_index_codebert.faiss and config_index_codebert.faiss with {functions_count} and {config_count} vectors.")
        else:
            print("❌ No indices were built - no valid data found")
        
        print(f"📊 Model: {self.model_name}")
        print(f"🔢 Embedding dimension: {self.embedding_dim}")
        print(f"📁 Functions processed: {functions_count}")
        print(f"⚙️  Config parameters processed: {config_count}")
        
        # Show output files
        output_files = []
        if functions_count > 0:
            output_files.extend(["functions_index_codebert.faiss", "functions_mapping_codebert.json"])
        if config_count > 0:
            output_files.extend(["config_index_codebert.faiss", "config_mapping_codebert.json"])
        
        if output_files:
            print(f"📄 Output files: {', '.join(output_files)}")
        
        print(f"{'='*60}")


def main():
    """Main function to run the FAISS index building."""
    try:
        # Create index builder
        builder = FAISSIndexBuilder()
        
        # Load input data
        builder.load_data()
        
        # Build indices
        functions_count, config_count = builder.build_indices()
        
        # Print summary
        builder.print_summary(functions_count, config_count)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Failed to build FAISS indices: {e}")
        exit(1)


if __name__ == "__main__":
    main()
