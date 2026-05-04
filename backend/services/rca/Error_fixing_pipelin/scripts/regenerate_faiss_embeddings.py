#!/usr/bin/env python3
"""
Regenerate FAISS Embeddings with Relative Paths
Optimized for GPU execution (Google Colab)

This script regenerates the FAISS indices for functions and configs with relative paths
instead of hardcoded 'openairinterface5g-develop' paths, making the system branch-agnostic.

Usage:
    python regenerate_faiss_embeddings.py --source-dir <path_to_oai_repo>

Author: AgenticRAN
"""

import os
import json
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print("=" * 80)
print("FAISS EMBEDDINGS REGENERATION - GPU OPTIMIZED")
print("=" * 80)

class FAISSEmbeddingGenerator:
    def __init__(self, source_dir: str, output_dir: str = "faiss_indices"):
        """
        Initialize the embedding generator
        
        Args:
            source_dir: Path to OpenAirInterface5G repository
            output_dir: Output directory for FAISS indices
        """
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🖥️  Device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load sentence transformer model - MUST match the RCA pipeline model!
        print(f"\n📥 Loading SentenceTransformer model...")
        # Using 'all-mpnet-base-v2' to match error_handling_pipeline.py and context_aware_retrieval.py
        self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        print(f"✅ Model loaded on {self.device}")
        print(f"   Model: all-mpnet-base-v2")
        print(f"   Dimension: {self.model.get_sentence_embedding_dimension()}")
        
    def load_database_files(self) -> Tuple[Dict, Dict]:
        """Load functions.json and config.json from database folder"""
        print("\n📂 Loading database files...")
        
        # Load functions
        functions_path = Path("functions.json")
        if not functions_path.exists():
            raise FileNotFoundError(f"Functions database not found: {functions_path}")
        
        with open(functions_path, 'r', encoding='utf-8') as f:
            functions_data = json.load(f)
        print(f"✅ Loaded {len(functions_data)} functions")
        
        # Load configs
        config_path = Path("config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config database not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"✅ Loaded {len(config_data)} config parameters")
        
        return functions_data, config_data
    
    def convert_to_relative_path(self, file_path: str) -> str:
        """
        Convert absolute or hardcoded path to relative path
        
        Args:
            file_path: Original file path (may contain 'openairinterface5g-develop')
        
        Returns:
            Relative path from repository root
        """
        # Remove any hardcoded folder names
        path = file_path.replace('openairinterface5g-develop/', '')
        path = path.replace('openairinterface5g-develop\\', '')
        path = path.replace('openairinterface5g\\', '')
        path = path.replace('openairinterface5g/', '')
        
        # Normalize path separators
        path = path.replace('\\', '/')
        
        return path
    
    def process_functions(self, functions_data) -> Tuple[np.ndarray, Dict]:
        """
        Process functions and generate embeddings
        
        Args:
            functions_data: List or Dictionary of function data
        
        Returns:
            Tuple of (embeddings array, mapping dictionary)
        """
        print("\n🔧 Processing functions...")
        
        texts = []
        mapping = {}
        
        # Handle both list and dict formats
        if isinstance(functions_data, list):
            items = enumerate(functions_data)
        else:
            items = functions_data.items()
        
        for idx, func in enumerate(tqdm(functions_data if isinstance(functions_data, list) else list(functions_data.values()), desc="Preparing functions")):
            # Handle both dict with key and direct list item
            if not isinstance(func, dict):
                continue
            # Convert to relative path
            original_path = func.get('file_path', '')
            relative_path = self.convert_to_relative_path(original_path)
            
            # Create searchable text
            function_name = func.get('function_name', '')
            code_body = func.get('code_body', '')
            signature = func.get('signature', '')
            return_type = func.get('return_type', '')
            
            # Combine for embedding - MUST match build_faiss_indices.py format exactly!
            text = f"{function_name}\n{code_body}\n{relative_path}"
            texts.append(text)
            
            # Store mapping with relative path
            mapping[str(idx)] = {
                "function_name": function_name,
                "file_path": relative_path,  # 🔧 NOW RELATIVE!
                "start_line": func.get('start_line'),
                "end_line": func.get('end_line'),
                "code_body": code_body,
                "return_type": return_type,
                "signature": signature
            }
        
        # Generate embeddings in batches (GPU optimized)
        print(f"\n🚀 Generating embeddings for {len(texts)} functions...")
        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        return embeddings, mapping
    
    def process_configs(self, config_data) -> Tuple[np.ndarray, Dict]:
        """
        Process config parameters and generate embeddings
        
        Args:
            config_data: List or Dictionary of config data
        
        Returns:
            Tuple of (embeddings array, mapping dictionary)
        """
        print("\n⚙️  Processing config parameters...")
        
        texts = []
        mapping = {}
        
        # Handle both list and dict formats
        for idx, config in enumerate(tqdm(config_data if isinstance(config_data, list) else list(config_data.values()), desc="Preparing configs")):
            # Handle both dict with key and direct list item
            if not isinstance(config, dict):
                continue
            # Convert to relative path
            original_path = config.get('file_path', '')
            relative_path = self.convert_to_relative_path(original_path)
            
            # Create searchable text
            param_name = config.get('param_name', '')
            param_value = config.get('param_value', '')
            
            # Combine for embedding - MUST match build_faiss_indices.py format exactly!
            text = f"{param_name}\n{param_value}\n{relative_path}"
            texts.append(text)
            
            # Store mapping with relative path
            mapping[str(idx)] = {
                "param_name": param_name,
                "param_value": param_value,
                "file_path": relative_path,  # 🔧 NOW RELATIVE!
                "line_number": config.get('line_number')
            }
        
        # Generate embeddings in batches (GPU optimized)
        print(f"\n🚀 Generating embeddings for {len(texts)} config parameters...")
        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        return embeddings, mapping
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "IVF") -> faiss.Index:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of FAISS index ("Flat" or "IVF")
        
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        print(f"\n📊 Creating FAISS index...")
        print(f"   Dimension: {dimension}")
        print(f"   Vectors: {n_vectors}")
        print(f"   Index Type: {index_type}")
        
        if index_type == "IVF" and n_vectors >= 1000:
            # Use IVF index for large datasets (faster search)
            nlist = min(int(np.sqrt(n_vectors)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            print(f"   Training IVF index with {nlist} clusters...")
            index.train(embeddings)
            index.add(embeddings)
            print(f"✅ IVF index created and trained")
        else:
            # Use flat index for small datasets (exact search)
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            print(f"✅ Flat index created")
        
        return index
    
    def save_outputs(self, 
                     functions_embeddings: np.ndarray,
                     functions_mapping: Dict,
                     functions_index: faiss.Index,
                     config_embeddings: np.ndarray,
                     config_mapping: Dict,
                     config_index: faiss.Index):
        """Save all outputs to disk"""
        print("\n💾 Saving outputs...")
        
        # Save functions
        functions_index_path = self.output_dir / "functions_index.faiss"
        faiss.write_index(functions_index, str(functions_index_path))
        print(f"✅ Saved functions index: {functions_index_path}")
        
        functions_mapping_path = self.output_dir / "functions_mapping.json"
        with open(functions_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(functions_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved functions mapping: {functions_mapping_path}")
        
        # Save configs
        config_index_path = self.output_dir / "config_index.faiss"
        faiss.write_index(config_index, str(config_index_path))
        print(f"✅ Saved config index: {config_index_path}")
        
        config_mapping_path = self.output_dir / "config_mapping.json"
        with open(config_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(config_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved config mapping: {config_mapping_path}")
        
        # Save metadata
        metadata = {
            "model_name": "all-mpnet-base-v2",
            "embedding_dimension": functions_embeddings.shape[1],
            "total_functions": len(functions_mapping),
            "total_configs": len(config_mapping),
            "path_format": "relative",
            "device_used": self.device,
            "note": "Paths are relative to repository root, not hardcoded to specific folder name"
        }
        
        metadata_path = self.output_dir / "embeddings_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")
        
        print(f"\n✨ All files saved to: {self.output_dir.absolute()}")
    
    def generate(self):
        """Main generation workflow"""
        try:
            # Load database files
            functions_data, config_data = self.load_database_files()
            
            # Process functions
            functions_embeddings, functions_mapping = self.process_functions(functions_data)
            functions_index = self.create_faiss_index(functions_embeddings, index_type="IVF")
            
            # Process configs
            config_embeddings, config_mapping = self.process_configs(config_data)
            config_index = self.create_faiss_index(config_embeddings, index_type="IVF")
            
            # Save everything
            self.save_outputs(
                functions_embeddings, functions_mapping, functions_index,
                config_embeddings, config_mapping, config_index
            )
            
            # Summary
            print("\n" + "=" * 80)
            print("✅ REGENERATION COMPLETE!")
            print("=" * 80)
            print(f"📊 Summary:")
            print(f"   Functions: {len(functions_mapping)}")
            print(f"   Configs: {len(config_mapping)}")
            print(f"   Embedding Dimension: {functions_embeddings.shape[1]}")
            print(f"   Device: {self.device.upper()}")
            print(f"   Path Format: RELATIVE (branch-agnostic)")
            print("\n🎯 Next Steps:")
            print("   1. Download the 'faiss_indices' folder from Colab")
            print("   2. Replace the existing 'faiss_indices' folder in your project")
            print("   3. The system will now work with any branch/folder name!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description="Regenerate FAISS embeddings with relative paths")
    parser.add_argument('--source-dir', type=str, default='openairinterface5g-develop',
                       help='Path to OpenAirInterface5G repository')
    parser.add_argument('--output-dir', type=str, default='faiss_indices',
                       help='Output directory for FAISS indices')
    
    args = parser.parse_args()
    
    print(f"\n🔧 Configuration:")
    print(f"   Source Directory: {args.source_dir}")
    print(f"   Output Directory: {args.output_dir}")
    
    # Check if running in Colab
    try:
        import google.colab
        print(f"   Environment: Google Colab ✅")
    except ImportError:
        print(f"   Environment: Local")
    
    # Generate embeddings
    generator = FAISSEmbeddingGenerator(args.source_dir, args.output_dir)
    generator.generate()


if __name__ == "__main__":
    main()

