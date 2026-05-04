#!/usr/bin/env python3
"""
Git Commit Embeddings Generator
Creates semantic embeddings from git commit history for similarity search
Optimized for Google Colab with GPU acceleration
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from datetime import datetime

class GitCommitEmbeddingGenerator:
    """Generate embeddings from git commit JSON"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        """
        Initialize the embedding generator
        
        Args:
            model_name: HuggingFace model name
                - 'all-MiniLM-L6-v2': Fast, 384 dims (recommended)
                - 'all-mpnet-base-v2': Accurate, 768 dims (slower)
            batch_size: Batch size for GPU processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load the model
        print(f"📥 Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def prepare_embedding_text(self, commit):
        """
        Prepare text for embedding from commit data
        
        Args:
            commit: Dictionary containing commit information
        
        Returns:
            str: Text ready for embedding
        """
        # Base: subject + body
        text_parts = []
        
        if commit.get('subject'):
            text_parts.append(commit['subject'])
        
        if commit.get('body'):
            # Limit body length to avoid extremely long texts
            body = commit['body'][:1000]  # First 1000 chars
            text_parts.append(body)
        
        # Add keywords for better semantic matching
        if commit.get('keywords') and len(commit['keywords']) > 0:
            keywords_str = ', '.join(commit['keywords'])
            text_parts.append(f"Keywords: {keywords_str}")
        
        # Add file context
        if commit.get('files_changed') and len(commit['files_changed']) > 0:
            files_str = ', '.join(commit['files_changed'][:5])  # Limit to 5 files
            text_parts.append(f"Files: {files_str}")
        
        # Combine all parts
        text = '\n\n'.join(text_parts)
        
        return text.strip()
    
    def prepare_metadata(self, commit, index):
        """
        Prepare metadata for storage with embeddings
        
        Args:
            commit: Dictionary containing commit information
            index: Index of commit in original list
        
        Returns:
            dict: Metadata for this commit
        """
        metadata = {
            'index': index,
            'commit_hash': commit.get('commit_hash', ''),
            'commit_hash_short': commit.get('commit_hash_short', ''),
            'author_name': commit.get('author_name', ''),
            'author_email': commit.get('author_email', ''),
            'date': commit.get('date', ''),
            'date_iso': commit.get('date_iso', ''),
            'subject': commit.get('subject', ''),
            'keywords': commit.get('keywords', []),
            'is_rca_commit': commit.get('rca_patches', {}).get('is_rca_commit', False),
            'files_changed': commit.get('files_changed', []),
        }
        
        # Add RCA patch information if available
        if metadata['is_rca_commit']:
            rca_patches = commit.get('rca_patches', {})
            metadata['code_patches'] = rca_patches.get('code_patches', [])
            metadata['config_patches'] = rca_patches.get('config_patches', [])
            metadata['code_patch_count'] = rca_patches.get('code_patch_count', 0)
            metadata['config_patch_count'] = rca_patches.get('config_patch_count', 0)
        
        return metadata
    
    def load_commits(self, json_path):
        """
        Load commits from JSON file
        
        Args:
            json_path: Path to git_log_commits.json
        
        Returns:
            list: List of commit dictionaries
        """
        print(f"📂 Loading commits from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        commits = data.get('commits', [])
        print(f"✅ Loaded {len(commits)} commits")
        
        return commits
    
    def generate_embeddings(self, commits, show_progress=True):
        """
        Generate embeddings for all commits with GPU acceleration
        
        Args:
            commits: List of commit dictionaries
            show_progress: Show progress bar
        
        Returns:
            tuple: (embeddings_array, metadata_list, texts_list)
        """
        print(f"\n🚀 Generating embeddings...")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total commits: {len(commits)}")
        
        # Prepare all texts and metadata
        texts = []
        metadata_list = []
        
        print("📝 Preparing texts...")
        for idx, commit in enumerate(tqdm(commits, disable=not show_progress)):
            text = self.prepare_embedding_text(commit)
            metadata = self.prepare_metadata(commit, idx)
            
            texts.append(text)
            metadata_list.append(metadata)
        
        # Generate embeddings in batches (GPU accelerated)
        print(f"\n🔥 Generating embeddings with GPU acceleration...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        
        return embeddings, metadata_list, texts
    
    def save_embeddings(self, embeddings, metadata_list, texts, output_dir):
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Numpy array of embeddings
            metadata_list: List of metadata dictionaries
            texts: List of original texts
            output_dir: Directory to save outputs
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving embeddings to: {output_dir}")
        
        # Save embeddings as numpy array
        embeddings_path = os.path.join(output_dir, 'git_commit_embeddings.npy')
        np.save(embeddings_path, embeddings)
        print(f"✅ Saved embeddings: {embeddings_path}")
        
        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, 'git_commit_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved metadata: {metadata_path}")
        
        # Save texts for reference
        texts_path = os.path.join(output_dir, 'git_commit_texts.json')
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved texts: {texts_path}")
        
        # Save config
        config = {
            'model_name': self.model_name,
            'embedding_dimension': embeddings.shape[1],
            'total_commits': len(metadata_list),
            'generated_at': datetime.now().isoformat(),
            'device': self.device,
            'batch_size': self.batch_size
        }
        
        config_path = os.path.join(output_dir, 'embedding_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved config: {config_path}")
        
        # Print statistics
        print(f"\n📊 Statistics:")
        print(f"   Total embeddings: {embeddings.shape[0]}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   File size: {embeddings.nbytes / 1e6:.2f} MB")
        
        # Count RCA commits
        rca_count = sum(1 for m in metadata_list if m['is_rca_commit'])
        print(f"   RCA commits: {rca_count}")
        print(f"   Regular commits: {len(metadata_list) - rca_count}")

def main():
    """Main function"""
    print("=" * 70)
    print("Git Commit Embeddings Generator")
    print("=" * 70)
    
    # Configuration
    INPUT_JSON = 'resources/git_log_commits.json'
    OUTPUT_DIR = 'resources/embeddings'
    
    # For Google Colab, you might need to adjust paths
    # INPUT_JSON = '/content/git_log_commits.json'
    # OUTPUT_DIR = '/content/embeddings'
    
    MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and efficient
    # MODEL_NAME = 'all-mpnet-base-v2'  # More accurate but slower
    
    BATCH_SIZE = 64  # Increase for better GPU utilization
    
    try:
        # Initialize generator
        generator = GitCommitEmbeddingGenerator(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE
        )
        
        # Load commits
        commits = generator.load_commits(INPUT_JSON)
        
        # Generate embeddings
        embeddings, metadata_list, texts = generator.generate_embeddings(
            commits,
            show_progress=True
        )
        
        # Save outputs
        generator.save_embeddings(
            embeddings,
            metadata_list,
            texts,
            OUTPUT_DIR
        )
        
        print("\n" + "=" * 70)
        print("✅ Embedding generation completed successfully!")
        print("=" * 70)
        
        # Sample test
        print("\n🔍 Sample embedding check:")
        print(f"   First commit: {metadata_list[0]['subject'][:60]}...")
        print(f"   Embedding shape: {embeddings[0].shape}")
        print(f"   Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

