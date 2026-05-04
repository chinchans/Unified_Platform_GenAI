"""
3GPP Embedding Generator with GPU Support for Google Colab

Requirements:
- torch (with CUDA support)
- sentence-transformers
- faiss-gpu (for GPU acceleration) or faiss-cpu
- pdfplumber
- numpy
- tqdm

Installation for Google Colab:
!pip install torch sentence-transformers faiss-gpu pdfplumber tqdm

For CPU-only:
!pip install torch sentence-transformers faiss-cpu pdfplumber tqdm

If you get dependency conflicts, use the CPU-only version:
3GPP_embedding_generator_cpu.py
"""

import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from tqdm import tqdm
import os
import torch

# ===============================
# 1. Load PDF
# ===============================
def load_pdf(file_path):
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting PDF pages"):
            pages_text.append(page.extract_text())
    return pages_text

# ===============================
# 2. Automatic Cleaning
# ===============================
def clean_text(pages_text):
    cleaned_paragraphs = []
    # Header/footer patterns: common for 3GPP ETSI PDFs
    header_footer_patterns = [
        r"ETSI TS \d+ \S+ V\d+\.\d+\.\d+.*",
        r"3GPP TS 38\.331 version \d+\.\d+\.\d+ Release \d+",
        r"© ETSI \d{4}\. All rights reserved\.",
        r"ETSI logo are trademarks.*"
    ]
    page_number_pattern = r"^\s*\d+\s*$"

    for page_num, page in enumerate(tqdm(pages_text, desc="Cleaning pages"), start=1):
        if not page:
            continue
        lines = page.splitlines()
        new_lines = []
        for line in lines:
            line_strip = line.strip()
            if any(re.match(pat, line_strip) for pat in header_footer_patterns):
                continue
            if re.match(page_number_pattern, line_strip):
                continue
            new_lines.append(line_strip if line_strip else "")

        # Merge lines into paragraphs
        para = ""
        for line in new_lines:
            if line == "":
                if para:
                    cleaned_paragraphs.append((page_num, para.strip()))
                    para = ""
            else:
                if para.endswith("-"):
                    para = para[:-1] + line  # handle hyphenation
                else:
                    para += " " + line if para else line
        if para:
            cleaned_paragraphs.append((page_num, para.strip()))

    return cleaned_paragraphs

# ===============================
# 3. Section Detection
# ===============================
def detect_sections(paragraphs):
    section_pattern = re.compile(r'^(\d+(\.\d+)*)\s+(.*)$')
    current_section_number = None
    current_section_title = None
    sectioned_paragraphs = []

    for page_num, para in paragraphs:
        match = section_pattern.match(para)
        if match:
            current_section_number = match.group(1)
            current_section_title = match.group(3)
        sectioned_paragraphs.append({
            "page_number": page_num,
            "section_number": current_section_number,
            "section_title": current_section_title,
            "text": para
        })
    return sectioned_paragraphs

# ===============================
# 4. Intelligent Chunking
# ===============================
def chunk_paragraphs(sectioned_paragraphs, max_words=300, overlap=50):
    """
    Intelligent chunking strategy that balances content completeness with retrieval quality.
    
    Strategy:
    1. Short paragraphs (< 300 words): Keep as single chunks
    2. Medium paragraphs (300-600 words): Split with moderate overlap
    3. Long paragraphs (> 600 words): Split with semantic boundaries when possible
    """
    chunks = []
    chunk_id = 0
    
    for para in tqdm(sectioned_paragraphs, desc="Chunking paragraphs"):
        words = para["text"].split()
        word_count = len(words)
        
        # Strategy 1: Short paragraphs - keep complete
        if word_count <= max_words:
            chunks.append({
                "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                "page_number": para["page_number"],
                "section_number": para["section_number"],
                "section_title": para["section_title"],
                "text": para["text"]
            })
            chunk_id += 1
            
        # Strategy 2: Medium paragraphs - simple split
        elif word_count <= max_words * 2:
            mid_point = word_count // 2
            chunk1_text = " ".join(words[:mid_point + overlap//2])
            chunk2_text = " ".join(words[mid_point - overlap//2:])
            
            chunks.append({
                "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                "page_number": para["page_number"],
                "section_number": para["section_number"],
                "section_title": para["section_title"],
                "text": f"[PART 1/2] {chunk1_text}"
            })
            chunk_id += 1
            
            chunks.append({
                "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                "page_number": para["page_number"],
                "section_number": para["section_number"],
                "section_title": para["section_title"],
                "text": f"[PART 2/2] {chunk2_text}"
            })
            chunk_id += 1
            
        # Strategy 3: Long paragraphs - semantic chunking
        else:
            # Try to find natural break points (sentences ending with periods)
            sentences = para["text"].split('. ')
            if len(sentences) > 1:
                # Group sentences into chunks
                current_chunk = []
                current_word_count = 0
                chunk_num = 1
                total_chunks = (word_count + max_words - 1) // max_words
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if current_word_count + len(sentence_words) <= max_words:
                        current_chunk.append(sentence)
                        current_word_count += len(sentence_words)
                    else:
                        # Save current chunk
                        if current_chunk:
                            chunk_text = '. '.join(current_chunk)
                            if not chunk_text.endswith('.'):
                                chunk_text += '.'
                            
                            chunks.append({
                                "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                                "page_number": para["page_number"],
                                "section_number": para["section_number"],
                                "section_title": para["section_title"],
                                "text": f"[PART {chunk_num}/{total_chunks}] {chunk_text}"
                            })
                            chunk_id += 1
                            chunk_num += 1
                        
                        # Start new chunk
                        current_chunk = [sentence]
                        current_word_count = len(sentence_words)
                
                # Add final chunk
                if current_chunk:
                    chunk_text = '. '.join(current_chunk)
                    if not chunk_text.endswith('.'):
                        chunk_text += '.'
                    
                    chunks.append({
                        "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                        "page_number": para["page_number"],
                        "section_number": para["section_number"],
                        "section_title": para["section_title"],
                        "text": f"[PART {chunk_num}/{total_chunks}] {chunk_text}"
                    })
                    chunk_id += 1
            else:
                # Fallback to word-based chunking for very long single sentences
                start = 0
                chunk_num = 1
                total_chunks = (word_count + max_words - 1) // max_words
                
                while start < len(words):
                    end = min(start + max_words, len(words))
                    chunk_text = " ".join(words[start:end])
                    
                    chunks.append({
                        "chunk_id": f"{para['section_number']}_chunk{chunk_id}",
                        "page_number": para["page_number"],
                        "section_number": para["section_number"],
                        "section_title": para["section_title"],
                        "text": f"[PART {chunk_num}/{total_chunks}] {chunk_text}"
                    })
                    chunk_id += 1
                    chunk_num += 1
                    
                    if end >= len(words):
                        break
                    start += max_words - overlap
    
    return chunks

# ===============================
# 5. Embedding Generation with GPU Support
# ===============================
def generate_embeddings(chunks, model_name="all-MPNet-base-v2"):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    if device == "cuda":
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize model with device
    model = SentenceTransformer(model_name, device=device)
    
    texts = [chunk["text"] for chunk in chunks]
    print(f"📝 Processing {len(texts)} text chunks...")
    
    # Generate embeddings with GPU acceleration
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32 if device == "cuda" else 8,  # Larger batch size for GPU
        device=device
    )
    
    # Store embeddings in chunks (optional)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding_vector"] = emb.tolist()
    
    print(f"✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    return chunks, np.array(embeddings)

# ===============================
# 6. FAISS Index with GPU Support
# ===============================
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    print(f"🔍 Building FAISS index with {embeddings.shape[0]} vectors of dimension {dim}")
    
    # Check if GPU is available for FAISS
    if torch.cuda.is_available():
        try:
            # Try to use GPU for FAISS
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embeddings.astype('float32'))
            
            # Convert back to CPU for saving
            index = faiss.index_gpu_to_cpu(gpu_index)
            print("✅ FAISS index built with GPU acceleration")
        except Exception as e:
            print(f"⚠️ GPU FAISS failed, using CPU: {e}")
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings.astype('float32'))
    else:
        # Use CPU
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        print("✅ FAISS index built on CPU")
    
    return index

# ===============================
# 7. Query Function
# ===============================
def query_index(query, index, chunks, model, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=top_k)
    results = []
    for idx in I[0]:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "page_number": chunks[idx]["page_number"],
            "section_number": chunks[idx]["section_number"],
            "section_title": chunks[idx]["section_title"],
            "text": chunks[idx]["text"]
        })
    return results

# ===============================
# 8. Main workflow with GPU setup
# ===============================
if __name__ == "__main__":
    # GPU Setup and Information
    print("🚀 3GPP Embedding Generator with GPU Support")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ No GPU detected, using CPU (will be slower)")
    
    print("=" * 50)
    
    pdf_path = "ts_138331v160301p.pdf"

    print("📄 Loading PDF...")
    pages = load_pdf(pdf_path)

    print("Cleaning text...")
    cleaned_paragraphs = clean_text(pages)

    print("Detecting sections...")
    sectioned_paragraphs = detect_sections(cleaned_paragraphs)

    print("Chunking paragraphs...")
    chunks = chunk_paragraphs(sectioned_paragraphs)

    print("Generating embeddings...")
    chunks, embeddings = generate_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save embeddings separately
    np.save("embeddings.npy", embeddings)

    # Save JSON metadata without NumPy arrays
    chunks_metadata = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        # Remove embedding if you don't want huge JSON
        if "embedding_vector" in chunk_copy:
            del chunk_copy["embedding_vector"]
        chunks_metadata.append(chunk_copy)

    with open("chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks_metadata, f, indent=2)

    # Save FAISS index
    faiss.write_index(index, "faiss_index.faiss")
    
    print(f"\n✅ Embedding generation complete!")
    print(f"📊 Generated {len(chunks)} chunks")
    print(f"📊 Embedding dimension: {embeddings.shape[1]}")
    print(f"💾 Saved embeddings.npy and chunks_metadata.json")
    print(f"💾 Saved FAISS index as faiss_index.faiss")
    
    # Example query
    # query_text = "No AMF associated with gNB"
    # model = SentenceTransformer("all-MPNet-base-v2")
    # results = query_index(query_text, index, chunks, model)
    # print("\nTop results for query:", query_text)
    # for res in results:
    #     print(f"Page {res['page_number']}, Section {res['section_number']} - {res['section_title']}")
    #     print(res['text'] + "...\n")