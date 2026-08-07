"""
Splits documents into chunks of a given size (in words), with optional overlap.

Why chunking matters for RAG: an LLM can't take a whole document collection as
context, so documents are split into smaller pieces (chunks), and retrieval
finds the most relevant chunks (not whole documents) for a given query.
Chunk size is a key tunable parameter: too small and chunks lose context,
too large and irrelevant text gets pulled in alongside the relevant part.
"""

import os
import glob


def load_documents(data_dir):
    """Load all .txt documents from a directory.

    Returns a list of dicts: {"doc_id": filename, "text": content}
    """
    documents = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "doc_id": os.path.basename(filepath),
            "text": text
        })
    return documents


def chunk_text(text, chunk_size, overlap=0):
    """Split a single text into word-based chunks.

    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks (helps avoid
             cutting a sentence's meaning in half at chunk boundaries)
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(documents, chunk_size, overlap=0):
    """Chunk a list of documents, tracking which document each chunk came from.

    Returns a list of dicts: {"chunk_id", "doc_id", "text"}
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk{i}",
                "doc_id": doc["doc_id"],
                "text": chunk
            })
    return all_chunks


if __name__ == "__main__":
    # quick sanity check when run directly
    docs = load_documents("../data")
    print(f"Loaded {len(docs)} documents")
    for size in [50, 100, 200]:
        chunks = chunk_documents(docs, chunk_size=size, overlap=10)
        print(f"chunk_size={size}: {len(chunks)} chunks total")
