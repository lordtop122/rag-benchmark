"""
Full RAG pipeline: retrieve relevant chunks for a query, then generate an
answer grounded in those chunks using the Claude API.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python rag_pipeline.py "What is overfitting?"

Uses the best-performing configuration found by the benchmark
(see results/benchmark_results.csv) as sensible defaults: TF-IDF retrieval,
chunk_size=200, top_k=3. Change these if you re-run the benchmark and find
a different config wins on your own corpus.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunking import load_documents, chunk_documents
from retrievers import RETRIEVER_REGISTRY
from generation import generate_answer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Defaults based on benchmark results — see results/benchmark_comparison.png
DEFAULT_METHOD = "tfidf"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_TOP_K = 3
DEFAULT_OVERLAP = 10


def build_pipeline(method=DEFAULT_METHOD, chunk_size=DEFAULT_CHUNK_SIZE,
                    overlap=DEFAULT_OVERLAP):
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    retriever_cls = RETRIEVER_REGISTRY[method]
    retriever = retriever_cls(chunks)
    return retriever


def answer_question(query, retriever, top_k=DEFAULT_TOP_K, verbose=True):
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    if verbose:
        print(f"\nRetrieved {len(retrieved_chunks)} chunks:")
        for c in retrieved_chunks:
            preview = c["text"][:80].replace("\n", " ")
            print(f"  [{c['chunk_id']}] {preview}...")

    answer = generate_answer(query, retrieved_chunks)
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag_pipeline.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    retriever = build_pipeline()
    answer = answer_question(query, retriever)

    print(f"\nQuestion: {query}")
    print(f"\nAnswer:\n{answer}")
