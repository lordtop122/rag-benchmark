"""
Runs the full benchmark: for every combination of chunk size, top-k, and
retrieval method, builds the retriever and evaluates it on the query set.
Results are saved to results/benchmark_results.csv for analysis.
"""

import os
import csv
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunking import load_documents, chunk_documents
from retrievers import RETRIEVER_REGISTRY
from evaluate import evaluate_retriever

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CHUNK_SIZES = [50, 100, 200]
TOP_K_VALUES = [1, 3, 5]
OVERLAP = 10


def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query": row["query"],
                "relevant_doc": row["relevant_doc"]
            })
    return queries


def main():
    documents = load_documents(DATA_DIR)
    queries = load_queries(os.path.join(DATA_DIR, "queries.csv"))
    print(f"Loaded {len(documents)} documents, {len(queries)} queries")

    results = []

    for chunk_size in CHUNK_SIZES:
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=OVERLAP)
        print(f"\nchunk_size={chunk_size}: {len(chunks)} chunks")

        for method_name, retriever_cls in RETRIEVER_REGISTRY.items():
            retriever = retriever_cls(chunks)

            for top_k in TOP_K_VALUES:
                metrics = evaluate_retriever(retriever, queries, top_k=top_k)
                row = {
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "method": method_name,
                    "num_chunks": len(chunks),
                    **metrics
                }
                results.append(row)
                print(f"  {method_name:8s} top_k={top_k}  "
                      f"P@k={metrics['precision@k']:.3f}  "
                      f"R@k={metrics['recall@k']:.3f}  "
                      f"MRR={metrics['mrr']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
