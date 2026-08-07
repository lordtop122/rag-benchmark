"""
Retrieval evaluation metrics. Each query has one known relevant document
(see data/queries.csv). Since documents are split into chunks, a retrieved
chunk counts as relevant if it came from the relevant document.

Metrics:
- Precision@k: of the top-k retrieved chunks, what fraction came from the
  relevant document? Measures how "clean" the results are.
- Recall@k: did the relevant document appear at all in the top-k? (binary
  per query, since here each query has exactly one relevant document)
- MRR (Mean Reciprocal Rank): 1 / (rank of the first relevant chunk).
  Rewards putting the relevant result near the top, not just somewhere
  in the top-k. MRR = 0 if the relevant document never appears.
"""

import numpy as np


def precision_at_k(retrieved_chunks, relevant_doc_id):
    if not retrieved_chunks:
        return 0.0
    hits = sum(1 for c in retrieved_chunks if c["doc_id"] == relevant_doc_id)
    return hits / len(retrieved_chunks)


def recall_at_k(retrieved_chunks, relevant_doc_id):
    hits = sum(1 for c in retrieved_chunks if c["doc_id"] == relevant_doc_id)
    return 1.0 if hits > 0 else 0.0


def reciprocal_rank(retrieved_chunks, relevant_doc_id):
    for rank, c in enumerate(retrieved_chunks, start=1):
        if c["doc_id"] == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(retriever, queries, top_k):
    """Run all queries through a retriever and average the metrics.

    queries: list of dicts {"query": ..., "relevant_doc": ...}
    """
    precisions, recalls, rr_scores = [], [], []

    for q in queries:
        retrieved = retriever.retrieve(q["query"], top_k=top_k)
        precisions.append(precision_at_k(retrieved, q["relevant_doc"]))
        recalls.append(recall_at_k(retrieved, q["relevant_doc"]))
        rr_scores.append(reciprocal_rank(retrieved, q["relevant_doc"]))

    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls),
        "mrr": np.mean(rr_scores),
    }
