"""
Three retrieval methods being benchmarked:

1. TF-IDF cosine similarity: represents each chunk and the query as a
   TF-IDF vector, ranks chunks by cosine similarity to the query vector.
   Good baseline, but treats all documents the same length and doesn't
   handle term frequency saturation (a word appearing 20 times isn't
   20x as important as it appearing once).

2. BM25: a probabilistic ranking function, an improvement over raw TF-IDF.
   It saturates term frequency (diminishing returns for repeated words)
   and normalizes for document length, which usually makes it a stronger
   baseline than TF-IDF for retrieval tasks.

3. Hybrid: combines TF-IDF and BM25 scores (each normalized to [0, 1] first)
   using a weighted average. The idea is that different methods might
   surface different relevant chunks, so combining them can be more robust
   than either alone.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _normalize(scores):
    """Min-max normalize a score array to [0, 1], avoiding divide-by-zero."""
    scores = np.array(scores, dtype=float)
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def score(self, query):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        return sims

    def retrieve(self, query, top_k):
        sims = self.score(query)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def score(self, query):
        tokenized_query = query.lower().split()
        return np.array(self.bm25.get_scores(tokenized_query))

    def retrieve(self, query, top_k):
        scores = self.score(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


class DenseRetriever:
    """Semantic (embedding-based) retrieval using a sentence-transformer model.

    Unlike TF-IDF and BM25, which match on exact keyword overlap, this method
    embeds each chunk and the query into a dense vector space where meaning
    (not just wording) determines closeness. It can match a query like
    "how does a model learn from data" to a chunk about "training via
    backpropagation" even with no shared keywords -- something keyword-based
    methods cannot do.

    Downside: much slower to build (needs to run every chunk through a
    neural network) and requires downloading a pretrained model on first use.
    """

    # all-MiniLM-L6-v2: small, fast, strong general-purpose sentence embedding
    # model -- good default for a benchmark like this (384-dim vectors).
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, chunks, model=None):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.model = model if model is not None else SentenceTransformer(self.MODEL_NAME)
        self.doc_embeddings = self.model.encode(self.texts, convert_to_numpy=True)

    def score(self, query):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        return sims

    def retrieve(self, query, top_k):
        sims = self.score(query)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


class HybridRetriever:
    def __init__(self, chunks, tfidf_weight=0.5):
        self.chunks = chunks
        self.tfidf_retriever = TfidfRetriever(chunks)
        self.bm25_retriever = BM25Retriever(chunks)
        self.tfidf_weight = tfidf_weight

    def score(self, query):
        tfidf_scores = _normalize(self.tfidf_retriever.score(query))
        bm25_scores = _normalize(self.bm25_retriever.score(query))
        combined = (self.tfidf_weight * tfidf_scores +
                    (1 - self.tfidf_weight) * bm25_scores)
        return combined

    def retrieve(self, query, top_k):
        scores = self.score(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


RETRIEVER_REGISTRY = {
    "tfidf": TfidfRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "dense": DenseRetriever,
}
