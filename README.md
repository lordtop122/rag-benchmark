# RAG Retrieval Benchmark

A benchmarking system that compares four retrieval methods — **TF-IDF cosine similarity**, **BM25**, a **Hybrid** of the two, and **Dense (embedding-based) retrieval** — across different chunk sizes and top-k values, to determine which retrieval strategy works best for a Retrieval-Augmented Generation (RAG) pipeline.

## Motivation

Retrieval quality is the foundation of any RAG system: if the wrong text gets retrieved, the generated answer will be wrong no matter how good the language model is. This project treats retrieval as an empirical question rather than assuming any one method is "best" — it measures precision, recall, and ranking quality across methods and configurations, and lets the data decide.

## Project structure

## Corpus

The corpus consists of five short original documents covering related AI/ML topics: Machine Learning, Deep Learning, Natural Language Processing, Neural Networks, and Artificial Intelligence. The topics deliberately overlap in vocabulary (e.g. "neural network" appears in multiple documents), so retrieval has to work to distinguish which document is genuinely most relevant to a given query, rather than relying on unique keywords.

## Methodology

**Retrievers compared:**
- **TF-IDF**: represents each chunk and query as a TF-IDF vector, ranks by cosine similarity.
- **BM25**: a probabilistic ranking function that saturates term frequency and normalizes for document length.
- **Hybrid**: min-max normalizes both TF-IDF and BM25 scores, then combines them with a weighted average (default: equal weight).
- **Dense**: embeds each chunk and query into a dense vector using a pretrained sentence-transformer model (`all-MiniLM-L6-v2`), ranks by cosine similarity in embedding space. Unlike the other three methods, this matches on *meaning* rather than exact keyword overlap.

**Parameters swept:**
- Chunk size: 50, 100, 200 words (with 10-word overlap between chunks)
- Top-k: 1, 3, 5

**Metrics:**
- **Precision@k**: of the top-k retrieved chunks, what fraction came from the correct document?
- **Recall@k**: did the correct document appear anywhere in the top-k?
- **MRR (Mean Reciprocal Rank)**: rewards ranking the correct chunk near the top, not just anywhere in the top-k.

## Results

Full results in [`results/benchmark_results.csv`](results/benchmark_results.csv), visualized in [`results/benchmark_comparison.png`](results/benchmark_comparison.png).

**Key findings:**

- **Dense retrieval decisively outperformed all keyword-based methods**, achieving perfect Recall@k (1.000) and perfect MRR (1.000) across every chunk size and top-k tested. The correct document was found and ranked #1 every single time. This is the headline result: queries in this benchmark were phrased as natural questions (e.g. "How does backpropagation work?"), while the source documents used related but not identical vocabulary — exactly the kind of vocabulary mismatch that embedding-based retrieval is designed to handle, and keyword methods (TF-IDF, BM25) cannot.
- **Among the keyword-based methods, TF-IDF outperformed BM25** — a mildly counterintuitive result, since BM25 is generally considered the stronger baseline in retrieval literature. The likely explanation: BM25's main advantages — document-length normalization and term-frequency saturation — matter most when documents vary substantially in length. This corpus's documents are all similar, short lengths, so those advantages don't get to show up.
- **Hybrid retrieval tracked close to TF-IDF** but did not clearly outperform it, which makes sense given Hybrid is a direct average of the two underlying keyword-based methods (it does not include dense scores in this implementation).
- **Chunk size is a real trade-off for keyword methods, but less so for dense retrieval:** smaller chunks (50 words) give TF-IDF/BM25/Hybrid higher precision but lower ranking quality; larger chunks (200 words) give better MRR but lower precision. Dense retrieval's MRR and recall stayed perfect regardless of chunk size — its precision did drop at higher top-k (more semantically-related-but-not-exact chunks get pulled in), but its ability to find and rank the right chunk first was unaffected by chunk size in this benchmark.
- **Trade-off to note:** dense retrieval is substantially slower to build than the keyword-based methods, since it requires running every chunk through a neural network to generate embeddings, versus fast keyword counting for TF-IDF/BM25. For a small corpus like this the difference is trivial; at production scale it becomes a real engineering consideration (embedding caching, batching, approximate nearest-neighbor indexes).

## How to run

```bash
pip install -r requirements.txt
cd src
python3 run_benchmark.py
```

This regenerates `results/benchmark_results.csv`. Note: the first run will download the `all-MiniLM-L6-v2` sentence-transformer model (~90MB) automatically.

### Generation (optional)

The retrieval benchmark above is fully self-contained and requires no API access. A generation step is also included, which takes retrieved chunks and calls the Claude API to produce a full natural-language answer grounded in that text:

```bash
export ANTHROPIC_API_KEY="your-key-here"
python3 rag_pipeline.py "What is overfitting and how can it be reduced?"
```

Requires an API key from [console.anthropic.com](https://console.anthropic.com/settings/keys).

## Possible extensions

- Expand the corpus and query set for a more statistically robust comparison
- Add dense scores into the Hybrid retriever (currently Hybrid only combines TF-IDF + BM25)
- Test with a corpus that has genuine vocabulary overlap between queries and documents, to see whether keyword methods close the gap with dense retrieval under those conditions
- Evaluate generation quality (faithfulness, answer correctness) in addition to retrieval quality
- Benchmark retrieval latency/throughput alongside quality metrics, since dense retrieval's compute cost is a real trade-off in production
