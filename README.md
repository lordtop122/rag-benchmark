# RAG Retrieval Benchmark

A benchmarking system that compares three retrieval methods — **TF-IDF cosine similarity**, **BM25**, and a **Hybrid** of the two — across different chunk sizes and top-k values, to determine which retrieval strategy works best for a Retrieval-Augmented Generation (RAG) pipeline.

## Motivation

Retrieval quality is the foundation of any RAG system: if the wrong text gets retrieved, the generated answer will be wrong no matter how good the language model is. This project treats retrieval as an empirical question rather than assuming any one method is "best" — it measures precision, recall, and ranking quality across methods and configurations, and lets the data decide.

## Project structure

```
rag-benchmark/
├── data/
│   ├── doc1_machine_learning.txt
│   ├── doc2_deep_learning.txt
│   ├── doc3_nlp.txt
│   ├── doc4_neural_networks.txt
│   ├── doc5_artificial_intelligence.txt
│   └── queries.csv              # 14 test queries with ground-truth relevant document
├── src/
│   ├── chunking.py              # splits documents into word-based chunks with overlap
│   ├── retrievers.py            # TF-IDF, BM25, and Hybrid retriever implementations
│   ├── evaluate.py              # Precision@k, Recall@k, MRR metrics
│   ├── run_benchmark.py         # sweeps chunk_size x top_k x method, saves results
│   ├── generation.py            # calls Claude API to generate answers from retrieved chunks
│   └── rag_pipeline.py          # full retrieve -> generate pipeline (CLI)
├── results/
│   ├── benchmark_results.csv
│   └── benchmark_comparison.png
└── requirements.txt
```

## Corpus

The corpus consists of five short original documents covering related AI/ML topics: Machine Learning, Deep Learning, Natural Language Processing, Neural Networks, and Artificial Intelligence. The topics deliberately overlap in vocabulary (e.g. "neural network" appears in multiple documents), so retrieval has to work to distinguish which document is genuinely most relevant to a given query, rather than relying on unique keywords.

## Methodology

**Retrievers compared:**
- **TF-IDF**: represents each chunk and query as a TF-IDF vector, ranks by cosine similarity.
- **BM25**: a probabilistic ranking function that saturates term frequency and normalizes for document length.
- **Hybrid**: min-max normalizes both TF-IDF and BM25 scores, then combines them with a weighted average (default: equal weight).

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

- **TF-IDF outperformed BM25** on this corpus, which is somewhat counterintuitive since BM25 is generally considered the stronger baseline in retrieval literature. The likely explanation: BM25's main advantages — document-length normalization and term-frequency saturation — matter most when documents vary substantially in length. This corpus's documents are all similar, short lengths, so those advantages don't get to show up.
- **Hybrid retrieval tracked close to TF-IDF** but did not clearly outperform it, which makes sense given Hybrid is a direct average of the two underlying methods.
- **Chunk size is a real trade-off, not a free lunch:** smaller chunks (50 words) give higher precision (less irrelevant text per chunk) but a lower chance the full relevant context sits in one chunk. Larger chunks (200 words) gave the best MRR (relevant chunk more likely to rank #1) but the lowest precision at higher top-k, since larger chunks are more likely to contain a mix of relevant and irrelevant text.
- **Best overall configuration:** TF-IDF, chunk_size=200, top_k=3 (MRR = 0.952).

## How to run

```bash
pip install -r requirements.txt
cd src
python3 run_benchmark.py
```

This regenerates `results/benchmark_results.csv`.

### Generation (optional)

The retrieval benchmark above is fully self-contained and requires no API access. A generation step is also included, which takes retrieved chunks and calls the Claude API to produce a full natural-language answer grounded in that text:

```bash
export ANTHROPIC_API_KEY="your-key-here"
python3 rag_pipeline.py "What is overfitting and how can it be reduced?"
```

Requires an API key from [console.anthropic.com](https://console.anthropic.com/settings/keys).

## Possible extensions

- Expand the corpus and query set for a more statistically robust comparison
- Add dense/embedding-based retrieval as a fourth method for comparison
- Tune the Hybrid weighting parameter rather than using a fixed 50/50 split
- Evaluate generation quality (faithfulness, answer correctness) in addition to retrieval quality
