# BBC News Information Retrieval System

A complete, reproducible Information Retrieval (IR) pipeline built over a categorized archive of 2,410 BBC News articles. The system implements and compares four distinct ranking strategies:

1. **TF-IDF with Cosine Similarity** (Unigrams + Bigrams, Porter Stemmed)
2. **Okapi BM25** (`rank_bm25` implementation with $k_1 = 1.5, b = 0.75$)
3. **Dense Semantic Retrieval** (`all-mpnet-base-v2`, 768-dim, with overlapping passage chunking & max-pooling)
4. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)** ($k = 60$, fusing the three base rankers)

The project also includes an interactive **Streamlit search interface** and an end-to-end evaluation benchmark measuring **Precision@10**, **Recall@10**, **Mean Average Precision (MAP)**, and **nDCG@10**, along with an **11-point Precision–Recall curve**.

---

## Performance Summary

Evaluated on 15 topical news queries with non-empty relevance sets against automated cross-encoder judgments:

| Model | Precision@10 | Recall@10 | MAP | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF** | 0.367 | 0.736 | 0.505 | 0.782 |
| **BM25** | **0.440** | 0.801 | 0.640 | 0.849 |
| **Dense (`all-mpnet-base-v2`)** | 0.373 | 0.666 | 0.473 | 0.750 |
| **Hybrid RRF** | **0.440** | **0.808** | **0.652** | **0.895** |

* **Hybrid RRF achieves the highest or tied-highest score across all four metrics**, attaining 0.652 MAP and 0.895 nDCG@10, with the lowest variance across queries ($\sigma = 0.248$).
* **BM25 performs strongly on keyword queries** (0.640 MAP), while **Dense retrieval excels on paraphrased queries** (e.g. 0.917 AP on *"music album sales charts"*).
* **RRF reliably bridges both paradigms**, matching or exceeding the median AP of the three base models on 12 of the 15 queries.

---

## Key Architecture & Features

### 1. Preprocessing & Indexing (`src/preprocess.py`)
- Uniform pipeline for documents and queries:
  1. Case folding (lowercasing)
  2. Punctuation stripping
  3. Whitespace normalization & tokenization
  4. Scikit-learn English stopword removal
  5. **Porter Stemming** via NLTK
- The dense retriever encodes the raw, natural text to retain grammatical cues and sentence context.

### 2. Passage Chunking for Dense Retrieval (`src/retrievers/dense_retriever.py`)
- Standard transformer models truncate inputs at 256–384 tokens, risking information loss on longer articles (max article length is 2,910 words).
- To eliminate truncation, documents are segmented into overlapping **150-word passages with a 30-word overlap** (8,293 chunks total across 2,410 articles).
- Encoded using `sentence-transformers/all-mpnet-base-v2` (768 dimensions).
- At query time, similarities are computed against all chunks and aggregated to document-level scores using **Max-Pooling** (`np.maximum.at`).

### 3. Reciprocal Rank Fusion (`src/retrievers/rrf_retriever.py`)
- Combines the top 50 ranked documents from TF-IDF, BM25, and Dense:
  $$\text{RRF}(d) = \sum_{r \in \{\text{TF-IDF, BM25, Dense}\}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$
- Documents missing from any retriever's list contribute $0$ for that ranker.

### 4. Interactive Search UI (`app.py`)
- Built with **Streamlit**.
- Offers single-model search and a **side-by-side comparison mode** displaying Rank, Score, Document ID, Category Tags, and Text snippets across all four models simultaneously.

---

## Project Structure

```text
IRProject/
├── app.py                             # Streamlit interactive search application
├── requirements.txt                   # Project dependencies (includes nltk, sentence-transformers)
├── data/
│   ├── raw/
│   │   └── BBCNews.csv                # Raw BBC News dataset (2,410 articles)
│   └── processed/
│       ├── processed_dataset.csv      # Cleaned and Porter-stemmed dataset
│       ├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer cache
│       ├── tfidf_matrix.pkl           # Precomputed TF-IDF sparse matrix
│       ├── dense_embeddings.npy       # Precomputed MPNet passage embeddings
│       └── dense_chunk_doc_ids.npy    # Chunk-to-document mapping array
├── src/
│   ├── config.py                      # Global path configuration
│   ├── preprocess.py                  # Text cleaning & Porter stemming
│   └── retrievers/
│       ├── tfidf_retriever.py         # TF-IDF sparse retriever
│       ├── bm25_retriever.py          # Okapi BM25 retriever
│       ├── dense_retriever.py         # MPNet passage-chunked dense retriever
│       ├── rrf_retriever.py           # Reciprocal Rank Fusion hybrid ranker
│       └── __init__.py                # Unified search API interface
├── evaluation/
│   ├── evaluation_queries.csv         # 25 topical evaluation queries
│   ├── generate_candidates.py         # Result pooling (top-10 TF-IDF, BM25, Dense)
│   ├── auto_label.py                  # Cross-encoder labelling (ms-marco-MiniLM-L-6-v2)
│   ├── relevance_judgments.csv        # Ground truth candidate relevance labels (499 pairs)
│   ├── relevance_audit.csv            # Labelling audit trail with confidence & explanations
│   ├── manual_review_sample.csv       # Stratified 60-pair sample for spot-checking
│   ├── evaluate.py                    # Metric computation (P@10, R@10, MAP, nDCG@10)
│   └── pr_curve.py                    # 11-point interpolated precision-recall curve
└── outputs/
    ├── metrics.csv                    # Aggregate performance metrics
    ├── per_query_metrics.csv          # Per-query evaluation metrics
    ├── error_analysis.csv             # Query win/loss case analysis
    ├── pr_curve_values.csv            # 11-point precision-recall table
    └── graphs/
        ├── precision_recall_curve.png # 11-point PR curve plot
        ├── all_metrics_comparison.png # Grouped metric bar chart
        └── tag_distribution.png       # Dataset category/tag distribution plot
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VarunPahuja/IRProject.git
   cd IRProject
   ```

2. **Set up a virtual environment (Python 3.8+ recommended):**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   venv\Scripts\activate

   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quickstart & Replication Workflow

Always execute evaluation and preprocessing scripts as Python modules using the `-m` flag from the `IRProject/` root directory so module imports resolve correctly:

### 1. Preprocess Dataset
Stem articles and verify dataset statistics:
```bash
python -m src.preprocess
```

### 2. Generate Candidate Pool (Result Pooling)
Collect top-10 candidates from TF-IDF, BM25, and Dense across 25 queries:
```bash
python -m evaluation.generate_candidates
```

### 3. Automated Relevance Labelling
Label candidate pairs using the cross-encoder (`ms-marco-MiniLM-L-6-v2` with threshold 0.6):
```bash
python -m evaluation.auto_label
```

### 4. Run Evaluation Benchmark
Compute P@10, Recall@10, MAP, and nDCG@10 across all four models:
```bash
python -m evaluation.evaluate
```

### 5. Generate Precision–Recall Curve
Generate the 11-point interpolated curve table and graph:
```bash
python -m evaluation.pr_curve
```

### 6. Launch the Streamlit Web App
Run the interactive search engine:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Evaluation Methodology & Statistical Notes

* **Pooling:** Evaluation uses standard trec-style result pooling. The top-10 hits from TF-IDF, BM25, and Dense are merged into a candidate pool (499 unique query–document pairs across 25 queries). RRF is excluded from the pool creation to avoid bias.
* **Scored Queries:** 10 of the 25 queries have zero relevant documents in the corpus (e.g. Premier League match results or Mars missions) and are excluded per standard IR methodology. Metrics are averaged over the 15 scored queries (containing 101 relevant pairs total).
* **Statistical Significance:** Paired Wilcoxon signed-rank tests ($\alpha = 0.05$) indicate nominal gains for BM25 over TF-IDF ($p = 0.041$), RRF over TF-IDF ($p = 0.043$), and RRF over Dense ($p = 0.013$). The difference between RRF and BM25 is not statistically significant ($p = 0.754$), showing that RRF achieves comparable top-end precision with higher consistency and lower cross-query variance.
