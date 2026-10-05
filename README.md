# BBC News Information Retrieval System

This project is a college-level experimental Information Retrieval (IR) system using a small dataset of BBC News articles. 
It compares four retrieval approaches:
1. TF-IDF
2. BM25
3. Dense Semantic Retrieval (Sentence Transformers: `all-MiniLM-L6-v2`)
4. Reciprocal Rank Fusion (RRF) combining the three above.

## 1. Dataset
The dataset `BBCNews.csv` contains 2,410 BBC News documents. The original raw data is kept unmodified in `data/raw/BBCNews.csv`.

**Limitations:**
- The corpus is small.
- There are no official query or relevance benchmarks for this dataset, meaning we must construct evaluation queries and manual relevance judgments.
- Dense retrieval relies on a generic pretrained model, and no task-specific neural reranking is performed.
- RRF uses a fixed constant `k=60`.

## 2. Installation
Ensure you have Python 3.8+ installed.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Preprocessing
To clean the data and inspect dataset statistics, run:
```bash
python src/preprocess.py
```
This script handles lowercasing, punctuation removal, whitespace normalization, and stopword removal, creating `data/processed/processed_dataset.csv`.

## 4, 5, 6. Building Indices and Running Qualitative Search
Instead of building models manually, you can run the candidate generation script, which lazily initializes TF-IDF, BM25, and Dense embeddings and uses them to run qualitative search queries.
```bash
python evaluation/generate_candidates.py
```
This generates:
- `outputs/qualitative_comparison.csv`
- `evaluation/evaluation_queries.csv`
- `evaluation/relevance_judgments.csv` (requires manual labeling)

*(Note: The first time Dense Retrieval runs, it will download the SentenceTransformer model and compute embeddings for all documents, saving them to `data/processed/dense_embeddings.npy`.)*

## 8. Creating Relevance Judgments
You MUST manually open `evaluation/relevance_judgments.csv` and inspect the `relevance` column. 
For every query-document pair, enter:
- `1` if relevant
- `0` if not relevant

Evaluation metrics **cannot** be computed until this file is populated.

## 9 & 10. Running Evaluation (including RRF)
Once `relevance_judgments.csv` has labels (1 or 0), run:
```bash
python evaluation/evaluate.py
```
This calculates Precision@10, Recall@10, MAP, and nDCG@10 for all models, including the RRF hybrid retriever.

**Outputs generated:**
- `outputs/metrics.csv`
- `outputs/per_query_metrics.csv`
- `outputs/error_analysis.csv`
- `outputs/graphs/` (bar charts for metrics)

## 11. Running Streamlit
To interact with the search engine visually:
```bash
streamlit run app.py
```

## Student Workflow Handoff
1. Ensure the environment is installed (`pip install -r requirements.txt`)
2. Run `python src/preprocess.py`
3. Run `python evaluation/generate_candidates.py`
4. **MANUAL STEP:** Open `evaluation/relevance_judgments.csv` and add 1s and 0s to the relevance column.
5. Run `python evaluation/evaluate.py`
6. Inspect the generated metrics in `outputs/` and charts in `outputs/graphs/`
7. Present the findings based on actual metrics!
