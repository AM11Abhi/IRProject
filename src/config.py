import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "BBCNews.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_dataset.csv")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.csv")
PER_QUERY_METRICS_PATH = os.path.join(OUTPUTS_DIR, "per_query_metrics.csv")
QUALITATIVE_COMP_PATH = os.path.join(OUTPUTS_DIR, "qualitative_comparison.csv")
ERROR_ANALYSIS_PATH = os.path.join(OUTPUTS_DIR, "error_analysis.csv")

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
EVAL_QUERIES_PATH = os.path.join(EVAL_DIR, "evaluation_queries.csv")
RELEVANCE_JUDGMENTS_PATH = os.path.join(EVAL_DIR, "relevance_judgments.csv")

# Model Specific Artifacts
TFIDF_VECTORIZER_PATH = os.path.join(DATA_DIR, "processed", "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(DATA_DIR, "processed", "tfidf_matrix.pkl")
DENSE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "processed", "dense_embeddings.npy")
