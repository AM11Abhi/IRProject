"""
Interpolated Precision-Recall curve for the four retrieval models.

Put this file in IRProject/evaluation/ and run it from anywhere:
    python evaluation/pr_curve.py

It uses the same queries and relevance judgments as evaluate.py
(queries with zero relevant documents are skipped, unjudged documents
count as non-relevant) and saves:
    outputs/graphs/precision_recall_curve.png
    outputs/pr_curve_values.csv
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.retrievers import search
from src.config import EVAL_QUERIES_PATH, RELEVANCE_JUDGMENTS_PATH, GRAPHS_DIR

MODELS = ["tfidf", "bm25", "dense", "rrf"]
LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense (SBERT)", "rrf": "Hybrid RRF"}
RETRIEVE_DEPTH = 50                      # deep enough for recall to reach its maximum
RECALL_LEVELS = np.linspace(0, 1, 11)    # standard 11-point interpolation


def interpolated_precision(rels, n_relevant):
    """Return interpolated precision at the 11 standard recall levels."""
    rels = np.asarray(rels, dtype=float)
    hits = np.cumsum(rels)
    precision = hits / (np.arange(len(rels)) + 1)
    recall = hits / n_relevant
    out = []
    for r in RECALL_LEVELS:
        mask = recall >= r - 1e-9
        out.append(precision[mask].max() if mask.any() else 0.0)
    return np.array(out)


def main():
    judgments = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    queries = pd.read_csv(EVAL_QUERIES_PATH)

    curves = {m: [] for m in MODELS}
    for _, row in queries.iterrows():
        qj = judgments[judgments["query_id"] == row["query_id"]]
        relevant = set(qj[qj["relevance"] >= 1]["document_id"].values)
        if not relevant:
            continue  # same rule as evaluate.py
        for m in MODELS:
            hits = search(row["query"], model=m, top_k=RETRIEVE_DEPTH)
            rels = [1 if h["document_id"] in relevant else 0 for h in hits]
            if len(rels) < RETRIEVE_DEPTH:
                rels += [0] * (RETRIEVE_DEPTH - len(rels))
            curves[m].append(interpolated_precision(rels, len(relevant)))

    n_queries = len(curves[MODELS[0]])
    mean_curves = {m: np.mean(curves[m], axis=0) for m in MODELS}

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5.5))
    styles = {"tfidf": "o--", "bm25": "s-", "dense": "^-", "rrf": "D-"}
    for m in MODELS:
        plt.plot(RECALL_LEVELS, mean_curves[m], styles[m], label=LABELS[m], linewidth=2, markersize=5)
    plt.xlabel("Recall")
    plt.ylabel("Interpolated precision")
    plt.title(f"11-point Precision-Recall Curve (mean over {n_queries} queries)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(GRAPHS_DIR, "precision_recall_curve.png")
    plt.savefig(out_png, dpi=200)

    table = pd.DataFrame({"recall": RECALL_LEVELS, **{LABELS[m]: mean_curves[m] for m in MODELS}})
    out_csv = os.path.join(ROOT, "outputs", "pr_curve_values.csv")
    table.round(4).to_csv(out_csv, index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {out_png}\nSaved {out_csv}")


if __name__ == "__main__":
    main()