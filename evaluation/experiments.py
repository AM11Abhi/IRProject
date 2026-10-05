import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from src.retrievers.rrf_retriever import RRFRetriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.config import (
    EVAL_QUERIES_PATH, 
    RELEVANCE_JUDGMENTS_PATH,
    PROCESSED_DATA_PATH
)
from evaluation.evaluate import average_precision, ndcg_at_k

def evaluate_retriever(retriever, name, queries_df, judgments_df, top_k=10):
    per_query_results = []
    
    for _, row in queries_df.iterrows():
        qid = row['query_id']
        qtext = row['query']
        
        q_judgments = judgments_df[judgments_df['query_id'] == qid]
        relevant_docs = set(q_judgments[q_judgments['relevance'] >= 1]['document_id'].values)
        ground_truth_count = len(relevant_docs)
        
        if ground_truth_count == 0:
            continue
            
        hits = retriever.search(qtext, top_k=top_k)
        r = [1 if hit["document_id"] in relevant_docs else 0 for hit in hits]
        
        k = top_k
        r_at_k = r[:k]
        retrieved_relevant_at_k = sum(r_at_k)
        
        p_at_k = retrieved_relevant_at_k / k
        recall_at_k = retrieved_relevant_at_k / ground_truth_count if ground_truth_count > 0 else 0.
        ap = average_precision(r, ground_truth_count)
        ndcg = ndcg_at_k(r, k, ground_truth_count)
        
        per_query_results.append({
            "model": name,
            "precision_at_10": p_at_k,
            "recall_at_10": recall_at_k,
            "map": ap,
            "ndcg_at_10": ndcg,
        })
        
    pq_df = pd.DataFrame(per_query_results)
    agg_df = pq_df.groupby('model')[['precision_at_10', 'recall_at_10', 'map', 'ndcg_at_10']].mean().reset_index()
    return agg_df

class RRFBM25Dense(RRFRetriever):
    def __init__(self, k: int = 60):
        # Override to remove tfidf
        self.k = k
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        
    def search(self, query: str, top_k: int = 10) -> list:
        candidate_pool_size = max(50, top_k * 2)
        bm25_res = self.bm25.search(query, top_k=candidate_pool_size)
        dense_res = self.dense.search(query, top_k=candidate_pool_size)
        
        doc_scores = {}
        def add_to_rrf(results_list):
            for res in results_list:
                doc_id = res["document_id"]
                rank = res["rank"]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0.0,
                        "text": res["text"],
                        "tags": res["tags"],
                        "document_id": doc_id
                    }
                doc_scores[doc_id]["score"] += 1.0 / (self.k + rank)
                
        add_to_rrf(bm25_res)
        add_to_rrf(dense_res)
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        final_results = []
        for rank, doc in enumerate(sorted_docs[:top_k], start=1):
            final_results.append({
                "document_id": doc["document_id"],
                "score": doc["score"],
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
        return final_results

class DenseTagsRetriever(DenseRetriever):
    def _initialize_model(self):
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        print("Computing dense embeddings for Tags + Text. This might take a minute...")
        # Use tags + text
        self.df['tags'] = self.df['tags'].fillna("")
        texts_to_encode = (self.df['tags'] + " " + self.df['text']).tolist()
        self.embeddings = self.model.encode(texts_to_encode, show_progress_bar=False)

def run_experiments():
    queries_df = pd.read_csv(EVAL_QUERIES_PATH)
    judgments_df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    
    # Check judgments
    if judgments_df['relevance'].isnull().any():
        print("ERROR: relevance judgments missing.")
        return
    judgments_df['relevance'] = pd.to_numeric(judgments_df['relevance'], errors='coerce')
    
    results = []

    print("\n--- EXPERIMENT 1: Tune RRF k ---")
    for k in [10, 30, 60, 100]:
        print(f"Evaluating RRF(k={k})...")
        retriever = RRFRetriever(k=k)
        res = evaluate_retriever(retriever, f"RRF (k={k})", queries_df, judgments_df)
        results.append(res)
        
    print("\n--- EXPERIMENT 2: Tune Dense retrieval preprocessing ---")
    print("Evaluating Dense (Raw Text)...")
    dense_raw = DenseRetriever()
    res_raw = evaluate_retriever(dense_raw, "Dense (Raw Text)", queries_df, judgments_df)
    results.append(res_raw)
    
    print("Evaluating Dense (Tags + Text)...")
    dense_tags = DenseTagsRetriever()
    res_tags = evaluate_retriever(dense_tags, "Dense (Tags + Text)", queries_df, judgments_df)
    results.append(res_tags)
    
    print("\n--- EXPERIMENT 3: Try RRF with only useful models (BM25 + Dense) ---")
    print("Evaluating RRF (BM25 + Dense, k=60)...")
    rrf_bm25_dense = RRFBM25Dense(k=60)
    res_bm25_dense = evaluate_retriever(rrf_bm25_dense, "RRF (BM25 + Dense, k=60)", queries_df, judgments_df)
    results.append(res_bm25_dense)
    
    # Print combined results
    final_df = pd.concat(results, ignore_index=True)
    print("\n=== ALL EXPERIMENT RESULTS ===")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_experiments()
