import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.retrievers import search
from src.config import (
    EVAL_QUERIES_PATH, 
    RELEVANCE_JUDGMENTS_PATH, 
    METRICS_PATH, 
    PER_QUERY_METRICS_PATH,
    ERROR_ANALYSIS_PATH,
    GRAPHS_DIR,
    PROCESSED_DATA_PATH
)

def dcg_at_k(r, k):
    """Discounted Cumulative Gain"""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k, ground_truth_count):
    """Normalized Discounted Cumulative Gain"""
    dcg_max = dcg_at_k(sorted(r, reverse=True) + [1] * max(0, ground_truth_count - sum(r)), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def average_precision(r, ground_truth_count):
    """Average Precision"""
    r = np.asarray(r)
    out = [np.sum(r[:k+1]) / (k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / ground_truth_count

def run_evaluation():
    if not os.path.exists(RELEVANCE_JUDGMENTS_PATH):
        print(f"ERROR: {RELEVANCE_JUDGMENTS_PATH} does not exist.")
        return
        
    judgments_df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    
    # Check if relevance labels are filled
    if judgments_df['relevance'].isnull().any() or judgments_df['relevance'].astype(str).str.strip().eq("").any():
        print("ERROR: relevance_judgments.csv contains empty or null relevance labels.")
        print("Please manually inspect the candidate pairs and assign 0 or 1 for relevance.")
        print("Metrics cannot be calculated until this is done.")
        return
        
    judgments_df['relevance'] = pd.to_numeric(judgments_df['relevance'], errors='coerce')
    if judgments_df['relevance'].isnull().any():
        print("ERROR: relevance labels must be numeric (0 or 1).")
        return
        
    queries_df = pd.read_csv(EVAL_QUERIES_PATH)
    
    models = ["tfidf", "bm25", "dense", "rrf"]
    
    per_query_results = []
    
    for _, row in queries_df.iterrows():
        qid = row['query_id']
        qtext = row['query']
        
        # Ground truth relevant documents for this query
        q_judgments = judgments_df[judgments_df['query_id'] == qid]
        relevant_docs = set(q_judgments[q_judgments['relevance'] >= 1]['document_id'].values)
        ground_truth_count = len(relevant_docs)
        
        if ground_truth_count == 0:
            # Skip query if there are no relevant documents at all in the judgments
            continue
            
        for model in models:
            # Retrieve top 10 for evaluation
            hits = search(qtext, model=model, top_k=10)
            
            # Binary relevance array for the retrieved hits
            r = [1 if hit["document_id"] in relevant_docs else 0 for hit in hits]
            
            # Calculate metrics
            k = 10
            r_at_k = r[:k]
            retrieved_relevant_at_k = sum(r_at_k)
            
            p_at_k = retrieved_relevant_at_k / k
            recall_at_k = retrieved_relevant_at_k / ground_truth_count if ground_truth_count > 0 else 0.
            ap = average_precision(r, ground_truth_count)
            ndcg = ndcg_at_k(r, k, ground_truth_count)
            
            # Top retrieved document info for error analysis
            top_doc_id = hits[0]["document_id"] if hits else None
            top_doc_text = hits[0]["text"] if hits else None
            top_doc_score = hits[0]["score"] if hits else None
            top_doc_rel = 1 if top_doc_id in relevant_docs else 0
            
            per_query_results.append({
                "query_id": qid,
                "query": qtext,
                "model": model,
                "precision_at_10": p_at_k,
                "recall_at_10": recall_at_k,
                "average_precision": ap,
                "ndcg_at_10": ndcg,
                "top_doc_id": top_doc_id,
                "top_doc_text": top_doc_text,
                "top_doc_score": top_doc_score,
                "top_doc_relevance": top_doc_rel
            })
            
    if not per_query_results:
        print("No evaluation data could be processed (maybe no relevant documents found for any query).")
        return
        
    pq_df = pd.DataFrame(per_query_results)
    
    # Save per_query_metrics.csv
    pq_df[['query_id', 'model', 'precision_at_10', 'recall_at_10', 'average_precision', 'ndcg_at_10']].to_csv(PER_QUERY_METRICS_PATH, index=False)
    
    # Aggregate metrics
    agg_df = pq_df.groupby('model')[['precision_at_10', 'recall_at_10', 'average_precision', 'ndcg_at_10']].mean().reset_index()
    agg_df = agg_df.rename(columns={'average_precision': 'map'})
    
    # Sort models by a specific order for consistency
    model_order = {'tfidf': 0, 'bm25': 1, 'dense': 2, 'rrf': 3}
    agg_df['order'] = agg_df['model'].map(model_order)
    agg_df = agg_df.sort_values('order').drop(columns=['order'])
    
    agg_df.to_csv(METRICS_PATH, index=False)
    
    print("\n--- FINAL EVALUATION METRICS ---")
    print(agg_df.to_string(index=False))
    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved per-query metrics to {PER_QUERY_METRICS_PATH}")
    
    # Perform error analysis
    generate_error_analysis(pq_df)
    
    # Generate graphs
    generate_graphs(agg_df)


def generate_error_analysis(pq_df):
    """
    Identifies queries where specific models win or lose,
    and saves interesting cases for the presentation.
    """
    analysis_records = []
    
    queries = pq_df['query_id'].unique()
    for qid in queries:
        q_data = pq_df[pq_df['query_id'] == qid]
        qtext = q_data.iloc[0]['query']
        
        # Determine best model by MAP
        best_row = q_data.loc[q_data['average_precision'].idxmax()]
        worst_row = q_data.loc[q_data['average_precision'].idxmin()]
        
        rrf_row = q_data[q_data['model'] == 'rrf'].iloc[0]
        
        # Is RRF worse than any base model?
        base_models_max_map = q_data[q_data['model'] != 'rrf']['average_precision'].max()
        rrf_failed = rrf_row['average_precision'] < base_models_max_map
        
        # Collect interesting facts
        analysis_records.append({
            "query_id": qid,
            "query": qtext,
            "best_model": best_row['model'],
            "best_map": best_row['average_precision'],
            "worst_model": worst_row['model'],
            "rrf_failed_vs_best_base": rrf_failed,
            "top_doc_by_best": best_row['top_doc_text'],
            "top_doc_relevance": best_row['top_doc_relevance']
        })
        
    analysis_df = pd.DataFrame(analysis_records)
    analysis_df.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis to {ERROR_ANALYSIS_PATH}")

def generate_graphs(agg_df):
    metrics = ['precision_at_10', 'recall_at_10', 'map', 'ndcg_at_10']
    titles = ['Precision@10', 'Recall@10', 'Mean Average Precision (MAP)', 'nDCG@10']
    
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Individual bar charts
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(8, 5))
        plt.bar(agg_df['model'].str.upper(), agg_df[metric], color=['skyblue', 'lightgreen', 'salmon', 'orchid'])
        plt.title(f'Model vs {title}')
        plt.ylabel(title)
        plt.ylim(0, 1.05) # Metrics are [0, 1]
        for i, val in enumerate(agg_df[metric]):
            plt.text(i, val + 0.02, f"{val:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{metric}.png"))
        plt.close()
        
    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agg_df['model']))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width - width*1.5, agg_df[metric], width, label=titles[i])
        
    ax.set_title('Overall Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(agg_df['model'].str.upper())
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "all_metrics_comparison.png"))
    plt.close()
    
    print(f"Saved visualization graphs to {GRAPHS_DIR}")

if __name__ == "__main__":
    run_evaluation()
