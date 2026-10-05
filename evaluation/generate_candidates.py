import pandas as pd
import os
from src.retrievers import search
from src.config import QUALITATIVE_COMP_PATH, EVAL_QUERIES_PATH, RELEVANCE_JUDGMENTS_PATH

QUALITATIVE_QUERIES = [
    "technology companies and stock market prices", # Exact keyword/business
    "Apple product announcements", # Named entity / tech
    "football match results premier league", # Sports
    "british election prime minister vote", # Politics
    "oscar winning actors movies", # Entertainment
    "environmental impacts of global warming", # Semantic / paraphased
    "business strategies for making money", # Ambiguous / general
    "oil prices surge affecting economy", # Business
    "cyber security hackers attack networks", # Tech
    "cricket tournament champions", # Sports
    "budget cuts and taxes in parliament" # Politics
]

# We will use the same qualitative queries as our evaluation set for simplicity, 
# plus a few more to reach around 25-30 if needed, but 11 is a good start.
# Let's add more for the eval set to reach 25.

EVAL_ONLY_QUERIES = [
    "broadband internet speed in rural areas",
    "mobile phone battery technology",
    "new video game console releases",
    "music album sales charts",
    "inflation rate going up",
    "new hospital funding health care",
    "police investigation into robbery",
    "tennis grand slam tournament winners",
    "rugby world cup final",
    "space exploration mission to mars",
    "electric vehicles and battery production",
    "film director wins award",
    "theatre play opening night",
    "interest rates central bank decision"
]

ALL_EVAL_QUERIES = QUALITATIVE_QUERIES + EVAL_ONLY_QUERIES

def generate_qualitative_comparison():
    print("Running qualitative comparison on TF-IDF, BM25, and Dense...")
    results = []
    
    # We do top 5 for qualitative comparison
    models = ["tfidf", "bm25", "dense"]
    
    for q in QUALITATIVE_QUERIES:
        print(f"\nQuery: '{q}'")
        for model in models:
            hits = search(q, model=model, top_k=5)
            print(f"  [{model.upper()}] Top Result: {hits[0]['text'][:80]}..." if hits else f"  [{model.upper()}] No results.")
            for hit in hits:
                results.append({
                    "query": q,
                    "model": model,
                    "rank": hit["rank"],
                    "document_id": hit["document_id"],
                    "score": hit["score"],
                    "document_text": hit["text"],
                    "tags": hit["tags"]
                })
                
    df = pd.DataFrame(results)
    df.to_csv(QUALITATIVE_COMP_PATH, index=False)
    print(f"\nSaved qualitative comparison to {QUALITATIVE_COMP_PATH}")


def generate_evaluation_set():
    print("Generating reproducible evaluation set...")
    
    # Save queries
    queries_df = pd.DataFrame({
        "query_id": [f"q{str(i).zfill(2)}" for i in range(1, len(ALL_EVAL_QUERIES) + 1)],
        "query": ALL_EVAL_QUERIES
    })
    queries_df.to_csv(EVAL_QUERIES_PATH, index=False)
    
    # Generate candidate documents (union of top 10 from TF-IDF, BM25, Dense)
    judgments = []
    
    for _, row in queries_df.iterrows():
        qid = row['query_id']
        qtext = row['query']
        
        candidate_doc_ids = set()
        
        for model in ["tfidf", "bm25", "dense"]:
            hits = search(qtext, model=model, top_k=10)
            for hit in hits:
                candidate_doc_ids.add(hit["document_id"])
                
        for doc_id in candidate_doc_ids:
            judgments.append({
                "query_id": qid,
                "document_id": doc_id,
                "relevance": "" # To be filled manually
            })
            
    judgments_df = pd.DataFrame(judgments)
    judgments_df.to_csv(RELEVANCE_JUDGMENTS_PATH, index=False)
    
    print(f"Saved {len(ALL_EVAL_QUERIES)} queries to {EVAL_QUERIES_PATH}")
    print(f"Saved {len(judgments_df)} candidate pairs to {RELEVANCE_JUDGMENTS_PATH}")
    print("NOTE: You must manually inspect and set 'relevance' (0 or 1) in relevance_judgments.csv before calculating metrics.")

if __name__ == "__main__":
    generate_qualitative_comparison()
    generate_evaluation_set()
