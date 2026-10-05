import pandas as pd
import json
from src.config import EVAL_QUERIES_PATH, RELEVANCE_JUDGMENTS_PATH, RAW_DATA_PATH
import os

AUDIT_PATH = os.path.join(os.path.dirname(RELEVANCE_JUDGMENTS_PATH), "relevance_audit.csv")

def extract_batch(batch_size=50):
    queries_df = pd.read_csv(EVAL_QUERIES_PATH)
    judgments_df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    raw_docs_df = pd.read_csv(RAW_DATA_PATH)
    
    if "Unnamed: 0" in raw_docs_df.columns:
        raw_docs_df = raw_docs_df.rename(columns={"Unnamed: 0": "document_id", "descr": "text"})
        
    if os.path.exists(AUDIT_PATH):
        audit_df = pd.read_csv(AUDIT_PATH)
        judged_pairs = set(zip(audit_df['query_id'], audit_df['document_id']))
    else:
        judged_pairs = set()
        
    queries_dict = dict(zip(queries_df['query_id'], queries_df['query']))
    docs_dict = dict(zip(raw_docs_df['document_id'], raw_docs_df['text']))
    
    batch = []
    for idx, row in judgments_df.iterrows():
        qid = row['query_id']
        doc_id = row['document_id']
        if (qid, doc_id) not in judged_pairs:
            qtext = queries_dict.get(qid, "")
            doctext = docs_dict.get(doc_id, "")
            batch.append({
                "query_id": qid,
                "document_id": doc_id,
                "query": qtext,
                "text": doctext[:2000] # truncate to avoid excessive tokens
            })
            if len(batch) >= batch_size:
                break
                
    if not batch:
        print("DONE")
        return
        
    print(f"BATCH_START (Size: {len(batch)})")
    for item in batch:
        print(f"--- QID: {item['query_id']} | DOC: {item['document_id']} ---")
        print(f"QUERY: {item['query']}")
        print(f"TEXT: {item['text']}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    extract_batch()
