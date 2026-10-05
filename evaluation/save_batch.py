import pandas as pd
import json
import sys
from src.config import RELEVANCE_JUDGMENTS_PATH, EVAL_QUERIES_PATH, RAW_DATA_PATH
import os

AUDIT_PATH = os.path.join(os.path.dirname(RELEVANCE_JUDGMENTS_PATH), "relevance_audit.csv")

def save_batch():
    judgments_df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    
    if os.path.exists(AUDIT_PATH):
        audit_df = pd.read_csv(AUDIT_PATH)
        audit_records = audit_df.to_dict('records')
    else:
        audit_records = []
        
    # Read stdin for JSON
    input_data = sys.stdin.read()
    try:
        new_judgments = json.loads(input_data)
    except Exception as e:
        print("Failed to parse JSON:", e)
        return
        
    for item in new_judgments:
        qid = item['query_id']
        doc_id = item['document_id']
        relevance = item['relevance']
        confidence = item['confidence']
        reason = item['reason']
        
        audit_records.append({
            "query_id": qid,
            "document_id": doc_id,
            "relevance": relevance,
            "confidence": confidence,
            "reason": reason
        })
        
        # update relevance judgments
        idx = judgments_df[(judgments_df['query_id'] == qid) & (judgments_df['document_id'] == doc_id)].index
        judgments_df.loc[idx, 'relevance'] = relevance
        
    pd.DataFrame(audit_records).to_csv(AUDIT_PATH, index=False)
    judgments_df.to_csv(RELEVANCE_JUDGMENTS_PATH, index=False)
    print(f"Saved {len(new_judgments)} judgments. Total judged: {len(audit_records)}")

if __name__ == "__main__":
    save_batch()
