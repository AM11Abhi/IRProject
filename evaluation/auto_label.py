import os
import sys
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import warnings
from src.config import EVAL_QUERIES_PATH, RELEVANCE_JUDGMENTS_PATH, RAW_DATA_PATH

warnings.filterwarnings('ignore')

AUDIT_PATH = os.path.join(os.path.dirname(RELEVANCE_JUDGMENTS_PATH), "relevance_audit.csv")
SAMPLE_PATH = os.path.join(os.path.dirname(RELEVANCE_JUDGMENTS_PATH), "manual_review_sample.csv")

def extract_reason(query, text, dense_model):
    """
    Extracts the most relevant sentence from the text to serve as the reason.
    """
    sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if len(s.strip()) > 20]
    if not sentences:
        return "No extractable text content."
        
    # Cap sentences for speed
    sentences = sentences[:50]
    
    q_emb = dense_model.encode([query])
    s_embs = dense_model.encode(sentences)
    
    # Cosine similarity
    similarities = np.dot(s_embs, q_emb.T).flatten()
    best_idx = np.argmax(similarities)
    
    return f"Key match: '{sentences[best_idx][:150]}...'"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_judgments():
    print("Loading datasets and local AI models...")
    # Use cross-encoder for actual relevance judging
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Use dense model for extracting reason sentences
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    queries_df = pd.read_csv(EVAL_QUERIES_PATH)
    judgments_df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    raw_docs_df = pd.read_csv(RAW_DATA_PATH)
    
    if "Unnamed: 0" in raw_docs_df.columns:
        raw_docs_df = raw_docs_df.rename(columns={"Unnamed: 0": "document_id", "descr": "text"})
    
    if os.path.exists(AUDIT_PATH):
        print(f"Found existing audit file at {AUDIT_PATH}. Resuming from checkpoint...")
        audit_df = pd.read_csv(AUDIT_PATH)
        judged_pairs = set(zip(audit_df['query_id'], audit_df['document_id']))
        audit_records = audit_df.to_dict('records')
    else:
        judged_pairs = set()
        audit_records = []
        
    total_pairs = len(judgments_df)
    print(f"Total pairs to judge: {total_pairs}. Already judged: {len(judged_pairs)}.")
    
    queries_dict = dict(zip(queries_df['query_id'], queries_df['query']))
    docs_dict = dict(zip(raw_docs_df['document_id'], raw_docs_df['text']))
    
    for idx, row in judgments_df.iterrows():
        qid = row['query_id']
        doc_id = row['document_id']
        
        if (qid, doc_id) in judged_pairs:
            continue
            
        qtext = queries_dict.get(qid, "")
        doctext = docs_dict.get(doc_id, "")
        
        # MS-MARCO CrossEncoder takes (query, text) pairs
        # We truncate document text to avoid exceeding max_length of 512 tokens
        truncated_text = doctext[:2000] 
        
        score = cross_encoder.predict([qtext, truncated_text])
        # Convert logit to probability
        confidence = float(sigmoid(score))
        
        # Threshold: MS MARCO models usually have positive logits for relevant, negative for non-relevant
        # 0.5 probability (logit 0) is a reasonable threshold
        relevance = 1 if score > 0 else 0
        
        # Force strict rule: confidence needs to be reasonably high to be 1
        # otherwise we prefer 0 (the user instructed: "prefer relevance=0 unless clearly satisfies")
        if confidence < 0.6:
            relevance = 0
            
        if relevance == 1:
            reason = extract_reason(qtext, truncated_text, dense_model)
        else:
            reason = "Document lacks substantial semantic match to query."
            
        audit_records.append({
            "query_id": qid,
            "document_id": doc_id,
            "relevance": relevance,
            "confidence": confidence,
            "reason": reason
        })
        
        judgments_df.at[idx, 'relevance'] = relevance
        
        # Checkpoint
        if len(audit_records) % 10 == 0 or len(audit_records) == total_pairs:
            pd.DataFrame(audit_records).to_csv(AUDIT_PATH, index=False)
            judgments_df.to_csv(RELEVANCE_JUDGMENTS_PATH, index=False)
            
        print(f"[{len(audit_records)}/{total_pairs}] {qid} -> {doc_id} : Rel={relevance} ({confidence:.2f})")

    # Final save
    pd.DataFrame(audit_records).to_csv(AUDIT_PATH, index=False)
    judgments_df.to_csv(RELEVANCE_JUDGMENTS_PATH, index=False)
    print("\nAll 511 candidate pairs judged successfully!")
    
    # 10. Validation
    print("\n--- VALIDATION ---")
    final_audit = pd.read_csv(AUDIT_PATH)
    final_judgments = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    
    assert len(final_judgments) == 511, f"Expected 511 rows, got {len(final_judgments)}"
    assert len(final_judgments['query_id'].unique()) == 25, "Expected 25 queries"
    assert not final_judgments[['query_id', 'document_id']].duplicated().any(), "Duplicate query-doc pairs found"
    assert final_judgments['relevance'].isin([0, 1, '0', '1', 0.0, 1.0]).all(), "Relevance must be strictly 0 or 1"
    assert not final_judgments['relevance'].isnull().any(), "Blanks found in relevance column"
    missing_docs = set(final_judgments['document_id']) - set(raw_docs_df['document_id'])
    assert not missing_docs, f"Missing document IDs in raw data: {missing_docs}"
    print("Validation passed!")
    
    # 11. Summary
    print("\n--- SUMMARY ---")
    total_judged = len(final_audit)
    num_relevant = final_audit['relevance'].sum()
    num_non = total_judged - num_relevant
    print(f"Total judged documents: {total_judged}")
    print(f"Number relevant: {num_relevant}")
    print(f"Number non-relevant: {num_non}")
    print(f"Relevant percentage: {(num_relevant/total_judged)*100:.2f}%")
    print("\nRelevant documents per query:")
    print(final_audit.groupby('query_id')['relevance'].sum())
    
    # 12. Manual Review Sample (Stratified)
    print("\nGenerating Manual Review Sample...")
    
    rel_1 = final_audit[final_audit['relevance'] == 1]
    rel_0 = final_audit[final_audit['relevance'] == 0]
    
    low_conf_1 = rel_1.nsmallest(15, 'confidence')
    low_conf_0 = rel_0.nsmallest(15, 'confidence')
    
    rem_1_n = min(15, len(rel_1) - len(low_conf_1))
    rem_0_n = min(15, len(rel_0) - len(low_conf_0))
    
    rem_1 = rel_1.drop(low_conf_1.index).sample(n=rem_1_n, random_state=42) if rem_1_n > 0 else pd.DataFrame()
    rem_0 = rel_0.drop(low_conf_0.index).sample(n=rem_0_n, random_state=42) if rem_0_n > 0 else pd.DataFrame()
    
    sample_df = pd.concat([low_conf_1, low_conf_0, rem_1, rem_0]).sample(frac=1, random_state=42)
    
    sample_df = sample_df.merge(queries_df, on='query_id', how='left')
    sample_df = sample_df.merge(raw_docs_df[['document_id', 'text']], on='document_id', how='left')
    
    sample_df = sample_df[['query_id', 'query', 'document_id', 'text', 'relevance', 'confidence', 'reason']]
    sample_df = sample_df.rename(columns={'text': 'document_text'})
    
    sample_df.to_csv(SAMPLE_PATH, index=False)
    print(f"Saved manual review sample (size={len(sample_df)}) to {SAMPLE_PATH}")

if __name__ == "__main__":
    process_judgments()
