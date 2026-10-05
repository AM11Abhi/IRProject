import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import EVAL_QUERIES_PATH, RELEVANCE_JUDGMENTS_PATH, PROCESSED_DATA_PATH

st.set_page_config(page_title="Relevance Labeler", layout="wide")

st.title("IR Relevance Labeler")
st.markdown("Use this interface to manually label the 511 evaluation candidates. "
            "**Rule:** `1` if the document is directly relevant to the query and substantially answers it, "
            "`0` if irrelevant, loosely related, or just sharing a keyword.")

@st.cache_data
def load_data():
    queries = pd.read_csv(EVAL_QUERIES_PATH)
    judgments = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
    docs = pd.read_csv(PROCESSED_DATA_PATH)[['document_id', 'text', 'tags']]
    return queries, judgments, docs

queries, judgments, docs = load_data()

# Merge data for easier display
merged = judgments.merge(queries, on='query_id', how='left').merge(docs, on='document_id', how='left')

# Find the first unlabeled item
unlabeled = merged[merged['relevance'].isnull() | (merged['relevance'] == "")]

st.sidebar.header("Progress")
st.sidebar.write(f"Labeled: {len(merged) - len(unlabeled)} / {len(merged)}")
st.sidebar.progress((len(merged) - len(unlabeled)) / len(merged))

if len(unlabeled) == 0:
    st.success("All 511 candidate pairs have been labeled! You can now run the evaluation script.")
    st.balloons()
    
    if st.button("Verify Dataset Requirements"):
        j = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
        assert len(j) == 511, f"Expected 511 rows, got {len(j)}"
        assert j['relevance'].isin([0, 1, '0', '1', 0.0, 1.0]).all(), "All relevance values must be 0 or 1"
        assert not j['relevance'].isnull().any(), "No relevance values can be blank"
        assert len(j) == len(j.drop_duplicates(subset=['query_id', 'document_id'])), "No duplicate pairs allowed"
        assert len(j['query_id'].unique()) == 25, "All 25 queries must be represented"
        st.success("Verification passed: 511 valid labels across 25 queries!")
else:
    current = unlabeled.iloc[0]
    
    st.subheader(f"Query: {current['query']} ({current['query_id']})")
    st.markdown(f"**Document ID:** {current['document_id']}")
    st.markdown(f"**Tags:** {current['tags']}")
    
    # Simple heuristic to extract a 'title' or first sentence
    text = str(current['text'])
    title_end = text.find("  ")
    if title_end != -1 and title_end < 100:
        st.markdown(f"**Title (inferred):** {text[:title_end].strip()}")
        st.write(f"**Content:** {text[title_end:].strip()}")
    else:
        st.write(f"**Content:** {text}")
        
    st.markdown("---")
    
    col1, col2, _ = st.columns([1, 1, 4])
    
    def save_label(label):
        # Update the CSV file directly
        df = pd.read_csv(RELEVANCE_JUDGMENTS_PATH)
        idx = df[(df['query_id'] == current['query_id']) & (df['document_id'] == current['document_id'])].index
        df.loc[idx, 'relevance'] = label
        df.to_csv(RELEVANCE_JUDGMENTS_PATH, index=False)
        st.cache_data.clear()
        
    with col1:
        if st.button("👍 Relevant (1)", use_container_width=True, type="primary"):
            save_label(1)
            st.rerun()
            
    with col2:
        if st.button("👎 Not Relevant (0)", use_container_width=True):
            save_label(0)
            st.rerun()
            
st.sidebar.markdown("---")
st.sidebar.markdown("**Shortcuts aren't supported directly in this simple app, please click the buttons to label.**")
