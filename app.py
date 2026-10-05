import streamlit as st
import pandas as pd
from src.retrievers import search

st.set_page_config(page_title="BBC News IR System", layout="wide")

st.title("BBC News Information Retrieval System")
st.markdown("A simple demonstration of four retrieval models: TF-IDF, BM25, Dense Semantic (Sentence Transformers), and Reciprocal Rank Fusion (RRF).")

# Sidebar controls
st.sidebar.header("Search Settings")
query = st.sidebar.text_input("Enter your query:", "technology companies and stock market")
model = st.sidebar.selectbox("Select Model:", ["tfidf", "bm25", "dense", "rrf"], index=0)
top_k = st.sidebar.slider("Top K Results:", min_value=1, max_value=20, value=5)
compare_all = st.sidebar.checkbox("Compare all models side-by-side")

if st.sidebar.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        st.write(f"### Results for: *{query}*")
        
        if compare_all:
            cols = st.columns(4)
            models = ["tfidf", "bm25", "dense", "rrf"]
            
            for i, mod in enumerate(models):
                with cols[i]:
                    st.subheader(mod.upper())
                    try:
                        results = search(query, model=mod, top_k=top_k)
                        if not results:
                            st.write("No results found.")
                        for hit in results:
                            st.markdown(f"**Rank:** {hit['rank']} | **Score:** {hit['score']:.4f}")
                            st.markdown(f"**Doc ID:** {hit['document_id']}")
                            st.markdown(f"**Tags:** {hit['tags']}")
                            st.write(hit['text'][:200] + "...")
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error running {mod}: {str(e)}")
        else:
            try:
                results = search(query, model=model, top_k=top_k)
                
                if not results:
                    st.info("No results found.")
                else:
                    for hit in results:
                        with st.container():
                            st.markdown(f"#### Rank {hit['rank']} (Score: {hit['score']:.4f})")
                            st.markdown(f"**Document ID:** {hit['document_id']} | **Tags:** {hit['tags']}")
                            st.write(hit['text'])
                            st.markdown("---")
            except Exception as e:
                st.error(f"Error running {model}: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** Run `evaluation/generate_candidates.py` and `evaluation/evaluate.py` to evaluate these models offline.")
