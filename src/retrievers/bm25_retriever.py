import os
import pandas as pd
from rank_bm25 import BM25Okapi
from src.config import PROCESSED_DATA_PATH
from src.preprocess import preprocess_text

class BM25Retriever:
    def __init__(self):
        self.df = None
        self.bm25 = None
        
        # Load dataset
        if os.path.exists(PROCESSED_DATA_PATH):
            self.df = pd.read_csv(PROCESSED_DATA_PATH)
            self.df['clean_text'] = self.df['clean_text'].fillna("")
        else:
            raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
            
        self._initialize_model()
        
    def _initialize_model(self):
        """
        BM25Okapi takes a list of tokenized documents.
        We will split the clean_text into tokens.
        """
        print("Initializing BM25 Index...")
        tokenized_corpus = [text.split() for text in self.df['clean_text']]
        self.bm25 = BM25Okapi(tokenized_corpus)
                
    def search(self, query: str, top_k: int = 10) -> list:
        """
        1. Preprocess query
        2. Tokenize query
        3. Run BM25
        4. Return top K
        """
        clean_query = preprocess_text(query)
        if not clean_query:
            return []
            
        tokenized_query = clean_query.split()
        
        # Get BM25 scores for all documents
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        # argsort sorts ascending, so we reverse it
        top_indices = doc_scores.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = doc_scores[idx]
            if score <= 0:
                continue # Skip docs with zero or negative BM25 score
                
            doc = self.df.iloc[idx]
            results.append({
                "document_id": doc["document_id"],
                "score": score,
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
            
        return results
