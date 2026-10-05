import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_PATH, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH
from src.preprocess import preprocess_text

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df = None
        
        # Load dataset
        if os.path.exists(PROCESSED_DATA_PATH):
            self.df = pd.read_csv(PROCESSED_DATA_PATH)
            self.df['clean_text'] = self.df['clean_text'].fillna("")
        else:
            raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run preprocess.py first.")
            
        self._initialize_model()
        
    def _initialize_model(self):
        """Fit the TF-IDF vectorizer if not saved, otherwise load it."""
        if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(TFIDF_MATRIX_PATH, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
        else:
            print("Fitting TF-IDF Vectorizer...")
            # Unigrams + bigrams, english stopwords, L2 norm by default
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['clean_text'])
            
            # Save for future use
            with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(TFIDF_MATRIX_PATH, 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
                
    def search(self, query: str, top_k: int = 10) -> list:
        """
        1. Preprocess query
        2. Transform query
        3. Cosine similarity
        4. Return top K
        """
        clean_query = preprocess_text(query)
        if not clean_query:
            return []
            
        query_vec = self.vectorizer.transform([clean_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = similarities[idx]
            if score == 0:
                continue # Skip documents with zero similarity
                
            doc = self.df.iloc[idx]
            results.append({
                "document_id": doc["document_id"],
                "score": score,
                "text": doc["text"],
                "tags": doc["tags"],
                "rank": rank
            })
            
        return results
